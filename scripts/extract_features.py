"""Extract frozen FaRL features without training, changing splits or fitting transforms."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from agelens.farl import encode, encoder_state_sha256, file_sha256, load_encoder
from agelens.feature_cache import (
    PARTITIONS,
    ROW_COLUMNS,
    feature_stats,
    fingerprint,
    make_indices,
    safe_raw_path,
    verify_cache,
    write_json_atomic,
)
from scripts.audit_data import validate_manifests
from scripts.prepare_assets import ROOT


def load_inputs():
    protocol = json.loads((ROOT / "configs/protocol-v1.json").read_text())
    config = json.loads((ROOT / "configs/features-v1.json").read_text())
    required = {
        "input_resolution": 224,
        "output_dimensions": 512,
        "dtype": "float32",
        "normalization": "per_row_L2_no_fitted_statistics",
        "augmentation": "none",
        "preprocess": "EXIF_transpose_RGB_then_pinned_official_CLIP_transform",
        "encoder_mode": "eval_inference_mode_requires_grad_false",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Configuration is not supported by this extraction implementation")
    if config.get("device") not in {"cpu", "cuda"}:
        raise ValueError("Device must be explicitly cpu or cuda")
    if any(
        config[key] < 1
        for key in ["batch_size", "checkpoint_rows", "torch_threads", "preprocessing_threads"]
    ):
        raise ValueError("Extraction sizes and worker counts must be positive")
    frozen = protocol["data_audit"]
    for name, expected in frozen["manifest_sha256"].items():
        if file_sha256(ROOT / "data/manifests" / name) != expected:
            raise ValueError(f"Frozen manifest changed: {name}")
    if file_sha256(ROOT / "configs/audit-v1.json") != frozen["audit_policy_sha256"]:
        raise ValueError("Audit policy changed")
    if file_sha256(ROOT / "scripts/audit_data.py") != frozen["audit_code_sha256"]:
        raise ValueError("Audit implementation changed")
    rows = pd.read_csv(ROOT / "data/manifests/selected.csv")
    validate_manifests(rows, protocol["dataset"]["max_total_samples"])
    if rows.split.value_counts().to_dict() != frozen["split_sizes"]:
        raise ValueError("Frozen partition counts changed")
    if not protocol["backbone"]["frozen"] or protocol["trained"] or config["trained"]:
        raise ValueError("Frozen untrained protocol required")
    for partition in PARTITIONS:
        part = pd.read_csv(ROOT / f"data/manifests/{partition}.csv")
        if not np.array_equal(part.sample_id, rows.loc[rows.split == partition, "sample_id"]):
            raise ValueError("Partition row order differs from frozen selection")
    return protocol, config, rows


def setup_torch(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; refuse silent CPU fallback")
    if config["device"] == "cuda":
        cache = ROOT / "runtime/cuda-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_CACHE_PATH"] = str(cache)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def image_tensor(row, preprocess):
    path = safe_raw_path(ROOT, row.relative_path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row.image_sha256:
        raise ValueError("Audited input image bytes changed")
    with Image.open(io.BytesIO(content)) as image:
        return preprocess(ImageOps.exif_transpose(image).convert("RGB"))


def make_spec(protocol, config):
    import clip
    import torch

    return {
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "device": config["device"],
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0))
            if config["device"] == "cuda"
            else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "checkpoint_sha256": protocol["backbone"]["checkpoint_sha256"],
        "frozen_manifests": protocol["data_audit"]["manifest_sha256"],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "clip", "numpy", "Pillow", "pandas"]
        },
        "clip_source_sha256": file_sha256(Path(clip.__file__).parent / "clip.py"),
        "clip_model_source_sha256": file_sha256(Path(clip.__file__).parent / "model.py"),
        "implementation": {
            name: file_sha256(ROOT / name)
            for name in [
                "src/agelens/farl.py",
                "src/agelens/feature_cache.py",
                "scripts/extract_features.py",
            ]
        },
    }


def benchmark(encoder, preprocess, rows, config):
    import torch

    sizes = [4, 8, 16, 32] if config["device"] == "cuda" else [4, 8, 16]
    sample = rows[rows.split == "train"].head(max(sizes))
    batch = torch.stack([image_tensor(row, preprocess) for row in sample.itertuples()])
    results = []
    for threads in [4] if config["device"] == "cuda" else [4, 8]:
        torch.set_num_threads(threads)
        for size in sizes:
            encode(encoder, batch[:size])
            start = time.perf_counter()
            for _ in range(2):
                encode(encoder, batch[:size])
            seconds = (time.perf_counter() - start) / 2
            results.append(
                {
                    "threads": threads,
                    "batch_size": size,
                    "seconds": round(seconds, 4),
                    "images_per_second": round(size / seconds, 3),
                }
            )
            print(json.dumps(results[-1]), flush=True)
    torch.set_num_threads(config["torch_threads"])
    path = ROOT / "artifacts/private/features-benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"train_images_only": True, "results": results, "trained": False})


def publish_cache_result(directory, selected, metadata):
    identity = metadata["fingerprint"]
    metadata_digest = file_sha256(directory / "metadata.json")
    output = ROOT / "reports/task05"
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output / "extraction.json",
        {
            "task": "05",
            "cache_id": identity[:32],
            "fingerprint": identity,
            "samples": len(selected),
            "split_sizes": selected.split.value_counts().to_dict(),
            "extraction_runtime": metadata["spec"].get("runtime", {}),
            "library_versions": metadata["spec"].get("versions", {}),
            "feature_statistics": metadata["feature_statistics"],
            "encoder_state_unchanged": metadata["encoder_state_before"]
            == metadata["encoder_state_after"],
            "checkpoint_precision_preserved": metadata["encoder"].get(
                "checkpoint_precision_preserved", False
            ),
            "eval_mode": metadata["eval_mode"],
            "all_parameters_frozen": metadata["all_parameters_frozen"],
            "gradients_absent": metadata["gradients_absent"],
            "no_fitted_transforms": True,
            "metadata_sha256": metadata_digest,
            "file_sha256": metadata["file_sha256"],
            "elapsed_seconds": metadata["elapsed_this_session_seconds"],
            "trained": False,
            "features_private": True,
            "commit_created": False,
            "push_executed": False,
        },
    )
    write_json_atomic(
        directory.parent / "active.json",
        {"cache_id": identity[:32], "fingerprint": identity, "metadata_sha256": metadata_digest},
    )


def extract(benchmark_only=False):
    import torch

    protocol, config, selected = load_inputs()
    setup_torch(config)
    registry = json.loads((ROOT / "configs/assets.json").read_text())
    checkpoint = ROOT / registry["backbone"]["path"]
    encoder, preprocess, encoder_info = load_encoder(
        checkpoint, protocol["backbone"]["checkpoint_sha256"], device=config["device"]
    )
    if benchmark_only:
        benchmark(encoder, preprocess, selected, config)
        return
    spec = make_spec(protocol, config)
    identity = fingerprint(spec)
    base = ROOT / "data/processed/farl_ep16"
    directory = base / identity[:32]
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        _, _, _, metadata = verify_cache(directory, spec, selected)
        publish_cache_result(directory, selected, metadata)
        print("Existing completed cache verified; no inference or training repeated.", flush=True)
        return
    before = encoder_state_sha256(encoder)
    progress_path = directory / "progress.json"
    partial = directory / "features.partial.npy"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != identity or progress["encoder_state_before"] != before:
            raise ValueError("Partial cache belongs to a different encoder/specification")
        resume_path = partial if partial.exists() else directory / "features.npy"
        matrix = np.lib.format.open_memmap(resume_path, mode="r+")
        expected_shape = (len(selected), config["output_dimensions"])
        if matrix.shape != expected_shape or matrix.dtype != np.float32:
            raise ValueError("Partial cache shape/dtype mismatch")
        completed = 0
        for chunk in progress["chunks"]:
            if chunk["start"] != completed or not completed < chunk["end"] <= len(selected):
                raise ValueError("Partial cache chunks are not contiguous")
            observed = hashlib.sha256(matrix[chunk["start"] : chunk["end"]].tobytes()).hexdigest()
            if observed != chunk["sha256"]:
                raise ValueError("Partial feature chunk checksum mismatch")
            completed = chunk["end"]
        if resume_path != partial and completed != len(selected):
            raise ValueError("Published matrix has incomplete progress; refuse recovery")
    else:
        if partial.exists() or (directory / "features.npy").exists():
            raise ValueError("Orphan partial cache; inspect it before restarting")
        matrix = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(len(selected), config["output_dimensions"])
        )
        matrix[:] = np.nan
        matrix.flush()
        progress = {"fingerprint": identity, "encoder_state_before": before, "chunks": []}
        write_json_atomic(progress_path, progress)
        completed = 0
    initial_completed = completed
    start = time.perf_counter()
    rows = list(selected.itertuples())
    try:
        with ThreadPoolExecutor(max_workers=config["preprocessing_threads"]) as pool:
            for begin in range(completed, len(selected), config["checkpoint_rows"]):
                end = min(begin + config["checkpoint_rows"], len(selected))
                for batch_start in range(begin, end, config["batch_size"]):
                    batch_end = min(batch_start + config["batch_size"], end)
                    tensors = list(
                        pool.map(
                            lambda row: image_tensor(row, preprocess), rows[batch_start:batch_end]
                        )
                    )
                    features = encode(encoder, torch.stack(tensors))
                    matrix[batch_start:batch_end] = features
                matrix.flush()
                digest = hashlib.sha256(matrix[begin:end].tobytes()).hexdigest()
                progress["chunks"].append({"start": begin, "end": end, "sha256": digest})
                write_json_atomic(progress_path, progress)
                elapsed = time.perf_counter() - start
                rate = (end - initial_completed) / elapsed
                print(
                    f"{end}/{len(selected)} images | {rate:.2f} images/s | "
                    f"remaining {(len(selected) - end) / rate / 60:.1f} min",
                    flush=True,
                )
        stats = feature_stats(matrix)
    finally:
        matrix.flush()
        matrix._mmap.close()
    after = encoder_state_sha256(encoder)
    if (
        before != after
        or encoder.training
        or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in encoder.parameters()
        )
    ):
        raise ValueError("Encoder state/gradients changed; refuse cache publication")
    if file_sha256(checkpoint) != protocol["backbone"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint file changed during extraction")
    if partial.exists():
        partial.replace(directory / "features.npy")
    row_table = selected[list(ROW_COLUMNS)].reset_index(drop=True)
    row_table.to_csv(directory / "rows.csv", index=False, lineterminator="\n")
    for name, indices in make_indices(row_table).items():
        np.save(directory / name, indices, allow_pickle=False)
    metadata = {
        "fingerprint": identity,
        "spec": spec,
        "complete": True,
        "trained": False,
        "eval_mode": not encoder.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in encoder.parameters()
        ),
        "gradients_absent": all(parameter.grad is None for parameter in encoder.parameters()),
        "encoder": encoder_info,
        "encoder_state_before": before,
        "encoder_state_after": after,
        "feature_statistics": stats,
        "elapsed_this_session_seconds": time.perf_counter() - start,
        "resumed_rows": initial_completed,
        "file_sha256": {
            path.name: file_sha256(path)
            for path in sorted(directory.iterdir())
            if path.name in {"features.npy", "rows.csv", *make_indices(row_table)}
        },
    }
    write_json_atomic(metadata_path, metadata)
    verify_cache(directory, spec, selected)
    publish_cache_result(directory, selected, metadata)
    print("Feature cache completed and verified. No models trained.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    extract(parser.parse_args().benchmark_only)
