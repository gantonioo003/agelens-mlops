"""Verify all feature-cache contracts and independently recompute held-out check rows."""

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np

from agelens.cache_access import load_active_cache
from agelens.farl import encode, encoder_state_sha256, file_sha256, load_encoder
from agelens.feature_cache import PARTITIONS, safe_raw_path, write_json_atomic
from scripts.extract_features import image_tensor, load_inputs, make_spec, setup_torch
from scripts.prepare_assets import ROOT


def verify(require_frozen=True):
    protocol, config, selected = load_inputs()
    setup_torch(config)
    directory, features, rows, stats, metadata = load_active_cache(
        ROOT, require_frozen=require_frozen
    )
    expected_spec = make_spec(protocol, config)
    if metadata["spec"] != expected_spec:
        raise ValueError("Current runtime/source differs from the extracted representation")
    for number, row in enumerate(selected.itertuples(), 1):
        if file_sha256(safe_raw_path(ROOT, row.relative_path)) != row.image_sha256:
            raise ValueError("An image changed after its feature was cached")
        if number % 5000 == 0:
            print(f"Rechecked selected image hashes {number}/{len(selected)}", flush=True)
    if stats["min_column_std"] <= 1e-8:
        raise ValueError("Unexpected constant feature dimension; inspect this cache")
    registry = json.loads((ROOT / "configs/assets.json").read_text())
    encoder, preprocess, encoder_info = load_encoder(
        ROOT / registry["backbone"]["path"],
        protocol["backbone"]["checkpoint_sha256"],
        device=config["device"],
    )
    if not encoder_info["checkpoint_precision_preserved"]:
        raise ValueError("Original checkpoint precision was not preserved")
    before = encoder_state_sha256(encoder)
    if before != metadata["encoder_state_before"]:
        raise ValueError("Reconstructed encoder differs from the extraction encoder")
    indices = []
    split_checks = {}
    for partition in PARTITIONS:
        part = np.load(directory / f"indices_{partition}.npy", allow_pickle=False)
        indices.extend(part[np.linspace(0, len(part) - 1, 4, dtype=int)].tolist())
        split_checks[partition] = {
            "rows": len(part),
            "features": 512,
            "cv_folds": sorted(map(int, rows.iloc[part].cv_fold.unique())),
        }
    largest_error = 0.0
    for index in sorted(set(indices)):
        tensor = image_tensor(selected.iloc[index], preprocess).unsqueeze(0)
        actual = encode(encoder, tensor)[0]
        repeated = encode(encoder, tensor)[0]
        if not np.array_equal(actual, repeated):
            raise ValueError("Single-image frozen inference was not repeatable")
        difference = float(np.abs(actual - features[index]).max())
        largest_error = max(largest_error, difference)
        if not np.allclose(actual, features[index], atol=2e-6, rtol=1e-5):
            raise ValueError("Independently recomputed features differ from cache")
    after = encoder_state_sha256(encoder)
    if before != after or any(parameter.grad is not None for parameter in encoder.parameters()):
        raise ValueError("Verification changed model weights or gradients")
    cpu_error = None
    cpu_rows = 0
    if config["device"] == "cuda":
        encoder.cpu()
        cpu_error = 0.0
        for index in indices[::4]:
            tensor = image_tensor(selected.iloc[index], preprocess).unsqueeze(0)
            cpu_features = encode(encoder, tensor)[0]
            cpu_error = max(cpu_error, float(np.abs(cpu_features - features[index]).max()))
            if not np.allclose(cpu_features, features[index], atol=2e-6, rtol=1e-5):
                raise ValueError("CUDA feature cache differs from independent FP32 CPU inference")
            cpu_rows += 1
        if encoder_state_sha256(encoder) != before:
            raise ValueError("Device transfer changed checkpoint weights")
    tests = list(ET.parse(ROOT / "artifacts/private/task05-tests.xml").getroot().iter("testsuite"))
    failures = sum(
        int(test.attrib.get("failures", 0)) + int(test.attrib.get("errors", 0)) for test in tests
    )
    if not tests or failures:
        raise ValueError("Unit-test results missing or failing")
    result = {
        "task": "05",
        "completed": True,
        "samples": len(rows),
        "feature_shape": stats["shape"],
        "dtype": stats["dtype"],
        "finite": True,
        "l2_normalized": True,
        "row_alignment_verified": True,
        "all_selected_image_hashes_verified": True,
        "all_cache_file_hashes_verified": True,
        "independent_recomputed_rows": len(set(indices)),
        "recomputation_max_absolute_error": largest_error,
        "recomputation_tolerance": {"atol": 2e-6, "rtol": 1e-5},
        "single_image_repeats_bitwise_equal": True,
        "checkpoint_precision_preserved": True,
        "extraction_device": config["device"],
        "cpu_crosscheck_rows": cpu_rows,
        "cpu_crosscheck_max_absolute_error": cpu_error,
        "encoder_state_unchanged": before == after,
        "encoder_state_sha256": before,
        "eval_mode": not encoder.training,
        "all_parameters_frozen": True,
        "gradients_absent": True,
        "fitted_transforms": False,
        "partitions": split_checks,
        "cache_id": directory.name,
        "cache_metadata_sha256": file_sha256(directory / "metadata.json"),
        "cache_fingerprint": metadata["fingerprint"],
        "unit_tests_passed": sum(
            int(test.attrib["tests"]) - int(test.attrib.get("skipped", 0)) for test in tests
        ),
        "trained": False,
        "age_predictions_generated": False,
        "commit_created": False,
        "push_executed": False,
        "next_task": "06_MLflow_tracking_setup_before_model_training",
    }
    report = ROOT / "reports/task05/verification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-freeze",
        action="store_true",
        help="Bootstrap verification before protocol registration",
    )
    verify(require_frozen=not parser.parse_args().before_freeze)
