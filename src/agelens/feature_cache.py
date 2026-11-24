"""Path-safe, hash-bound, row-aligned feature-cache contracts; no fitting logic."""

import hashlib
import json

import numpy as np
import pandas as pd

from agelens.farl import file_sha256

PARTITIONS = ("train", "val_select", "val_calibration", "test")
ROW_COLUMNS = ("sample_id", "age", "split", "cv_fold", "duplicate_group", "image_sha256")


def fingerprint(spec):
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomic(path, data):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def feature_stats(features):
    if features.ndim != 2 or features.shape[1] != 512 or features.dtype != np.float32:
        raise ValueError("Expected an N x 512 float32 feature matrix")
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("Empty or non-finite feature cache")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1, atol=2e-6, rtol=0):
        raise ValueError("Feature rows must have unit L2 norm")
    return {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "finite": True,
        "min_row_norm": float(norms.min()),
        "max_row_norm": float(norms.max()),
        "min_column_std": float(np.std(features, axis=0, dtype=np.float64).min()),
    }


def safe_raw_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "data/raw/UTKFace").resolve()):
        raise ValueError("Image path escapes the audited dataset")
    return path


def make_indices(rows):
    result = {
        f"indices_{partition}.npy": np.flatnonzero(rows.split.to_numpy() == partition)
        for partition in PARTITIONS
    }
    for fold in range(3):
        result[f"indices_cv_{fold}.npy"] = np.flatnonzero(
            (rows.split.to_numpy() == "train") & (rows.cv_fold.to_numpy() == fold)
        )
    return result


def verify_cache(directory, expected_spec, selected):
    metadata = json.loads((directory / "metadata.json").read_text())
    expected_id = fingerprint(expected_spec)
    if metadata["fingerprint"] != expected_id or metadata["spec"] != expected_spec:
        raise ValueError("Feature cache specification changed")
    if not metadata.get("complete") or metadata.get("trained"):
        raise ValueError("Refuse incomplete or trained-state cache")
    if not all(
        metadata.get(name) is True
        for name in ["eval_mode", "all_parameters_frozen", "gradients_absent"]
    ):
        raise ValueError("Cache is not verified as frozen inference")
    for name, digest in metadata["file_sha256"].items():
        path = (directory / name).resolve()
        if path.parent != directory.resolve() or file_sha256(path) != digest:
            raise ValueError(f"Feature cache checksum mismatch: {name}")
    rows = pd.read_csv(directory / "rows.csv")
    if not rows.equals(selected[list(ROW_COLUMNS)].reset_index(drop=True)):
        raise ValueError("Feature rows do not match the frozen selected manifest")
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    if len(features) != len(rows):
        raise ValueError("Feature row count mismatch")
    stats = feature_stats(features)
    expected_indices = make_indices(rows)
    for name, indices in expected_indices.items():
        if not np.array_equal(np.load(directory / name, allow_pickle=False), indices):
            raise ValueError(f"Partition/fold feature indices changed: {name}")
    if set(metadata["file_sha256"]) != {"features.npy", "rows.csv", *expected_indices}:
        raise ValueError("Cache file inventory mismatch")
    if metadata["encoder_state_before"] != metadata["encoder_state_after"]:
        raise ValueError("Encoder weights changed during feature extraction")
    return features, rows, stats, metadata
