"""Materialize only the approved, labeled training partition. No fitting."""

import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_training(root=Path("/agelens"), destination=Path("/opt/airflow/prepared")):
    protocol_path = root / "inputs/protocol.json"
    protocol = read_json(protocol_path)
    frozen = protocol["feature_extraction"]
    cache = root / "cache"
    pointer = read_json(cache / "active.json")
    for key in ("cache_id", "fingerprint", "metadata_sha256"):
        if pointer[key] != frozen[key]:
            raise ValueError("Unapproved input cache")
    cache_id = frozen["cache_id"]
    if len(cache_id) != 32 or any(c not in "0123456789abcdef" for c in cache_id):
        raise ValueError("Invalid cache identifier")
    source = cache / cache_id
    metadata_file = source / "metadata.json"
    if sha256(metadata_file) != frozen["metadata_sha256"]:
        raise ValueError("Unapproved cache metadata")
    metadata = read_json(metadata_file)
    if not metadata["complete"] or metadata["trained"]:
        raise ValueError("Training preparation requires a verified frozen backbone")
    manifest = root / "inputs/train.csv"
    train_hash = protocol["data_audit"]["manifest_sha256"]["train.csv"]
    files = metadata["file_sha256"]
    index_file = source / "indices_train.npy"
    matrix_file = source / "features.npy"
    for path, expected in (
        (manifest, train_hash),
        (index_file, files[index_file.name]),
        (matrix_file, files[matrix_file.name]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path.name}")
    contract = {
        "partition": "train",
        "label_source": "audited_dataset_age_annotations",
        "rows": protocol["data_audit"]["split_sizes"]["train"],
        "protocol_sha256": sha256(protocol_path),
        "train_manifest_sha256": train_hash,
        "cache_fingerprint": frozen["fingerprint"],
        "feature_dimensions": frozen["dimensions"],
        "operational_predictions_used": False,
        "calibration_or_test_materialized": False,
    }
    output = destination / cache_id
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "train.npz"
    evidence = output / "verification.json"
    if evidence.exists():
        observed = read_json(evidence)
        if all(observed[key] == value for key, value in contract.items()) and (
            sha256(archive) == observed["archive_sha256"]
        ):
            return observed
        raise ValueError("Existing prepared dataset failed integrity verification")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indices = np.load(index_file, allow_pickle=False)
    matrix = np.load(matrix_file, mmap_mode="r", allow_pickle=False)
    if (
        len(rows) != len(indices)
        or len(rows) != contract["rows"]
        or not all(row["split"] == "train" for row in rows)
        or indices.ndim != 1
        or indices.dtype.kind not in {"i", "u"}
        or not np.all(indices[1:] > indices[:-1])
        or indices.min() < 0
        or indices.max() >= len(matrix)
    ):
        raise ValueError("Training index or manifest alignment changed")
    features = matrix[indices]
    labels = np.array([int(row["age"]) for row in rows], dtype=np.int16)
    folds = np.array([int(row["cv_fold"]) for row in rows], dtype=np.int8)
    if (
        features.shape != (contract["rows"], contract["feature_dimensions"])
        or features.dtype != np.float32
        or not np.isfinite(features).all()
        or labels.min() < 0
        or labels.max() >= protocol["training"]["age_classes"]
        or set(map(int, folds)) != set(range(protocol["tuning"]["folds"]))
    ):
        raise ValueError("Invalid feature or label contract")
    with TemporaryDirectory(dir=output) as temp:
        staged = Path(temp) / "train.npz"
        np.savez(staged, features=features, age_years=labels, folds=folds)
        contract["archive_sha256"] = sha256(staged)
        report = Path(temp) / "verification.json"
        report.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, archive)
        os.replace(report, evidence)
    return contract
