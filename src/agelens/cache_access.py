"""Read only the published, protocol-approved feature cache; never reshuffle or fit."""

import json

import numpy as np
import pandas as pd

from agelens.farl import file_sha256
from agelens.feature_cache import PARTITIONS, verify_cache


def open_approved_cache(root, require_frozen=True):
    protocol = json.loads((root / "configs/protocol-v1.json").read_text())
    pointer = json.loads((root / "data/processed/farl_ep16/active.json").read_text())
    cache_id = pointer["cache_id"]
    if len(cache_id) != 32 or any(character not in "0123456789abcdef" for character in cache_id):
        raise ValueError("Invalid feature cache identifier")
    directory = root / "data/processed/farl_ep16" / cache_id
    if directory.resolve().parent != (root / "data/processed/farl_ep16").resolve():
        raise ValueError("Feature cache path escapes its root")
    metadata_path = directory / "metadata.json"
    digest = file_sha256(metadata_path)
    if digest != pointer["metadata_sha256"]:
        raise ValueError("Active feature cache metadata checksum mismatch")
    if require_frozen:
        frozen = protocol["feature_extraction"]
        if any(
            pointer[key] != frozen[key] for key in ["cache_id", "fingerprint", "metadata_sha256"]
        ):
            raise ValueError("Active cache differs from the feature cache frozen in the protocol")
    metadata = json.loads(metadata_path.read_text())
    if metadata["fingerprint"] != pointer["fingerprint"]:
        raise ValueError("Cache pointer fingerprint mismatch")
    spec = metadata["spec"]
    if spec["frozen_manifests"] != protocol["data_audit"]["manifest_sha256"]:
        raise ValueError("Feature cache uses different dataset manifests")
    if spec["checkpoint_sha256"] != protocol["backbone"]["checkpoint_sha256"]:
        raise ValueError("Feature cache uses a different backbone checkpoint")
    if spec["config"] != json.loads((root / "configs/features-v1.json").read_text()):
        raise ValueError("Feature cache configuration changed")
    return directory, metadata


def load_active_cache(root, require_frozen=True):
    directory, metadata = open_approved_cache(root, require_frozen=require_frozen)
    spec = metadata["spec"]
    for name, expected in spec["frozen_manifests"].items():
        if file_sha256(root / "data/manifests" / name) != expected:
            raise ValueError(f"Frozen data manifest changed: {name}")
    selected = pd.read_csv(root / "data/manifests/selected.csv")
    features, rows, stats, metadata = verify_cache(directory, spec, selected)
    return directory, features, rows, stats, metadata


def get_feature_split(root, partition):
    if partition not in PARTITIONS:
        raise ValueError("Select exactly one frozen data partition")
    directory, features, rows, _, _ = load_active_cache(root)
    indices = np.load(directory / f"indices_{partition}.npy", allow_pickle=False)
    return features[indices], rows.iloc[indices].reset_index(drop=True)
