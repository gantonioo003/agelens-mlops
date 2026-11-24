import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from agelens.farl import file_sha256
from agelens.feature_cache import (
    ROW_COLUMNS,
    feature_stats,
    fingerprint,
    make_indices,
    safe_raw_path,
    verify_cache,
    write_json_atomic,
)


def cache_fixture(tmp_path):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(6, 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    rows = pd.DataFrame(
        {
            "sample_id": list("abcdef"),
            "age": [12, 15, 18, 21, 22, 23],
            "split": ["train"] * 3 + ["val_select", "val_calibration", "test"],
            "cv_fold": [0, 1, 2, -1, -1, -1],
            "duplicate_group": list("abcdef"),
            "image_sha256": list("abcdef"),
        }
    )
    rows = rows[list(ROW_COLUMNS)]
    np.save(tmp_path / "features.npy", features, allow_pickle=False)
    rows.to_csv(tmp_path / "rows.csv", index=False)
    for name, indices in make_indices(rows).items():
        np.save(tmp_path / name, indices, allow_pickle=False)
    spec = {"test": "frozen"}
    metadata = {
        "spec": spec,
        "fingerprint": fingerprint(spec),
        "complete": True,
        "trained": False,
        "eval_mode": True,
        "all_parameters_frozen": True,
        "gradients_absent": True,
        "encoder_state_before": "unchanged",
        "encoder_state_after": "unchanged",
        "file_sha256": {path.name: file_sha256(path) for path in tmp_path.iterdir()},
    }
    write_json_atomic(tmp_path / "metadata.json", metadata)
    return rows, spec


def test_cache_row_alignment_and_partition_indices(tmp_path):
    selected, spec = cache_fixture(tmp_path)
    features, rows, stats, _ = verify_cache(tmp_path, spec, selected)
    assert features.shape == (6, 512) and stats["finite"]
    assert rows.sample_id.tolist() == list("abcdef")
    assert np.array_equal(np.load(tmp_path / "indices_test.npy"), [5])


def test_fingerprint_canonical_and_bound_to_parameters():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


@pytest.mark.parametrize("issue", ["nan", "zero", "dtype", "dimension", "non_unit"])
def test_feature_stats_reject_invalid_matrices(issue):
    values = np.ones((3, 512), dtype=np.float32) / np.sqrt(np.float32(512))
    if issue == "nan":
        values[0, 0] = np.nan
    elif issue == "zero":
        values[0] = 0
    elif issue == "dtype":
        values = values.astype(np.float64)
    elif issue == "dimension":
        values = values[:, :511]
    elif issue == "non_unit":
        values *= 2
    with pytest.raises(ValueError):
        feature_stats(values)


def test_cache_detects_bit_corruption(tmp_path):
    rows, spec = cache_fixture(tmp_path)
    path = tmp_path / "features.npy"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        verify_cache(tmp_path, spec, rows)


def test_cache_rejects_row_reordering(tmp_path):
    rows, spec = cache_fixture(tmp_path)
    with pytest.raises(ValueError, match="rows"):
        verify_cache(tmp_path, spec, rows.iloc[::-1].reset_index(drop=True))


def test_cache_rejects_encoder_or_manifest_change(tmp_path):
    rows, spec = cache_fixture(tmp_path)
    with pytest.raises(ValueError, match="specification"):
        verify_cache(tmp_path, {"test": "different"}, rows)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["encoder_state_after"] = "changed"
    write_json_atomic(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError, match="weights changed"):
        verify_cache(tmp_path, spec, rows)


def test_cache_rejects_incomplete_state(tmp_path):
    rows, spec = cache_fixture(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["complete"] = False
    write_json_atomic(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError, match="incomplete"):
        verify_cache(tmp_path, spec, rows)


def test_paths_cannot_escape_raw_dataset(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_raw_path(tmp_path, "../elsewhere.jpg")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional local PyTorch check"
)
def test_frozen_toy_encoder_only_infers_without_gradients():
    import torch

    from agelens.farl import encode, encoder_state_sha256

    torch.manual_seed(42)
    encoder = torch.nn.Linear(3, 512).eval().requires_grad_(False)
    before = encoder_state_sha256(encoder)
    result = encode(encoder, torch.ones(2, 3))
    feature_stats(result)
    assert before == encoder_state_sha256(encoder)
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert np.array_equal(result, encode(encoder, torch.ones(2, 3)))
    encoder.train()
    with pytest.raises(ValueError, match="unfrozen/training"):
        encode(encoder, torch.ones(2, 3))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional local PyTorch check"
)
def test_zero_toy_encoder_output_is_rejected():
    import torch

    from agelens.farl import encode

    encoder = torch.nn.Linear(3, 512, bias=False).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="Zero"):
        encode(encoder, torch.zeros(2, 3))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional local PyTorch check"
)
def test_evaluation_mode_alone_is_not_sufficient_without_freezing_parameters():
    import torch

    from agelens.farl import encode

    encoder = torch.nn.Linear(3, 512).eval()
    with pytest.raises(ValueError, match="unfrozen/training"):
        encode(encoder, torch.ones(2, 3))
