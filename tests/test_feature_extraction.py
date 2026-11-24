import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from agelens.farl import file_sha256
from scripts import extract_features as extraction


def test_verifier_configures_determinism_before_comparing_runtime_fingerprint(monkeypatch):
    from scripts import verify_features as verification

    initialized = []
    config = {"device": "cuda"}
    monkeypatch.setattr(verification, "load_inputs", lambda: ({}, config, None))
    monkeypatch.setattr(verification, "setup_torch", lambda value: initialized.append(value))
    monkeypatch.setattr(
        verification, "load_active_cache", lambda *a, **k: (None, None, None, None, {"spec": {}})
    )

    def expected_spec(*args):
        assert initialized == [config]
        raise RuntimeError("runtime configured before fingerprint")

    monkeypatch.setattr(verification, "make_spec", expected_spec)
    with pytest.raises(RuntimeError, match="runtime configured before fingerprint"):
        verification.verify(require_frozen=False)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional local PyTorch check"
)
def test_requested_cuda_never_silently_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="refuse silent CPU fallback"):
        extraction.setup_torch({"device": "cuda"})


def test_image_tensor_uses_audited_bytes_and_rejects_modification(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    directory = tmp_path / "data/raw/UTKFace"
    directory.mkdir(parents=True)
    path = directory / "fixture.png"
    Image.new("RGB", (8, 8), (21, 42, 84)).save(path)
    row = SimpleNamespace(
        relative_path="data/raw/UTKFace/fixture.png", image_sha256=file_sha256(path)
    )
    output = extraction.image_tensor(row, lambda image: np.asarray(image))
    assert output.shape == (8, 8, 3) and np.array_equal(output[0, 0], [21, 42, 84])
    path.write_bytes(b"changed after audit")
    with pytest.raises(ValueError, match="bytes changed"):
        extraction.image_tensor(row, lambda image: np.asarray(image))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional local PyTorch check"
)
@pytest.mark.parametrize("failure", ["partial", "publication"])
def test_extraction_resumes_and_publishes_only_complete_cache(tmp_path, monkeypatch, failure):
    import torch

    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    checkpoint = tmp_path / "models/fixture.bin"
    checkpoint.write_bytes(b"frozen-test-fixture")
    (tmp_path / "configs/assets.json").write_text(
        json.dumps({"backbone": {"path": "models/fixture.bin"}})
    )
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
    config = {
        "device": "cpu",
        "preprocessing_threads": 1,
        "checkpoint_rows": 4,
        "batch_size": 2,
        "output_dimensions": 512,
    }
    protocol = {"backbone": {"checkpoint_sha256": file_sha256(checkpoint)}}
    torch.manual_seed(42)
    encoder = torch.nn.Linear(3, 512).eval().requires_grad_(False)
    monkeypatch.setattr(extraction, "load_inputs", lambda: (protocol, config, rows))
    monkeypatch.setattr(extraction, "setup_torch", lambda _: None)
    monkeypatch.setattr(extraction, "load_encoder", lambda *a, **k: (encoder, None, {}))
    monkeypatch.setattr(extraction, "make_spec", lambda *a: {"fixture": True})
    seen = []
    first_error = True

    def reader(row, _):
        nonlocal first_error
        seen.append(row.sample_id)
        if failure == "partial" and row.sample_id == "e" and first_error:
            first_error = False
            raise RuntimeError("simulated interruption")
        return torch.tensor([ord(row.sample_id) / 100, 1.0, 2.0])

    monkeypatch.setattr(extraction, "image_tensor", reader)
    original_publish = extraction.publish_cache_result
    if failure == "publication":

        def fail_publish(*args):
            raise RuntimeError("simulated interruption")

        monkeypatch.setattr(extraction, "publish_cache_result", fail_publish)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        extraction.extract()
    assert not (tmp_path / "data/processed/farl_ep16/active.json").exists()
    monkeypatch.setattr(extraction, "publish_cache_result", original_publish)
    extraction.extract()
    pointer = json.loads((tmp_path / "data/processed/farl_ep16/active.json").read_text())
    directory = tmp_path / "data/processed/farl_ep16" / pointer["cache_id"]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["complete"] and not metadata["trained"]
    assert np.load(directory / "features.npy").shape == (6, 512)
    assert all(seen.count(value) == 1 for value in "abcd")
    calls = len(seen)
    extraction.extract()
    assert len(seen) == calls
