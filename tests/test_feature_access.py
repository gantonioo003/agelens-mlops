import json

import pytest

from agelens.cache_access import get_feature_split, load_active_cache


def setup_pointer(tmp_path, pointer):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/protocol-v1.json").write_text("{}")
    directory = tmp_path / "data/processed/farl_ep16"
    directory.mkdir(parents=True)
    (directory / "active.json").write_text(json.dumps(pointer))
    return directory


def test_active_cache_rejects_unsafe_identifier(tmp_path):
    setup_pointer(tmp_path, {"cache_id": "../escape"})
    with pytest.raises(ValueError, match="identifier"):
        load_active_cache(tmp_path)


def test_active_cache_checks_metadata_before_reading_vectors(tmp_path):
    identifier = "a" * 32
    directory = setup_pointer(
        tmp_path, {"cache_id": identifier, "fingerprint": "a" * 64, "metadata_sha256": "0" * 64}
    )
    (directory / identifier).mkdir()
    (directory / identifier / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata checksum"):
        load_active_cache(tmp_path)


def test_split_loader_refuses_unrestricted_all_data(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        get_feature_split(tmp_path, "all")
