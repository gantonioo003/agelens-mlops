import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agelens.farl import file_sha256
from agelens.tracking import (
    ALLOWED_URI,
    EXPERIMENT_KIND,
    collect_provenance,
    ensure_experiment,
    tracked_run,
    tracking_config,
)


def test_tracking_refuses_nonlocal_override(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    config = {
        "tracking_uri": ALLOWED_URI,
        "research_experiment": "research",
        "smoke_experiment": "smoke",
    }
    (tmp_path / "configs/tracking-v1.json").write_text(json.dumps(config))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking_config(tmp_path) == config
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.org")
    with pytest.raises(ValueError, match="local AgeLens HTTP server"):
        tracking_config(tmp_path)


def test_existing_experiment_must_be_owned_and_proxy_artifacts():
    client = Mock()
    experiment = SimpleNamespace(
        experiment_id="2",
        lifecycle_stage="active",
        artifact_location="mlflow-artifacts:/2",
        tags={EXPERIMENT_KIND: "smoke"},
    )
    client.get_experiment_by_name.return_value = experiment
    config = {"smoke_experiment": "smoke"}
    assert ensure_experiment(client, config, "smoke") == "2"
    client.create_experiment.assert_not_called()
    experiment.artifact_location = "file:///a-local-path"
    with pytest.raises(ValueError, match="artifact store"):
        ensure_experiment(client, config, "smoke")


def test_provenance_matches_approved_cache_and_rejects_changed_inputs(tmp_path):
    (tmp_path / "configs").mkdir()
    manifest = tmp_path / "data/manifests/train.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("opaque-manifest-fixture")
    cache = tmp_path / "data/processed/farl_ep16"
    cache_id = "a" * 32
    metadata = cache / cache_id / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}")
    pointer = {
        "cache_id": cache_id,
        "fingerprint": "b" * 64,
        "metadata_sha256": file_sha256(metadata),
    }
    (cache / "active.json").write_text(json.dumps(pointer))
    (tmp_path / "configs/protocol-v1.json").write_text(
        json.dumps(
            {
                "protocol_id": "test",
                "feature_extraction": pointer,
                "backbone": {"checkpoint_sha256": "c" * 64},
                "data_audit": {
                    "manifest_sha256": {"train.csv": file_sha256(manifest)},
                    "split_sizes": {"train": 1},
                },
            }
        )
    )
    provenance = collect_provenance(tmp_path)
    assert provenance["feature_cache"] == pointer
    assert "opaque-manifest-fixture" not in json.dumps(provenance)
    manifest.write_text("modified")
    with pytest.raises(ValueError, match="Frozen manifest checksum mismatch"):
        collect_provenance(tmp_path)
    metadata.write_text('{"modified": true}')
    with pytest.raises(ValueError, match="cache metadata checksum mismatch"):
        collect_provenance(tmp_path)


@pytest.mark.parametrize(
    ("error", "status"),
    [(None, "FINISHED"), (RuntimeError, "FAILED"), (KeyboardInterrupt, "KILLED")],
)
def test_tracked_run_status_and_provenance(error, status):
    pytest.importorskip("mlflow")
    client = Mock()
    client.get_experiment.return_value.tags = {EXPERIMENT_KIND: "smoke"}
    client.create_run.return_value.info.run_id = "recorded-run"
    provenance = {
        "protocol_id": "test",
        "feature_cache": {"fingerprint": "test-fingerprint"},
        "code_sha256": "test-hash",
    }

    def body():
        with tracked_run(
            client, "1", name="test", kind="smoke", provenance=provenance, params={"num": 1}
        ) as run_id:
            assert run_id == "recorded-run"
            if error:
                raise error()

    if error:
        with pytest.raises(error):
            body()
    else:
        body()
    client.log_dict.assert_called_once_with("recorded-run", provenance, "metadata/provenance.json")
    client.set_terminated.assert_called_once_with("recorded-run", status)
