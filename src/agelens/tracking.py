"""Explicit MLflow runs with reproducible lineage and no training side effects."""

import json
import os
import platform
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

from agelens.farl import file_sha256
from agelens.feature_cache import fingerprint

PROVENANCE_PATH = "metadata/provenance.json"
EXPERIMENT_KIND = "agelens.kind"
ALLOWED_URI = "http://127.0.0.1:15000"


def tracking_config(root):
    config = json.loads((root / "configs/tracking-v1.json").read_text(encoding="utf-8"))
    uri = config["tracking_uri"]
    if uri != ALLOWED_URI or os.environ.get("MLFLOW_TRACKING_URI", uri) != uri:
        raise ValueError("Tracking must use the configured local AgeLens HTTP server")
    if config["research_experiment"] == config["smoke_experiment"]:
        raise ValueError("Research and infrastructure must use separate experiments")
    return config


def tracking_client(root):
    config = tracking_config(root)
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    # The MLflow run link contains Unicode that can fail on Windows cp1252 consoles.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=config["tracking_uri"])


def ensure_experiment(client, config, kind):
    if kind not in {"research", "smoke"}:
        raise ValueError("Unknown AgeLens experiment kind")
    name = config[f"{kind}_experiment"]
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        client.create_experiment(name, tags={EXPERIMENT_KIND: kind})
        experiment = client.get_experiment_by_name(name)
    if (
        experiment.lifecycle_stage != "active"
        or experiment.tags.get(EXPERIMENT_KIND) != kind
        or not experiment.artifact_location.startswith("mlflow-artifacts:/")
    ):
        raise ValueError(f"Unexpected experiment ownership, lifecycle or artifact store: {name}")
    return experiment.experiment_id


def collect_provenance(root):
    protocol_path = root / "configs/protocol-v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = protocol["feature_extraction"]
    cache_root = root / "data/processed/farl_ep16"
    pointer = json.loads((cache_root / "active.json").read_text(encoding="utf-8"))
    cache_id = frozen["cache_id"]
    if len(cache_id) != 32 or any(c not in "0123456789abcdef" for c in cache_id):
        raise ValueError("Invalid approved FaRL cache ID")
    if any(pointer[key] != frozen[key] for key in ("cache_id", "fingerprint", "metadata_sha256")):
        raise ValueError("Active feature cache does not match the approved protocol")
    metadata = cache_root / cache_id / "metadata.json"
    if file_sha256(metadata) != frozen["metadata_sha256"]:
        raise ValueError("Feature cache metadata checksum mismatch")
    manifests = protocol["data_audit"]["manifest_sha256"]
    for name, expected in manifests.items():
        if file_sha256(root / "data/manifests" / name) != expected:
            raise ValueError(f"Frozen manifest checksum mismatch: {name}")
    sources = sorted([*(root / "src/agelens").rglob("*.py"), *(root / "scripts").rglob("*.py")])
    code = {path.relative_to(root).as_posix(): file_sha256(path) for path in sources}
    libraries = {}
    for name in (
        "mlflow",
        "mlflow-skinny",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
    ):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoint_sha256": protocol["backbone"]["checkpoint_sha256"],
        "manifests_sha256": manifests,
        "split_sizes": protocol["data_audit"]["split_sizes"],
        "feature_cache": {
            key: frozen[key] for key in ("cache_id", "fingerprint", "metadata_sha256")
        },
        "code_sha256": fingerprint(code),
        "code_files_sha256": code,
        "client_libraries": libraries,
        "python_version": platform.python_version(),
    }


@contextmanager
def tracked_run(client, experiment_id, *, name, kind, provenance, params, tags=None):
    from mlflow.entities import Param

    if kind not in {"research", "smoke"}:
        raise ValueError("Unknown AgeLens run kind")
    experiment = client.get_experiment(experiment_id)
    if experiment.tags.get(EXPERIMENT_KIND) != kind:
        raise ValueError("Run kind must match its experiment")
    run_tags = {
        **(tags or {}),
        EXPERIMENT_KIND: kind,
        "agelens.protocol": provenance["protocol_id"],
        "agelens.cache_fingerprint": provenance["feature_cache"]["fingerprint"],
        "agelens.code_sha256": provenance["code_sha256"],
    }
    if kind == "smoke":
        run_tags["agelens.trained"] = "false"
    run_id = client.create_run(experiment_id, run_name=name, tags=run_tags).info.run_id
    status = "FAILED"
    try:
        client.log_batch(
            run_id,
            params=[Param(key, str(value)) for key, value in params.items()],
            synchronous=True,
        )
        client.log_dict(run_id, provenance, PROVENANCE_PATH)
        yield run_id
        status = "FINISHED"
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    finally:
        client.set_terminated(run_id, status)
