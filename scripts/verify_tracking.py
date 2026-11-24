"""Validate the local MLflow Tracking round trip without fitting any model."""

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from uuid import uuid4

from agelens.farl import file_sha256
from agelens.feature_cache import write_json_atomic
from agelens.tracking import (
    EXPERIMENT_KIND,
    PROVENANCE_PATH,
    collect_provenance,
    ensure_experiment,
    tracked_run,
    tracking_client,
    tracking_config,
)
from scripts.prepare_assets import ROOT

SMOKE_ARTIFACT = "checks/synthetic.json"
EXPECTED_PAYLOAD = {"synthetic": True, "trained": False, "samples": [1, 2, 3]}
EXPECTED_PARAMS = {"source": "synthetic", "task": "06", "sample_count": "3"}
EXPECTED_HISTORY = [(0, 1.0), (1, 2.0)]


class IntentionalSmokeFailure(Exception):
    pass


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_smoke(client, experiment_id, provenance):
    from mlflow.entities import Metric

    name = f"task06-roundtrip-{uuid4().hex[:8]}"
    options = {
        "experiment_id": experiment_id,
        "kind": "smoke",
        "provenance": provenance,
        "params": EXPECTED_PARAMS,
        "tags": {"agelens.task": "06"},
    }
    with tracked_run(client, name=name, **options) as finished:
        timestamp = time.time_ns() // 1_000_000
        client.log_batch(
            finished,
            metrics=[
                Metric("synthetic_value", value, timestamp + step, step)
                for step, value in EXPECTED_HISTORY
            ],
            synchronous=True,
        )
        client.log_dict(finished, EXPECTED_PAYLOAD, SMOKE_ARTIFACT)
    failed = None
    options["tags"] = {**options["tags"], "agelens.expected_failure": "true"}
    try:
        with tracked_run(client, name=f"{name}-expected-failure", **options) as failed:
            raise IntentionalSmokeFailure
    except IntentionalSmokeFailure:
        check(failed is not None, "Failure fixture did not create a run")
    return {"FINISHED": finished, "FAILED": failed}


def inspect_runs(client, run_ids, smoke_id, fingerprint, artifact_hash=None, provenance=None):
    check(len(set(run_ids.values())) == 2, "Runs must be separate")
    temporary_root = ROOT / "runtime"
    temporary_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=temporary_root) as temporary:
        for status, run_id in run_ids.items():
            run = client.get_run(run_id)
            check(run.info.status == status, f"Unexpected run status: {run.info.status}")
            check(run.info.experiment_id == smoke_id, "Smoke run outside smoke experiment")
            check(
                run.info.artifact_uri.startswith("mlflow-artifacts:/"), "No server artifact proxy"
            )
            check(run.data.params == EXPECTED_PARAMS, "Stored parameters differ")
            tags = run.data.tags
            check(tags.get(EXPERIMENT_KIND) == "smoke", "Missing synthetic run tag")
            check(tags.get("agelens.task") == "06", "Missing task tag")
            check(tags.get("agelens.trained") == "false", "Smoke run marked as training")
            check(tags.get("agelens.cache_fingerprint") == fingerprint, "Wrong FaRL cache")
            if status == "FAILED":
                check(not run.data.metrics, "Failed fixture must not receive another run's metrics")
                check(tags.get("agelens.expected_failure") == "true", "Missing failure fixture tag")
        run_id = run_ids["FINISHED"]
        history = client.get_metric_history(run_id, "synthetic_value")
        check(
            sorted((metric.step, metric.value) for metric in history) == EXPECTED_HISTORY,
            "Metric history differs",
        )
        check(
            client.get_run(run_id).data.metrics == {"synthetic_value": 2.0}, "Wrong latest metric"
        )
        paths = {item.path for item in client.list_artifacts(run_id) if item.is_dir}
        check(paths == {"checks", "metadata"}, "Unexpected artifact directories")
        downloaded = Path(client.download_artifacts(run_id, SMOKE_ARTIFACT, temporary))
        check(
            json.loads(downloaded.read_text(encoding="utf-8")) == EXPECTED_PAYLOAD,
            "Synthetic artifact content differs",
        )
        metadata = Path(client.download_artifacts(run_id, PROVENANCE_PATH, temporary))
        stored = json.loads(metadata.read_text(encoding="utf-8"))
        check(stored["feature_cache"]["fingerprint"] == fingerprint, "Artifact lineage differs")
        check(
            stored["code_sha256"] == client.get_run(run_id).data.tags["agelens.code_sha256"],
            "Artifact code provenance differs from run tag",
        )
        if provenance is not None:
            check(stored == provenance, "Provenance artifact content differs")
        observed = file_sha256(metadata)
        if artifact_hash is not None:
            check(observed == artifact_hash, "Stored provenance changed after restart")
    return observed


def main(recheck=False):
    config = tracking_config(ROOT)
    uri = config["tracking_uri"]
    with urlopen(f"{uri}/health", timeout=5) as response:
        check(response.status == 200, "MLflow health request failed")
    with urlopen(f"{uri}/version", timeout=5) as response:
        server_version = response.read().decode().strip()
    with urlopen(uri, timeout=5) as response:
        check(response.headers.get_content_type() == "text/html", "Tracking UI unavailable")
    client = tracking_client(ROOT)
    report_path = ROOT / "reports/task06/verification.json"
    if recheck:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        smoke = client.get_experiment_by_name(config["smoke_experiment"])
        research = client.get_experiment_by_name(config["research_experiment"])
        check(smoke is not None and research is not None, "Experiments not found after restart")
        smoke_id, research_id = smoke.experiment_id, research.experiment_id
        check(smoke_id == report["smoke_experiment_id"], "Smoke experiment ID changed")
        check(research_id == report["research_experiment_id"], "Research experiment ID changed")
        inspect_runs(
            client,
            report["run_ids"],
            smoke_id,
            report["cache_fingerprint"],
            artifact_hash=report["provenance_artifact_sha256"],
        )
        report["read_only_recheck_passed"] = True
    else:
        provenance = collect_provenance(ROOT)
        research_id = ensure_experiment(client, config, "research")
        smoke_id = ensure_experiment(client, config, "smoke")
        run_ids = run_smoke(client, smoke_id, provenance)
        fingerprint = provenance["feature_cache"]["fingerprint"]
        artifact_hash = inspect_runs(client, run_ids, smoke_id, fingerprint, provenance=provenance)
        report = {
            "task": "06",
            "tracking_uri": uri,
            "server_version": server_version,
            "client_version": provenance["client_libraries"]["mlflow"],
            "smoke_experiment": config["smoke_experiment"],
            "smoke_experiment_id": smoke_id,
            "research_experiment": config["research_experiment"],
            "research_experiment_id": research_id,
            "run_ids": run_ids,
            "synthetic_metric_history": EXPECTED_HISTORY,
            "cache_fingerprint": fingerprint,
            "protocol_sha256": provenance["protocol_sha256"],
            "code_sha256": provenance["code_sha256"],
            "provenance_artifact_sha256": artifact_hash,
            "parameters_tags_metrics_artifacts_verified": True,
            "tracking_ui_http_verified": True,
            "read_only_recheck_passed": False,
            "trained": False,
            "personal_data_uploaded": False,
        }
    report["research_experiment_empty"] = not client.search_runs([research_id], max_results=1)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recheck", action="store_true", help="Read saved runs without new logging"
    )
    main(recheck=parser.parse_args().recheck)
