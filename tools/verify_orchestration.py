"""Validate scheduled MQTT-to-PostgreSQL inference and the train-only snapshot."""

import hashlib
import json
import time
from uuid import uuid4

import httpx
from PIL import Image

from scripts.prepare_assets import ROOT
from tools.verify_ingestion import compose, send, simulate, sql


def await_value(read, seconds=190):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        result = read()
        if result:
            return result
        time.sleep(2)
    raise ValueError("Scheduled orchestration did not produce evidence")


def lookup(event_id):
    raw = sql(
        "SELECT json_build_object('status', e.status, 'attempts', e.inference_attempts, "
        "'event_id', r.event_id, 'model_version', r.model_version, "
        "'model_run_id', r.model_run_id, 'age', r.age_estimate_years, "
        "'duration', r.duration_ms, 'decisions', r.decisions, 'processed_at', r.processed_at) "
        "FROM ingest_events e JOIN inference_results r USING (event_id) "
        f"WHERE e.event_id = '{event_id}';"
    )
    return json.loads(raw) if raw else None


def training_evidence():
    return compose(
        "exec",
        "-T",
        "airflow",
        "python",
        "-c",
        "from pathlib import Path; import json, sys; "
        "f=json.loads(Path('/agelens/inputs/protocol.json').read_text())['feature_extraction']; "
        "p=Path('/opt/airflow/prepared')/f['cache_id']/'verification.json'; "
        "sys.stdout.write(p.read_text() if p.exists() else '')",
    )


def main():
    inbox = ROOT / "runtime/ingest/inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    image = inbox / f"orchestration-{uuid4().hex}.png"
    with Image.new("RGB", (112, 96), color="#577b62") as fixture:
        fixture.save(image)
    data = image.read_bytes()
    try:
        await_value(lambda: sql("SELECT to_regclass('public.inference_results') IS NOT NULL;") == "t")
        event = simulate(image.name)
        event_id = event["event_id"]
        durable = await_value(lambda: lookup(event_id))
        assert durable["status"] == "completed" and durable["event_id"] == event_id
        assert len(durable["decisions"]) == 6
        # The actual scheduled DAG produced the first prediction. These requests only verify replay.
        with httpx.Client(base_url="http://127.0.0.1:18080", trust_env=False, timeout=15) as client:
            remote = client.get(f"/v1/results/{event_id}")
            remote.raise_for_status()
            expected = remote.json()
            replay = client.post(
                "/v1/predict",
                content=data,
                headers={"Content-Type": "image/png", "X-Event-ID": event_id},
            )
            assert replay.status_code == 200 and replay.json() == expected
            conflict = client.post(
                "/v1/predict",
                content=b"different",
                headers={"Content-Type": "image/png", "X-Event-ID": event_id},
            )
            assert conflict.status_code == 409
        assert (
            durable["decisions"] == expected["thresholds"]
            and durable["age"] == expected["age_estimate_years"]
        )
        assert durable["model_run_id"] == expected["model"]["mlflow_run_id"]
        assert durable["duration"] == expected["duration_ms"]
        send(json.dumps(event))
        marker = f'"state": "duplicate", "detail": "{event_id}"'
        await_value(lambda: marker in compose("logs", "--tail", "60", "ingestion"), 30)
        assert lookup(event_id) == durable
        count = int(sql(f"SELECT count(*) FROM inference_results WHERE event_id = '{event_id}';"))
        assert count == 1
        prepared = json.loads(await_value(training_evidence))
        assert prepared["partition"] == "train" and prepared["rows"] == 9001
        assert (
            not prepared["operational_predictions_used"]
            and not prepared["calibration_or_test_materialized"]
        )
        recheck = compose(
            "exec",
            "-T",
            "airflow",
            "python",
            "-c",
            "import json; from services.orchestration.training_data import prepare_training; "
            "print(json.dumps(prepare_training()))",
        )
        assert json.loads(recheck) == prepared
        result = {
            "task": "12",
            "completed": True,
            "scheduled_without_manual_trigger": True,
            "fixture_sha256": hashlib.sha256(data).hexdigest(),
            "event_id": event_id,
            "status": durable["status"],
            "attempts": durable["attempts"],
            "database_result_count": count,
            "model_version": durable["model_version"],
            "model_run_id": durable["model_run_id"],
            "processed_at": durable["processed_at"],
            "duration_ms": durable["duration"],
            "same_id_mqtt_api_postgres": True,
            "api_replay_same_record": True,
            "api_conflicting_replay_http_status": 409,
            "mqtt_replay_no_new_result": True,
            "labeled_training_snapshot": prepared,
            "prepared_snapshot_reused": True,
            "trained": False,
            "evaluation_test_used": False,
            "evaluation_calibration_used": False,
        }
        report = ROOT / "reports/task12/verification.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        image.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
