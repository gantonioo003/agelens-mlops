"""Exercise the local MQTT pipeline from the host using only a synthetic fixture."""

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from PIL import Image

from scripts.prepare_assets import ROOT


def compose(*args, stdin=None):
    result = subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        input=stdin,
        timeout=125,
    )
    return result.stdout.strip()


def sql(query):
    return compose(
        "exec",
        "-T",
        "database",
        "psql",
        "-X",
        "-qAt",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "agelens",
        "-d",
        "agelens",
        "-f",
        "-",
        stdin=query,
    )


def until(call, expected):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        actual = call()
        if actual == expected:
            return actual
        time.sleep(0.35)
    raise ValueError(f"Ingestion condition did not reach {expected!r}; observed {actual!r}")


def simulate(filename):
    value = compose(
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "simulator",
        "python",
        "-m",
        "services.ingest.simulator",
        "--image",
        filename,
    )
    event = json.loads(value.splitlines()[-1])
    assert str(UUID(event["event_id"])) == event["event_id"]
    return event


def send(payload):
    compose(
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "simulator",
        "python",
        "-c",
        "import sys; from services.ingest.simulator import publish; "
        "publish(sys.stdin.buffer.read(), 'broker')",
        stdin=payload,
    )


def row(event_id):
    uuid = str(UUID(event_id))
    return sql(
        "SELECT status || '|' || encode(image_data,'hex') "
        f"FROM ingest_events WHERE event_id = '{uuid}';"
    )


def check_rejection(payload, reason):
    digest = hashlib.sha256(payload.encode()).hexdigest()
    send(payload)
    until(
        lambda: sql(f"SELECT reason FROM ingest_rejections WHERE payload_sha256 = '{digest}';"),
        reason,
    )


def main():
    inbox = ROOT / "runtime/ingest/inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    fixture = inbox / f"synthetic-{uuid4().hex}.png"
    with Image.new("RGB", (32, 24), (27, 107, 164)) as im:
        im.save(fixture)
    encoded = fixture.read_bytes().hex()
    observed = f"pending|{encoded}"
    compose("up", "-d", "--wait", "--wait-timeout", "100", "ingestion")
    first = simulate(fixture.name)
    until(lambda: row(first["event_id"]), observed)
    send(json.dumps(first, indent=2))
    marker = f'"state": "duplicate", "detail": "{first["event_id"]}"'
    until(lambda: marker in compose("logs", "--tail", "60", "ingestion"), True)
    check_rejection(json.dumps({**first, "source": "sim-conflict"}), "event_id_conflict")
    check_rejection(
        json.dumps({**first, "event_id": str(uuid4()), "image_sha256": "0" * 64}),
        "image_checksum_mismatch",
    )
    check_rejection(json.dumps({"invalid": str(uuid4())}), "invalid_contract")
    compose("stop", "ingestion")
    try:
        second = simulate(fixture.name)
        if row(second["event_id"]):
            raise ValueError("Consumer was expected to be stopped")
    finally:
        compose("up", "-d", "--wait", "--wait-timeout", "100", "ingestion")
    until(lambda: row(second["event_id"]), observed)
    until(lambda: row(first["event_id"]), observed)
    compose("restart", "broker")
    compose("up", "-d", "--wait", "--wait-timeout", "100", "ingestion")
    third = simulate(fixture.name)
    until(lambda: row(third["event_id"]), observed)
    ids = [first["event_id"], second["event_id"], third["event_id"]]
    result = {
        "task": "10",
        "completed": True,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "synthetic_only": True,
        "event_ids": ids,
        "image_sha256": first["image_sha256"],
        "mqtt_qos": 1,
        "postgres_status": "pending",
        "exact_bytes_readback": True,
        "duplicate_ignored": True,
        "rejections_verified": ["event_id_conflict", "image_checksum_mismatch", "invalid_contract"],
        "offline_delivery_verified": True,
        "broker_restart_recovery_verified": True,
        "test_used": False,
        "calibration_used": False,
    }
    out = ROOT / "reports/task10/verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # The persisted synthetic image is now in PostgreSQL, not in the staging inbox.
    fixture.unlink()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
