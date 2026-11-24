"""Exercise real initial Docker services without creating ML experiment results."""

import json
import subprocess
import time
import urllib.request

from scripts.prepare_assets import ROOT


def compose(*args):
    return ["docker", "compose", *args]


def run(*args):
    return subprocess.run(
        compose(*args), cwd=ROOT, check=True, text=True, capture_output=True, timeout=40
    ).stdout.strip()


def main():
    topic = "agelens/setup/roundtrip"
    payload = "setup-check-no-personal-data"
    subscriber = subprocess.Popen(
        compose(
            "exec",
            "-T",
            "broker",
            "mosquitto_sub",
            "-h",
            "localhost",
            "-t",
            topic,
            "-C",
            "1",
            "-W",
            "10",
        ),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1)
        run("exec", "-T", "broker", "mosquitto_pub", "-h", "localhost", "-t", topic, "-m", payload)
        received, error = subscriber.communicate(timeout=15)
        if subscriber.returncode or received.strip() != payload:
            raise RuntimeError(f"MQTT roundtrip failed: {error}")
    finally:
        if subscriber.poll() is None:
            subscriber.kill()
            subscriber.communicate()
    query = run(
        "exec", "-T", "database", "psql", "-U", "agelens", "-d", "agelens", "-Atc", "SELECT 1;"
    )
    assert query == "1"
    with urllib.request.urlopen("http://127.0.0.1:15000/health", timeout=5) as response:
        assert response.status == 200
    report = {
        "mqtt_roundtrip": True,
        "postgres_select_1": True,
        "mlflow_http_health": 200,
        "trained": False,
    }
    path = ROOT / "artifacts/private/services-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
