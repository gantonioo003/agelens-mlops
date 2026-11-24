"""Verify a running loopback API against a separately reloaded offline model."""

import io
import json
import os
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw

from agelens.farl import file_sha256
from scripts.prepare_assets import ROOT
from services.inference.model import Engine
from services.ingest.contract import MAX_IMAGE


def fixture():
    data = io.BytesIO()
    with Image.new("RGB", (96, 112), "#eeead9") as image:
        draw = ImageDraw.Draw(image)
        draw.polygon([(9, 95), (44, 18), (82, 69)], fill="#31717c")
        draw.rectangle((53, 33, 84, 95), outline="#221d21", width=4)
        image.save(data, format="PNG")
    return data.getvalue()


def main():
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    port = int(os.environ.get("AGELENS_API_PORT", "18080"))
    origin = f"http://127.0.0.1:{port}"
    raw = fixture()
    offline = Engine(ROOT).predict(raw, "image/png")
    with httpx.Client(base_url=origin, trust_env=False, timeout=25) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json() == {"status": "ready", "model_version": offline["model"]["version"]}
        assert client.get("/v1/model").json() == offline["model"]
        response = client.post("/v1/predict", content=raw, headers={"Content-Type": "image/png"})
        response.raise_for_status()
        online = response.json()
        assert online["model"] == offline["model"]
        error = abs(online["age_estimate_years"] - offline["age_estimate_years"])
        assert error <= 1e-6
        assert online["thresholds"].keys() == offline["thresholds"].keys()
        for threshold, expected in offline["thresholds"].items():
            actual = online["thresholds"][threshold]
            assert abs(actual["score"] - expected["score"]) <= 1e-6
            assert actual["cutoff"] == expected["cutoff"]
            assert actual["estimated_under_age"] is expected["estimated_under_age"]
        stored = client.get(f"/v1/results/{online['id']}")
        assert stored.status_code == 200 and stored.json() == online
        assert client.get(f"/v1/results/{uuid4()}").status_code == 404
        rejected = {}
        for label, data, media, status in (
            ("wrong_type", raw, "text/plain", 415),
            ("corrupt_png", b"invalid", "image/png", 422),
            ("wrong_signature", raw, "image/jpeg", 422),
            ("oversized", b"x" * (MAX_IMAGE + 1), "image/png", 413),
        ):
            invalid = client.post("/v1/predict", content=data, headers={"Content-Type": media})
            assert invalid.status_code == status and "id" in invalid.json()["detail"]
            rejected[label] = status
        assert client.get("/health").status_code == 200 and client.get("/docs").status_code == 200
    report = {
        "task": "11",
        "completed": True,
        "origin": origin,
        "model": offline["model"],
        "synthetic_fixture_only": True,
        "offline_http_age_abs_error_years": error,
        "threshold_decisions_equal": True,
        "result_lookup_verified": True,
        "rejected_status_codes": rejected,
        "health_after_rejections": True,
        "api_duration_ms": online["duration_ms"],
        "test_used": False,
        "selection_sha256": file_sha256(ROOT / "configs/deployment-selection-v1.json"),
        "calibration_sha256": file_sha256(ROOT / "configs/deployment-calibration-v1.json"),
    }
    path = ROOT / "reports/task11/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
