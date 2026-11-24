import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from services.inference.api import create_app
from services.inference.model import decisions, decode_image
from services.inference.prepare import operating_point, rank
from services.ingest.contract import MAX_IMAGE, Rejected
from tools.verify_inference import fixture


def test_operating_point_preserves_ties_and_uses_highest_valid_cutoff():
    scores = np.array([0.1, 0.2, 0.2] + [0.9] * 98 + [0.0, 0.5])
    positives = np.array([True] * 101 + [False] * 2)
    point = operating_point(scores, positives, 0.99)
    assert point["cutoff"] == 0.2
    assert point["metrics"]["fn"] == 1
    assert decisions(12.0, {"18": -12.0})["18"]["estimated_under_age"] is True


def test_rank_excludes_trivial_all_minor_and_requires_all_seeds():
    def rows(recall, tn, f2):
        return [
            {"seed": seed, "recall": recall, "tn": tn, "fn": 0.0, "f2": f2, "fpr": 0.1, "mae": 4.0}
            for seed in (17, 42, 2026)
        ]

    ranking = rank(
        {"trivial": rows(1.0, 0, 0.99), "low": rows(0.98, 3, 0.99), "valid": rows(0.99, 5, 0.8)},
        [17, 42, 2026],
        0.99,
    )
    assert ranking[0]["variant"] == "valid"


def test_http_rejection_does_not_break_queries(tmp_path):
    class Stub:
        def __init__(self, root):
            pass

        def metadata(self):
            return {"version": "synthetic-test"}

        def predict(self, raw, media):
            with decode_image(raw, media) as image:
                if image.width != 96:
                    raise Rejected("unexpected_fixture")
            return {"age_estimate_years": 20.0, "model": self.metadata()}

    app = create_app(Stub, tmp_path)
    with TestClient(app) as client:
        assert client.post("/v1/predict", content=b"x").status_code == 415
        assert (
            client.post(
                "/v1/predict", content=b"x", headers={"content-type": "image/png"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/predict", content=b"x" * (MAX_IMAGE + 1), headers={"content-type": "image/png"}
            ).status_code
            == 413
        )
        response = client.post(
            "/v1/predict", content=fixture(), headers={"content-type": "image/png"}
        )
        assert response.status_code == 200
        record = response.json()
        assert client.get(f"/v1/results/{record['id']}").json() == record
        assert client.get("/health").status_code == 200
