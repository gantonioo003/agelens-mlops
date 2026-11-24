"""Reload the frozen operating point and estimate age without training."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image, ImageOps

from agelens.age_mlp import AgeMLP, predict_age
from agelens.farl import encode, encoder_state_sha256, file_sha256, load_encoder
from agelens.tracking import tracking_client
from services.ingest.contract import MAX_EDGE, MAX_IMAGE, Rejected

MODEL_ARTIFACT = "model/age_mlp.pt"
MEDIA = {"image/jpeg": "JPEG", "image/png": "PNG"}
NOTICE = "Experimental estimate; not age verification or access control."


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_head(root, protocol, reference):
    if reference["variant"] != "R0":
        raise ValueError("This version serves only the selected R0 architecture")
    training = protocol["training"]
    client = tracking_client(root)
    run_id = reference["run_id"]
    run = client.get_run(run_id)
    if run.info.status != "FINISHED" or run.data.params.get("variant") != "R0":
        raise ValueError("The approved R0 run is not available")
    with TemporaryDirectory(dir=root / "runtime") as temp:
        path = Path(client.download_artifacts(run_id, MODEL_ARTIFACT, temp))
        if file_sha256(path) != reference["checkpoint_sha256"]:
            raise ValueError("The served model checkpoint does not match its frozen hash")
        state = torch.load(path, weights_only=True, map_location="cpu")
    head = AgeMLP(training["shared_mlp_dimensions"], training["age_classes"])
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


def decode_image(data, content_type):
    if content_type not in MEDIA:
        raise Rejected("unsupported_media_type")
    if not 0 < len(data) <= MAX_IMAGE:
        raise Rejected("invalid_image_size")
    try:
        with Image.open(io.BytesIO(data), formats=[MEDIA[content_type]]) as src:
            if max(src.size) > MAX_EDGE or getattr(src, "n_frames", 1) != 1:
                raise Rejected("invalid_image_dimensions")
            src.verify()
        with Image.open(io.BytesIO(data), formats=[MEDIA[content_type]]) as src:
            return ImageOps.exif_transpose(src).convert("RGB")
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise Rejected("invalid_image") from exc


def decisions(age, operating_points):
    if not np.isfinite(age):
        raise ValueError("Invalid model prediction")
    score = -age
    return {
        str(t): {"score": score, "cutoff": q, "estimated_under_age": score >= q}
        for t, q in operating_points.items()
    }


class Engine:
    def __init__(self, root):
        from agelens.age_mlp import setup_torch

        setup_torch("cpu", 4)
        protocol_path = root / "configs/protocol-v1.json"
        self.protocol = read_json(protocol_path)
        selection_path = root / "configs/deployment-selection-v1.json"
        selection = read_json(selection_path)
        policy = read_json(root / "configs/deployment-calibration-v1.json")
        if (
            selection["protocol_sha256"] != file_sha256(protocol_path)
            or policy["selection_sha256"] != file_sha256(selection_path)
            or policy["partition"] != "val_calibration"
            or policy["cache_fingerprint"] != self.protocol["feature_extraction"]["fingerprint"]
        ):
            raise ValueError("The approved deployment policy does not match the protocol")
        self.version = policy["version"]
        self.ref = selection["deployment"]
        points = policy["by_seed"][str(self.ref["seed"])]
        if points["run_id"] != self.ref["run_id"] or set(points["thresholds"]) != {
            str(t) for t in self.protocol["evaluation"]["thresholds"]
        }:
            raise ValueError("Calibrated decisions do not match the deployed checkpoint")
        self.cutoffs = {t: details["cutoff"] for t, details in points["thresholds"].items()}
        if not all(np.isfinite(list(self.cutoffs.values()))):
            raise ValueError("Non-finite cutoff")
        self.head = load_head(root, self.protocol, self.ref)
        assets = read_json(root / "configs/assets.json")["backbone"]
        digest = self.protocol["backbone"]["checkpoint_sha256"]
        self.encoder, self.preprocess, _ = load_encoder(root / assets["path"], digest)
        if (
            encoder_state_sha256(self.encoder)
            != self.protocol["feature_extraction"]["encoder_state_sha256"]
        ):
            raise ValueError("FaRL parameters differ from the verified extractor")

    def metadata(self):
        return {
            "version": self.version,
            "variant": self.ref["variant"],
            "seed": self.ref["seed"],
            "mlflow_run_id": self.ref["run_id"],
            "checkpoint_sha256": self.ref["checkpoint_sha256"],
            "backbone_sha256": self.protocol["backbone"]["checkpoint_sha256"],
            "score_direction": "negative_predicted_age",
            "notice": NOTICE,
        }

    def predict(self, data, content_type):
        with decode_image(data, content_type) as image:
            features = encode(self.encoder, self.preprocess(image).unsqueeze(0))
        age = float(predict_age(self.head, features)[0])
        return {
            "age_estimate_years": age,
            "thresholds": decisions(age, self.cutoffs),
            "model": self.metadata(),
        }
