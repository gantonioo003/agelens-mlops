"""Load official FaRL weights and test one synthetic input, without extracting the dataset."""

import json
import time

import torch

from agelens.farl import load_encoder
from scripts.prepare_assets import ROOT


def main():
    config = json.loads((ROOT / "configs/assets.json").read_text())
    path = ROOT / config["backbone"]["path"]
    model, _, info = load_encoder(path, config["backbone"]["sha256"])
    start = time.perf_counter()
    with torch.inference_mode():
        features = model(torch.zeros(1, 3, 224, 224))
    assert tuple(features.shape) == (1, 512)
    assert torch.isfinite(features).all()
    assert not any(p.requires_grad for p in model.parameters())
    report = {
        "checkpoint_sha256": info["checkpoint_sha256"],
        "strict_architecture_load": True,
        "ignored_pretraining_keys": info["ignored_pretraining_keys"],
        "checkpoint_precision_preserved": info["checkpoint_precision_preserved"],
        "device": "cpu",
        "shape": list(features.shape),
        "finite": True,
        "frozen": True,
        "synthetic_forward_seconds": round(time.perf_counter() - start, 3),
        "trained": False,
    }
    out = ROOT / "artifacts/private/backbone-check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
