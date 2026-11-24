"""Check configuration and installed packages without training."""

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_protocol(protocol):
    assert protocol["trained"] is False
    assert abs(sum(protocol["splits"]["fallback_ratios"]) - 1) < 1e-9
    assert len(set(protocol["training"]["seeds"])) == 3
    assert set(protocol["experiments"]["R6"]["binary_thresholds"]) - set(
        protocol["experiments"]["R4"]["binary_thresholds"]
    ) == {13, 16}
    assert protocol["backbone"]["frozen"]
    assert not protocol["splits"]["test_used_for_selection"]
    assert not protocol["evaluation"]["test_threshold_retuning"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/protocol-v1.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    report = {"python": platform.python_version(), "protocol": protocol["protocol_id"]}
    packages = (
        ["mlflow"]
        if args.minimal
        else ["numpy", "pandas", "sklearn", "PIL", "torch", "torchvision", "mlflow", "clip"]
    )
    for name in packages:
        print(f"Checking {name}...", file=sys.stderr, flush=True)
        module = importlib.import_module(name)
        report[name] = getattr(module, "__version__", "import_ok")
    if not args.minimal:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_smoke"] = (torch.ones(2) + 1).tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
