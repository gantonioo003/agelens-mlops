"""Check acquisition provenance and a small deterministic image sample, not EDA."""

import json
import random
from collections import Counter

from PIL import Image

from scripts.prepare_assets import ROOT, sha256


def main():
    registry = json.loads((ROOT / "configs/assets.json").read_text())
    acquisition = json.loads((ROOT / "artifacts/private/acquisition.json").read_text())
    report = {"trained": False, "assets": {}}
    for key, asset in registry.items():
        path = ROOT / asset["path"]
        digest = sha256(path)
        assert path.stat().st_size == asset["size"]
        assert digest == acquisition["assets"][key]["sha256"]
        if asset.get("sha256"):
            assert digest == asset["sha256"]
        report["assets"][key] = {"bytes": path.stat().st_size, "sha256": digest}
    files = sorted((ROOT / "data/raw/UTKFace").glob("*.jpg"))
    ages = Counter()
    invalid = 0
    for path in files:
        parts = path.name.split("_")
        try:
            age, gender, race = map(int, parts[:3])
            assert len(parts) >= 4 and 0 <= age <= 116 and gender in (0, 1) and 0 <= race <= 4
            ages[age] += 1
        except (ValueError, AssertionError):
            invalid += 1
    sample = random.Random(42).sample(files, min(32, len(files)))
    for path in sample:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    report["images"] = {
        "jpg_files": len(files),
        "parseable_labels": sum(ages.values()),
        "invalid_filename_labels": invalid,
        "sample_images_decoded": len(sample),
        "eligible_ages_0_to_101_before_deduplication": sum(
            n for age, n in ages.items() if age <= 101
        ),
        "full_image_audit_done": False,
        "splits_created": False,
    }
    output = ROOT / "artifacts/private/asset-check.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
