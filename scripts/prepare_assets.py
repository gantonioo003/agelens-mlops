"""Acquire pinned research assets without training or feature extraction."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(asset, root=ROOT):
    target = root / asset["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = sha256(target)
        if target.stat().st_size != asset["size"]:
            raise ValueError(f"Existing asset has wrong size: {target.name}")
        if asset.get("sha256") and digest != asset["sha256"]:
            raise ValueError(f"Existing asset has wrong SHA-256: {target.name}")
        return target, digest
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "AgeLens-research/0.1"})
    digest = hashlib.sha256()
    total = 0
    print(f"Downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB)", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > asset["size"]:
                    raise ValueError("Asset exceeds its declared size")
                digest.update(chunk)
                out.write(chunk)
        if total != asset["size"]:
            raise ValueError("Truncated asset")
        observed = digest.hexdigest()
        if asset.get("sha256") and observed != asset["sha256"]:
            raise ValueError("Asset SHA-256 mismatch")
        partial.replace(target)
        return target, observed
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def extract_images(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            # Never delegate filesystem paths, links or device nodes to tar.extractall.
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Unsafe archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Links and special archive entries are forbidden")
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if member.size > 20 * 1024 * 1024:
                raise ValueError("Unexpected image size")
            total += member.size
            count += 1
            if total > 2 * 1024**3 or count > 100000:
                raise ValueError("Archive exceeds extraction budget")
            if path.name in seen:
                raise ValueError("Duplicate image filename in archive")
            seen.add(path.name)
            target = destination / path.name
            if target.resolve().parent != destination or target.is_symlink():
                raise ValueError("Unsafe output path")
            with tar.extractfile(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
    if not count:
        raise ValueError("Archive contains no images")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-research-terms", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files and record provenance; do not download/extract",
    )
    args = parser.parse_args()
    if not args.accept_research_terms:
        parser.error("Read docs/data-sources.md and explicitly accept research-only terms")
    registry = json.loads((ROOT / "configs/assets.json").read_text(encoding="utf-8"))
    report = {"checked_at": datetime.now(UTC).isoformat(), "assets": {}}
    for key, asset in registry.items():
        if args.verify_only and not (ROOT / asset["path"]).is_file():
            raise FileNotFoundError(f"Missing local asset: {asset['path']}")
        path, digest = download(asset)
        report["assets"][key] = {
            "name": asset["name"],
            "sha256": digest,
            "bytes": path.stat().st_size,
            "source_kind": asset["source_kind"],
            "url": asset["url"],
        }
        if key == "dataset":
            count = (
                len(list((ROOT / "data/raw/UTKFace").glob("*.jpg")))
                if args.verify_only
                else extract_images(path, ROOT / "data/raw/UTKFace")
            )
            if not count:
                raise ValueError("No extracted images found")
            report["assets"][key]["extracted_images"] = count
            print(f"Extracted {count} images; SHA-256 {digest}", flush=True)
        output = ROOT / "artifacts/private/acquisition.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Assets verified. No models trained.", flush=True)


if __name__ == "__main__":
    main()
