"""Audit every local image and freeze deterministic, duplicate-group-aware manifests."""

import argparse
import hashlib
import io
import json
import struct
import tarfile
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from scripts.prepare_assets import ROOT, sha256


def parse_label(name):
    parts = name.split("_")
    if len(parts) != 4 or not parts[3].endswith(".jpg.chip.jpg"):
        raise ValueError("unexpected_filename_format")
    age, gender, race = map(int, parts[:3])
    stamp = parts[3].removesuffix(".jpg.chip.jpg")
    if not (0 <= age <= 116 and gender in (0, 1) and 0 <= race <= 4 and stamp.isdigit()):
        raise ValueError("invalid_filename_label")
    return age, gender, race


def difference_hash(gray):
    values = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    bits = (values[:, 1:] > values[:, :-1]).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def inspect_image(path):
    record = {
        "sample_id": hashlib.sha256(path.name.encode()).hexdigest()[:24],
        "relative_path": f"data/raw/UTKFace/{path.name}",
        "filename": path.name,
        "age": None,
        "gender": None,
        "race": None,
        "label_valid": False,
        "image_valid": False,
        "error": "",
        "bytes": path.stat().st_size,
    }
    errors = []
    try:
        record["age"], record["gender"], record["race"] = parse_label(path.name)
        record["label_valid"] = True
    except (ValueError, TypeError) as exc:
        errors.append(f"label:{exc}")
    thumbnail = None
    try:
        content = path.read_bytes()
        record["image_sha256"] = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            record["original_mode"] = image.mode
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.load()
            width, height = image.size
            if width < 1 or height < 1:
                raise ValueError("empty_image")
            record.update(width=width, height=height)
            record["pixel_sha256"] = hashlib.sha256(
                struct.pack(">II", width, height) + image.tobytes()
            ).hexdigest()
            gray = image.convert("L")
            small = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS), dtype=float)
            center = small[1:-1, 1:-1]
            laplacian = (
                small[:-2, 1:-1] + small[2:, 1:-1] + small[1:-1, :-2] + small[1:-1, 2:] - 4 * center
            )
            record.update(
                brightness=float(small.mean()),
                contrast=float(small.std()),
                sharpness=float(laplacian.var()),
                dhash=f"{difference_hash(gray):016x}",
                dhash_flip=f"{difference_hash(ImageOps.mirror(gray)):016x}",
                image_valid=True,
            )
            thumbnail = np.asarray(image.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.uint8)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        errors.append(f"image:{type(exc).__name__}:{str(exc)[:120]}")
    record["error"] = ";".join(errors)
    return record, thumbnail


class Groups:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


class HashTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        node = self.root
        if node is None:
            self.root = (value, [index], {})
            return
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = (value, [index], {})
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        matches = []
        while stack:
            stored, indices, children = stack.pop()
            distance = (value ^ stored).bit_count()
            if distance <= radius:
                matches.extend(indices)
            stack.extend(
                child
                for edge, child in children.items()
                if distance - radius <= edge <= distance + radius
            )
        return matches


def assign_duplicates(records, thumbnails, settings):
    groups = Groups(len(records))
    pixels = defaultdict(list)
    for index, record in enumerate(records):
        record["exclusion_reason"] = (
            "invalid_image"
            if not record["image_valid"]
            else "invalid_label"
            if not record["label_valid"]
            else ""
        )
        if record["image_valid"]:
            pixels[record["pixel_sha256"]].append(index)
    exact_sets = 0
    conflicting_sets = 0
    for indices in pixels.values():
        for index in indices[1:]:
            groups.union(indices[0], index)
        if len(indices) < 2:
            continue
        exact_sets += 1
        valid = sorted(
            (index for index in indices if records[index]["label_valid"]),
            key=lambda index: records[index]["filename"],
        )
        labels = {
            (records[index]["age"], records[index]["gender"], records[index]["race"])
            for index in valid
        }
        if len(labels) > 1:
            conflicting_sets += 1
            for index in valid:
                records[index]["exclusion_reason"] = "conflicting_exact_duplicate_labels"
        elif valid:
            for index in valid[1:]:
                records[index]["exclusion_reason"] = "exact_pixel_duplicate"
    tree = HashTree()
    edges = []
    candidates_checked = 0
    for index, record in enumerate(records):
        if not record["image_valid"] or record["contrast"] < settings["min_gray_std_0_to_255"]:
            continue
        candidates = set(tree.query(int(record["dhash"], 16), settings["max_hamming"]))
        candidates.update(tree.query(int(record["dhash_flip"], 16), settings["max_hamming"]))
        current = thumbnails[index].astype(np.float32) / 255
        for other in sorted(candidates):
            if record["pixel_sha256"] == records[other]["pixel_sha256"]:
                continue
            candidates_checked += 1
            previous = thumbnails[other].astype(np.float32) / 255
            mae = float(
                min(np.abs(current - previous).mean(), np.abs(current - previous[:, ::-1]).mean())
            )
            if mae <= settings["max_rgb_mae_0_to_1"]:
                groups.union(index, other)
                edges.append(
                    {
                        "left": record["sample_id"],
                        "right": records[other]["sample_id"],
                        "rgb_mae": mae,
                        "status": "conservative_similarity_group_not_identity",
                    }
                )
        tree.add(int(record["dhash"], 16), index)
    members = defaultdict(list)
    for index, record in enumerate(records):
        members[groups.find(index)].append(record["sample_id"])
    identifiers = {
        key: hashlib.sha256("|".join(sorted(value)).encode()).hexdigest()[:24]
        for key, value in members.items()
    }
    for index, record in enumerate(records):
        record["duplicate_group"] = identifiers[groups.find(index)]
    return edges, {
        "exact_pixel_duplicate_groups": exact_sets,
        "conflicting_exact_label_groups": conflicting_sets,
        "near_candidate_comparisons": candidates_checked,
        "near_edges": len(edges),
        "largest_conservative_group": max(map(len, members.values()), default=0),
    }


def make_strata(frame, boundaries):
    strata = pd.cut(frame.age, bins=boundaries, right=False, labels=False).astype(int)
    strata.index = frame.index
    # Merge only strata lacking enough distinct groups for five grouped folds.
    while True:
        counts = frame.assign(_stratum=strata).groupby("_stratum").duplicate_group.nunique()
        insufficient = counts[counts < 5]
        if insufficient.empty or len(counts) == 1:
            break
        old = int(insufficient.index[0])
        nearest = min(
            (int(key) for key in counts.index if key != old), key=lambda key: (abs(key - old), key)
        )
        strata = strata.replace(old, nearest)
    return strata


def grouped_folds(frame, n_splits, seed, strata):
    if frame.duplicate_group.nunique() < n_splits:
        raise ValueError("Insufficient independent groups for requested folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = pd.Series(-1, index=frame.index, dtype=int)
    for fold, (_, heldout) in enumerate(splitter.split(frame, strata, frame.duplicate_group)):
        result.iloc[heldout] = fold
    if (result < 0).any():
        raise AssertionError("Missing fold assignment")
    return result


def capped_group_sample(frame, maximum, seed):
    if len(frame) <= maximum:
        return frame.index.tolist()
    rng = np.random.default_rng(seed)
    group_table = frame.groupby("duplicate_group", sort=True).agg(
        size=("sample_id", "size"), stratum=("stratum", "min"), base_split=("base_split", "first")
    )
    group_table["random_order"] = rng.random(len(group_table))
    cells = list(group_table.groupby(["base_split", "stratum"], sort=True))
    quotas = {key: maximum * int(cell["size"].sum()) / len(frame) for key, cell in cells}
    consumed = Counter()
    accepted = set()
    remainder = []
    count = 0
    for key, cell in cells:
        for group, row in cell.sort_values("random_order").iterrows():
            size = int(row["size"])
            if consumed[key] + size <= quotas[key] and count + size <= maximum:
                accepted.add(group)
                consumed[key] += size
                count += size
            else:
                remainder.append((key, group, size, float(row["random_order"])))
    while remainder and count < maximum:
        remainder.sort(key=lambda value: (consumed[value[0]] - quotas[value[0]], value[3]))
        key, group, size, _ = remainder.pop(0)
        if count + size <= maximum:
            accepted.add(group)
            consumed[key] += size
            count += size
    return frame.index[frame.duplicate_group.isin(accepted)].tolist()


def validate_manifests(frame, maximum):
    allowed = {"train", "val_select", "val_calibration", "test"}
    assert set(frame.split) == allowed, "Empty or invalid partition"
    assert len(frame) <= maximum and frame.sample_id.is_unique
    assert frame.age.between(0, 101).all()
    for key in ["sample_id", "image_sha256", "pixel_sha256", "duplicate_group"]:
        assert frame.groupby(key).split.nunique().max() == 1, f"Leakage through {key}"
    train = frame[frame.split == "train"]
    assert set(train.cv_fold.astype(int)) == {0, 1, 2}
    for key in ["image_sha256", "pixel_sha256", "duplicate_group"]:
        assert train.groupby(key).cv_fold.nunique().max() == 1, f"CV leakage through {key}"
    assert frame[frame.split != "train"].cv_fold.eq(-1).all()
    numeric = [
        name
        for name in [
            "age",
            "gender",
            "race",
            "width",
            "height",
            "bytes",
            "brightness",
            "contrast",
            "sharpness",
            "reference_folder",
            "stratum",
            "cv_fold",
        ]
        if name in frame
    ]
    assert np.isfinite(frame[numeric].to_numpy(dtype=float)).all()
    for name in ["image_valid", "label_valid"]:
        if name in frame:
            assert frame[name].eq(True).all()


def load_reference(settings):
    path = ROOT / "data/raw/reference/UTKFace_aligned.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(
            settings["reference_url"], headers={"User-Agent": "AgeLens-research"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(20 * 1024**2 + 1)
        if len(payload) > 20 * 1024**2:
            raise ValueError("Unexpected reference metadata size")
        json.loads(payload)
        path.write_bytes(payload)
    reference = json.loads(path.read_text())
    by_name = defaultdict(list)
    for row in reference:
        by_name[Path(row["img_path"]).name].append(row)
    return by_name, sha256(path)


def run_audit(workers=4):
    protocol = json.loads((ROOT / "configs/protocol-v1.json").read_text())
    settings = json.loads((ROOT / "configs/audit-v1.json").read_text())
    registry = json.loads((ROOT / "configs/assets.json").read_text())
    archive = ROOT / registry["dataset"]["path"]
    if sha256(archive) != registry["dataset"]["sha256"]:
        raise ValueError("Raw archive changed")
    files = sorted((ROOT / "data/raw/UTKFace").iterdir())
    files = [path for path in files if path.is_file()]
    if not files:
        raise ValueError("No input images")
    records, thumbnails = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (record, thumbnail) in enumerate(pool.map(inspect_image, files), 1):
            records.append(record)
            thumbnails.append(thumbnail)
            if index % 2000 == 0:
                print(f"Decoded and hashed {index}/{len(files)}", flush=True)
    with tarfile.open(archive, "r:gz") as source:
        expected = {}
        for member in source:
            if member.isfile() and Path(member.name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                with source.extractfile(member) as stream:
                    expected[Path(member.name).name] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
    if set(expected) != {record["filename"] for record in records}:
        raise ValueError("Extracted file inventory differs from verified archive")
    if any(expected[record["filename"]] != record.get("image_sha256") for record in records):
        raise ValueError("Extracted image bytes differ from verified archive")
    print("Checking exact and conservative perceptual duplicates...", flush=True)
    edges, duplicate_stats = assign_duplicates(records, thumbnails, settings["near_duplicates"])
    for record in records:
        if (
            not record["exclusion_reason"]
            and not protocol["dataset"]["age_min"]
            <= record["age"]
            <= protocol["dataset"]["age_max"]
        ):
            record["exclusion_reason"] = "outside_age_scope"
    audit = pd.DataFrame(records)
    eligible = audit[audit.exclusion_reason == ""].copy()
    eligible[["age", "gender", "race"]] = eligible[["age", "gender", "race"]].astype(int)
    reference, reference_digest = load_reference(settings)
    reference_matches = 0
    eligible["reference_folder"] = -1
    eligible["base_split"] = pd.Series(index=eligible.index, dtype="object")
    for index, row in eligible.iterrows():
        matches = reference.get(row.filename.removesuffix(".chip.jpg"), [])
        valid = [
            entry
            for entry in matches
            if entry.get("age") == row.age
            and isinstance(entry.get("folder"), int)
            and 0 <= entry["folder"] <= 9
        ]
        folders = {entry["folder"] for entry in valid}
        if valid and len(folders) == 1 and len(valid) == len(matches):
            folder = next(iter(folders))
            eligible.loc[index, "reference_folder"] = folder
            eligible.loc[index, "base_split"] = (
                "train" if folder < 6 else "validation" if folder < 8 else "test"
            )
            reference_matches += 1
    reference_group_conflicts = int(
        eligible.groupby("duplicate_group").base_split.nunique().gt(1).sum()
    )
    use_reference = reference_matches == len(eligible) and reference_group_conflicts == 0
    eligible["stratum"] = make_strata(eligible, protocol["splits"]["fallback_age_boundaries"])
    if not use_reference:
        assignments = grouped_folds(eligible, 5, 42, eligible.stratum)
        eligible["base_split"] = assignments.map(
            {0: "train", 1: "train", 2: "train", 3: "validation", 4: "test"}
        )
    eligible["reference_folder"] = eligible.reference_folder.fillna(-1).astype(int)
    chosen = capped_group_sample(eligible, protocol["dataset"]["max_total_samples"], 42)
    selected = eligible.loc[chosen].copy()
    selected["split"] = selected.base_split.replace("validation", "val_select")
    validation = selected[selected.base_split == "validation"]
    val_folds = grouped_folds(validation, 2, 42, validation.stratum)
    selected.loc[validation.index, "split"] = val_folds.map({0: "val_select", 1: "val_calibration"})
    selected["cv_fold"] = -1
    train = selected[selected.split == "train"]
    selected.loc[train.index, "cv_fold"] = grouped_folds(train, 3, 42, train.stratum)
    selected["cv_fold"] = selected.cv_fold.astype(int)
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    selected["source"] = "UTKFace_py97_mirror"
    validate_manifests(selected, protocol["dataset"]["max_total_samples"])
    audit["selected"] = audit.sample_id.isin(selected.sample_id)
    audit.loc[(audit.exclusion_reason == "") & ~audit.selected, "exclusion_reason"] = (
        "outside_sample_cap"
    )
    directory = ROOT / "data/manifests"
    directory.mkdir(parents=True, exist_ok=True)
    audit.to_csv(directory / "audit.csv", index=False, lineterminator="\n")
    selected.to_csv(directory / "selected.csv", index=False, lineterminator="\n")
    for partition in ["train", "val_select", "val_calibration", "test"]:
        selected[selected.split == partition].to_csv(
            directory / f"{partition}.csv", index=False, lineterminator="\n"
        )
    pd.DataFrame(edges, columns=["left", "right", "rgb_mae", "status"]).to_csv(
        directory / "near_duplicate_pairs.csv", index=False, lineterminator="\n"
    )
    split_support = {
        str(threshold): {
            partition: {
                "minor": int((part.age < threshold).sum()),
                "adult": int((part.age >= threshold).sum()),
            }
            for partition, part in selected.groupby("split")
        }
        for threshold in protocol["evaluation"]["thresholds"]
    }
    summary = {
        "audit_id": settings["audit_id"],
        "raw_archive_sha256": registry["dataset"]["sha256"],
        "files_audited": len(audit),
        "images_decoded": int(audit.image_valid.sum()),
        "valid_filename_labels": int(audit.label_valid.sum()),
        "eligible_before_cap": len(eligible),
        "selected": len(selected),
        "exclusion_counts": audit.exclusion_reason.replace("", "selected").value_counts().to_dict(),
        "duplicates": duplicate_stats,
        "byte_duplicate_groups": int(
            audit[audit.image_valid].groupby("image_sha256").size().gt(1).sum()
        ),
        "audit_policy_sha256": sha256(ROOT / "configs/audit-v1.json"),
        "audit_code_sha256": sha256(ROOT / "scripts/audit_data.py"),
        "reference": {
            "revision": settings["reference_revision"],
            "sha256": reference_digest,
            "eligible_matches": reference_matches,
            "eligible_total": len(eligible),
            "cross_partition_similarity_groups": reference_group_conflicts,
            "used": use_reference,
        },
        "split_method": "reference_split_0"
        if use_reference
        else "registered_grouped_stratified_fallback",
        "split_sizes": selected.split.value_counts().to_dict(),
        "support": split_support,
        "manifest_sha256": {file.name: sha256(file) for file in sorted(directory.glob("*.csv"))},
        "subject_exclusive_verified": False,
        "trained": False,
        "archive_inventory_verified": True,
        "full_technical_image_audit_done": True,
        "invariants_passed": True,
        "near_duplicate_identity_verified": False,
    }
    output = ROOT / "reports/task04"
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "files_audited",
                    "images_decoded",
                    "eligible_before_cap",
                    "selected",
                    "split_method",
                    "split_sizes",
                    "duplicates",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    run_audit(parser.parse_args().workers)
