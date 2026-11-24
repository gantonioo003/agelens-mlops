import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image, PngImagePlugin

from scripts.audit_data import (
    HashTree,
    assign_duplicates,
    capped_group_sample,
    grouped_folds,
    inspect_image,
    parse_label,
    validate_manifests,
)
from scripts.prepare_assets import ROOT


@pytest.mark.parametrize("age", [0, 12, 18, 101, 116])
def test_parse_age_boundaries(age):
    assert parse_label(f"{age}_1_4_20170101000000.jpg.chip.jpg") == (age, 1, 4)


@pytest.mark.parametrize(
    "name",
    [
        "18_2_0_123.jpg.chip.jpg",
        "117_0_0_123.jpg.chip.jpg",
        "18_0_5_123.jpg.chip.jpg",
        "broken.jpg",
        "18_0_0_x.jpg.chip.jpg",
    ],
)
def test_reject_bad_labels(name):
    with pytest.raises(ValueError):
        parse_label(name)


def fixture_image(tmp_path, age, suffix, values=None, metadata=None):
    path = tmp_path / f"{age}_0_0_{suffix}.jpg.chip.jpg"
    if values is None:
        values = np.random.default_rng(42).integers(20, 220, (64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(values)
    image.save(path, format="PNG", pnginfo=metadata)
    return inspect_image(path)


def settings():
    return json.loads((ROOT / "configs/audit-v1.json").read_text())["near_duplicates"]


def test_decoded_pixel_duplicates_and_conflicting_labels(tmp_path):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("example", "different encoding metadata")
    first, thumb_a = fixture_image(tmp_path, 18, "123")
    second, thumb_b = fixture_image(tmp_path, 18, "124", metadata=meta)
    assert first["image_sha256"] != second["image_sha256"]
    assert first["pixel_sha256"] == second["pixel_sha256"]
    edges, stats = assign_duplicates([first, second], [thumb_a, thumb_b], settings())
    assert not edges and stats["exact_pixel_duplicate_groups"] == 1
    assert second["exclusion_reason"] == "exact_pixel_duplicate"
    third, thumb_c = fixture_image(tmp_path, 21, "125")
    assign_duplicates([first, third], [thumb_a, thumb_c], settings())
    assert (
        first["exclusion_reason"]
        == third["exclusion_reason"]
        == "conflicting_exact_duplicate_labels"
    )


def test_similarity_is_grouped_not_removed(tmp_path):
    values = np.random.default_rng(11).integers(30, 200, (64, 64, 3), dtype=np.uint8)
    first, a = fixture_image(tmp_path, 16, "100", values)
    second, b = fixture_image(tmp_path, 17, "101", values + 1)
    edges, _ = assign_duplicates([first, second], [a, b], settings())
    assert len(edges) == 1
    assert first["duplicate_group"] == second["duplicate_group"]
    assert first["exclusion_reason"] == second["exclusion_reason"] == ""


def test_corrupt_image_is_recorded(tmp_path):
    path = tmp_path / "18_0_0_123.jpg.chip.jpg"
    path.write_bytes(b"not an image")
    record, thumbnail = inspect_image(path)
    assert record["label_valid"] and not record["image_valid"]
    assert record["error"].startswith("image:") and thumbnail is None


def test_hash_tree_matches_brute_force():
    rng = np.random.default_rng(41)
    values = [int(value) for value in rng.integers(0, 2**32, size=150, dtype=np.uint64)]
    tree = HashTree()
    for index, value in enumerate(values):
        tree.add(value, index)
    for value in values[::7]:
        expected = {
            index for index, stored in enumerate(values) if (value ^ stored).bit_count() <= 4
        }
        assert set(tree.query(value, 4)) == expected


def test_grouped_splits_and_cap_are_deterministic():
    frame = pd.DataFrame(
        {
            "sample_id": range(300),
            "duplicate_group": np.repeat(range(100), 3),
            "stratum": np.repeat(np.arange(100) % 5, 3),
        }
    )
    folds = grouped_folds(frame, 5, 42, frame.stratum)
    assert folds.equals(grouped_folds(frame, 5, 42, frame.stratum))
    assert frame.assign(fold=folds).groupby("duplicate_group").fold.nunique().max() == 1
    frame["base_split"] = folds.map(
        {0: "train", 1: "train", 2: "train", 3: "validation", 4: "test"}
    )
    chosen = capped_group_sample(frame, 101, 42)
    assert len(chosen) <= 101 and chosen == capped_group_sample(frame, 101, 42)
    assert frame.loc[chosen].groupby("duplicate_group").size().eq(3).all()


def test_leakage_invariants_reject_cross_split_duplicates():
    frame = pd.DataFrame(
        {
            "sample_id": list("abcdef"),
            "age": [18] * 6,
            "image_sha256": list("abcdef"),
            "pixel_sha256": list("abcdef"),
            "duplicate_group": list("abcdef"),
            "split": ["train"] * 3 + ["val_select", "val_calibration", "test"],
            "cv_fold": [0, 1, 2, -1, -1, -1],
        }
    )
    validate_manifests(frame, 6)
    frame.loc[5, "pixel_sha256"] = "a"
    with pytest.raises(AssertionError, match="Leakage"):
        validate_manifests(frame, 6)
    frame.loc[5, "pixel_sha256"] = "f"
    frame.loc[1, "image_sha256"] = "a"
    with pytest.raises(AssertionError, match="CV leakage"):
        validate_manifests(frame, 6)


def test_csv_roundtrip_allows_empty_error_column():
    frame = pd.DataFrame(
        {
            "sample_id": list("abcdef"),
            "age": [18] * 6,
            "image_sha256": list("abcdef"),
            "pixel_sha256": list("abcdef"),
            "duplicate_group": list("abcdef"),
            "error": [np.nan] * 6,
            "split": ["train"] * 3 + ["val_select", "val_calibration", "test"],
            "cv_fold": [0, 1, 2, -1, -1, -1],
        }
    )
    validate_manifests(frame, 6)
