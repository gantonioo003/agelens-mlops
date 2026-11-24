import json
from pathlib import Path

import pytest

from agelens.contribution import MAE, PRIMARY_F2, add_primary, f2_key, paired_outcome

PROTOCOL = json.loads((Path(__file__).parents[1] / "configs/protocol-v1.json").read_text())
COMMON = (12, 15, 18, 21)


def fixture(f2_13, f2_16, mae=4.5, f2_18=0.75):
    return add_primary(
        {
            **{f2_key(age): 0.75 for age in COMMON},
            f2_key(13): f2_13,
            f2_key(16): f2_16,
            f2_key(18): f2_18,
            MAE: mae,
            "val_select_age10_21_mae_years": 3.5,
        }
    )


def test_primary_average_then_paired_delta_with_three_seeds():
    rows = [
        {"seed": 17, "r4": fixture(0.5, 0.6), "r6": fixture(0.5, 0.8)},
        {"seed": 42, "r4": fixture(0.4, 0.4), "r6": fixture(0.6, 0.6)},
        {"seed": 2026, "r4": fixture(0.2, 0.4), "r6": fixture(0.2, 0.4)},
    ]
    result = paired_outcome(rows, COMMON, PROTOCOL["evaluation"])
    primary = result["metrics"][PRIMARY_F2]
    assert primary["delta_by_seed"] == pytest.approx({"17": 0.1, "42": 0.2, "2026": 0.0})
    assert primary["delta_mean"] == pytest.approx(0.1)
    assert primary["delta_std"] == pytest.approx(0.1)
    assert result["primary_delta_positive_on_val_select"]


def test_common_head_and_mae_degradation_flags_are_separate_from_primary():
    rows = [
        {"seed": seed, "r4": fixture(0.6, 0.6), "r6": fixture(0.7, 0.7, mae=5.1, f2_18=0.73)}
        for seed in (17, 42, 2026)
    ]
    result = paired_outcome(rows, COMMON, PROTOCOL["evaluation"])
    assert result["common_f2_drop_beyond_margin"] == [18]
    assert result["mae_increase_beyond_margin"]
    assert result["primary_delta_positive_on_val_select"]
    with pytest.raises(ValueError, match="three finite paired"):
        paired_outcome(rows[:2], COMMON, PROTOCOL["evaluation"])
