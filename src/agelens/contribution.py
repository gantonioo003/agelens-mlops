"""Paired contribution metrics for the preregistered R4 versus R6 comparison."""

import numpy as np

PRIMARY_AGES = (13, 16)
PRIMARY_F2 = "val_select_primary_mean_f2_13_16"
MAE = "val_select_mae_years"


def f2_key(age):
    return f"val_select_provisional_recall_lt{age}_f2"


def add_primary(metrics):
    return {**metrics, PRIMARY_F2: sum(metrics[f2_key(age)] for age in PRIMARY_AGES) / 2}


def paired_outcome(rows, common_ages, evaluation):
    """Measure change per pair first; the two thresholds are not separate seeds."""

    def statistics(key):
        r4 = np.array([pair["r4"][key] for pair in rows], dtype=float)
        r6 = np.array([pair["r6"][key] for pair in rows], dtype=float)
        delta = r6 - r4
        if len(rows) != 3 or not np.isfinite([r4, r6]).all():
            raise ValueError("Expected exactly three finite paired observations")
        return {
            "r4_mean": float(r4.mean()),
            "r6_mean": float(r6.mean()),
            "r4_std": float(r4.std(ddof=1)),
            "r6_std": float(r6.std(ddof=1)),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(ddof=1)),
            "delta_by_seed": {
                str(row["seed"]): float(d) for row, d in zip(rows, delta, strict=True)
            },
        }

    metrics = {
        key: statistics(key)
        for key in (
            PRIMARY_F2,
            MAE,
            "val_select_age10_21_mae_years",
            *(f2_key(age) for age in sorted((*common_ages, *PRIMARY_AGES))),
        )
    }
    f2_margin = evaluation["secondary_common_F2_degradation_margin"]
    mae_margin = evaluation["secondary_MAE_degradation_margin_years"]
    return {
        "metrics": metrics,
        "primary_delta_positive_on_val_select": metrics[PRIMARY_F2]["delta_mean"] > 0,
        "common_f2_drop_beyond_margin": [
            age for age in common_ages if metrics[f2_key(age)]["delta_mean"] < -f2_margin
        ],
        "mae_increase_beyond_margin": metrics[MAE]["delta_mean"] > mae_margin,
        "common_f2_margin": f2_margin,
        "mae_margin_years": mae_margin,
    }
