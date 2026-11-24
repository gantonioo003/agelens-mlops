"""Protocol-bound data loading, candidates and selection diagnostics for task 07."""

import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agelens.cache_access import open_approved_cache
from agelens.farl import file_sha256

FAMILIES = ("dummy", "knn", "random_forest", "mlp")


@dataclass(frozen=True)
class Split:
    features: np.ndarray
    age: np.ndarray
    folds: np.ndarray


def load_comparison_data(root):
    protocol = json.loads((root / "configs/protocol-v1.json").read_text(encoding="utf-8"))
    directory, metadata = open_approved_cache(root)
    matrix_path = directory / "features.npy"
    # The full-file hash verifies storage integrity; only permitted rows are materialized.
    if file_sha256(matrix_path) != metadata["file_sha256"]["features.npy"]:
        raise ValueError("Frozen feature matrix checksum mismatch")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    loaded = {}
    groups = {}
    positions = {}
    for name in ("train", "val_select"):
        manifest = root / "data/manifests" / f"{name}.csv"
        expected = protocol["data_audit"]["manifest_sha256"][manifest.name]
        index_path = directory / f"indices_{name}.npy"
        if file_sha256(manifest) != expected:
            raise ValueError(f"Frozen {name} manifest changed")
        if file_sha256(index_path) != metadata["file_sha256"][index_path.name]:
            raise ValueError(f"Frozen {name} indices changed")
        table = pd.read_csv(manifest, usecols=["split", "age", "cv_fold", "duplicate_group"])
        index = np.load(index_path, allow_pickle=False)
        if (
            not table["split"].eq(name).all()
            or len(table) != len(index)
            or len(table) != protocol["data_audit"]["split_sizes"][name]
            or not np.all(np.diff(index) > 0)
        ):
            raise ValueError(f"Unexpected frozen {name} row alignment")
        positions[name] = index
        groups[name] = table["duplicate_group"]
        loaded[name] = Split(
            matrix[index],
            table.age.to_numpy(dtype=np.int64),
            table.cv_fold.to_numpy(dtype=np.int64),
        )
        if table.groupby("duplicate_group")["cv_fold"].nunique().max() != 1:
            raise ValueError("A duplicate group crosses CV folds")
    if (
        set(groups["train"]) & set(groups["val_select"])
        or np.intersect1d(positions["train"], positions["val_select"]).size
    ):
        raise ValueError("Training and selection overlap")
    train, selection = loaded["train"], loaded["val_select"]
    if set(train.folds) != set(range(protocol["tuning"]["folds"])) or set(selection.folds) != {-1}:
        raise ValueError("Unexpected frozen fold IDs")
    return protocol, train, selection


def candidate_grid(protocol, family):
    spec = protocol["tuning"][family]
    if family == "dummy":
        return [dict(spec)]
    keys = {
        "knn": ("n_neighbors", "weights", "p"),
        "random_forest": ("n_estimators", "max_depth", "min_samples_leaf", "max_features"),
        "mlp": ("lr", "weight_decay"),
    }[family]
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(spec[k] for k in keys))
    ]


def cv_summary(fold_mae):
    if len(fold_mae) != 3 or not np.isfinite(fold_mae).all():
        raise ValueError("Expected three finite fold MAEs")
    return {
        "cv_mean_mae_years": float(np.mean(fold_mae)),
        "cv_std_mae_years": float(np.std(fold_mae, ddof=1)),
    }


def select_candidate(records):
    return min(
        records,
        key=lambda row: (row["cv_mean_mae_years"], row["cv_std_mae_years"], row["candidate"]),
    )


def sklearn_estimator(family, parameters, seed, jobs):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if family == "dummy":
        return DummyRegressor(**parameters)
    if family == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(**parameters, n_jobs=jobs))
    if family == "random_forest":
        return RandomForestRegressor(**parameters, random_state=seed, n_jobs=jobs)
    raise ValueError(f"Unknown sklearn model {family}")


def mae_years(age, predicted):
    if predicted.shape != age.shape or not np.isfinite(predicted).all():
        raise ValueError("Invalid age predictions")
    return float(np.mean(np.abs(age - predicted)))


def binary_counts(positive, predicted_positive):
    tp = int(np.count_nonzero(positive & predicted_positive))
    fp = int(np.count_nonzero(~positive & predicted_positive))
    fn = int(np.count_nonzero(positive & ~predicted_positive))
    tn = int(np.count_nonzero(~positive & ~predicted_positive))
    if not (tp + fn and fp + tn):
        raise ValueError("Both age groups must be represented")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "f2": 5 * tp / (5 * tp + fp + 4 * fn),
        "fpr": fp / (fp + tn),
        "fnr": fn / (tp + fn),
    }


def selection_metrics(ages, predictions, evaluation):
    mae = mae_years(ages, predictions)
    target = evaluation["target_calibration_recall"]
    teenager = (ages >= 10) & (ages <= 21)
    metrics = {"val_select_mae_years": mae}
    if teenager.any():
        metrics["val_select_age10_21_mae_years"] = mae_years(ages[teenager], predictions[teenager])
    scores = -predictions
    for threshold in evaluation["thresholds"]:
        positive = ages < threshold
        eligible = np.sort(scores[positive])
        # Highest observed score that preserves at least the requested positives (including ties).
        cutoff = float(eligible[len(eligible) - math.ceil(target * len(eligible))])
        for label, decision in (
            ("raw", predictions < threshold),
            ("provisional_recall", scores >= cutoff),
        ):
            prefix = f"val_select_{label}_lt{threshold}"
            for name, value in binary_counts(positive, decision).items():
                metrics[f"{prefix}_{name}"] = value
        metrics[f"val_select_provisional_recall_lt{threshold}_cutoff"] = cutoff
    return metrics
