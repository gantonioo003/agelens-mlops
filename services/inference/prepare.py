"""Freeze deployment selection before materializing the calibration split."""

import csv
import json
import math
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from agelens.age_mlp import predict_age, setup_torch
from agelens.baselines import binary_counts
from agelens.cache_access import open_approved_cache
from agelens.farl import file_sha256
from agelens.tracking import PROVENANCE_PATH, tracking_client
from scripts.prepare_assets import ROOT
from services.inference.model import load_head, read_json

PREFIX = "val_select_provisional_recall_lt18_"
VERSION = "agelens-r0-v1"


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen policy cannot be replaced: {path.name}")
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def rank(candidates, seeds, recall):
    ranking = []
    for name, rows in candidates.items():
        if {row["seed"] for row in rows} != set(seeds) or len(rows) != len(seeds):
            raise ValueError("Incomplete seed inventory")
        if not all(
            math.isfinite(row[key]) for row in rows for key in ("f2", "recall", "fpr", "mae")
        ):
            raise ValueError("Non-finite selection metrics")
        ranking.append(
            {
                "variant": name,
                "eligible": all(
                    row["recall"] >= recall and row["tn"] + row["fn"] > 0 for row in rows
                ),
                "mean_f2": mean(row["f2"] for row in rows),
                "mean_fpr": mean(row["fpr"] for row in rows),
                "mean_mae": mean(row["mae"] for row in rows),
                "runs": rows,
            }
        )

    def key(row):
        return (not row["eligible"], -row["mean_f2"], row["mean_fpr"], row["mean_mae"])

    ranking.sort(key=key)
    if not ranking[0]["eligible"]:
        raise ValueError("No eligible nontrivial model")
    if len(ranking) > 1 and key(ranking[0]) == key(ranking[1]):
        raise ValueError("Measure serving cost before resolving this tie")
    return ranking


def archived_runs(root):
    task07 = read_json(root / "reports/task07/verification.json")
    task08 = read_json(root / "reports/task08/verification.json")
    task09 = read_json(root / "reports/task09/verification.json")
    with (root / "reports/task07/by_seed.csv").open(encoding="utf-8", newline="") as stream:
        runs = [
            ("R0" if row["family"] == "mlp" else row["family"], int(row["seed"]), row["run_id"])
            for row in csv.DictReader(stream)
            if row["run_id"] in task07["run_ids_verified"]
        ]
    runs += [("R4", row["seed"], row["r4"]) for row in task08["paired_runs"]]
    runs += [("R6", row["seed"], row["r6_run"]) for row in task09["pairs"]]
    if not all(
        proof["completed"] and proof["test_rows_materialized"] == 0
        for proof in (task07, task08, task09)
    ):
        raise ValueError("Selection needs verified results without test use")
    if len({task07["experiment_id"], task08["experiment_id"], task09["experiment_id"]}) != 1:
        raise ValueError("Archive experiment mismatch")
    return task07["experiment_id"], runs


def select(root, protocol, protocol_hash):
    experiment, runs = archived_runs(root)
    client = tracking_client(root)
    candidates = {}
    for name, seed, run_id in runs:
        run = client.get_run(run_id)
        params, metrics = run.data.params, run.data.metrics
        if (
            run.info.experiment_id != experiment
            or run.info.status != "FINISHED"
            or params["seed"] != str(seed)
            or params["evaluation_partition"] != "val_select"
            or run.data.tags["agelens.cache_fingerprint"]
            != protocol["feature_extraction"]["fingerprint"]
            or (params.get("variant") or params.get("family")) != name
        ):
            raise ValueError(f"Unapproved selection run: {run_id}")
        row = {"seed": seed, "run_id": run_id, "mae": metrics["val_select_mae_years"]}
        row.update({key: metrics[PREFIX + key] for key in ("f2", "recall", "fpr", "tn", "fn")})
        candidates.setdefault(name, []).append(row)
    ranking = rank(
        candidates,
        protocol["training"]["seeds"],
        protocol["evaluation"]["target_calibration_recall"],
    )
    winner = ranking[0]
    if winner["variant"] != "R0":
        raise ValueError("The winning architecture requires a different serving adapter")
    refs = {}
    for row in winner["runs"]:
        run_id = row["run_id"]
        with TemporaryDirectory(dir=root / "runtime") as temp:
            record = read_json(
                Path(client.download_artifacts(run_id, "evaluation/result.json", temp))
            )
            origin = read_json(Path(client.download_artifacts(run_id, PROVENANCE_PATH, temp)))
        if origin["protocol_sha256"] != protocol_hash or record["run_id"] != run_id:
            raise ValueError("R0 reference differs from archive")
        refs[str(row["seed"])] = {
            "variant": "R0",
            "seed": row["seed"],
            "run_id": run_id,
            "checkpoint_sha256": record["model"]["sha256"],
        }
    # The already-registered tuning seed is the representative serving seed, not the best validation seed.
    serving_seed = str(protocol["tuning"]["seed"])
    return {
        "partition": "val_select",
        "criterion": protocol["evaluation"]["deployment_selection"],
        "protocol_sha256": protocol_hash,
        "experiment_id": experiment,
        "ranking": ranking,
        "deployment": refs[serving_seed],
        "checkpoints_by_seed": refs,
    }


def calibration_split(root, protocol):
    directory, metadata = open_approved_cache(root)
    matrix, index = directory / "features.npy", directory / "indices_val_calibration.npy"
    manifest = root / "data/manifests/val_calibration.csv"
    for path in (matrix, index):
        if file_sha256(path) != metadata["file_sha256"][path.name]:
            raise ValueError(f"Invalid cache file: {path.name}")
    if file_sha256(manifest) != protocol["data_audit"]["manifest_sha256"][manifest.name]:
        raise ValueError("Calibration manifest hash changed")
    rows = pd.read_csv(manifest, usecols=["age", "split"])
    indices = np.load(index, allow_pickle=False)
    if (
        len(indices) != len(rows)
        or len(rows) != protocol["data_audit"]["split_sizes"]["val_calibration"]
        or not rows["split"].eq("val_calibration").all()
        or not np.all(np.diff(indices) > 0)
    ):
        raise ValueError("Calibration alignment mismatch")
    return np.load(matrix, mmap_mode="r", allow_pickle=False)[indices], rows.age.to_numpy()


def operating_point(scores, positive, target):
    if not np.isfinite(scores).all() or positive.sum() < 100 or not (~positive).any():
        raise ValueError("Insufficient class support or invalid scores")
    values = np.sort(scores[positive])
    cutoff = float(values[len(values) - math.ceil(target * len(values))])
    metrics = binary_counts(positive, scores >= cutoff)
    if metrics["recall"] < target:
        raise ValueError("Calibration recall below target")
    return {"cutoff": cutoff, "metrics": metrics}


def main():
    setup_torch("cpu", 4)
    protocol_path = ROOT / "configs/protocol-v1.json"
    protocol = read_json(protocol_path)
    selection_path = ROOT / "configs/deployment-selection-v1.json"
    selection = select(ROOT, protocol, file_sha256(protocol_path))
    freeze(selection_path, selection)
    print("Frozen selection:", selection["deployment"], flush=True)
    features, ages = calibration_split(ROOT, protocol)
    evaluation = protocol["evaluation"]
    target = evaluation["target_calibration_recall"]
    results = {}
    for seed, ref in selection["checkpoints_by_seed"].items():
        scores = -predict_age(load_head(ROOT, protocol, ref), features)
        results[seed] = {
            "run_id": ref["run_id"],
            "thresholds": {
                str(t): operating_point(scores, ages < t, target) for t in evaluation["thresholds"]
            },
        }
    policy = {
        "version": VERSION,
        "partition": "val_calibration",
        "selection_sha256": file_sha256(selection_path),
        "cache_fingerprint": protocol["feature_extraction"]["fingerprint"],
        "samples": len(ages),
        "target_recall": target,
        "test_used": False,
        "by_seed": results,
    }
    freeze(ROOT / "configs/deployment-calibration-v1.json", policy)
    print(
        f"Calibrated {len(results)} seeds on {len(ages)} val_calibration rows; test untouched.",
        flush=True,
    )


if __name__ == "__main__":
    main()
