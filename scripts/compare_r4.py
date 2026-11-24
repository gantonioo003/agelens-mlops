"""Task 08: compare R4 with the verified task 07 R0 checkpoints on val_select."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from mlflow.entities import Metric

from agelens.age_mlp import AgeMLP, predict_age, setup_torch
from agelens.baselines import binary_counts, load_comparison_data, selection_metrics
from agelens.farl import file_sha256
from agelens.feature_cache import write_json_atomic
from agelens.multi_age import (
    COMMON_THRESHOLDS,
    MultiAgeMLP,
    age_gap_bounds,
    binary_probabilities,
    draw_digest,
    fit_multi_age,
    paired_model,
)
from agelens.tracking import (
    PROVENANCE_PATH,
    collect_provenance,
    ensure_experiment,
    tracked_run,
    tracking_client,
    tracking_config,
)
from scripts.compare_baselines import RESULT, log_metrics
from scripts.prepare_assets import ROOT

CAMPAIGN = "task08-r4-v1"
HISTORY_PATH = "evaluation/epochs.json"
MODEL_PATH = "model/age_mlp_r4.pt"
PRIVATE = ROOT / "artifacts/private/task08" / CAMPAIGN
REPORT = ROOT / "reports/task08"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_artifact(client, run_id, artifact):
    with TemporaryDirectory(dir=ROOT / "runtime") as tmp:
        return read_json(Path(client.download_artifacts(run_id, artifact, tmp)))


def checkpoint(client, record, model, artifact, device):
    with TemporaryDirectory(dir=ROOT / "runtime") as tmp:
        path = Path(client.download_artifacts(record["run_id"], artifact, tmp))
        require(file_sha256(path) == record["model"]["sha256"], "Checkpoint checksum mismatch")
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
    return model.to(device).eval()


def scores_r0(model, selection, protocol):
    evaluation = {**protocol["evaluation"], "thresholds": list(COMMON_THRESHOLDS)}
    return selection_metrics(selection.age, predict_age(model, selection.features), evaluation)


def scores_r4(model, selection, protocol):
    evaluation = {**protocol["evaluation"], "thresholds": []}
    metrics = selection_metrics(selection.age, predict_age(model, selection.features), evaluation)
    probs = binary_probabilities(model, selection.features)
    require(np.isfinite(probs).all(), "Invalid binary probabilities")
    for offset, age in enumerate(model.thresholds):
        scores = probs[:, offset]
        positive = selection.age < age
        ordered = np.sort(scores[positive])
        count = math.ceil(evaluation["target_calibration_recall"] * len(ordered))
        require(count > 0, "Threshold must have eligible positives")
        cutoff = float(ordered[len(ordered) - count])
        for label, decision in (("raw", scores >= 0.5), ("provisional_recall", scores >= cutoff)):
            prefix = f"val_select_{label}_lt{age}"
            metrics.update(
                {f"{prefix}_{key}": val for key, val in binary_counts(positive, decision).items()}
            )
        metrics[f"val_select_provisional_recall_lt{age}_cutoff"] = cutoff
    return metrics


def match_metrics(actual, expected, label):
    for key, value in actual.items():
        if not math.isclose(value, expected[key], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{label} metric mismatch: {key}")


def verified_r0(client, protocol, selection, experiment_id, provenance, device):
    frozen = read_json(ROOT / "reports/task07/verification.json")
    prefix = ROOT / "artifacts/private/task07" / frozen["campaign"] / "mlp"
    require(frozen["completed"] and frozen["test_rows_materialized"] == 0, "Task 07 not verified")
    require(frozen["experiment_id"] == experiment_id, "R0 MLflow experiment mismatch")
    require(frozen["protocol_sha256"] == provenance["protocol_sha256"], "R0 used another protocol")
    require(
        frozen["cache_fingerprint"] == provenance["feature_cache"]["fingerprint"],
        "R0 used another cache",
    )
    training, results = protocol["training"], {}
    for seed in training["seeds"]:
        record = read_json(prefix / f"refit-{seed}.json")
        run_id = record["run_id"]
        run = client.get_run(run_id)
        require(
            run_id in frozen["run_ids_verified"] and run.info.status == "FINISHED", "Unknown R0 run"
        )
        require(read_artifact(client, run_id, RESULT) == record, "Archived R0 metrics changed")
        origin = read_artifact(client, run_id, PROVENANCE_PATH)
        require(origin["code_sha256"] == frozen["code_sha256"], "Wrong R0 code provenance")
        for key in ("protocol_sha256", "feature_cache", "manifests_sha256", "checkpoint_sha256"):
            require(origin[key] == provenance[key], f"R0 provenance mismatch: {key}")
        for name, digest in origin["code_files_sha256"].items():
            require(
                provenance["code_files_sha256"].get(name) == digest,
                f"Historical R0 source changed: {name}",
            )
        settings = record["logged_params"]
        require(
            run.data.params == {k: str(v) for k, v in settings.items()},
            "Archived R0 parameters changed",
        )
        require(
            settings["seed"] == seed
            and settings["variant"] == "R0"
            and settings["fit_partition"] == "train"
            and settings["evaluation_partition"] == "val_select"
            and settings["device"] == device
            and origin["client_libraries"]["torch"] == torch.__version__,
            "R0 must use the same seed, partitions and PyTorch runtime",
        )
        winning = read_json(prefix / f"cv-{record['selected_candidate']:02}.json")
        require(
            winning["run_id"] == record["cv_run_id"]
            and winning["run_id"] in frozen["run_ids_verified"]
            and winning["params"] == {k: settings[k] for k in ("lr", "weight_decay")},
            "R0 does not match the task 07 winning candidate",
        )
        model = checkpoint(
            client,
            record,
            AgeMLP(training["shared_mlp_dimensions"], training["age_classes"]),
            "model/age_mlp.pt",
            device,
        )
        computed = scores_r0(model, selection, protocol)
        match_metrics(computed, record["metrics"], "R0 replay")
        match_metrics(computed, run.data.metrics, "R0 server")
        results[seed] = {**record, "selection_metrics": computed, "settings": winning["params"]}
    require(
        all(
            row["settings"] == results[training["seeds"][0]]["settings"] for row in results.values()
        ),
        "Inconsistent R0 winning settings",
    )
    return results


def train_r4(client, experiment_id, protocol, train, selection, provenance, initial, device):
    training = protocol["training"]
    thresholds = protocol["experiments"]["R4"]["binary_thresholds"]
    require(tuple(thresholds) == COMMON_THRESHOLDS, "Unexpected registered R4 heads")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for seed, r0 in initial.items():
        path = PRIVATE / f"seed-{seed}.json"
        if path.exists():
            print(f"R4 seed {seed} exists; run --verify to check saved results.", flush=True)
            continue
        settings = {
            **r0["logged_params"],
            "variant": "R4",
            "binary_thresholds": ",".join(map(str, thresholds)),
            "binary_loss": training["binary_loss"],
            "binary_weight_each": training["binary_head_weight"],
            "age_gap_factor": training["age_gap_factor"],
            "age_gap_bands": str(age_gap_bounds(thresholds, training["age_gap_factor"])),
            "r0_cv_run_id": r0["cv_run_id"],
            "r0_run_id": r0["run_id"],
            "warm_started": False,
        }
        start = time.perf_counter()
        with tracked_run(
            client,
            experiment_id,
            name=f"{CAMPAIGN}/seed-{seed}",
            kind="research",
            provenance=provenance,
            params=settings,
            tags={
                "agelens.task": "08",
                "agelens.campaign": CAMPAIGN,
                "agelens.variant": "R4",
                "agelens.r0_run_id": r0["run_id"],
            },
        ) as run_id:
            model, history, pairing = fit_multi_age(
                train,
                selection,
                training,
                r0["settings"],
                seed=seed,
                thresholds=thresholds,
                device=device,
            )
            require(
                pairing["sampler_sha256"] == draw_digest(train.age, training, seed, len(history)),
                "Actual training sample order differs from the paired sampler",
            )
            metrics = scores_r4(model, selection, protocol)
            duration = time.perf_counter() - start
            age = metrics["val_select_mae_years"]
            require(
                abs(age - history[pairing["best_epoch"] - 1]["val_select_mae_years"]) <= 1e-6,
                "Best checkpoint was not restored",
            )
            now = time.time_ns() // 1_000_000
            values = [
                Metric(f"epoch_{key}", value, now + row["epoch"], row["epoch"])
                for row in history
                for key, value in row.items()
                if key != "epoch"
            ]
            for offset in range(0, len(values), 500):
                client.log_batch(run_id, metrics=values[offset : offset + 500], synchronous=True)
            for key in ("best_epoch", "epochs_run"):
                metrics[key] = pairing[key]
            metrics["fit_predict_seconds"] = duration
            log_metrics(client, run_id, metrics)
            folder = PRIVATE / run_id
            folder.mkdir()
            checkpoint_path = folder / Path(MODEL_PATH).name
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                checkpoint_path,
            )
            client.log_artifact(run_id, str(checkpoint_path), artifact_path="model")
            record = {
                "run_id": run_id,
                "seed": seed,
                "r0_run_id": r0["run_id"],
                "r0_checkpoint_sha256": r0["model"]["sha256"],
                "protocol_sha256": provenance["protocol_sha256"],
                "pairing": pairing,
                "metrics": metrics,
                "params": settings,
                "model": {"sha256": file_sha256(checkpoint_path), "filename": checkpoint_path.name},
            }
            client.log_dict(run_id, {"epochs": history}, HISTORY_PATH)
            client.log_dict(run_id, record, RESULT)
        write_json_atomic(path, record)
        print(
            f"R4 seed {seed}: MAE {age:.4f}, best epoch {pairing['best_epoch']}/{len(history)}",
            flush=True,
        )


def verify(client, experiment_id, protocol, train, selection, provenance, initial, device):
    train_cfg = protocol["training"]
    thresholds = protocol["experiments"]["R4"]["binary_thresholds"]
    runs = client.search_runs(
        [experiment_id], filter_string=f"tags.`agelens.campaign` = '{CAMPAIGN}'"
    )
    records = {seed: read_json(PRIVATE / f"seed-{seed}.json") for seed in initial}
    require(
        {row["run_id"] for row in records.values()} == {r.info.run_id for r in runs}
        and all(r.info.status == "FINISHED" for r in runs),
        "Incomplete R4 campaign inventory",
    )
    for seed, record in records.items():
        run_id = record["run_id"]
        run = client.get_run(run_id)
        r0 = initial[seed]
        require(
            run.data.tags.get("agelens.r0_run_id") == r0["run_id"]
            and record["r0_checkpoint_sha256"] == r0["model"]["sha256"],
            "Wrong R0 pair",
        )
        require(
            run.data.params == {k: str(v) for k, v in record["params"].items()},
            "R4 parameter mismatch",
        )
        require(read_artifact(client, run_id, RESULT) == record, "R4 result artifact mismatch")
        require(
            read_artifact(client, run_id, PROVENANCE_PATH) == provenance, "R4 provenance mismatch"
        )
        history = read_artifact(client, run_id, HISTORY_PATH)["epochs"]
        best, best_epoch = float("inf"), 0
        for row in history:
            if row["val_select_mae_years"] < best - train_cfg["early_stopping"]["min_delta_years"]:
                best, best_epoch = row["val_select_mae_years"], row["epoch"]
        require(
            best_epoch == record["pairing"]["best_epoch"]
            and len(history) == record["pairing"]["epochs_run"]
            and (
                len(history) - best_epoch == train_cfg["early_stopping"]["patience"]
                or len(history) == train_cfg["max_epochs"]
            )
            and math.isclose(best, record["metrics"]["val_select_mae_years"], abs_tol=1e-6),
            "Incorrect stopping or best epoch",
        )
        _, initial_hash = paired_model(train_cfg, thresholds, seed)
        require(
            initial_hash == record["pairing"]["paired_shared_initial_sha256"]
            and draw_digest(train.age, train_cfg, seed, len(history))
            == record["pairing"]["sampler_sha256"],
            "Pairing hashes differ",
        )
        for key in history[0].keys() - {"epoch"}:
            remote = sorted(client.get_metric_history(run_id, f"epoch_{key}"), key=lambda m: m.step)
            require(
                [(m.step, m.value) for m in remote] == [(r["epoch"], r[key]) for r in history],
                f"Epoch history mismatch: {key}",
            )
        match_metrics(record["metrics"], run.data.metrics, "R4 server")
        model = checkpoint(
            client,
            record,
            MultiAgeMLP(train_cfg["shared_mlp_dimensions"], train_cfg["age_classes"], thresholds),
            MODEL_PATH,
            device,
        )
        match_metrics(
            scores_r4(model, selection, protocol), record["metrics"], "R4 checkpoint replay"
        )
        print(
            f"R0/R4 seed {seed}: checkpoint hashes and reproduced selection metrics verified.",
            flush=True,
        )
    return records


def make_report(protocol, provenance, experiment, r0, r4):
    import pandas as pd
    from matplotlib.figure import Figure

    REPORT.mkdir(parents=True, exist_ok=True)
    seeds = protocol["training"]["seeds"]
    measurements = ["val_select_mae_years", "val_select_age10_21_mae_years"]
    measurements += [
        f"val_select_{kind}_lt{age}_{metric}"
        for age in COMMON_THRESHOLDS
        for kind in ("raw", "provisional_recall")
        for metric in ("f2", "f1", "precision", "recall", "fpr", "fnr", "tp", "fp", "tn", "fn")
    ]
    paired = []
    for seed in seeds:
        for key in measurements:
            a, b = r0[seed]["selection_metrics"][key], r4[seed]["metrics"][key]
            paired.append(
                {
                    "seed": seed,
                    "metric": key,
                    "r0": a,
                    "r4": b,
                    "delta_r4_minus_r0": b - a,
                    "r0_run_id": r0[seed]["run_id"],
                    "r4_run_id": r4[seed]["run_id"],
                }
            )
    frame = pd.DataFrame(paired)
    frame.to_csv(REPORT / "paired_by_seed.csv", index=False)
    summary = frame.groupby("metric", sort=False)[["r0", "r4", "delta_r4_minus_r0"]].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{key}_{stat}" for key, stat in summary.columns]
    summary.to_csv(REPORT / "paired_summary.csv")
    figures = Figure(figsize=(9.2, 4.2), layout="constrained")
    age_ax, f2_ax = figures.subplots(1, 2)
    key = "val_select_mae_years"
    for seed in seeds:
        age_ax.plot(
            ["R0", "R4"],
            [r0[seed]["selection_metrics"][key], r4[seed]["metrics"][key]],
            marker="o",
            label=f"Seed {seed}",
        )
    age_ax.set_ylabel("Age MAE (years)")
    age_ax.legend()
    series = summary.loc[[f"val_select_provisional_recall_lt{age}_f2" for age in COMMON_THRESHOLDS]]
    for variant, color in [("r0", "#167c80"), ("r4", "#ba5941")]:
        f2_ax.errorbar(
            COMMON_THRESHOLDS,
            series[f"{variant}_mean"],
            yerr=series[f"{variant}_std"],
            marker="o",
            capsize=3,
            label=variant.upper(),
            color=color,
        )
    f2_ax.set_xticks(COMMON_THRESHOLDS)
    f2_ax.set_xlabel("Age threshold T")
    f2_ax.set_ylabel("F2 at provisional recall cutoff")
    f2_ax.set_ylim(0, 1.02)
    f2_ax.legend()
    figures.suptitle("AgeLens | R0 vs. R4 on val_select")
    figures.savefig(REPORT / "comparison.png", dpi=150)
    write_json_atomic(
        REPORT / "verification.json",
        {
            "task": "08",
            "completed": True,
            "campaign": CAMPAIGN,
            "experiment_id": experiment,
            "seeds": seeds,
            "thresholds": COMMON_THRESHOLDS,
            "checkpoints_verified": 2 * len(seeds),
            "paired_runs": [
                {"seed": s, "r0": r0[s]["run_id"], "r4": r4[s]["run_id"], **r4[s]["pairing"]}
                for s in seeds
            ],
            "cache_fingerprint": provenance["feature_cache"]["fingerprint"],
            "protocol_sha256": provenance["protocol_sha256"],
            "code_sha256": provenance["code_sha256"],
            "shared_initialization_and_sampler_verified": True,
            "test_rows_materialized": 0,
            "calibration_rows_materialized": 0,
            "r0_retrained": False,
            "backbone_trained": False,
            "evaluation_partition": "val_select",
        },
    )
    print(summary.loc[["val_select_mae_years", *series.index]].round(6).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "runtime/cuda-cache"))
    device = "cuda"
    setup_torch(device, 2)
    client = tracking_client(ROOT)
    experiment = ensure_experiment(client, tracking_config(ROOT), "research")
    provenance = collect_provenance(ROOT)
    protocol, train, selection = load_comparison_data(ROOT)
    r0 = verified_r0(client, protocol, selection, experiment, provenance, device)
    if args.verify:
        records = verify(client, experiment, protocol, train, selection, provenance, r0, device)
        make_report(protocol, provenance, experiment, r0, records)
    else:
        train_r4(client, experiment, protocol, train, selection, provenance, r0, device)


if __name__ == "__main__":
    main()
