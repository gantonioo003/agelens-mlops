"""Train R6 and verify the preregistered R4/R6 comparison on val_select only."""

import argparse
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from mlflow.entities import Metric

from agelens.age_mlp import predict_age, setup_torch
from agelens.baselines import load_comparison_data, selection_metrics
from agelens.contribution import MAE, PRIMARY_AGES, PRIMARY_F2, add_primary, f2_key, paired_outcome
from agelens.farl import file_sha256
from agelens.feature_cache import write_json_atomic
from agelens.multi_age import (
    COMMON_THRESHOLDS,
    MultiAgeMLP,
    age_gap_bounds,
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
from scripts.compare_r4 import (
    HISTORY_PATH,
    checkpoint,
    match_metrics,
    read_artifact,
    read_json,
    require,
    scores_r4,
)
from scripts.compare_r4 import (
    MODEL_PATH as R4_MODEL,
)
from scripts.prepare_assets import ROOT

CAMPAIGN = "task09-r6-v1"
PRIVATE = ROOT / "artifacts/private/task09" / CAMPAIGN
REPORT = ROOT / "reports/task09"
MODEL = "model/age_mlp_r6.pt"
DEVICE = "cuda"


def load_r4(client, experiment, protocol, selection, provenance):
    proof = read_json(ROOT / "reports/task08/verification.json")
    seeds = protocol["training"]["seeds"]
    require(
        proof["completed"]
        and proof["seeds"] == seeds
        and proof["experiment_id"] == experiment
        and proof["protocol_sha256"] == provenance["protocol_sha256"]
        and proof["cache_fingerprint"] == provenance["feature_cache"]["fingerprint"]
        and not (proof["test_rows_materialized"] or proof["calibration_rows_materialized"]),
        "R4 evidence does not match the frozen experiment",
    )
    config = protocol["training"]
    r4 = {}
    for pair in proof["paired_runs"]:
        seed, run_id = pair["seed"], pair["r4"]
        record = read_artifact(client, run_id, RESULT)
        run = client.get_run(run_id)
        old = read_artifact(client, run_id, PROVENANCE_PATH)
        require(
            run.info.status == "FINISHED"
            and run.info.experiment_id == experiment
            and run.data.tags.get("agelens.campaign") == proof["campaign"]
            and record["run_id"] == run_id
            and record["seed"] == seed
            and record["r0_run_id"] == pair["r0"]
            and old["code_sha256"] == proof["code_sha256"]
            and all(
                old[key] == provenance[key]
                for key in (
                    "protocol_sha256",
                    "feature_cache",
                    "manifests_sha256",
                    "checkpoint_sha256",
                )
            )
            and all(
                provenance["code_files_sha256"].get(name) == digest
                for name, digest in old["code_files_sha256"].items()
            )
            and old["client_libraries"]["torch"] == torch.__version__,
            "R4 provenance or runtime differs",
        )
        params = record["params"]
        require(
            run.data.params == {key: str(value) for key, value in params.items()}
            and params["variant"] == "R4"
            and params["device"] == DEVICE
            and params["fit_partition"] == "train"
            and params["evaluation_partition"] == "val_select"
            and params["seed"] == seed,
            "R4 settings differ from the archive",
        )
        r0 = client.get_run(params["r0_cv_run_id"])
        require(
            r0.info.status == "FINISHED"
            and all(r0.data.params[key] == str(params[key]) for key in ("lr", "weight_decay")),
            "R6 must reuse the R0-selected hyperparameters",
        )
        model = checkpoint(
            client,
            record,
            MultiAgeMLP(config["shared_mlp_dimensions"], config["age_classes"], COMMON_THRESHOLDS),
            R4_MODEL,
            DEVICE,
        )
        common = scores_r4(model, selection, protocol)
        match_metrics(common, record["metrics"], "R4 checkpoint replay")
        match_metrics(common, run.data.metrics, "R4 server")
        age_metrics = selection_metrics(
            selection.age,
            predict_age(model, selection.features),
            {**protocol["evaluation"], "thresholds": PRIMARY_AGES},
        )
        extra = {key: value for key, value in age_metrics.items() if key not in common}
        r4[seed] = {**record, "comparison": add_primary({**common, **extra})}
    require(list(r4) == seeds, "Unexpected R4 seed inventory")
    return r4


def verify_pairing(train, config, thresholds, seed, pairing, reference):
    r4, shared_r4 = paired_model(config, COMMON_THRESHOLDS, seed)
    r6, shared_r6 = paired_model(config, thresholds, seed)
    require(
        shared_r4
        == shared_r6
        == pairing["paired_shared_initial_sha256"]
        == reference["paired_shared_initial_sha256"]
        and all(torch.equal(value, r6.state_dict()[key]) for key, value in r4.state_dict().items()),
        "Initialization of shared layers or common binary heads changed",
    )
    # Both checks independently reproduce the same CPU generator; stopping may differ.
    for current in (reference, pairing):
        require(
            current["sampler_sha256"]
            == draw_digest(train.age, config, seed, current["epochs_run"]),
            "Training order is not paired",
        )


def settings_for(r4, thresholds, config):
    return {
        **r4["params"],
        "variant": "R6",
        "binary_thresholds": ",".join(map(str, thresholds)),
        "age_gap_bands": str(
            dict(zip(thresholds, age_gap_bounds(thresholds, config["age_gap_factor"]), strict=True))
        ),
        "r4_run_id": r4["run_id"],
        "reference_13_16": "negative_predicted_age",
    }


def train_r6(client, experiment, protocol, train, selection, provenance, references):
    config = protocol["training"]
    thresholds = protocol["experiments"]["R6"]["binary_thresholds"]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for seed, r4 in references.items():
        path = PRIVATE / f"seed-{seed}.json"
        if path.exists():
            continue
        settings = settings_for(r4, thresholds, config)
        with tracked_run(
            client,
            experiment,
            name=f"{CAMPAIGN}/seed-{seed}",
            kind="research",
            provenance=provenance,
            params=settings,
            tags={
                "agelens.task": "09",
                "agelens.campaign": CAMPAIGN,
                "agelens.variant": "R6",
                "agelens.r4_run_id": r4["run_id"],
            },
        ) as run_id:
            model, history, pairing = fit_multi_age(
                train,
                selection,
                config,
                {key: settings[key] for key in ("lr", "weight_decay")},
                seed=seed,
                thresholds=thresholds,
                device=DEVICE,
            )
            verify_pairing(train, config, thresholds, seed, pairing, r4["pairing"])
            metrics = add_primary(scores_r4(model, selection, protocol))
            log_metrics(client, run_id, metrics)
            log_metrics(client, run_id, {key: pairing[key] for key in ("best_epoch", "epochs_run")})
            now = time.time_ns() // 1_000_000
            values = [
                Metric(f"epoch_{key}", value, now + row["epoch"], row["epoch"])
                for row in history
                for key, value in row.items()
                if key != "epoch"
            ]
            for offset in range(0, len(values), 500):
                client.log_batch(run_id, metrics=values[offset : offset + 500], synchronous=True)
            out = PRIVATE / run_id / Path(MODEL).name
            out.parent.mkdir()
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}, out
            )
            client.log_artifact(run_id, str(out), artifact_path="model")
            record = {
                "run_id": run_id,
                "seed": seed,
                "metrics": metrics,
                "pairing": pairing,
                "params": settings,
                "model": {"filename": out.name, "sha256": file_sha256(out)},
                "r4_checkpoint_sha256": r4["model"]["sha256"],
            }
            client.log_dict(run_id, {"epochs": history}, HISTORY_PATH)
            client.log_dict(run_id, record, RESULT)
        write_json_atomic(path, record)
        print(
            f"R6 seed {seed}: MAE {metrics[MAE]:.4f}, F2 13/16 {metrics[PRIMARY_F2]:.4f}",
            flush=True,
        )


def verify(client, experiment, protocol, train, selection, provenance, references):
    config = protocol["training"]
    thresholds = protocol["experiments"]["R6"]["binary_thresholds"]
    records = {seed: read_json(PRIVATE / f"seed-{seed}.json") for seed in references}
    runs = client.search_runs([experiment], filter_string=f"tags.`agelens.campaign` = '{CAMPAIGN}'")
    require(
        {r.info.run_id for r in runs} == {record["run_id"] for record in records.values()}
        and all(r.info.status == "FINISHED" for r in runs),
        "Incomplete R6 campaign",
    )
    rows = []
    for seed, record in records.items():
        r4 = references[seed]
        run_id = record["run_id"]
        run = client.get_run(run_id)
        params = settings_for(r4, thresholds, config)
        require(
            run.data.params == {key: str(value) for key, value in params.items()}
            and record["params"] == params
            and record["seed"] == seed
            and record["r4_checkpoint_sha256"] == r4["model"]["sha256"]
            and read_artifact(client, run_id, RESULT) == record
            and read_artifact(client, run_id, PROVENANCE_PATH) == provenance,
            "Run configuration or evidence changed",
        )
        pairing = record["pairing"]
        verify_pairing(train, config, thresholds, seed, pairing, r4["pairing"])
        model = checkpoint(
            client,
            record,
            MultiAgeMLP(config["shared_mlp_dimensions"], config["age_classes"], thresholds),
            MODEL,
            DEVICE,
        )
        replay = add_primary(scores_r4(model, selection, protocol))
        match_metrics(replay, record["metrics"], "R6 checkpoint replay")
        match_metrics(
            {**replay, **{key: pairing[key] for key in ("best_epoch", "epochs_run")}},
            run.data.metrics,
            "R6 server",
        )
        history = read_artifact(client, run_id, HISTORY_PATH)["epochs"]
        early = config["early_stopping"]
        best, epoch = float("inf"), 0
        require(
            [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
            "Invalid epoch sequence",
        )
        for row in history:
            if row["val_select_mae_years"] < best - early["min_delta_years"]:
                best, epoch = row["val_select_mae_years"], row["epoch"]
        require(
            epoch == pairing["best_epoch"]
            and len(history) == pairing["epochs_run"]
            and (len(history) - epoch == early["patience"] or len(history) == config["max_epochs"])
            and abs(best - replay[MAE]) <= 1e-6,
            "Early stopping or restored checkpoint does not match",
        )
        for key in history[0].keys() - {"epoch"}:
            remote = sorted(client.get_metric_history(run_id, f"epoch_{key}"), key=lambda m: m.step)
            require(
                [(m.step, m.value) for m in remote] == [(r["epoch"], r[key]) for r in history],
                f"Epoch readback differs: {key}",
            )
        rows.append(
            {
                "seed": seed,
                "r4": r4["comparison"],
                "r6": replay,
                "r4_run": r4["run_id"],
                "r6_run": run_id,
                "r6_pairing": pairing,
            }
        )
        print(f"Verified both checkpoints and metrics for seed {seed}", flush=True)
    return rows


def report(rows, experiment, protocol, provenance):
    import pandas as pd
    from matplotlib.figure import Figure

    REPORT.mkdir(parents=True, exist_ok=True)
    ages = sorted(protocol["experiments"]["R6"]["binary_thresholds"])
    outcome = paired_outcome(rows, COMMON_THRESHOLDS, protocol["evaluation"])
    frame = pd.DataFrame(
        [
            {
                "seed": row["seed"],
                "metric": key,
                "r4": value,
                "r6": row["r6"][key],
                "delta_r6_minus_r4": row["r6"][key] - value,
                "r4_run_id": row["r4_run"],
                "r6_run_id": row["r6_run"],
            }
            for row in rows
            for key, value in row["r4"].items()
            if not key.endswith("_cutoff")
        ]
    )
    frame.to_csv(REPORT / "paired_by_seed.csv", index=False)
    stats = frame.groupby("metric", sort=False)[["r4", "r6", "delta_r6_minus_r4"]].agg(
        ["mean", "std"]
    )
    stats.columns = [f"{key}_{mode}" for key, mode in stats.columns]
    stats.to_csv(REPORT / "paired_summary.csv")
    write_json_atomic(REPORT / "outcome.json", outcome)
    fig = Figure(figsize=(9.2, 4.0), layout="constrained")
    ax, delta = fig.subplots(1, 2)
    for label, color in [("r4", "#19778b"), ("r6", "#bc5323")]:
        points = [outcome["metrics"][f2_key(age)] for age in ages]
        ax.errorbar(
            ages,
            [p[f"{label}_mean"] for p in points],
            yerr=[p[f"{label}_std"] for p in points],
            capsize=3,
            marker="o",
            label=label.upper(),
            color=color,
        )
    ax.set(xticks=ages, ylim=(0, 1.02), xlabel="Age threshold T", ylabel="Provisional F2")
    ax.legend()
    ds = outcome["metrics"][PRIMARY_F2]["delta_by_seed"]
    delta.bar(
        list(ds), list(ds.values()), color=["#19778b" if d >= 0 else "#bc5323" for d in ds.values()]
    )
    delta.axhline(0, linewidth=1, color="black")
    delta.set(xlabel="Paired seed", ylabel="R6 - R4: mean F2 at 13 and 16")
    fig.suptitle("AgeLens | R4 vs. R6 on val_select")
    fig.savefig(REPORT / "comparison.png", dpi=150)
    proof = {
        "completed": True,
        "experiment_id": experiment,
        "campaign": CAMPAIGN,
        "evaluation_partition": "val_select",
        "test_rows_materialized": 0,
        "calibration_rows_materialized": 0,
        "backbone_trained": False,
        "r4_retrained": False,
        "checkpoints_verified": 2 * len(rows),
        "pairs": [
            {key: value for key, value in row.items() if key not in {"r4", "r6"}} for row in rows
        ],
        "code_sha256": provenance["code_sha256"],
        "protocol_sha256": provenance["protocol_sha256"],
        "cache_fingerprint": provenance["feature_cache"]["fingerprint"],
    }
    write_json_atomic(REPORT / "verification.json", proof)
    return outcome


def log_summary(client, experiment, provenance, outcome):
    campaign = f"{CAMPAIGN}-summary"
    existing = client.search_runs(
        [experiment], filter_string=f"tags.`agelens.campaign` = '{campaign}'"
    )
    names = (
        "paired_by_seed.csv",
        "paired_summary.csv",
        "comparison.png",
        "outcome.json",
        "verification.json",
    )
    metrics = {
        f"{name}_{key}": row[name]
        for key, row in outcome["metrics"].items()
        for name in ("r4_mean", "r6_mean", "delta_mean", "delta_std")
    }
    if existing:
        require(len(existing) == 1, "Unexpected duplicate summary runs")
        run_id = existing[0].info.run_id
    else:
        with tracked_run(
            client,
            experiment,
            name=f"{CAMPAIGN}/paired-summary",
            kind="research",
            provenance=provenance,
            params={"evaluation_partition": "val_select", "training_performed": False},
            tags={"agelens.task": "09", "agelens.campaign": campaign},
        ) as run_id:
            log_metrics(client, run_id, metrics)
            for name in names:
                client.log_artifact(run_id, str(REPORT / name), artifact_path="comparison")
    run = client.get_run(run_id)
    require(
        run.info.status == "FINISHED" and run.data.metrics == metrics, "Summary readback differs"
    )
    with TemporaryDirectory(dir=ROOT / "runtime") as tmp:
        for name in names:
            target = Path(client.download_artifacts(run_id, "comparison/" + name, tmp))
            require(file_sha256(target) == file_sha256(REPORT / name), "Summary artifact differs")
    print(f"Verified MLflow summary: {run_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "runtime/cuda-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/matplotlib"))
    setup_torch(DEVICE, 2)
    client = tracking_client(ROOT)
    experiment = ensure_experiment(client, tracking_config(ROOT), "research")
    provenance = collect_provenance(ROOT)
    protocol, train, selection = load_comparison_data(ROOT)
    thresholds = protocol["experiments"]["R6"]["binary_thresholds"]
    require(sorted((*COMMON_THRESHOLDS, *PRIMARY_AGES)) == thresholds, "Invalid R6 protocol")
    r4 = load_r4(client, experiment, protocol, selection, provenance)
    if args.verify:
        rows = verify(client, experiment, protocol, train, selection, provenance, r4)
        log_summary(client, experiment, provenance, report(rows, experiment, protocol, provenance))
    else:
        train_r6(client, experiment, protocol, train, selection, provenance, r4)


if __name__ == "__main__":
    main()
