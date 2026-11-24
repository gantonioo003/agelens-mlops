"""Run or verify task 07 on train and val_select only."""

import argparse
import json
import os
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from agelens.baselines import (
    FAMILIES,
    candidate_grid,
    cv_summary,
    load_comparison_data,
    mae_years,
    select_candidate,
    selection_metrics,
    sklearn_estimator,
)
from agelens.farl import file_sha256
from agelens.feature_cache import write_json_atomic
from agelens.tracking import (
    PROVENANCE_PATH,
    collect_provenance,
    ensure_experiment,
    tracked_run,
    tracking_client,
    tracking_config,
)
from scripts.prepare_assets import ROOT

RESULT = "evaluation/result.json"


def log_metrics(client, run_id, values, step=0):
    from mlflow.entities import Metric

    now = time.time_ns() // 1_000_000
    client.log_batch(
        run_id,
        metrics=[Metric(key, float(value), now, step) for key, value in values.items()],
        synchronous=True,
    )


def log_history(client, run_id, history, prefix):
    from mlflow.entities import Metric

    now = time.time_ns() // 1_000_000
    client.log_batch(
        run_id,
        metrics=[
            Metric(f"{prefix}{key}", value, now + row["epoch"], row["epoch"])
            for row in history
            for key, value in row.items()
            if key != "epoch"
        ],
        synchronous=True,
    )


def fit_predict(family, params, seed, jobs, device, protocol, x, y, vx, vy, *, cv):
    if family == "mlp":
        from agelens.age_mlp import fit_age_mlp, predict_age

        training = protocol["training"]
        epochs = protocol["tuning"]["mlp"]["cv_epochs"] if cv else training["max_epochs"]
        model, history, best_epoch = fit_age_mlp(
            x,
            y,
            training,
            params,
            seed=seed,
            device=device,
            epochs=epochs,
            validation=None if cv else (vx, vy),
        )
        return model, predict_age(model, vx), history, best_epoch
    from threadpoolctl import threadpool_limits

    model = sklearn_estimator(family, params, seed, jobs)
    with threadpool_limits(limits=4, user_api="blas"):
        model.fit(x, y)
        return model, model.predict(vx), [], None


def save_model(client, run_id, folder, family, model, x, predictions, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    if family == "mlp":
        import torch

        from agelens.age_mlp import AgeMLP, predict_age

        path = folder / "age_mlp.pt"
        training = protocol["training"]
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
        reloaded = AgeMLP(training["shared_mlp_dimensions"], training["age_classes"])
        reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
        new_predictions = predict_age(reloaded.to(next(model.parameters()).device), x)
    else:
        import joblib
        from threadpoolctl import threadpool_limits

        path = folder / f"{family}.joblib"
        joblib.dump(model, path, compress=3)
        reloaded = joblib.load(path)
        with threadpool_limits(limits=4, user_api="blas"):
            new_predictions = reloaded.predict(x)
    error = float(np.max(np.abs(new_predictions - predictions)))
    if not np.allclose(new_predictions, predictions, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Model reload changed age predictions for {family}")
    upload = family != "knn"
    if upload:
        client.log_artifact(run_id, str(path), artifact_path="model")
    return {
        "filename": path.name,
        "sha256": file_sha256(path),
        "uploaded_to_local_mlflow": upload,
        "reload_max_abs_error_years": error,
        "contains_training_vectors": not upload,
    }


def read_record(path, client, provenance):
    record = json.loads(path.read_text(encoding="utf-8"))
    run = client.get_run(record["run_id"])
    tags = run.data.tags
    if (
        run.info.status != "FINISHED"
        or tags.get("agelens.code_sha256") != provenance["code_sha256"]
        or tags.get("agelens.cache_fingerprint") != provenance["feature_cache"]["fingerprint"]
        or record["protocol_sha256"] != provenance["protocol_sha256"]
    ):
        raise ValueError(f"Stale or unverified completion record: {path.name}")
    return record


def run_family(client, experiment, campaign, family, protocol, train, selection, provenance, args):
    folder = ROOT / "artifacts/private/task07" / campaign / family
    folder.mkdir(parents=True, exist_ok=True)
    device = args.device if family == "mlp" else "cpu"
    if family == "mlp":
        from agelens.age_mlp import setup_torch

        setup_torch(device, args.torch_threads)
    common_tags = {"agelens.task": "07", "agelens.campaign": campaign, "agelens.family": family}
    tuning = protocol["tuning"]
    training = protocol["training"]

    def run_settings(params, seed, phase):
        result = {
            **params,
            "seed": seed,
            "family": family,
            "device": device,
            "phase": phase,
            "fit_partition": "train",
            "evaluation_partition": "train_cv" if phase == "cv" else "val_select",
        }
        if family == "knn":
            result["scaler"] = tuning["knn"]["scaler"]
        if family == "mlp":
            result.update(
                architecture="512-256-512-102",
                variant="R0",
                activation="ReLU",
                optimizer=training["optimizer"],
                loss=training["age_loss"],
                sampler=training["sampling"],
                batch_size=training["batch_size"],
                epoch_budget=tuning["mlp"]["cv_epochs"]
                if phase == "cv"
                else training["max_epochs"],
            )
            if phase == "refit":
                result.update(
                    early_stopping_patience=training["early_stopping"]["patience"],
                    early_stopping_min_delta=training["early_stopping"]["min_delta_years"],
                )
        return result

    cv_records = []
    for candidate, params in enumerate(candidate_grid(protocol, family), start=1):
        path = folder / f"cv-{candidate:02}.json"
        if path.exists():
            cv_records.append(read_record(path, client, provenance))
            continue
        name = f"{campaign}/{family}/cv-{candidate:02}"
        settings = run_settings(params, tuning["seed"], "cv")
        settings["candidate"] = candidate
        times, scores = [], []
        with tracked_run(
            client,
            experiment,
            name=name,
            kind="research",
            provenance=provenance,
            params=settings,
            tags={**common_tags, "agelens.phase": "cv"},
        ) as run_id:
            for fold in range(tuning["folds"]):
                hold = train.folds == fold
                start = time.perf_counter()
                _, pred, history, _ = fit_predict(
                    family,
                    params,
                    tuning["seed"],
                    args.jobs,
                    device,
                    protocol,
                    train.features[~hold],
                    train.age[~hold],
                    train.features[hold],
                    train.age[hold],
                    cv=True,
                )
                elapsed = time.perf_counter() - start
                score = mae_years(train.age[hold], pred)
                times.append(elapsed)
                scores.append(score)
                log_metrics(
                    client, run_id, {"cv_fold_mae_years": score, "cv_fold_seconds": elapsed}, fold
                )
                if history:
                    log_history(client, run_id, history, f"cv{fold}_")
                print(
                    f"{family} CV {candidate}, fold {fold}: MAE {score:.4f} ({elapsed:.1f}s)",
                    flush=True,
                )
            record = {
                "run_id": run_id,
                "family": family,
                "phase": "cv",
                "candidate": candidate,
                "params": params,
                "logged_params": settings,
                "fold_mae": scores,
                "fold_seconds": times,
                "protocol_sha256": provenance["protocol_sha256"],
                **cv_summary(scores),
            }
            log_metrics(client, run_id, cv_summary(scores))
            client.log_dict(run_id, record, RESULT)
        write_json_atomic(path, record)
        cv_records.append(record)
    selected = select_candidate(cv_records)
    for seed in training["seeds"]:
        path = folder / f"refit-{seed}.json"
        if path.exists():
            read_record(path, client, provenance)
            continue
        params = selected["params"]
        settings = run_settings(params, seed, "refit")
        settings["selected_cv_candidate"] = selected["candidate"]
        start = time.perf_counter()
        with tracked_run(
            client,
            experiment,
            name=f"{campaign}/{family}/seed-{seed}",
            kind="research",
            provenance=provenance,
            params=settings,
            tags={**common_tags, "agelens.phase": "refit", "agelens.cv_run_id": selected["run_id"]},
        ) as run_id:
            model, predictions, history, best_epoch = fit_predict(
                family,
                params,
                seed,
                args.jobs,
                device,
                protocol,
                train.features,
                train.age,
                selection.features,
                selection.age,
                cv=False,
            )
            scores = selection_metrics(selection.age, predictions, protocol["evaluation"])
            scores["fit_predict_seconds"] = time.perf_counter() - start
            if history:
                log_history(client, run_id, history, "epoch_")
                scores.update(epochs_run=len(history), best_epoch=best_epoch)
            log_metrics(client, run_id, scores)
            model_info = save_model(
                client,
                run_id,
                folder / run_id,
                family,
                model,
                selection.features,
                predictions,
                protocol,
            )
            record = {
                "run_id": run_id,
                "phase": "refit",
                "family": family,
                "seed": seed,
                "cv_run_id": selected["run_id"],
                "selected_candidate": selected["candidate"],
                "protocol_sha256": provenance["protocol_sha256"],
                "logged_params": settings,
                "metrics": scores,
                "model": model_info,
            }
            client.log_dict(run_id, record, RESULT)
        write_json_atomic(path, record)
        print(
            f"{family} seed {seed}: val_select MAE {scores['val_select_mae_years']:.4f}", flush=True
        )


def verify_and_report(client, experiment, campaign, protocol, provenance):
    import pandas as pd
    from matplotlib.figure import Figure

    base = ROOT / "artifacts/private/task07" / campaign
    out = ROOT / "reports/task07"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    summary = []
    for family in FAMILIES:
        folder = base / family
        cv = [
            read_record(folder / f"cv-{n:02}.json", client, provenance)
            for n in range(1, len(candidate_grid(protocol, family)) + 1)
        ]
        best = select_candidate(cv)
        refit = [
            read_record(folder / f"refit-{seed}.json", client, provenance)
            for seed in protocol["training"]["seeds"]
        ]
        if not all(row["cv_run_id"] == best["run_id"] for row in refit):
            raise ValueError("Refit must use the winner from training CV only")
        if [row["params"] for row in cv] != candidate_grid(protocol, family):
            raise ValueError("CV inventory does not match protocol")
        records += cv + refit
        row = {
            "family": family,
            "cv_candidate": best["candidate"],
            "params": best["params"],
            **cv_summary(best["fold_mae"]),
        }
        for key in refit[0]["metrics"]:
            values = [record["metrics"][key] for record in refit]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=1))
        summary.append(row)

    ids = {row["run_id"] for row in records}
    actual = client.search_runs(
        [experiment], filter_string=f"tags.`agelens.campaign` = '{campaign}'"
    )
    if {run.info.run_id for run in actual if run.info.status == "FINISHED"} != ids:
        raise ValueError("Unexpected finished runs for this campaign")
    if any(run.info.status == "RUNNING" for run in actual):
        raise ValueError("Campaign still has active runs")
    for record in records:
        run_id = record["run_id"]
        run = next(run for run in actual if run.info.run_id == run_id)
        if run.data.params != {key: str(value) for key, value in record["logged_params"].items()}:
            raise ValueError(f"Parameters differ for {run_id}")
        if record["phase"] == "cv":
            expected = cv_summary(record["fold_mae"])
            history = client.get_metric_history(run_id, "cv_fold_mae_years")
            if [(m.step, m.value) for m in sorted(history, key=lambda m: m.step)] != list(
                enumerate(record["fold_mae"])
            ):
                raise ValueError(f"CV fold history differs for {run_id}")
            if record["family"] == "mlp":
                for fold in range(protocol["tuning"]["folds"]):
                    history = client.get_metric_history(run_id, f"cv{fold}_train_age_ce")
                    if sorted(m.step for m in history) != list(
                        range(1, protocol["tuning"]["mlp"]["cv_epochs"] + 1)
                    ):
                        raise ValueError(f"MLP training epoch history differs for {run_id}")
        else:
            expected = record["metrics"]
            if record["family"] == "mlp":
                history = client.get_metric_history(run_id, "epoch_val_select_mae_years")
                epochs = expected["epochs_run"]
                if sorted(m.step for m in history) != list(range(1, epochs + 1)):
                    raise ValueError("MLP refit validation history differs")
            model = record["model"]
            path = base / record["family"] / run_id / model["filename"]
            if file_sha256(path) != model["sha256"]:
                raise ValueError(f"Saved model hash mismatch for {run_id}")
            remote = client.list_artifacts(run_id, path="model")
            if [item.path for item in remote] != (
                [f"model/{path.name}"] if model["uploaded_to_local_mlflow"] else []
            ):
                raise ValueError("Incorrect remote model artifact inventory")
            if remote and remote[0].file_size != path.stat().st_size:
                raise ValueError(f"Remote model artifact size mismatch for {run_id}")
        if any(run.data.metrics.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Readback metrics differ for {run_id}")
        with TemporaryDirectory(dir=ROOT / "runtime") as tmp:
            for artifact, value in ((RESULT, record), (PROVENANCE_PATH, provenance)):
                data = json.loads(
                    Path(client.download_artifacts(run_id, artifact, tmp)).read_text()
                )
                if artifact == PROVENANCE_PATH:
                    filtered = {key: val for key, val in value.items() if key != "client_libraries"}
                    observed = {key: val for key, val in data.items() if key != "client_libraries"}
                    if filtered != observed:
                        raise ValueError(f"Run provenance differs: {run_id}")
                elif data != value:
                    raise ValueError(f"Run artifact differs: {run_id}")
        print(f"Verified {record['family']} {record['phase']} {run_id}", flush=True)
    trials = pd.DataFrame(
        [
            {
                **r["params"],
                "family": r["family"],
                "candidate": r["candidate"],
                **cv_summary(r["fold_mae"]),
                "run_id": r["run_id"],
            }
            for r in records
            if r["phase"] == "cv"
        ]
    )
    trials.to_csv(out / "cv_candidates.csv", index=False)
    pd.DataFrame(
        [
            {"family": r["family"], "seed": r["seed"], "run_id": r["run_id"], **r["metrics"]}
            for r in records
            if r["phase"] == "refit"
        ]
    ).to_csv(out / "by_seed.csv", index=False)
    write_json_atomic(out / "summary.json", {"campaign": campaign, "models": summary})
    write_json_atomic(
        out / "verification.json",
        {
            "campaign": campaign,
            "experiment_id": experiment,
            "completed": True,
            "cv_runs": sum(r["phase"] == "cv" for r in records),
            "refit_runs": sum(r["phase"] == "refit" for r in records),
            "run_ids_verified": sorted(ids),
            "cache_fingerprint": provenance["feature_cache"]["fingerprint"],
            "protocol_sha256": provenance["protocol_sha256"],
            "code_sha256": provenance["code_sha256"],
            "mlflow_metrics_params_artifacts_verified": True,
            "private_knn_artifacts_not_uploaded": True,
            "test_rows_materialized": 0,
            "calibration_rows_materialized": 0,
            "trained": True,
            "backbone_trained": False,
            "registered_models": False,
        },
    )
    figure = Figure(figsize=(9.4, 4.3), layout="constrained")
    axes = figure.subplots(1, 2)
    labels = ["Median", "KNN", "Random Forest", "MLP (R0)"]
    for ax, key, label in (
        (axes[0], "val_select_mae_years", "Age MAE (years)"),
        (axes[1], "val_select_provisional_recall_lt18_f2", "Under 18 F2 at provisional cutoff"),
    ):
        ax.barh(
            labels,
            [row[f"{key}_mean"] for row in summary],
            xerr=[row[f"{key}_std"] for row in summary],
            color="#167c80",
            capsize=4,
        )
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    axes[1].set_xlim(0, 1.02)
    figure.suptitle("AgeLens | Selection split only (mean and sample SD, three seeds)")
    figure.savefig(out / "comparison.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="task07-v1")
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"task07-[a-z0-9-]+", args.campaign):
        parser.error("Invalid task 07 campaign name")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    client = tracking_client(ROOT)
    experiment = ensure_experiment(client, tracking_config(ROOT), "research")
    provenance = collect_provenance(ROOT)
    protocol, train, selection = load_comparison_data(ROOT)
    if args.verify:
        verify_and_report(client, experiment, args.campaign, protocol, provenance)
        return
    for family in FAMILIES if args.family == "all" else (args.family,):
        run_family(
            client, experiment, args.campaign, family, protocol, train, selection, provenance, args
        )


if __name__ == "__main__":
    main()
