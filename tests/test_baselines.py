import json
from pathlib import Path

import numpy as np
import pytest

from agelens.baselines import (
    FAMILIES,
    candidate_grid,
    cv_summary,
    load_comparison_data,
    select_candidate,
    selection_metrics,
    sklearn_estimator,
)
from agelens.farl import file_sha256

PROTOCOL = json.loads((Path(__file__).parents[1] / "configs/protocol-v1.json").read_text())


def test_all_declared_candidates_in_fixed_order():
    grids = {name: candidate_grid(PROTOCOL, name) for name in FAMILIES}
    assert [len(grid) for grid in grids.values()] == [1, 8, 6, 4]
    assert grids["knn"][0] == {"n_neighbors": 3, "weights": "uniform", "p": 2}
    assert grids["random_forest"][0]["max_depth"] is None
    assert grids["mlp"][-1] == {"lr": 0.0003, "weight_decay": 0.0001}


def test_candidate_selection_uses_mean_std_then_order():
    mae = ([1, 2, 3], [2, 2, 2], [2, 2, 2], [1, 1, 5])
    rows = [{"candidate": i, **cv_summary(scores)} for i, scores in enumerate(mae)]
    assert select_candidate(rows[::-1])["candidate"] == 1
    assert rows[0]["cv_std_mae_years"] == 1.0


def test_knn_uses_only_fit_rows_for_standardization():
    model = sklearn_estimator("knn", {"n_neighbors": 1, "p": 2, "weights": "uniform"}, 17, 1)
    x = np.array([[0, 10], [2, 12]])
    model.fit(x, [10, 20])
    model.predict(np.array([[100, 200]]))
    np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, [1, 11])


def test_threshold_strictness_and_provisional_score_ties():
    age = np.array([10, 11, 17, 18, 25])
    prediction = np.array([12.0, 15.0, 18.0, 18.0, 40.0])
    metrics = selection_metrics(
        age, prediction, {"thresholds": [18], "target_calibration_recall": 0.99}
    )
    assert metrics["val_select_raw_lt18_tp"] == 2
    assert metrics["val_select_raw_lt18_fp"] == 0
    assert metrics["val_select_provisional_recall_lt18_cutoff"] == -18
    assert metrics["val_select_provisional_recall_lt18_tp"] == 3
    assert metrics["val_select_provisional_recall_lt18_fp"] == 1


def test_mlp_balances_bins_and_predictions_are_expectation():
    torch = pytest.importorskip("torch")
    from agelens.age_mlp import AgeMLP, predict_age, sampling_weights

    np.testing.assert_array_equal(
        sampling_weights(np.array([0, 3, 4, 8, 9]), [4, 8]), [0.5, 0.5, 1.0, 0.5, 0.5]
    )
    model = AgeMLP((2, 3, 4), 102)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    np.testing.assert_allclose(predict_age(model, np.ones((3, 2))), 50.5, atol=1e-5)


def test_comparison_reads_only_permitted_manifests_and_materializes_only_permitted_rows(
    tmp_path, monkeypatch
):
    from agelens import baselines

    cache = tmp_path / "mock_cache"
    manifests = tmp_path / "data/manifests"
    (tmp_path / "configs").mkdir()
    cache.mkdir()
    manifests.mkdir(parents=True)
    np.save(cache / "features.npy", np.array([[1], [2], [3], [4], [np.nan]], dtype=np.float32))
    sizes = {"train": 3, "val_select": 1}
    hashes = {}
    for name, entries, indices in (
        ("train", ["train,11,0,a", "train,12,1,b", "train,13,2,c"], [0, 1, 2]),
        ("val_select", ["val_select,14,-1,d"], [3]),
    ):
        np.save(cache / f"indices_{name}.npy", np.array(indices))
        csv = manifests / f"{name}.csv"
        csv.write_text("split,age,cv_fold,duplicate_group\n" + "\n".join(entries) + "\n")
        hashes[csv.name] = file_sha256(csv)
    metadata = {"file_sha256": {path.name: file_sha256(path) for path in cache.iterdir()}}
    monkeypatch.setattr(baselines, "open_approved_cache", lambda _: (cache, metadata))
    protocol = {
        "data_audit": {"split_sizes": sizes, "manifest_sha256": hashes},
        "tuning": {"folds": 3},
    }
    (tmp_path / "configs/protocol-v1.json").write_text(json.dumps(protocol))
    _, train, selection = load_comparison_data(tmp_path)
    np.testing.assert_array_equal(train.features, [[1], [2], [3]])
    np.testing.assert_array_equal(selection.features, [[4]])
    np.save(cache / "features.npy", np.zeros((5, 1)))
    with pytest.raises(ValueError, match="matrix checksum mismatch"):
        load_comparison_data(tmp_path)
