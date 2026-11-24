import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from services.orchestration.operational import fail
from services.orchestration.training_data import prepare_training, sha256


def test_failure_status_and_backoff_are_bounded():
    writes = []

    def execute(statement, values):
        writes.append(values)
        return SimpleNamespace(rowcount=1)

    db = SimpleNamespace(execute=execute)
    event, token = uuid4(), uuid4()
    assert fail(db, event, token, 1, "network", permanent=False) == "pending"
    assert writes[-1][2:] == ("pending", 15, event, token)
    assert fail(db, event, token, 2, "network", permanent=False) == "pending"
    assert writes[-1][2:] == ("pending", 30, event, token)
    assert fail(db, event, token, 3, "network", permanent=False) == "failed"
    assert fail(db, event, token, 1, "invalid", permanent=True) == "failed"


def test_prepared_training_reads_original_labels_and_reuses_matching_archive(tmp_path):
    root, output = tmp_path / "root", tmp_path / "out"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    cache_id = "a" * 32
    cache = root / "cache" / cache_id
    cache.mkdir(parents=True)
    manifest = inputs / "train.csv"
    manifest.write_text("age,cv_fold,split\n18,0,train\n20,1,train\n41,2,train\n")
    np.save(
        cache / "features.npy",
        np.array([[1, 2], [np.nan, np.nan], [3, 4], [5, 6]], dtype=np.float32),
    )
    np.save(cache / "indices_train.npy", [0, 2, 3])
    path = cache / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "trained": False,
                "file_sha256": {p.name: sha256(p) for p in cache.glob("*.npy")},
            }
        )
    )
    frozen = {
        "cache_id": cache_id,
        "fingerprint": "f" * 64,
        "metadata_sha256": sha256(path),
        "dimensions": 2,
    }
    (cache.parent / "active.json").write_text(json.dumps(frozen))
    (inputs / "protocol.json").write_text(
        json.dumps(
            {
                "feature_extraction": frozen,
                "training": {"age_classes": 102},
                "tuning": {"folds": 3},
                "data_audit": {
                    "manifest_sha256": {"train.csv": sha256(manifest)},
                    "split_sizes": {"train": 3},
                },
            }
        )
    )
    result = prepare_training(root, output)
    assert result["rows"] == 3 and not result["operational_predictions_used"]
    archive = Path(output / cache_id / "train.npz")
    with np.load(archive) as data:
        np.testing.assert_array_equal(data["age_years"], [18, 20, 41])
        np.testing.assert_array_equal(data["features"], [[1, 2], [3, 4], [5, 6]])
    assert prepare_training(root, output) == result
    archive.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity"):
        prepare_training(root, output)
