import json
from pathlib import Path

import numpy as np
import pytest

PROTOCOL = json.loads((Path(__file__).parents[1] / "configs/protocol-v1.json").read_text())


def test_r4_mask_boundaries_and_strict_minor_labels():
    torch = pytest.importorskip("torch")
    from agelens.multi_age import COMMON_THRESHOLDS, age_gap_bounds, binary_targets

    bounds = age_gap_bounds(COMMON_THRESHOLDS, 1.2)
    assert bounds == [(10, 14), (12, 18), (15, 21), (17, 25)]
    age = torch.tensor([9, 10, 11, 12, 13, 14])
    targets, keep = binary_targets(age, COMMON_THRESHOLDS, bounds)
    assert targets[:, 0].tolist() == [True, True, True, False, False, False]
    assert keep[:, 0].tolist() == [True, False, False, False, False, True]


def test_empty_masks_have_zero_loss_and_zero_gradient():
    torch = pytest.importorskip("torch")
    from agelens.multi_age import masked_bce

    logits = torch.tensor([[0.0, 4.0], [0.0, -5.0]], requires_grad=True)
    targets = torch.tensor([[True, False], [False, True]])
    valid = torch.tensor([[True, False], [False, False]])
    loss = masked_bce(logits, targets, valid)
    assert loss.tolist() == pytest.approx([np.log(2.0), 0.0])
    loss.sum().backward()
    assert logits.grad.tolist() == [[-0.5, 0.0], [0.0, 0.0]]


def test_r0_r4_and_later_r6_shared_initialization():
    torch = pytest.importorskip("torch")
    from agelens.multi_age import paired_model

    training = PROTOCOL["training"]
    protocol_experiments = PROTOCOL["experiments"]
    for seed in training["seeds"]:
        r4, hash_r4 = paired_model(training, protocol_experiments["R4"]["binary_thresholds"], seed)
        r6, hash_r6 = paired_model(training, protocol_experiments["R6"]["binary_thresholds"], seed)
        assert hash_r4 == hash_r6
        assert all(
            torch.equal(value, r6.state_dict()[key]) for key, value in r4.state_dict().items()
        )


def test_r4_batch_sequence_matches_the_original_r0_sampler():
    torch = pytest.importorskip("torch")
    from agelens.age_mlp import sampling_weights
    from agelens.multi_age import paired_draws, paired_model

    training = PROTOCOL["training"]
    ages = np.array([8, 12, 13, 16, 17, 29, 30, 31, 45, 51])
    weights = torch.as_tensor(sampling_weights(ages, training["age_sampling_bin_upper_edges"]))
    for seed in training["seeds"]:
        rng = torch.Generator().manual_seed(seed)
        historical = [
            torch.multinomial(weights, len(weights), replacement=True, generator=rng)
            for _ in range(3)
        ]
        paired_model(training, [12, 15, 18, 21], seed)
        assert all(
            torch.equal(a, b)
            for a, b in zip(historical, paired_draws(ages, training, seed, 3), strict=True)
        )
