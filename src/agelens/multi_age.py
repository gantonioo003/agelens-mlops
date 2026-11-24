"""Frozen-feature MLP with the paired age trunk and masked binary age heads."""

import hashlib
from copy import deepcopy
from fractions import Fraction

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agelens.age_mlp import AgeMLP, predict_age, sampling_weights
from agelens.farl import encoder_state_sha256

COMMON_THRESHOLDS = (12, 15, 18, 21)


class MultiAgeMLP(AgeMLP):
    def __init__(self, dimensions, classes, thresholds):
        super().__init__(dimensions, classes)
        # Initialize common heads first so adding R6 heads does not shift their RNG draws.
        order = [age for age in COMMON_THRESHOLDS if age in thresholds]
        order += sorted(set(thresholds) - set(order))
        self.thresholds = tuple(order)
        self.binary_heads = nn.ModuleDict(
            {str(age): nn.Linear(dimensions[-1], 1) for age in self.thresholds}
        )

    def binary_logits(self, shared):
        return torch.cat([self.binary_heads[str(age)](shared) for age in self.thresholds], dim=1)

    def joint_logits(self, features):
        shared = self.trunk(features)
        return self.age_head(shared), self.binary_logits(shared)


def paired_model(training, thresholds, seed):
    dims, classes = training["shared_mlp_dimensions"], training["age_classes"]
    torch.manual_seed(seed)
    reference = AgeMLP(dims, classes)
    torch.manual_seed(seed)
    model = MultiAgeMLP(dims, classes, thresholds)
    current = model.state_dict()
    if not all(torch.equal(value, current[key]) for key, value in reference.state_dict().items()):
        raise ValueError("Shared initialization must match R0 exactly")
    return model, encoder_state_sha256(reference)


def age_gap_bounds(thresholds, factor):
    ratio = Fraction(str(factor))
    return [(int(t // ratio), int(t * ratio // 1)) for t in thresholds]


def binary_targets(ages, thresholds, bounds):
    limits = ages.new_tensor(thresholds)
    bands = ages.new_tensor(bounds)
    column = ages[:, None]
    targets = column < limits
    valid = (column < bands[:, 0]) | (column >= bands[:, 1])
    return targets, valid


def masked_bce(logits, targets, valid):
    raw = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    return (raw * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)


def paired_draws(ages, training, seed, epochs):
    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(sampling_weights(ages, training["age_sampling_bin_upper_edges"]))
    for _ in range(epochs):
        yield torch.multinomial(weights, len(weights), replacement=True, generator=generator)


def draw_digest(ages, training, seed, epochs):
    digest = hashlib.sha256()
    for rows in paired_draws(ages, training, seed, epochs):
        digest.update(rows.numpy().tobytes())
    return digest.hexdigest()


def binary_probabilities(model, features):
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        values = torch.as_tensor(features, dtype=torch.float32, device=device)
        return (
            torch.cat(
                [model.binary_logits(model.trunk(batch)).sigmoid() for batch in values.split(1024)]
            )
            .cpu()
            .numpy()
        )


def fit_multi_age(train, selection, training, params, *, seed, thresholds, device):
    model, initial_hash = paired_model(training, thresholds, seed)
    model.to(device)
    x = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    y = torch.tensor(train.age, dtype=torch.long, device=device)
    bounds = age_gap_bounds(model.thresholds, training["age_gap_factor"])
    targets, valid = binary_targets(y, model.thresholds, bounds)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    early = training["early_stopping"]
    history = []
    best_mae, best_epoch, best_state = float("inf"), 0, None
    digest = hashlib.sha256()
    draws = paired_draws(train.age, training, seed, training["max_epochs"])
    for epoch, sampled in enumerate(draws, start=1):
        digest.update(sampled.numpy().tobytes())
        model.train()
        loss_sums = torch.zeros(2 + len(thresholds), device=device)
        for rows in sampled.to(device).split(training["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            age_logits, binary_logits = model.joint_logits(x[rows])
            age_loss = F.cross_entropy(age_logits, y[rows])
            binary_losses = masked_bce(binary_logits, targets[rows], valid[rows])
            total = age_loss + training["binary_head_weight"] * binary_losses.sum()
            total.backward()
            optimizer.step()
            loss_sums += (
                torch.cat(
                    [age_loss.detach().view(1), total.detach().view(1), binary_losses.detach()]
                )
                * rows.numel()
            )
        losses = (loss_sums / len(y)).tolist()
        mae = float(np.mean(np.abs(predict_age(model, selection.features) - selection.age)))
        if not np.isfinite([mae, *losses]).all():
            raise ValueError("Non-finite R4 training or validation result")
        history.append(
            {
                "epoch": epoch,
                "val_select_mae_years": mae,
                "train_age_ce": losses[0],
                "train_total_loss": losses[1],
                **{
                    f"train_bce_lt{t}": v for t, v in zip(model.thresholds, losses[2:], strict=True)
                },
            }
        )
        if mae < best_mae - early["min_delta_years"]:
            best_mae, best_epoch, best_state = mae, epoch, deepcopy(model.state_dict())
        if epoch - best_epoch >= early["patience"]:
            break
    if best_state is None:
        raise ValueError("No valid validation checkpoint")
    model.load_state_dict(best_state)
    return (
        model.eval(),
        history,
        {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "paired_shared_initial_sha256": initial_hash,
            "sampler_sha256": digest.hexdigest(),
        },
    )
