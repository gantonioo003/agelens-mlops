"""The protocol's age-only R0 head, operating exclusively on frozen features."""

import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class AgeMLP(nn.Module):
    def __init__(self, dimensions, classes):
        super().__init__()
        inputs, hidden, outputs = dimensions
        self.trunk = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs), nn.ReLU()
        )
        self.age_head = nn.Linear(outputs, classes)

    def forward(self, features):
        return self.age_head(self.trunk(features))

    def ages(self, features):
        logits = self(features)
        ages = torch.arange(logits.shape[1], dtype=logits.dtype, device=logits.device)
        return (logits.softmax(dim=1) * ages).sum(dim=1)


def setup_torch(device, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sampling_weights(ages, edges):
    bins = np.searchsorted(edges, ages, side="right")
    return 1.0 / np.bincount(bins, minlength=len(edges) + 1)[bins]


def predict_age(model, features):
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        values = torch.as_tensor(features, dtype=torch.float32, device=device)
        return torch.cat([model.ages(batch) for batch in values.split(1024)]).cpu().numpy()


def fit_age_mlp(x, y, training, params, *, seed, device, epochs, validation=None):
    # The sampler's RNG is independent of initialization, including later extra heads.
    torch.manual_seed(seed)
    model = AgeMLP(training["shared_mlp_dimensions"], training["age_classes"]).to(device)
    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(sampling_weights(y, training["age_sampling_bin_upper_edges"]))
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_age_mae = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        sampled = torch.multinomial(weights, len(weights), replacement=True, generator=generator)
        epoch_loss = torch.zeros((), device=device)
        for rows in sampled.to(device).split(training["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[rows]), labels[rows])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * rows.numel()
        record = {"epoch": epoch, "train_age_ce": float(epoch_loss.item() / len(y))}
        if not np.isfinite(record["train_age_ce"]):
            raise ValueError("Non-finite MLP training loss")
        history.append(record)
        if validation is None:
            continue
        x_val, y_val = validation
        mae = float(np.mean(np.abs(predict_age(model, x_val) - y_val)))
        record["val_select_mae_years"] = mae
        early = training["early_stopping"]
        if mae < best_age_mae - early["min_delta_years"]:
            best_age_mae = mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= early["patience"]:
            break
    if validation is not None:
        if best_state is None:
            raise ValueError("No valid MLP validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model.eval(), history, best_epoch
