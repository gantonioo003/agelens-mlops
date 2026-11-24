"""Measure preprocessing versus forward time on training images; no fitting."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from agelens.farl import encode, load_encoder
from scripts.extract_features import image_tensor, load_inputs, setup_torch
from scripts.prepare_assets import ROOT


def main():
    protocol, config, selected = load_inputs()
    setup_torch(config)
    assets = json.loads((ROOT / "configs/assets.json").read_text())
    encoder, preprocess, _ = load_encoder(
        ROOT / assets["backbone"]["path"],
        protocol["backbone"]["checkpoint_sha256"],
        device=config["device"],
    )
    rows = list(selected[selected.split == "train"].head(32).itertuples())
    results = []
    for workers in [0, 1, 2]:
        for threads in [4, 8]:
            torch.set_num_threads(threads)
            preprocessing = 0.0
            forward = 0.0
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for begin in range(0, len(rows), 8):
                    start = time.perf_counter()
                    batch_rows = rows[begin : begin + 8]
                    tensors = (
                        [image_tensor(row, preprocess) for row in batch_rows]
                        if workers == 0
                        else list(pool.map(lambda row: image_tensor(row, preprocess), batch_rows))
                    )
                    batch = torch.stack(tensors)
                    preprocessing += time.perf_counter() - start
                    start = time.perf_counter()
                    encode(encoder, batch)
                    forward += time.perf_counter() - start
            row = {
                "workers": workers,
                "torch_threads": threads,
                "preprocess_seconds": preprocessing,
                "forward_seconds": forward,
                "images_per_second": len(rows) / (preprocessing + forward),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    torch.set_num_threads(8)
    batch = torch.stack([image_tensor(row, preprocess) for row in rows[:8]])
    outputs = []
    for flush in [False, True]:
        torch.set_flush_denormal(flush)
        start = time.perf_counter()
        output = encode(encoder, batch)
        outputs.append(output)
        print(
            json.dumps({"flush_denormal": flush, "seconds": time.perf_counter() - start}),
            flush=True,
        )
    print("Denormal comparison max error", float(np.abs(outputs[0] - outputs[1]).max()), flush=True)
    torch.set_flush_denormal(False)
    (ROOT / "artifacts/private/features-profile.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
