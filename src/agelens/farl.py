"""Frozen official FaRL image encoder with CLIP's pinned preprocessing."""

import hashlib


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_encoder(checkpoint_path, expected_sha256, device="cpu"):
    import clip
    import torch
    from clip.clip import _transform

    if file_sha256(checkpoint_path) != expected_sha256:
        raise ValueError("FaRL checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = dict(checkpoint["state_dict"])
    prefixes = ("visual.lm_transformer.", "visual.ln_lm.", "visual.lm_head.")
    ignored = [key for key in state if key == "visual.mask_token" or key.startswith(prefixes)]
    for key in ignored:
        state.pop(key)
    model = clip.model.build_model(state).float().eval()
    # CLIP's builder initially casts weights to fp16; reload after .float() to retain FaRL fp32.
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    encoder = model.visual
    if encoder.input_resolution != 224 or encoder.output_dim != 512:
        raise ValueError("Unexpected FaRL image encoder architecture")
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Encoder must stay frozen in evaluation mode")
    if not all(
        torch.equal(value, state[f"visual.{name}"].to(dtype=value.dtype))
        for name, value in encoder.state_dict().items()
    ):
        raise ValueError("Image encoder weights differ from the original checkpoint precision")
    encoder = encoder.to(device=device, dtype=torch.float32)
    return (
        encoder,
        _transform(224),
        {
            "checkpoint_sha256": expected_sha256,
            "ignored_pretraining_keys": sorted(ignored),
            "strict_encoder_load": True,
            "checkpoint_precision_preserved": True,
            "device": str(next(encoder.parameters()).device),
            "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "input_resolution": 224,
            "features": 512,
        },
    )


def encoder_state_sha256(encoder):
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def encode(encoder, batch):
    import torch
    from torch.nn import functional

    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Refuse feature extraction from an unfrozen/training encoder")
    with torch.inference_mode():
        batch = batch.to(device=next(encoder.parameters()).device, dtype=torch.float32)
        raw = encoder(batch)
        if raw.ndim != 2 or raw.shape[1] != 512 or not torch.isfinite(raw).all():
            raise ValueError("Invalid encoder output")
        if (torch.linalg.vector_norm(raw, dim=1) <= 1e-12).any():
            raise ValueError("Zero feature vector")
        result = functional.normalize(raw.float(), p=2, dim=1)
    return result.cpu().numpy()
