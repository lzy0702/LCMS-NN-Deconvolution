"""Export a trained charge model to ONNX and load it back."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_torch_model(checkpoint: str | Path):
    import torch

    from .model import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "unet")
    ladder = {}
    if arch == "ladder":
        # checkpoints written before these were recorded hold the 32 768-bin network
        ladder = {"window": int(ckpt.get("window") or 32768), "active_top": ckpt.get("active_top"),
                  "iso": bool(ckpt.get("iso_channels", False))}
    model = build_model(ckpt.get("model_size", "proof"), z_max=ckpt.get("z_max", 100),
                        in_ch=ckpt.get("in_ch", 4), comb_prior=ckpt.get("comb_prior", False),
                        arch=arch, **ladder)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _example_input(model, ckpt, length: int):
    """A random input of the model's shape: log-SNR in [0, 5] and, for a network reading
    isotope channels, whole charges (0 = none) with strengths in [0, 1]."""
    import torch

    x = torch.rand(1, ckpt.get("in_ch", 4), length)
    x[:, 0] *= 5.0
    if getattr(model, "iso", False):
        g = torch.Generator().manual_seed(0)
        z = torch.randint(0, 25, (length,), generator=g).float()
        z[torch.rand(length, generator=g) < 0.5] = 0.0
        x[:, 4] = z
        x[:, 5] = torch.where(z > 0, x[:, 5], torch.zeros_like(x[:, 5]))
    return x


def export_onnx(checkpoint: str | Path, out_path: str | Path, length: int = 32768) -> Path:
    import torch

    model, ckpt = load_torch_model(checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ladder = ckpt.get("arch", "unet") == "ladder"
    if ladder:
        # static shifts over a fixed window: the exported graph is for that length only
        model.dense = True
        length = model.window
    dummy = _example_input(model, ckpt, length)
    axes = {0: "batch"} if ladder else {0: "batch", 2: "length"}
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=["charge_logits", "apex_logit"],
        dynamic_axes={"features": axes, "charge_logits": axes, "apex_logit": axes},
        opset_version=17,
        dynamo=False,
    )
    # write a small sidecar with metadata
    meta = {"z_max": ckpt.get("z_max", 100), "model_size": ckpt.get("model_size", "proof"),
            "in_ch": ckpt.get("in_ch", 4), "evidence_channels": ckpt.get("evidence_channels", 0),
            "arch": ckpt.get("arch", "unet"), "singly_charged": ckpt.get("singly_charged", "keep"),
            "grid_step": ckpt.get("grid_step", 2e-5),
            "window": int(model.window) if ladder else ckpt.get("window", 32768),
            "active_top": int(model.active_top) if ladder else None,
            "iso_channels": bool(getattr(model, "iso", False))}
    import json

    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return out_path


def verify_parity(checkpoint: str | Path, onnx_path: str | Path, length: int = 8192,
                  tol: float = 1e-3) -> dict:
    import onnxruntime as ort
    import torch

    model, ckpt = load_torch_model(checkpoint)
    if ckpt.get("arch", "unet") == "ladder":
        length = model.window  # the sparse path is the reference for the dense export
    x = _example_input(model, ckpt, length)
    with torch.no_grad():
        ref = model(x)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"features": x.numpy().astype(np.float32)})
    onx = {"charge_logits": outs[0], "apex_logit": outs[1]}
    dc = float(np.abs(ref["charge_logits"].numpy() - onx["charge_logits"]).max())
    da = float(np.abs(ref["apex_logit"].numpy() - onx["apex_logit"]).max())
    return {"charge_max_abs_diff": dc, "apex_max_abs_diff": da, "ok": dc < tol and da < tol}
