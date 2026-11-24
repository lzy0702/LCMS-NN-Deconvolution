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
        # checkpoints written before these were recorded hold the 32 768-bin network, a window
        # view, and (with isotope channels, in_ch 6) the first layout's two isotope channels
        ladder = {"window": int(ckpt.get("window") or 32768), "active_top": ckpt.get("active_top"),
                  "iso": bool(ckpt.get("iso_channels", False)), "iso_count": ckpt.get("iso_count"),
                  "view": ckpt.get("view") or "window"}
    model = build_model(ckpt.get("model_size", "proof"), z_max=ckpt.get("z_max", 100),
                        in_ch=ckpt.get("in_ch", 4), comb_prior=ckpt.get("comb_prior", False),
                        arch=arch, **ladder)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _example_input(model, ckpt, length: int):
    """A random input of the model's shape: log-SNR in [0, 5] and, for a network reading
    isotope channels, whole charges (0 = none) with strengths in [0, 1] and, in the current
    layout, whole isotope limits; for a comb-view network, the last eighth unmeasured (zero
    input, measured channel 0) as padding is."""
    import torch

    x = torch.rand(1, ckpt.get("in_ch", 4), length)
    x[:, 0] *= 5.0
    if getattr(model, "iso", False):
        g = torch.Generator().manual_seed(0)
        z = torch.randint(0, 25, (length,), generator=g).float()
        z[torch.rand(length, generator=g) < 0.5] = 0.0
        x[:, 4] = z
        x[:, 5] = torch.where(z > 0, x[:, 5], torch.zeros_like(x[:, 5]))
        if model.iso_count > 2:
            x[:, 6] = torch.randint(0, 40, (length,), generator=g).float()
    if getattr(model, "view", "window") == "comb":
        measured = length - length // 8
        x[:, :, measured:] = 0.0
        x[:, -1, :measured] = 1.0
    return x


#: lengths a comb-view export is checked at against PyTorch: short, long, and one that is no
#: power of two (a 398-2 500 m/z grid, padded to a multiple of 64)
COMB_PARITY_LENGTHS = (8192, 65536, 91712)


def export_onnx(checkpoint: str | Path, out_path: str | Path, length: int = 32768) -> Path:
    import torch

    model, ckpt = load_torch_model(checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ladder = ckpt.get("arch", "unet") == "ladder"
    comb_view = ladder and model.view == "comb"
    if ladder:
        # static shifts over a fixed window: the exported graph is for that length only. The
        # comb view's shifts hold at any length, so its graph keeps the length axis open (and
        # is traced at a short one)
        model.dense = True
        length = 8192 if comb_view else model.window
    dummy = _example_input(model, ckpt, length)
    axes = {0: "batch"} if ladder and not comb_view else {0: "batch", 2: "length"}
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
            "iso_channels": bool(getattr(model, "iso", False)),
            "iso_count": int(getattr(model, "iso_count", 0)),
            "view": getattr(model, "view", "window")}
    import json

    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return out_path


def verify_parity(checkpoint: str | Path, onnx_path: str | Path, length: int = 8192,
                  tol: float = 1e-3, lengths: tuple[int, ...] | None = None) -> dict:
    """Largest differences between the ONNX model and the checkpoint run in PyTorch (a
    LadderNet's sparse path, the reference for the dense export) on random inputs: at the
    window of a window-view LadderNet, at each of ``lengths`` (default
    :data:`COMB_PARITY_LENGTHS`) for a comb-view one, else at ``length``."""
    import onnxruntime as ort
    import torch

    model, ckpt = load_torch_model(checkpoint)
    if ckpt.get("arch", "unet") == "ladder":
        if model.view == "comb":
            lengths = tuple(lengths or COMB_PARITY_LENGTHS)
        else:
            lengths = (model.window,)
    else:
        lengths = tuple(lengths or (length,))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dc = da = 0.0
    for n in lengths:
        x = _example_input(model, ckpt, n)
        with torch.no_grad():
            ref = model(x)
        outs = sess.run(None, {"features": x.numpy().astype(np.float32)})
        dc = max(dc, float(np.abs(ref["charge_logits"].numpy() - outs[0]).max()))
        da = max(da, float(np.abs(ref["apex_logit"].numpy() - outs[1]).max()))
    return {"charge_max_abs_diff": dc, "apex_max_abs_diff": da, "ok": dc < tol and da < tol,
            "lengths": list(lengths)}
