"""Export a trained charge model to ONNX and load it back."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_torch_model(checkpoint: str | Path):
    import torch

    from .model import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("model_size", "proof"), z_max=ckpt.get("z_max", 100))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export_onnx(checkpoint: str | Path, out_path: str | Path, length: int = 32768) -> Path:
    import torch

    model, ckpt = load_torch_model(checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 4, length)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=["charge_logits", "apex_logit"],
        dynamic_axes={
            "features": {0: "batch", 2: "length"},
            "charge_logits": {0: "batch", 2: "length"},
            "apex_logit": {0: "batch", 2: "length"},
        },
        opset_version=17,
        dynamo=False,
    )
    # write a small sidecar with metadata
    meta = {"z_max": ckpt.get("z_max", 100), "model_size": ckpt.get("model_size", "proof")}
    import json

    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return out_path


def verify_parity(checkpoint: str | Path, onnx_path: str | Path, length: int = 8192,
                  tol: float = 1e-3) -> dict:
    import onnxruntime as ort
    import torch

    model, _ = load_torch_model(checkpoint)
    x = torch.randn(1, 4, length)
    with torch.no_grad():
        ref = model(x)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"features": x.numpy().astype(np.float32)})
    onx = {"charge_logits": outs[0], "apex_logit": outs[1]}
    dc = float(np.abs(ref["charge_logits"].numpy() - onx["charge_logits"]).max())
    da = float(np.abs(ref["apex_logit"].numpy() - onx["apex_logit"]).max())
    return {"charge_max_abs_diff": dc, "apex_max_abs_diff": da, "ok": dc < tol and da < tol}
