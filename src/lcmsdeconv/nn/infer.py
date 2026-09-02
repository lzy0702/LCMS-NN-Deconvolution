"""Charge-state inference on a full grid spectrum using onnxruntime (or a torch fallback)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import featurize
from .grid import LogMzGrid
from .model import DOWNSAMPLE


@dataclass
class ChargePrediction:
    """Per-bin charge assignment on a grid."""

    top1_z: np.ndarray  # int16 [B]
    top1_p: np.ndarray  # float32 [B]
    top2_z: np.ndarray  # int16 [B]
    top2_p: np.ndarray  # float32 [B]
    apex: np.ndarray  # float32 [B] in [0, 1]
    noise_sigma: float


def bundled_model_path() -> Path | None:
    d = Path(__file__).resolve().parent.parent / "models"
    for name in ("charge_unet.onnx", "charge_unet_proof.onnx"):
        if (d / name).exists():
            return d / name
    onnxs = sorted(d.glob("*.onnx")) if d.exists() else []
    return onnxs[0] if onnxs else None


class ChargePredictor:
    def __init__(self, model_path: str | Path | None = None,
                 providers: list[str] | None = None, window: int = 32768, overlap: float = 0.25):
        self.window = window
        self.overlap = overlap
        self.backend = None
        self.z_max = 100
        path = Path(model_path) if model_path else bundled_model_path()
        self.model_path = path
        if path is not None and str(path).endswith((".onnx",)):
            self._init_onnx(path, providers)
        elif path is not None:
            self._init_torch(path)
        else:
            raise FileNotFoundError(
                "No charge model found. Train one with `lcmsdeconv train` and export to ONNX, "
                "or pass model_path explicitly."
            )

    def _init_onnx(self, path: Path, providers):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        req = providers or ["CPUExecutionProvider"]
        use = [p for p in req if p in avail] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=use)
        self.backend = "onnx"
        meta = path.with_suffix(".json")
        if meta.exists():
            self.z_max = json.loads(meta.read_text()).get("z_max", 100)

    def _init_torch(self, path: Path):
        from .export import load_torch_model

        self.torch_model, ckpt = load_torch_model(path)
        self.z_max = ckpt.get("z_max", 100)
        self.backend = "torch"

    # ------------------------------------------------------------- run
    def _run(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """feats [C, L] -> (charge_logits [Cclass, L], apex_logit [L])."""
        x = feats[None].astype(np.float32)
        if self.backend == "onnx":
            outs = self.session.run(None, {"features": x})
            return outs[0][0], outs[1][0, 0]
        import torch

        with torch.no_grad():
            out = self.torch_model(torch.from_numpy(x))
        return out["charge_logits"][0].numpy(), out["apex_logit"][0, 0].numpy()

    def predict_grid(self, grid: LogMzGrid, intensity: np.ndarray, noise_sigma: float) -> ChargePrediction:
        B = intensity.size
        top1_z = np.zeros(B, dtype=np.int16)
        top1_p = np.zeros(B, dtype=np.float32)
        top2_z = np.zeros(B, dtype=np.int16)
        top2_p = np.zeros(B, dtype=np.float32)
        apex = np.zeros(B, dtype=np.float32)
        best_w = np.zeros(B, dtype=np.float32)  # blending weight already applied

        W = self.window
        step = max(1, int(W * (1 - self.overlap)))
        starts = list(range(0, max(1, B - W + 1), step))
        if not starts or starts[-1] + W < B:
            starts.append(max(0, B - W))
        # cosine window weights (high at center)
        ramp = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(W) + 0.5) / W)
        ramp = ramp.astype(np.float32) + 1e-3

        thr = 3.0 * max(noise_sigma, 1e-9)
        for s0 in starts:
            s1 = min(B, s0 + W)
            if intensity[s0:s1].max(initial=0.0) <= thr:
                continue  # nothing above noise: every bin is class 0 anyway
            seg = np.zeros(W, dtype=np.float64)
            seg[: s1 - s0] = intensity[s0:s1]
            pad = (-W) % DOWNSAMPLE
            feats = featurize(seg, noise_sigma, grid.step, z_max=min(self.z_max, 64))
            if pad:
                feats = np.pad(feats, ((0, 0), (0, pad)))
            logits, apx = self._run(feats)
            logits = logits[:, :W]
            apx = apx[:W]
            probs = _softmax(logits, axis=0)  # [C, W]
            order = np.argsort(-probs, axis=0)[:2]  # [2, W]
            w1 = probs[order[0], np.arange(W)]
            z1 = order[0].astype(np.int16)
            w2 = probs[order[1], np.arange(W)]
            z2 = order[1].astype(np.int16)
            wl = ramp[: s1 - s0]
            local = slice(s0, s1)
            better = wl > best_w[local]
            idx = np.nonzero(better)[0]
            gi = idx + s0
            top1_z[gi] = z1[idx]
            top1_p[gi] = w1[idx]
            top2_z[gi] = z2[idx]
            top2_p[gi] = w2[idx]
            apex[gi] = _sigmoid(apx[idx])
            best_w[gi] = wl[idx]
        return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
