"""Charge-state inference on a full grid spectrum using onnxruntime (or a torch fallback)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DOWNSAMPLE
from .features import ISOTOPE_CHANNELS_FIRST_LAYOUT, featurize, isotope_channels
from .grid import LogMzGrid

#: a comb-view network's features are computed on the grid followed by at least this many zero
#: bins (and on no fewer than COMB_MIN_FEATURE_LENGTH bins): the training frames' grids sit at
#: the start of a 131 072-bin crop with zeros after them, and the local-maximum feature reads
#: 2 000 bins either side (its filter is shorter on inputs under 8 000 bins)
COMB_FEATURE_TAIL = 2048
COMB_MIN_FEATURE_LENGTH = 8192
#: unmeasured bins a comb-view network reads after the grid: its encoder reaches 60 bins either
#: side, and what it computes at the grid's last bins is what it computed in training, where
#: zeros followed the grid, only if that many zero bins follow the grid here too
COMB_TAIL = 64


def comb_length(bins: int) -> int:
    """The input length a comb-view network reads a grid of ``bins`` bins at: the grid and at
    least :data:`COMB_TAIL` zero bins, up to a multiple of 64."""
    n = int(bins) + COMB_TAIL
    return n + (-n) % DOWNSAMPLE


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
    """The model loaded automatically, if one is installed.

    Only ``charge_unet.onnx`` is picked up. Other models in the directory (such as the small
    CPU-trained proof model) must be named explicitly, so a model is never used by default
    until it has been shown to beat the deterministic estimator it would replace.
    """
    d = Path(__file__).resolve().parent.parent / "models"
    path = d / "charge_unet.onnx"
    return path if path.exists() else None


class ChargePredictor:
    """A trained charge network run over a whole grid spectrum, in windows of its own length.

    A LadderNet reads windows of the length it was trained at (recorded with it): the whole grid
    in one zero-padded window when it fits, else windows overlapping by ``overlap``, each bin
    read from the window it is most central in. A comb-view LadderNet (``view == "comb"``)
    reads the whole grid in one pass at its own length, padded to a multiple of 64
    (:func:`comb_length`), with the measured channel 1 on the grid's bins: no windows. A
    network with isotope-charge input channels (``iso_channels``; ``iso_count`` of them) needs
    ``instrument``, whose resolving power those channels are computed with
    (:func:`lcmsdeconv.nn.features.isotope_channels`, over the whole grid, then cut to each
    window; empty for a time-of-flight instrument, as in training); it can also be set on the
    predictor before ``predict_grid`` is called.
    """

    def __init__(self, model_path: str | Path | None = None,
                 providers: list[str] | None = None, window: int = 32768, overlap: float = 0.25,
                 instrument=None):
        self.window = window
        self.overlap = overlap
        self.backend = None
        self.z_max = 100
        self.evidence_channels = 0
        #: grid step a model's fixed offsets assume (LadderNet), None when it does not matter
        self.grid_step: float | None = None
        #: whether the network reads the isotope-charge channels, which need ``instrument``,
        #: and how many (2 in the first layout, 3 with the isotope limit)
        self.iso_channels = False
        self.iso_count = 0
        #: "window" (windows of the training length) or "comb" (the whole grid in one pass)
        self.view = "window"
        #: the LadderNet's active-bin cap (recorded with the model; informational)
        self.active_top: int | None = None
        self.instrument = instrument
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
            info = json.loads(meta.read_text())
            self.z_max = info.get("z_max", 100)
            self.evidence_channels = int(info.get("evidence_channels", 0))
            if info.get("arch") == "ladder":
                self.grid_step = float(info.get("grid_step", 2e-5))
                self.window = int(info.get("window", self.window))
                self.active_top = info.get("active_top")
                self.iso_channels = bool(info.get("iso_channels", False))
                # sidecars written before these were recorded: the first layout, a window view
                self.iso_count = int(info.get("iso_count") or (ISOTOPE_CHANNELS_FIRST_LAYOUT
                                                               if self.iso_channels else 0))
                self.view = info.get("view") or "window"

    def _init_torch(self, path: Path):
        from .export import load_torch_model

        self.torch_model, ckpt = load_torch_model(path)
        self.z_max = ckpt.get("z_max", 100)
        self.evidence_channels = int(ckpt.get("evidence_channels", 0))
        if ckpt.get("arch") == "ladder":
            self.grid_step = float(ckpt.get("grid_step", 2e-5))
            # the network's own window: its ladder offsets and active-bin cap are fixed for it
            self.window = int(self.torch_model.window)
            self.active_top = int(self.torch_model.active_top)
            self.iso_channels = bool(self.torch_model.iso)
            self.iso_count = int(self.torch_model.iso_count)
            self.view = self.torch_model.view
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
        if self.grid_step is not None and abs(grid.step - self.grid_step) > 1e-12:
            raise ValueError(
                f"this charge model reads charge ladders at offsets fixed for a grid step of "
                f"{self.grid_step:g}; the method's grid step is {grid.step:g}. Use a method with "
                f"grid_step {self.grid_step:g}, or another model.")
        iso = None
        if self.iso_channels:
            if self.instrument is None:
                raise ValueError(
                    "this charge model reads charges from resolved isotope spacing, which needs "
                    "the instrument's resolving power: pass instrument= to ChargePredictor "
                    "(make_predictor and process_run do), or set predictor.instrument before "
                    "calling predict_grid")
            # over the whole grid (absolute m/z), once; each window takes its slice
            iso = isotope_channels(grid, intensity, noise_sigma, self.instrument,
                                   count=self.iso_count)
        if self.view == "comb":
            return self._predict_whole(grid, intensity, noise_sigma, iso)
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
            feats = featurize(seg, noise_sigma, grid.step, z_max=min(self.z_max, 64),
                              evidence_channels=self.evidence_channels)
            if iso is not None:
                part = np.zeros((iso.shape[0], W), dtype=np.float32)
                part[:, : s1 - s0] = iso[:, s0:s1]
                feats = np.concatenate([feats, part])
            if pad:
                feats = np.pad(feats, ((0, 0), (0, pad)))
            logits, apx = self._run(feats)
            logits = logits[:, :W]
            apx = apx[:W]
            probs = _softmax(logits, axis=0)  # [C, W]
            # Which charge, given that the bin is an ion: the decoder decides whether a bin is
            # signal from its intensity, so the charge is read off the ion classes alone and its
            # probability renormalised. Taking the argmax over all classes instead lets a
            # background share beat a charge distribution spread over a few classes, and the bin
            # then casts no vote at all.
            ion = probs[1:]
            ion = ion / np.maximum(ion.sum(axis=0, keepdims=True), 1e-12)
            order = np.argsort(-ion, axis=0)[:2]  # [2, W]
            w1 = ion[order[0], np.arange(W)]
            z1 = (order[0] + 1).astype(np.int16)
            w2 = ion[order[1], np.arange(W)]
            z2 = (order[1] + 1).astype(np.int16)
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

    def comb_input(self, grid: LogMzGrid, intensity: np.ndarray, noise_sigma: float,
                   iso: np.ndarray | None) -> np.ndarray:
        """A comb-view network's input for the whole grid ``[C, L]``: the grid followed by zeros
        (:func:`comb_length`), the base features (computed as on a training frame, see
        :data:`COMB_FEATURE_TAIL`), the isotope channels and the measured channel last."""
        B = intensity.size
        L = comb_length(B)
        seg = np.zeros(max(B + COMB_FEATURE_TAIL, COMB_MIN_FEATURE_LENGTH), dtype=np.float64)
        seg[:B] = intensity
        parts = [featurize(seg, noise_sigma, grid.step, z_max=min(self.z_max, 64),
                           evidence_channels=self.evidence_channels)[:, :L]]
        if iso is not None:
            part = np.zeros((iso.shape[0], L), dtype=np.float32)
            part[:, :B] = iso
            parts.append(part)
        measured = np.zeros((1, L), dtype=np.float32)
        measured[0, :B] = 1.0
        parts.append(measured)
        return np.concatenate(parts)

    def _predict_whole(self, grid: LogMzGrid, intensity: np.ndarray, noise_sigma: float,
                       iso: np.ndarray | None) -> ChargePrediction:
        """A comb-view network over the whole grid in one pass."""
        B = intensity.size
        top1_z = np.zeros(B, dtype=np.int16)
        top1_p = np.zeros(B, dtype=np.float32)
        top2_z = np.zeros(B, dtype=np.int16)
        top2_p = np.zeros(B, dtype=np.float32)
        apex = np.zeros(B, dtype=np.float32)
        if B == 0 or intensity.max(initial=0.0) <= 3.0 * max(noise_sigma, 1e-9):
            return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)
        logits, apx = self._run(self.comb_input(grid, intensity, noise_sigma, iso))
        probs = _softmax(logits[:, :B], axis=0)
        # the charge among the ion classes, renormalised (see predict_grid)
        ion = probs[1:]
        ion = ion / np.maximum(ion.sum(axis=0, keepdims=True), 1e-12)
        order = np.argsort(-ion, axis=0)[:2]
        cols = np.arange(B)
        top1_z[:] = order[0] + 1
        top1_p[:] = ion[order[0], cols]
        top2_z[:] = order[1] + 1
        top2_p[:] = ion[order[1], cols]
        apex[:] = _sigmoid(apx[:B])
        return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
