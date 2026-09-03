"""Instrument response models: resolving power, peak shape, sampling, saturation, centroiding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FWHM_TO_SIGMA = 1.0 / 2.354820045


@dataclass
class InstrumentModel:
    """Parametric description of a mass analyzer's line shape and dynamic range.

    ``resolution`` is the FWHM resolving power at ``mz_ref``; the resolving power varies as
    ``R(mz) = resolution * (mz / mz_ref) ** mz_exponent`` (0 for TOF, -0.5 for Orbitrap,
    -1 for FT-ICR).
    """

    kind: str = "tof"
    resolution: float = 30000.0
    mz_ref: float = 1000.0
    mz_exponent: float | None = None
    shape: str = "gaussian"  # gaussian | lorentzian | mixed
    eta: float = 0.0  # Lorentzian fraction when shape == "mixed"
    saturation_level: float | None = None
    saturation_kind: str = "clip"  # clip (ADC) | tdc (dead-time compression)
    points_per_fwhm: float = 5.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mz_exponent is None:
            self.mz_exponent = {"tof": 0.0, "orbitrap": -0.5, "fticr": -1.0}.get(self.kind, 0.0)
        if self.kind == "orbitrap" and self.mz_ref == 1000.0:
            self.mz_ref = 200.0
        if self.kind == "fticr" and self.mz_ref == 1000.0:
            self.mz_ref = 400.0

    # ------------------------------------------------------------- line width
    def resolving_power(self, mz):
        mz = np.asarray(mz, dtype=float)
        return self.resolution * (mz / self.mz_ref) ** float(self.mz_exponent)

    def fwhm(self, mz):
        return np.asarray(mz, dtype=float) / self.resolving_power(mz)

    def sigma_mz(self, mz):
        return self.fwhm(mz) * FWHM_TO_SIGMA

    def sigma_ln(self, mz):
        """Peak sigma in ln(m/z) units (dimensionless)."""
        return FWHM_TO_SIGMA / self.resolving_power(mz)

    # ------------------------------------------------------------- sampling axis
    def profile_axis(self, mz_min: float, mz_max: float, points_per_fwhm: float | None = None) -> np.ndarray:
        """m/z axis with roughly constant number of points per FWHM (piecewise log-uniform)."""
        ppf = points_per_fwhm or self.points_per_fwhm
        edges = np.exp(np.linspace(np.log(mz_min), np.log(mz_max), 33))
        pieces = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            r = float(self.resolving_power(np.sqrt(lo * hi)))
            step = 1.0 / (r * ppf)
            n = max(2, int(np.ceil(np.log(hi / lo) / step)))
            pieces.append(np.exp(np.linspace(np.log(lo), np.log(hi), n, endpoint=False)))
        pieces.append(np.array([mz_max]))
        return np.concatenate(pieces)

    # ------------------------------------------------------------- saturation
    def apply_saturation(self, intensity: np.ndarray) -> np.ndarray:
        if not self.saturation_level:
            return intensity
        s = float(self.saturation_level)
        if self.saturation_kind == "tdc":
            return intensity * np.exp(-intensity / s)
        return np.minimum(intensity, s)

    # ------------------------------------------------------------- kernel
    def kernel(self, sigma_bins: float, half_width: int | None = None) -> np.ndarray:
        """Normalized (peak height 1) line-shape kernel on an evenly spaced grid."""
        sigma_bins = max(float(sigma_bins), 0.3)
        if half_width is None:
            half_width = int(np.ceil(sigma_bins * (4.0 if self.shape == "gaussian" else 12.0)))
        x = np.arange(-half_width, half_width + 1, dtype=float)
        g = np.exp(-0.5 * (x / sigma_bins) ** 2)
        if self.shape == "gaussian" or self.eta <= 0:
            return g
        gamma = sigma_bins * 2.354820045 / 2.0
        lor = 1.0 / (1.0 + (x / gamma) ** 2)
        eta = 1.0 if self.shape == "lorentzian" else float(self.eta)
        return (1 - eta) * g + eta * lor

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "resolution": self.resolution,
            "mz_ref": self.mz_ref,
            "mz_exponent": self.mz_exponent,
            "shape": self.shape,
            "eta": self.eta,
            "saturation_level": self.saturation_level,
            "saturation_kind": self.saturation_kind,
            "points_per_fwhm": self.points_per_fwhm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InstrumentModel:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**known)


def estimate_noise_sigma(intensity: np.ndarray) -> float:
    """Robust estimate of the additive noise level of a profile or grid spectrum.

    Uses the median absolute deviation of first differences over the lower half of the
    intensity distribution, which is insensitive to peaks.
    """
    y = np.asarray(intensity, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return 1.0
    nz = y[y > 0]
    if nz.size < 8:
        return 1.0
    thr = np.percentile(nz, 60)
    low = y[(y > 0) & (y <= thr)]
    if low.size < 8:
        low = nz
    d = np.diff(low)
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(float(np.std(low)), 1e-3)
    return float(sigma)


def is_profile(mz: np.ndarray, intensity: np.ndarray) -> bool:
    """Heuristic: profile data has many zero/near-zero points and dense, regular spacing."""
    if mz.size < 50:
        return False
    zero_frac = float(np.mean(intensity <= 0))
    if zero_frac > 0.05:
        return True
    d = np.diff(mz) / mz[:-1]
    reg = np.median(d)
    spread = np.percentile(d, 90) / max(reg, 1e-12)
    return spread < 3.0 and reg < 5e-4


def centroid_profile(
    mz: np.ndarray,
    intensity: np.ndarray,
    noise_sigma: float | None = None,
    min_snr: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local-maximum centroiding with parabolic apex interpolation.

    Returns (mz_centroid, apex_intensity, fwhm) arrays. The FWHM is measured from the
    half-height crossings when available.
    """
    y = np.asarray(intensity, dtype=float)
    x = np.asarray(mz, dtype=float)
    if y.size < 3:
        return np.array([]), np.array([]), np.array([])
    sigma = estimate_noise_sigma(y) if noise_sigma is None else float(noise_sigma)
    thr = max(min_snr * sigma, 1e-12)
    mid = y[1:-1]
    is_max = (mid > y[:-2]) & (mid >= y[2:]) & (mid >= thr)
    idx = np.nonzero(is_max)[0] + 1
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([])
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom < 0, 0.5 * (y0 - y2) / denom, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    dx = np.where(delta >= 0, x[np.minimum(idx + 1, x.size - 1)] - x[idx], x[idx] - x[idx - 1])
    mzc = x[idx] + delta * dx
    apex = y1 - 0.25 * (y0 - y2) * delta
    # FWHM by half-height crossing search (bounded walk)
    fwhm = np.empty(idx.size)
    for k, i in enumerate(idx):
        half = 0.5 * y[i]
        j = i
        while j > 0 and y[j] > half:
            j -= 1
        left = x[j] if j == 0 else x[j] + (half - y[j]) / max(y[j + 1] - y[j], 1e-12) * (x[j + 1] - x[j])
        j = i
        n = y.size - 1
        while j < n and y[j] > half:
            j += 1
        right = x[j] if j == n else x[j - 1] + (y[j - 1] - half) / max(y[j - 1] - y[j], 1e-12) * (x[j] - x[j - 1])
        fwhm[k] = max(right - left, 1e-9)
    return mzc, apex, fwhm
