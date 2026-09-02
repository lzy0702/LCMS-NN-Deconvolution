"""Logarithmic m/z grid used by the neural network and the synthetic renderer.

The grid coordinate is ``u = ln(m/z - carrier)`` where ``carrier`` is ``+m_proton`` in positive
mode and ``-m_proton`` in negative mode. Under this transform a proton-charged ion of neutral
mass ``M`` at charge ``z`` sits at ``u = ln(M) - ln(z)``, independent of ``M`` up to a shift, so
adjacent charge states are a fixed distance ``ln((z+1)/z)`` apart and the charge-envelope pattern
is translation-equivariant. An adduct that adds mass ``d`` at constant charge is simply a
proton-charged ion of mass ``M + d`` (``u = ln(M + d) - ln(z)``), so adducts need no special
encoding in the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chem.adducts import carrier_mass


@dataclass
class LogMzGrid:
    mz_min: float = 50.0
    mz_max: float = 10000.0
    step: float = 2e-5
    polarity: int = 1

    def __post_init__(self) -> None:
        self.carrier = carrier_mass(self.polarity)
        self.u_min = float(np.log(self.mz_min - self.carrier))
        self.u_max = float(np.log(self.mz_max - self.carrier))
        self.size = int(np.ceil((self.u_max - self.u_min) / self.step)) + 1

    # --------------------------------------------------------------- axes
    @property
    def u(self) -> np.ndarray:
        return self.u_min + np.arange(self.size) * self.step

    @property
    def mz(self) -> np.ndarray:
        return np.exp(self.u) + self.carrier

    # --------------------------------------------------------------- transforms
    def mz_to_u(self, mz: np.ndarray | float) -> np.ndarray:
        return np.log(np.asarray(mz, dtype=float) - self.carrier)

    def u_to_mz(self, u: np.ndarray | float) -> np.ndarray:
        return np.exp(np.asarray(u, dtype=float)) + self.carrier

    def mz_to_bin(self, mz: np.ndarray | float) -> np.ndarray:
        return np.round((self.mz_to_u(mz) - self.u_min) / self.step).astype(np.int64)

    def u_to_bin(self, u: np.ndarray | float) -> np.ndarray:
        return np.round((np.asarray(u, dtype=float) - self.u_min) / self.step).astype(np.int64)

    def bin_to_u(self, b: np.ndarray | float) -> np.ndarray:
        return self.u_min + np.asarray(b, dtype=float) * self.step

    def bin_to_mz(self, b: np.ndarray | float) -> np.ndarray:
        return self.u_to_mz(self.bin_to_u(b))

    def mass_charge_to_u(self, mass: float, z: int) -> float:
        """u-position of a proton-charged ion of neutral mass ``mass`` at charge ``z``."""
        return float(np.log(mass)) - float(np.log(z))

    def mass_charge_to_bin(self, mass: float, z: int) -> int:
        return int(round((self.mass_charge_to_u(mass, z) - self.u_min) / self.step))

    def bin_to_mass(self, b: float, z: int) -> float:
        """Neutral (proton-charged) mass implied by grid bin ``b`` at charge ``z``."""
        return float(z * np.exp(self.bin_to_u(b)))

    def in_range_mz(self, mz: float) -> bool:
        return self.mz_min <= mz <= self.mz_max

    # --------------------------------------------------------------- resampling
    def resample_profile(self, mz: np.ndarray, intensity: np.ndarray,
                         centroided: bool | None = None) -> np.ndarray:
        """Resample a raw spectrum onto the grid, preserving peak shape.

        Profile data is bin-averaged (the raw axis is usually denser than the grid, so summing
        would turn peaks into spikes whose shape no longer matches the instrument line shape)
        and empty bins between samples are linearly interpolated, so a rendered template and a
        resampled measurement have the same shape. Centroided data is splatted with the
        instrument line width instead, by :meth:`render_centroids`.
        """
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        out = np.zeros(self.size, dtype=np.float64)
        m = (mz > self.mz_min) & (mz < self.mz_max)
        if not np.any(m):
            return out
        mz, intensity = mz[m], intensity[m]
        b = np.clip(self.mz_to_bin(mz), 0, self.size - 1)
        sums = np.bincount(b, weights=intensity, minlength=self.size)
        counts = np.bincount(b, minlength=self.size)
        filled = counts > 0
        out[filled] = sums[filled] / counts[filled]
        if centroided is False or filled.sum() < 2:
            return out
        # linear interpolation across bins that received no raw sample (upsampling regions)
        idx = np.nonzero(filled)[0]
        if idx.size >= 2 and idx.size < self.size:
            lo, hi = idx[0], idx[-1]
            span = np.arange(lo, hi + 1)
            out[lo:hi + 1] = np.interp(span, idx, out[idx])
        return out

    def render_centroids(self, mz: np.ndarray, intensity: np.ndarray, instrument) -> np.ndarray:
        """Place centroided peaks on the grid with the instrument line shape."""
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        out = np.zeros(self.size, dtype=np.float64)
        m = (mz > self.mz_min) & (mz < self.mz_max) & (intensity > 0)
        if not np.any(m):
            return out
        mz, intensity = mz[m], intensity[m]
        centers = (self.mz_to_u(mz) - self.u_min) / self.step
        sigma = np.maximum(instrument.sigma_mz(mz) / (mz - self.carrier) / self.step, 0.5)
        hw = int(np.ceil(4 * sigma.max())) + 1
        offs = np.arange(-hw, hw + 1)
        bins = np.floor(centers).astype(np.int64)[:, None] + offs[None, :]
        d = centers[:, None] - bins
        vals = intensity[:, None] * np.exp(-0.5 * (d / sigma[:, None]) ** 2)
        keep = (bins >= 0) & (bins < self.size)
        np.add.at(out, bins[keep], vals[keep])
        return out

    def resample_max(self, mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Resample by taking the max intensity in each bin (peak-preserving for display)."""
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        out = np.zeros(self.size, dtype=np.float64)
        m = (mz > self.mz_min) & (mz < self.mz_max)
        if not np.any(m):
            return out
        b = np.clip(self.mz_to_bin(mz[m]), 0, self.size - 1)
        np.maximum.at(out, b, intensity[m])
        return out


def default_grid(polarity: int = 1, step: float = 2e-5, mz_max: float = 10000.0) -> LogMzGrid:
    return LogMzGrid(50.0, mz_max, step, polarity)
