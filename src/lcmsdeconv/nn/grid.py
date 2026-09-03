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


@dataclass(frozen=False, eq=False)
class LogMzGrid:
    mz_min: float = 50.0
    mz_max: float = 10000.0
    step: float = 2e-5
    polarity: int = 1

    def __hash__(self):
        return hash((self.mz_min, self.mz_max, self.step, self.polarity))

    def __eq__(self, other):
        return isinstance(other, LogMzGrid) and (
            self.mz_min, self.mz_max, self.step, self.polarity
        ) == (other.mz_min, other.mz_max, other.step, other.polarity)

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
        """Resample a raw spectrum onto the grid, preserving both shape and scale.

        Each raw sample is treated as covering its own slice of the axis, so a bin's value is
        the integral of intensity over that bin divided by the bin width. For densely sampled
        data this equals the mean of the samples in the bin, so peak shape is preserved; for
        the thresholded profiles that vendor software writes, where only points near peaks
        survive, it correctly reports less signal instead of averaging the surviving peak tops
        up to full height. Empty bins between surviving samples are left at zero.
        """
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        out = np.zeros(self.size, dtype=np.float64)
        m = (mz > self.mz_min) & (mz < self.mz_max)
        if not np.any(m):
            return out
        mz, intensity = mz[m], intensity[m]
        if mz.size == 1:
            b = int(np.clip(self.mz_to_bin(mz), 0, self.size - 1))
            out[b] = intensity[0]
            return out

        u = self.mz_to_u(mz)
        # width in u represented by each sample (midpoints between neighbours)
        du = np.empty_like(u)
        du[1:-1] = 0.5 * (u[2:] - u[:-2])
        du[0] = u[1] - u[0]
        du[-1] = u[-1] - u[-2]
        # a sample never represents more than a bin's worth when data is denser than the grid,
        # and never less than its own spacing when it is sparser
        du = np.clip(du, 0.0, None)
        b = np.clip(self.mz_to_bin(mz), 0, self.size - 1)
        np.add.at(out, b, intensity * du / self.step)
        if centroided is False:
            return out
        # bins that received several samples are already integrated; nothing else to do
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
