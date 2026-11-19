"""Co-elution of a species' charge states across the scans of a summed spectrum.

The charge states of one molecule are the same molecules seen at different m/z, so their
intensities rise and fall together over the scans the summed spectrum was made of. A charge
ladder assembled from unrelated ions does not: each of them follows its own elution profile.
Above the noise this is the test that tells the two apart. The noise threshold cannot, once
the spectrum is resampled correctly (:mod:`lcmsdeconv.core.profile`): vendor software has
already removed the noise, and what false ladders are built from is real, low-level signal of
other ions.
"""

from __future__ import annotations

import numpy as np

#: two charge states co-elute when their scan-by-scan traces correlate at least this well
COELUTION_R = 0.5
#: fewer scans carrying signal than this and a trace cannot show a profile: the test abstains
MIN_SCANS = 4


class FrameStack:
    """The scans a summed spectrum was made of, ready for windowed intensity traces."""

    def __init__(self, frames):
        self.mz: list[np.ndarray] = []
        self.cum: list[np.ndarray] = []
        for f in frames:
            mz = np.asarray(f.mz, dtype=float)
            y = np.clip(np.asarray(f.intensity, dtype=float), 0.0, None)
            if mz.size > 1 and np.any(np.diff(mz) < 0):
                order = np.argsort(mz, kind="stable")
                mz, y = mz[order], y[order]
            self.mz.append(mz)
            self.cum.append(np.concatenate([[0.0], np.cumsum(y)]))

    @classmethod
    def from_spectrum(cls, spectrum, min_frames: int = MIN_SCANS) -> FrameStack | None:
        """The stack of a summed spectrum's scans (``meta["frames"]``), if it has enough."""
        frames = (spectrum.meta or {}).get("frames")
        if not frames or len(frames) < min_frames:
            return None
        return cls(frames)

    def __len__(self) -> int:
        return len(self.mz)

    def traces(self, centres, halves) -> np.ndarray:
        """Summed intensity within ``centre +- half`` in every scan: shape (windows, scans)."""
        c = np.atleast_1d(np.asarray(centres, dtype=float))
        h = np.broadcast_to(np.asarray(halves, dtype=float), c.shape)
        lo, hi = c - h, c + h
        out = np.empty((c.size, len(self.mz)))
        for j, (mz, cum) in enumerate(zip(self.mz, self.cum)):
            out[:, j] = cum[np.searchsorted(mz, hi)] - cum[np.searchsorted(mz, lo)]
        return out


def shows_profile(trace: np.ndarray, min_scans: int = MIN_SCANS) -> bool:
    """Does ``trace`` carry signal in enough scans to have an elution profile at all?"""
    return int(np.count_nonzero(trace > 0)) >= min_scans


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two traces; 0 when either is constant (no profile to share)."""
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def coeluting(traces: np.ndarray, ref: int, r_min: float = COELUTION_R,
              min_scans: int = MIN_SCANS) -> np.ndarray:
    """Which rows of ``traces`` co-elute with row ``ref`` (which always does).

    When the reference itself shows no profile (signal in fewer than ``min_scans`` scans, or
    fewer scans than that in all) the test abstains and keeps every row: it does not veto what
    it cannot see. A weak row is still judged: if the few scans that carry it are the
    reference's strongest, it correlates.
    """
    keep = np.ones(traces.shape[0], dtype=bool)
    if traces.shape[1] < min_scans or not shows_profile(traces[ref], min_scans):
        return keep
    for k in range(traces.shape[0]):
        if k != ref:
            keep[k] = correlation(traces[ref], traces[k]) >= r_min
    return keep
