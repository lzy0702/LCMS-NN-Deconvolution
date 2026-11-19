"""Non-negative least squares through the normal equations.

The design matrices here are very tall and narrow: tens of thousands of grid bins against a few
hundred templates. Solving that shape directly is slow, but the problem only depends on the data
through the Gram matrix, so a Cholesky factor of ``AᵀA`` turns it into an equivalent problem of
size (columns x columns). For 15 000 rows and 100 columns this is about two orders of magnitude
faster and returns the same solution.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.optimize import nnls


def fit_weights(observed: np.ndarray, noise_sigma: float, grid_step: float = 2e-5,
                resolution: float = 30000.0) -> np.ndarray:
    """Least-squares weights of a grid spectrum: one over the standard deviation of each bin.

    A bin's variance is its expected signal (shot noise) plus the noise floor. The expected
    signal is taken as the largest observed value within one peak width (at least two bins) to
    either side, not as the bin's own value: a template whose peak sits a fraction of a bin off
    the data, or is a little narrower, predicts more than is observed in a peak's flank bins,
    and weighting those by their own small values pulls every strong species' coefficient down
    (Neyman's chi-square bias) - to a fifth of the envelope on real Orbitrap data once the
    noise floor stopped hiding it - while weak species, nearer the floor, keep theirs.
    """
    from scipy.ndimage import maximum_filter1d

    y = np.clip(np.asarray(observed, dtype=float), 0.0, None)
    half = max(2, int(np.ceil(1.0 / (max(resolution, 1.0) * max(grid_step, 1e-12)))))
    local = maximum_filter1d(y, size=2 * half + 1, mode="constant") if y.size else y
    return 1.0 / np.sqrt(local + noise_sigma ** 2 + 1e-9)


def weighted_nnls(A: np.ndarray, b: np.ndarray, w: np.ndarray | None = None,
                  ridge: float = 1e-10, maxiter_factor: int = 10) -> np.ndarray:
    """Solve ``min ||W(Ax - b)||`` subject to ``x >= 0``.

    ``ridge`` stabilizes the factorization when templates are nearly collinear, which happens
    when an adduct spacing falls below the peak width.
    """
    if A.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)
    if w is not None:
        Aw = A * w[:, None]
        bw = b * w
    else:
        Aw, bw = A, b
    n = Aw.shape[1]
    if Aw.shape[0] <= 2 * n:  # already small; solve directly
        try:
            x, _ = nnls(Aw, bw, maxiter=maxiter_factor * n)
            return x
        except Exception:
            return np.zeros(n)

    gram = Aw.T @ Aw
    rhs = Aw.T @ bw
    scale = float(np.trace(gram)) / max(n, 1)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(n)
    for extra in (ridge, 1e-8, 1e-6, 1e-4):
        try:
            lower = cholesky(gram + extra * scale * np.eye(n), lower=True)
            d = solve_triangular(lower, rhs, lower=True)
            x, _ = nnls(lower.T, d, maxiter=maxiter_factor * n)
            return x
        except (LinAlgError, ValueError):
            continue
    try:
        x, _ = nnls(Aw, bw, maxiter=maxiter_factor * n)
        return x
    except Exception:
        return np.zeros(n)
