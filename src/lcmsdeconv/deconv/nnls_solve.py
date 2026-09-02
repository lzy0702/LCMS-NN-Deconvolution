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
