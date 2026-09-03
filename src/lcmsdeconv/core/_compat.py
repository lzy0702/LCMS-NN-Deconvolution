"""Small numpy-version compatibility helpers."""

from __future__ import annotations

import numpy as np

try:
    trapezoid = np.trapezoid  # numpy >= 2.0
except AttributeError:  # pragma: no cover
    trapezoid = np.trapz
