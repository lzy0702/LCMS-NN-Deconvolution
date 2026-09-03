"""Additive/multiplicative noise and instrument artifacts for the grid tier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    gain: float = 10.0  # counts per ion (shot-noise scale)
    electronic_sigma: float = 2.0
    baseline: float = 0.0
    chemical_density: float = 0.2  # z=1 chemical-noise peaks per grid "decade" region
    chemical_max: float = 30.0
    spike_prob: float = 0.0
    shot_noise: bool = True


def add_noise(
    signal: np.ndarray,
    cfg: NoiseConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (observed_without_saturation, non_ion_noise) arrays on the same grid.

    ``non_ion_noise`` is the electronic/chemical/baseline content (class 0 in the labels);
    shot noise is a property of the ion signal and is folded into the observed array only.
    """
    L = signal.size
    noise = np.full(L, cfg.baseline, dtype=np.float64)
    if cfg.electronic_sigma > 0:
        noise += np.abs(rng.normal(0.0, cfg.electronic_sigma, L))
    # chemical noise: sparse singly-charged-like sticks (already-on-grid spikes)
    n_chem = int(cfg.chemical_density * L / 4000)
    if n_chem > 0:
        pos = rng.integers(0, L, n_chem)
        amp = rng.uniform(1.0, cfg.chemical_max, n_chem)
        np.add.at(noise, pos, amp)
    observed = signal.copy()
    if cfg.shot_noise and cfg.gain > 0:
        lam = np.clip(signal / cfg.gain, 0, None)
        observed = rng.poisson(lam).astype(np.float64) * cfg.gain
    observed = observed + noise
    if cfg.spike_prob > 0:
        spikes = rng.random(L) < cfg.spike_prob
        observed[spikes] += rng.uniform(cfg.chemical_max, 5 * cfg.chemical_max, int(spikes.sum()))
    return observed, noise
