"""A deterministic charge estimator used as a fallback and as a baseline for the network.

The same charge-evidence comb that feeds the network as an input channel can be used on its own:
for each hypothesised charge z, the spectrum is shifted onto the positions where that charge's
neighbouring charge states would lie, and the charge whose neighbours line up best wins. This
needs no training, runs in about a second over a full grid, and gives the decoder something
sensible to work with on a machine where no model has been trained yet.

It is weaker than a trained network on overlapping envelopes and on low charge states whose
neighbours fall outside the analysis window, which is why the network exists; but it makes the
package useful out of the box and provides the baseline any model must beat.
"""

from __future__ import annotations

import numpy as np

from ..nn.features import isotope_reading, shifted
from ..nn.grid import LogMzGrid
from ..nn.infer import ChargePrediction


class CombPredictor:
    """Charge assignment by neighbour-spacing coincidence, with no neural network."""

    def __init__(self, z_max: int = 60, neighbours: tuple[int, ...] = (1, 2, 3),
                 window: int = 65536, min_snr: float = 3.0, instrument=None,
                 isotope_resolution_factor: float = 2.5, isotope_agreement: float = 0.6):
        self.z_max = z_max
        self.neighbours = neighbours
        self.window = window
        self.min_snr = min_snr
        self.backend = "comb"
        self.model_path = None
        #: with an instrument model, charges are read from resolved isotope spacing wherever the
        #: instrument resolves 1.00336/z at that m/z (see :meth:`isotope_charges`)
        self.instrument = instrument
        self.isotope_resolution_factor = isotope_resolution_factor
        self.isotope_agreement = isotope_agreement

    def predict_grid(self, grid: LogMzGrid, intensity: np.ndarray, noise_sigma: float) -> ChargePrediction:
        B = intensity.size
        top1_z = np.zeros(B, dtype=np.int16)
        top1_p = np.zeros(B, dtype=np.float32)
        top2_z = np.zeros(B, dtype=np.int16)
        top2_p = np.zeros(B, dtype=np.float32)
        apex = np.zeros(B, dtype=np.float32)

        ns = max(noise_sigma, 1e-9)
        s = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
        active = intensity > self.min_snr * ns
        if not active.any():
            return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)

        best = np.zeros(B)
        second = np.zeros(B)
        for z in range(1, self.z_max + 1):
            acc = np.zeros(B)
            n = 0
            for k in self.neighbours:
                for zz in (z + k, z - k):
                    if zz < 1:
                        continue
                    shift = int(round(np.log(z / zz) / grid.step))
                    if shift == 0 or abs(shift) >= B:
                        continue
                    acc += shifted(s, shift)
                    n += 1
            if n == 0:
                continue
            ev = s * (acc / n)
            better = ev > best
            second[better] = best[better]
            top2_z[better] = top1_z[better]
            best[better] = ev[better]
            top1_z[better] = z
            between = (~better) & (ev > second)
            second[between] = ev[between]
            top2_z[between] = z

        total = best + second + 1e-12
        top1_p = (best / total).astype(np.float32)
        top2_p = (second / total).astype(np.float32)
        top1_z[~active] = 0
        top1_p[~active] = 0.0
        top2_z[~active] = 0
        top2_p[~active] = 0.0

        if self.instrument is not None:
            zi, conf = self.isotope_charges(grid, s, active)
            use = zi > 0
            # resolved isotope spacing is direct evidence of the charge; it overrides the ladder
            # comb, whose evidence is weakest exactly where isotopes are easiest to resolve (low
            # charge states, few neighbours in the window)
            top2_z[use] = np.where(top1_z[use] != zi[use], top1_z[use], top2_z[use])
            top2_p[use] = np.where(top1_z[use] != zi[use], 1.0 - conf[use], top2_p[use])
            top1_z[use] = zi[use]
            top1_p[use] = conf[use]

        # apex map: local maxima of the measured signal
        mid = intensity[1:-1]
        ismax = (mid >= intensity[:-2]) & (mid > intensity[2:]) & (mid > self.min_snr * ns)
        apex[1:-1] = ismax.astype(np.float32)
        return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)


    def isotope_charges(self, grid: LogMzGrid, s: np.ndarray, active: np.ndarray):
        """Charge per bin from resolved isotope spacing: (charge or 0, confidence).

        For each charge z whose isotope spacing 1.00336/z the instrument resolves at a bin's m/z,
        the evidence is the bin's log-SNR times the mean log-SNR one spacing either side. A z
        envelope also scores at the divisors of z (two or more isotopes away) but not at 2z,
        whose half-spacing positions fall between its peaks; the largest charge scoring within
        ``isotope_agreement`` of the best is therefore the answer
        (:func:`lcmsdeconv.nn.features.isotope_reading`, which the charge network's isotope
        channels share). It counts only where the neighbours carry a real share of the bin's
        signal (best evidence at least a quarter of the bin's own log-SNR squared).
        """
        B = s.size
        zout = np.zeros(B, dtype=np.int16)
        conf = np.zeros(B, dtype=np.float32)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            return zout, conf
        zraw, best_all = isotope_reading(grid, s, active, self.instrument, self.z_max,
                                         self.isotope_resolution_factor, self.isotope_agreement)
        chosen = zraw[idx]
        best = best_all[idx]
        strong = best >= 0.25 * s[idx] ** 2  # neighbours carry a real share of the bin's signal
        chosen[~strong] = 0
        zout[idx] = chosen
        conf[idx] = np.where(chosen > 0, 0.9, 0.0)
        return zout, conf


def make_predictor(model_path: str | None, providers=None, z_max: int = 60, instrument=None):
    """Return a network predictor, or the deterministic comb predictor when asked for it.

    ``model_path`` of ``"comb"`` or ``"none"`` selects the deterministic estimator; ``None``
    uses the bundled model and falls back to the comb estimator if no model is installed.
    ``instrument`` turns on the comb's isotope-spacing charges and is handed to a network too,
    whose isotope-charge input channels (if it has them) need its resolving power.
    """
    from ..nn.infer import ChargePredictor, bundled_model_path

    if model_path in ("comb", "none", "classical"):
        return CombPredictor(z_max=z_max, instrument=instrument)
    if model_path is None and bundled_model_path() is None:
        return CombPredictor(z_max=z_max, instrument=instrument)
    return ChargePredictor(model_path, providers=providers, instrument=instrument)
