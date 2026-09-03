"""Quantify discovered components in every frame by NNLS against fixed templates.

Discovery (on the summed spectrum of a region) already determines each component's envelope
shape: which charges it carries and in what proportion. Across a chromatographic peak that
shape is essentially constant, so per-frame quantification only needs one scale factor per
component and adduct state. Collapsing the charge dimension into the template keeps the design
matrix small (a few dozen columns instead of thousands) and the per-frame solve fast, while
still reporting how each adduct's contribution changes across the peak.
"""

from __future__ import annotations

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState
from ..chem.instrument import InstrumentModel, estimate_noise_sigma
from ..core.model import Component, FrameResult, Spectrum
from ..deconv.nnls_solve import weighted_nnls
from ..deconv.templates import build_template
from ..nn.grid import LogMzGrid


def _keep_top(values: np.ndarray, fraction: float) -> np.ndarray:
    """Indices of the largest entries whose sum reaches ``fraction`` of the total."""
    if values.size == 0 or fraction >= 1.0:
        return np.arange(values.size)
    order = np.argsort(-values)
    csum = np.cumsum(values[order])
    n = int(np.searchsorted(csum, fraction * csum[-1]) + 1)
    keep = np.zeros(values.size, dtype=bool)
    keep[order[:n]] = True
    return np.nonzero(keep)[0]


class TemplateBank:
    """One column per (component, adduct state), with the charge distribution baked in."""

    def __init__(self, components: list[Component], grid: LogMzGrid, instrument: InstrumentModel,
                 library: AdductLibrary, compound_class: str = "peptide",
                 max_components: int = 20, min_adduct_fraction: float = 0.02,
                 keep_fraction: float = 0.995, max_rows: int = 20000):
        self.grid = grid
        self.instrument = instrument
        by_intensity = sorted(components, key=lambda c: -c.intensity)[:max_components]
        self.components = by_intensity
        states = {s.label or "base": s for s in library.states()}
        states.setdefault("base", AdductState())

        self.keys: list[tuple[int, str]] = []
        cols_bins: list[np.ndarray] = []
        cols_vals: list[np.ndarray] = []
        for ci, comp in enumerate(by_intensity):
            total_charge = sum(comp.charges.values()) or 1.0
            fractions = comp.adduct_fractions() or {"base": 1.0}
            for label, frac in fractions.items():
                if frac < min_adduct_fraction:
                    continue
                st = states.get(label)
                if st is None:
                    continue
                acc: dict[int, float] = {}
                for z, w in comp.charges.items():
                    t = build_template(comp.mass, int(z), grid, instrument,
                                       comp.compound_class or compound_class,
                                       adduct_mass=st.mass, adduct_label=st.label)
                    if t is None:
                        continue
                    wz = w / total_charge
                    for b, v in zip(t.bins, t.values):
                        acc[int(b)] = acc.get(int(b), 0.0) + v * wz
                if not acc:
                    continue
                bins = np.fromiter(acc.keys(), dtype=np.int64)
                vals = np.fromiter(acc.values(), dtype=np.float64)
                order = np.argsort(bins)
                bins, vals = bins[order], vals[order]
                s = vals.sum()
                if s <= 0:
                    continue
                vals = vals / s
                # Drop the faint tails: they add rows to every frame's solve without carrying
                # information, and a per-frame scale factor is set by the envelope's body.
                keep = _keep_top(vals, keep_fraction)
                cols_bins.append(bins[keep])
                cols_vals.append(vals[keep])
                self.keys.append((ci, label))

        if cols_bins:
            support = np.unique(np.concatenate(cols_bins))
            if support.size > max_rows:
                # a uniform thinning keeps every envelope represented and the scale unbiased
                support = support[np.linspace(0, support.size - 1, max_rows).astype(np.int64)]
            self.support = support
            self.A = np.zeros((self.support.size, len(cols_bins)))
            for j, (bins, vals) in enumerate(zip(cols_bins, cols_vals)):
                idx = np.searchsorted(self.support, bins)
                idx = np.clip(idx, 0, self.support.size - 1)
                hit = self.support[idx] == bins
                np.add.at(self.A[:, j], idx[hit], vals[hit])
        else:
            self.support = np.array([], dtype=np.int64)
            self.A = np.zeros((0, 0))

    @property
    def n_columns(self) -> int:
        return len(self.keys)

    def quantify(self, spectrum: Spectrum) -> FrameResult:
        if self.A.size == 0:
            return FrameResult(spectrum.rt, spectrum.polarity, [], 0.0, 0.0)
        if spectrum.is_profile is False:
            observed = self.grid.render_centroids(spectrum.mz, spectrum.intensity, self.instrument)
        else:
            observed = self.grid.resample_profile(spectrum.mz, spectrum.intensity)
        noise = estimate_noise_sigma(observed)
        b = observed[self.support]
        if b.max() <= 0:
            return FrameResult(spectrum.rt, spectrum.polarity, [], noise, 0.0)
        w = 1.0 / np.sqrt(np.clip(b, 0, None) + noise**2 + 1e-9)
        x = weighted_nnls(self.A, b, w)

        per_comp: dict[int, dict[str, float]] = {}
        for j, (ci, label) in enumerate(self.keys):
            if x[j] <= 0:
                continue
            per_comp.setdefault(ci, {})[label] = float(x[j])
        comps: list[Component] = []
        for ci, adducts in per_comp.items():
            base = self.components[ci]
            total = sum(adducts.values())
            if total <= 0:
                continue
            scale = total / max(sum(base.charges.values()), 1e-12)
            comps.append(Component(mass=base.mass, intensity=total, mass_type=base.mass_type,
                                   charges={z: v * scale for z, v in base.charges.items()},
                                   adducts=adducts, score=base.score,
                                   compound_class=base.compound_class, id=base.id,
                                   mass_spread_ppm=base.mass_spread_ppm))
        model = self.A @ x
        resid = float(np.clip(b - model, 0, None).sum() / (b.sum() + 1e-9))
        return FrameResult(spectrum.rt, spectrum.polarity, comps, noise, resid)
