"""Provenance: where each reported species was read from.

``process_run(..., provenance=True)`` (``lcmsdeconv process --provenance``) keeps, for every
species it reports, the chain of evidence the processing followed, and
:func:`write_provenance` writes it beside the other outputs:

- ``provenance.json`` (schema :data:`SCHEMA`, version :data:`SCHEMA_VERSION`): the run's regions
  (their scans, shared boundaries and the fitted columns that the steps after the fit dropped),
  and per species its regions, the pieces it was made of (each with the charges it was linked
  with) and how they were joined, the components of the regions' summed spectra its mass and
  amount come from (fit pass and id, mass, template shift, score, mass spread, and where the fit
  placed each part of a component it merged from several), each charge state's m/z window with
  the signal the region's scans measured in it, the fitted templates and what the fit left
  there, the adduct-form columns, the charge estimator's reading on the envelope's bins and the
  species' per-scan trace; and the SHA-256 of the ``results.json`` it was written with;
- ``evidence/species-<id>-<mass>.png``: the region-summed spectrum with the species' charge
  states shaded, its fitted templates and its trace over the scans, for the
  :data:`MAX_FIGURES` largest species (the title, which names the run and the species, also
  kept as text in the file, for the leak check of name aliases);
- ``provenance.html``: an index of the species linking their figures.

Everything is taken from the objects the processing makes anyway (a region's scans, summed
spectrum, grid view, the fit's residual and the estimator's output are kept only while the region
is processed, its grid view and fitted model until the species are final, and only with
provenance on); nothing the processing computes changes. docs/usage.md, "Provenance", describes
the schema field by field.

The species of the grid decoder are described by the components of a region's summed spectrum
(``evidence.kind == "grid_components"``). A decoder that reports whole regions instead (a
maximum-entropy engine, say) will describe its species with ``evidence.kind ==
"region_report"`` and fields of its own; a region names its engine in ``deconvolution.engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .chem.adducts import AdductState, carrier_mass
from .chem.classes import class_isotope_pattern
from .deconv.templates import build_template
from .features.quantify import _keep_top

#: what ``provenance.json`` says it is, and the version of its layout
SCHEMA = "lcmsdeconv.provenance"
SCHEMA_VERSION = 1

#: the most evidence figures one run gets: those of the largest species by base-form amount
MAX_FIGURES = 300

#: a charge state's m/z window spans the grid bins that hold this share of its fitted template's
#: area (the share :class:`~lcmsdeconv.features.quantify.TemplateBank` keeps of each of its
#: columns, each of which adds up every charge state of one form); the fit of the summed spectrum
#: reads each template's whole support (``support_mz``)
WINDOW_FRACTION = 0.995

#: the monoisotopic m/z is given where the class pattern's monoisotopic peak holds at least this
#: share of its tallest isotopologue (beyond about 15 kDa it is not seen)
MONO_MIN = 0.01

#: the 13C - 12C spacing (Da)
ISOTOPE_DA = 1.0033548

#: the charge states a figure shows close up, the strongest first
ZOOM_STATES = 4

#: at most about this many points per curve in a figure's whole-span panel (peaks kept: the
#: largest value of each run of grid bins)
SPAN_POINTS = 1600

#: the colours of a merged species' pieces in its figure's trace panel, each named in the
#: legend: every piece of a species of up to seven, the first six of one of more (the others
#: drawn grey under one entry, which the panel has room for)
PIECE_COLOURS = ("#7b3294", "#008837", "#a6611a", "#e66101", "#5e3c99", "#018571", "#d01c8b")

#: merge steps whose joined pieces are not counted in the species' amount
_NOT_COUNTED = ("set_aside", "other_polarity")

#: the merge steps, as provenance.json names them
MERGE_STEPS = {
    "adduct_fold": "an adduct form reported apart, folded into its parent (its trace counted "
                   "under the combined adduct label)",
    "adjacent_pieces": "the same mass linked as another species whose trace touches or overlaps "
                       "this one's as one elution band (a neighbouring region's piece of the "
                       "band, or a reading in the same region that linking kept apart, as when "
                       "its trace has a gap longer than linking.max_gap_frames scans), joined (a "
                       "scan both hold counted once, with the larger reading)",
    "boundary_piece": "a piece of its band that a region boundary cut off and its region read up "
                      "to a dalton off, joined as the same molecule by its isotopes",
    "set_aside": "the same molecule found later in the run (its tail, the column wash): named in "
                 "the flags, not counted",
    "other_polarity": "the reading of the other polarity of a polarity-switching run: named in "
                      "the flags, not counted",
}

#: the steps after the fit that take fitted columns away, as provenance.json names them
DROP_STEPS = {
    "multimer": "a multimer of a stronger component it co-elutes with (drop_multimers)",
    "harmonic_reading": "a stronger component's ions read at other charges "
                        "(drop_harmonic_readings)",
    "shared_evidence": "envelopes that stronger components' columns already explain "
                       "(drop_shared_evidence)",
    "coincident_adducts": "adduct columns that coincide with a stronger component's adduct "
                          "ladder, given to it (drop_coincident_adducts); the component is "
                          "dropped where no column is left",
}


def _num(x) -> float | None:
    """A finite float, or None (JSON has no NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _floats(a) -> list:
    return [_num(v) for v in np.asarray(a, dtype=float).ravel()]


def _sha256(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def _sorted(mz, intensity) -> tuple[np.ndarray, np.ndarray]:
    """A spectrum's points as float arrays, by m/z."""
    mz = np.asarray(mz, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if mz.size > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz, kind="stable")
        mz, y = mz[order], y[order]
    return mz, y


def _adduct(label) -> str:
    return "" if label in ("", "base", None) else str(label)


def describe_estimator(predictor) -> tuple[dict, str | None]:
    """What assigned the charges: ``({"kind": "comb", ...} or {"kind": "network", "file": ...,
    "sha256": ...}, the network's file name or None)``."""
    backend = getattr(predictor, "backend", None)
    if backend == "comb":
        return {"kind": "comb", "z_max": int(getattr(predictor, "z_max", 0)),
                "isotope_charges": getattr(predictor, "instrument", None) is not None}, None
    path = getattr(predictor, "model_path", None)
    if path is not None and backend in ("onnx", "torch"):
        path = Path(path)
        return {"kind": "network", "backend": backend, "file": path.name,
                "sha256": _sha256(path), "view": getattr(predictor, "view", None),
                "isotope_channels": bool(getattr(predictor, "iso_channels", False)),
                "z_max": int(getattr(predictor, "z_max", 0))}, path.name
    return {"kind": type(predictor).__name__}, None


def piece_charges(species) -> list[dict]:
    """A piece's charges as it was linked: each charge state's readings added up, as
    results.json gives a species' charges (a merged species' are its counted pieces' added
    up)."""
    return [{"z": int(z), "value": float(v)} for z, v in sorted(species.charges.items())]


def _mass_key(mass: float) -> int:
    """A template mass as :func:`~lcmsdeconv.deconv.templates.build_template` keys it (0.1 mDa):
    two masses with one key have one template."""
    return int(round(float(mass) * 1e4))


def placements(c, columns: dict, parts: dict | None) -> dict | None:
    """Where the fit placed the templates of ``columns`` (keys of ``c.columns``) of component
    ``c``, where that is not one template position: ``{key: [(template mass, the adduct label
    of the part's template, fitted)]}``. A component the region's fit merged from several (of
    almost one mass, or an adduct form into its parent: refine's ``_merge_same_mass`` and
    ``_merge_adduct_components``) holds each part's templates where the part's own fit placed
    and subtracted them (``parts``: ``deconvolve_spectrum``'s ``keep["parts"]``). None where
    every one of them sits at the component's own template position, mass x exp(template_shift),
    under its own label: a component fitted as one."""
    entry = (parts or {}).get(id(c))
    if entry is None:
        return None
    own = _mass_key(c.mass * float(np.exp(c.template_shift)))
    out: dict = {}
    moved = False
    for tm, cols in entry[1]:
        for _z, lab, key, v in cols:
            if key not in columns or not v > 0:
                continue
            out.setdefault(key, []).append((float(tm), _adduct(lab), float(v)))
            moved = moved or _mass_key(tm) != own or _adduct(lab) != _adduct(key[1])
    return out if moved else None


def fitted_parts(c, columns: dict, parts: dict | None) -> list[dict] | None:
    """``fitted_parts`` of a component's record: for a component the region's fit merged from
    several (:func:`placements`), each part's template position and the columns of ``columns``
    it was fitted with (``adduct``: the component's column it is counted in;
    ``template_adduct``: the adduct state its own template carries); None for one fitted as
    one, its templates at its own template position."""
    entry = (parts or {}).get(id(c))
    if entry is None:
        return None
    out = []
    for tm, cols in entry[1]:
        rows = sorted((int(z), _adduct(key[1]), _adduct(lab), float(v))
                      for z, lab, key, v in cols if key in columns and v > 0)
        if rows:
            out.append({"template_mass": float(tm),
                        "columns": [{"z": z, "adduct": a, "template_adduct": t, "fitted": v}
                                    for z, a, t, v in rows]})
    if len(out) < 2 and placements(c, columns, parts) is None:
        return None
    return out


def dropped_records(dropped, parts: dict | None = None, pass_of: dict | None = None
                    ) -> list[dict]:
    """``regions[].deconvolution.dropped``: the fitted columns the steps after the fit took away
    (``deconvolve_spectrum``'s ``keep["dropped"]``), each with its step, the component (with
    the fit pass that found it, ``pass_of``: ``{id(component): pass}``, and where its templates
    sat, ``parts``: :func:`fitted_parts`) and the stronger components they were attributed
    to."""
    pass_of = pass_of or {}
    out = []
    for d in dropped or ():
        c = d["component"]
        cols = sorted(((int(z), _adduct(lab), float(v)) for (z, lab), v in d["columns"].items()
                       if v > 0), key=lambda x: (x[0], x[1]))
        out.append({
            "step": d["step"], "id": int(c.id), "pass": pass_of.get(id(c)),
            "mass": float(c.mass), "template_mass": float(c.mass * np.exp(c.template_shift)),
            "component_dropped": bool(d["whole"]),
            "attributed_to": [{"id": int(k.id), "pass": pass_of.get(id(k)),
                               "mass": float(k.mass)} for k in d["attributed_to"]],
            "columns": [{"z": z, "adduct": lab, "fitted": v} for z, lab, v in cols],
            "fitted_parts": fitted_parts(c, d["columns"], parts),
        })
    return out


# ------------------------------------------------------------------------------ one region
@dataclass
class _Column:
    z: int
    label: str  # "" for the base form
    coef: float
    template: Any  # lcmsdeconv.deconv.templates.Template, or a _Placed


@dataclass
class _Placed:
    """The templates the fit placed for one column of a component it merged from parts, added
    up: ``values * coef`` is every part's template times what the part's fit gave it."""

    bins: np.ndarray
    values: np.ndarray


class _Region:
    """What a region's species are read from, while the region is processed: its scans and
    their summed spectrum, its grid view, what the fit left of it, the model of every component
    it reports and of the columns dropped after the fit (each template where the fit placed and
    subtracted it), and the charge estimator's output of each pass. :meth:`release` keeps what
    its species' figures draw (the grid view, the models and where the templates sit) until the
    species are final."""

    def __init__(self, grid, keep: dict, summed, fr, instrument, snr: float):
        self.grid = grid
        self.instrument = instrument
        self.observed = np.asarray(keep["observed"], dtype=float)
        self.noise = float(keep.get("noise_sigma", fr.noise_sigma))
        self.thr = snr * self.noise  # the decoder's vote threshold
        self.carrier = carrier_mass(grid.polarity)
        self.raw_mz, self.raw_int = _sorted(summed.mz, summed.intensity)
        # each scan's own points, all scans together by m/z with the scan each is from: what the
        # instrument measured (the summed spectrum interpolates profile scans on axes of their
        # own onto the union of their points)
        frames = (getattr(summed, "meta", None) or {}).get("frames") or ()
        self.points = None
        self.n_scans = len(frames)
        if frames:
            mz = np.concatenate([np.asarray(f.mz, dtype=float) for f in frames])
            order = np.argsort(mz, kind="stable")
            self.points = (mz[order],
                           np.concatenate([np.asarray(f.intensity, dtype=float)
                                           for f in frames])[order],
                           np.repeat(np.arange(len(frames), dtype=np.int32),
                                     [len(f.mz) for f in frames])[order])
        residual = keep.get("residual")
        self.residual = None if residual is None else np.asarray(residual, dtype=float)
        self.dropped = list(keep.get("dropped", ()))
        self.parts = keep.get("parts") or {}
        self.cols: dict[int, list[_Column]] = {}
        self.owns: dict[int, tuple[int, np.ndarray]] = {}
        self.model = np.zeros(grid.size)
        for c in fr.components:
            for col in self.columns(c):
                self.model[col.template.bins] += col.template.values * col.coef
        self.dropped_model = None
        if any(v > 0 for d in self.dropped for v in d["columns"].values()):
            self.dropped_model = np.zeros(grid.size)
            for d in self.dropped:
                for col in self._columns(d["component"], d["columns"]):
                    self.dropped_model[col.template.bins] += col.template.values * col.coef
        self.predictions = {it: (pred, sig) for it, pred, sig in keep.get("predictions", ())}
        self.pass_of = {id(c): i for i, new in enumerate(keep.get("passes", ())) for c in new}
        self.evidence: dict[int, dict] = {}

    def release(self) -> None:
        """Let go of what only the region's own records read (its scans, summed spectrum, the
        fit's residual, the estimator's output, the templates); keep the grid view and the
        models, single precision, and where the fit placed the templates, for the figures."""
        self.observed = self.observed.astype(np.float32)
        self.model = self.model.astype(np.float32)
        if self.dropped_model is not None:
            self.dropped_model = self.dropped_model.astype(np.float32)
        self.raw_mz = self.raw_int = self.points = self.residual = None
        self.predictions, self.pass_of, self.evidence = {}, {}, {}
        self.dropped = []
        self.forget()

    def forget(self) -> None:
        """Let go of the templates built so far."""
        self.cols, self.owns = {}, {}

    # ----------------------------------------------------------------- templates
    def template(self, c, z: int, label: str = ""):
        st = AdductState.from_label(label)
        tmass = c.mass * float(np.exp(c.template_shift))
        return build_template(tmass, int(z), self.grid, self.instrument, c.compound_class,
                              adduct_mass=st.mass, adduct_label=st.label)

    def _columns(self, c, columns: dict) -> list[_Column]:
        placed = placements(c, columns, self.parts)
        out = []
        for key, v in columns.items():
            if not v > 0:
                continue
            z, label = int(key[0]), _adduct(key[1])
            if placed is None or key not in placed:
                t = self.template(c, z, label)
            else:
                t = self._placed(c, z, placed[key], float(v))
            if t is not None:
                out.append(_Column(z, label, float(v), t))
        return out

    def _placed(self, c, z: int, parts: list, coef: float) -> _Placed | None:
        """The templates of one column of a component merged from parts, each at its part's
        template mass with its part's adduct state, added up and scaled to ``coef``."""
        found = []
        for tm, label, v in parts:
            st = AdductState.from_label(label)
            t = build_template(tm, int(z), self.grid, self.instrument, c.compound_class,
                               adduct_mass=st.mass, adduct_label=st.label)
            if t is not None:
                found.append((t, v))
        if not found or not coef > 0:
            return None
        lo = min(int(t.bins.min()) for t, _v in found)
        y = np.zeros(max(int(t.bins.max()) for t, _v in found) - lo + 1)
        for t, v in found:
            y[t.bins - lo] += t.values * v
        nz = np.flatnonzero(y > 0)
        return _Placed(nz + lo, y[nz] / coef)

    def columns(self, c) -> list[_Column]:
        """The component's fitted (charge, adduct-state) columns with their templates, placed as
        the fit placed them (the class's envelope at mass * exp(template_shift); for a component
        the fit merged from parts, each part's at the part's own position, :func:`placements`)."""
        out = self.cols.get(id(c))
        if out is None:
            out = self.cols[id(c)] = self._columns(c, c.columns)
        return out

    def own(self, c) -> tuple[int, np.ndarray]:
        """The component's fitted templates added up: (first bin, values from that bin on)."""
        got = self.owns.get(id(c))
        if got is None:
            cols = self.columns(c)
            if not cols:
                got = (0, np.zeros(0))
            else:
                lo = min(int(col.template.bins.min()) for col in cols)
                y = np.zeros(max(int(col.template.bins.max()) for col in cols) - lo + 1)
                for col in cols:
                    y[col.template.bins - lo] += col.template.values * col.coef
                got = (lo, y)
            self.owns[id(c)] = got
        return got

    def own_in(self, comps, b0: int, b1: int) -> np.ndarray:
        """The fitted templates of ``comps`` on bins b0..b1."""
        out = np.zeros(b1 - b0 + 1)
        for c in comps:
            lo, y = self.own(c)
            a, b = max(b0, lo), min(b1, lo + y.size - 1)
            if b >= a:
                out[a - b0:b - b0 + 1] += y[a - lo:b - lo + 1]
        return out

    def window(self, t) -> tuple[int, int]:
        """The grid bins holding WINDOW_FRACTION of the template's area (first, last)."""
        kept = t.bins[_keep_top(t.values, WINDOW_FRACTION)]
        return int(kept.min()), int(kept.max())

    def span(self, b0: int, b1: int) -> list[float]:
        """The m/z span of grid bins b0..b1, their outer edges."""
        return [float(self.grid.bin_to_mz(b0 - 0.5)), float(self.grid.bin_to_mz(b1 + 0.5))]

    # ----------------------------------------------------------------- measurements
    def raw_signal(self, lo: float, hi: float) -> dict:
        """What the region's scans measured within [lo, hi] m/z: their own points there, all
        scans together (number, summed intensity, intensity-weighted mean m/z, scans with
        signal), and the tallest point of their sum, the region-summed spectrum."""
        i0 = int(np.searchsorted(self.raw_mz, lo, side="left"))
        i1 = int(np.searchsorted(self.raw_mz, hi, side="right"))
        peak = float(self.raw_int[i0:i1].max()) if i1 > i0 else 0.0
        mz, y, scan = self.points if self.points is not None else (self.raw_mz, self.raw_int,
                                                                   None)
        a = int(np.searchsorted(mz, lo, side="left"))
        b = int(np.searchsorted(mz, hi, side="right"))
        mz, y = mz[a:b], y[a:b]
        scans = None
        if scan is None:
            total = float(y.sum())
        else:  # each scan's points in the window added up, then the scans'
            per = np.bincount(scan[a:b], weights=y, minlength=self.n_scans)
            total, scans = float(per.sum()), int(np.count_nonzero(per > 0))
        moment = float(np.dot(y, mz))
        return {"points": b - a, "sum": total, "max": peak, "scans": scans,
                "centroid_mz": moment / total if total > 0 else None}

    def grid_signal(self, b0: int, b1: int, c) -> dict:
        """On the grid bins b0..b1: the grid view, the fitted templates of every reported
        component and of this one, the signal they leave unexplained and the amount by which they
        exceed it, what the fit itself left (clipped at zero after each component) and the
        templates of the columns dropped after the fit."""
        sl = slice(b0, b1 + 1)
        obs, model = self.observed[sl], self.model[sl]
        diff = obs - model
        return {"bins": [int(b0), int(b1)], "observed": float(obs.sum()),
                "model": float(model.sum()),
                "component_model": float(self.own_in([c], b0, b1).sum()),
                "unexplained": float(np.clip(diff, 0.0, None).sum()),
                "overshoot": float(np.clip(-diff, 0.0, None).sum()),
                "fit_residual": (None if self.residual is None
                                 else float(self.residual[sl].sum())),
                "dropped_model": (0.0 if self.dropped_model is None
                                  else float(self.dropped_model[sl].sum()))}

    def charge_map(self, c, z: int, b0: int, b1: int) -> dict | None:
        """The estimator's reading of charge ``z`` on bins b0..b1, from the pass that found the
        component: over the bins above the vote threshold, weighted by the signal it read, the
        probability of z (its top-two classes only: 0 where z is neither), the share of signal on
        bins whose first choice is z, and the strongest other charge of the top two."""
        found = self.pass_of.get(id(c))
        if found not in self.predictions:
            return None
        pred, signal = self.predictions[found]
        sl = slice(b0, b1 + 1)
        w = np.asarray(signal[sl], dtype=float)
        on = w > self.thr
        if not on.any():
            return {"pass": found, "bins": 0, "p_assigned": None, "top1_share": None,
                    "alternative": None}
        w = w[on]
        z1, p1 = np.asarray(pred.top1_z[sl])[on], np.asarray(pred.top1_p[sl], dtype=float)[on]
        z2, p2 = np.asarray(pred.top2_z[sl])[on], np.asarray(pred.top2_p[sl], dtype=float)[on]
        total = float(w.sum())
        pz = np.where(z1 == z, p1, np.where(z2 == z, p2, 0.0))
        other: dict[int, float] = {}
        for zz, pp in ((z1, p1), (z2, p2)):
            for v in np.unique(zz):
                if int(v) != z and int(v) > 0:
                    sel = zz == v
                    other[int(v)] = other.get(int(v), 0.0) + float((w[sel] * pp[sel]).sum())
        alt = max(other, key=other.get) if other else None
        return {"pass": found, "bins": int(on.sum()),
                "p_assigned": float((w * pz).sum() / total),
                "top1_share": float(w[z1 == z].sum() / total),
                "alternative": None if alt is None else {"z": alt, "p": other[alt] / total}}

    # ----------------------------------------------------------------- one component
    def component(self, c) -> dict:
        """Everything provenance.json says of one component of the region's summed spectrum."""
        if id(c) in self.evidence:
            return self.evidence[id(c)]
        cols = self.columns(c)
        pat = class_isotope_pattern(c.mass, c.compound_class)
        ab = pat.abundances / max(float(pat.abundances.max()), 1e-300)
        mono_offset = pat.average_mass - pat.mono_mass if ab[0] >= MONO_MIN else None
        charge_total = sum(v for v in c.charges.values() if v > 0)
        states, adducts = [], []
        for z in sorted(int(k) for k in c.charges):
            base = next((col for col in cols if col.z == z and not col.label), None)
            t = base.template if base is not None else self.template(c, z)
            mz = (c.mass + z * self.carrier) / z
            fwhm = float(self.instrument.fwhm(mz)) if self.instrument is not None else None
            rec = {"z": z, "mz_centroid": mz,
                   "mz_mono": (None if mono_offset is None
                               else (c.mass - mono_offset + z * self.carrier) / z),
                   "isotopes_resolved": None if fwhm is None else bool(ISOTOPE_DA / z >= fwhm),
                   "fitted": base.coef if base is not None else 0.0,
                   "fitted_all_forms": float(c.charges.get(z, 0.0)),
                   "share": (float(c.charges.get(z, 0.0)) / charge_total
                             if charge_total > 0 else None)}
            rec.update(self._window_record(c, t, z, 0.0))
            states.append(rec)
        for col in cols:
            if not col.label:
                continue
            st = AdductState.from_label(col.label)
            rec = {"z": col.z, "adduct": col.label, "fitted": col.coef,
                   "mz_centroid": (c.mass + st.avg_mass + col.z * self.carrier) / col.z}
            rec.update(self._window_record(c, col.template, col.z, st.avg_mass,
                                           charge_map=False))
            adducts.append(rec)
        adducts.sort(key=lambda r: (r["z"], r["adduct"]))
        out = {"id": int(c.id), "mass": float(c.mass), "mass_type": c.mass_type,
               "template_shift": float(c.template_shift),
               "template_mass": float(c.mass * np.exp(c.template_shift)),
               "comb_placed": bool(getattr(c, "comb_placed", False)),
               "score": float(c.score), "mass_spread_ppm": float(c.mass_spread_ppm),
               "compound_class": c.compound_class, "intensity": float(c.intensity),
               "pass": self.pass_of.get(id(c)), "flags": list(c.flags),
               "fitted_parts": fitted_parts(c, c.columns, self.parts),
               "charge_states": states, "adduct_columns": adducts}
        self.evidence[id(c)] = out
        return out

    def _window_record(self, c, t, z: int, shift: float, charge_map: bool = True) -> dict:
        """The m/z window of template ``t`` and what lies in it."""
        if t is None:  # beyond the grid: nothing was fitted there
            return {"window_mz": None, "support_mz": None, "observed": None,
                    "centroid_mass": None, "grid": None,
                    **({"charge_map": None} if charge_map else {})}
        b0, b1 = self.window(t)
        lo, hi = self.span(b0, b1)
        raw = self.raw_signal(lo, hi)
        centre = raw["centroid_mz"]
        out = {"window_mz": [lo, hi],
               "support_mz": self.span(int(t.bins.min()), int(t.bins.max())),
               "observed": raw,
               "centroid_mass": (None if centre is None
                                 else z * (centre - self.carrier) - shift),
               "grid": self.grid_signal(b0, b1, c)}
        if charge_map:
            out["charge_map"] = self.charge_map(c, z, b0, b1)
        return out

    # ----------------------------------------------------------------- figure data
    def figure(self, comps: list, evidence: list[dict], joined=(), joined_evidence=()
               ) -> dict | None:
        """What an evidence figure draws of a species read from ``comps`` here, with
        ``joined``, the components of its other counted pieces of this region (adduct forms
        folded into it, its pieces joined): the grid view, the templates of each and the other
        components' over the span of their charge states (reduced to about SPAN_POINTS points,
        each the largest of its run of bins), the windows of their charge states, and its
        strongest charge states close up."""
        spans = [self.own(c) for c in (*comps, *joined)]
        spans = [(lo, y) for lo, y in spans if y.size]
        if not spans:
            return None
        b_lo = min(lo for lo, _y in spans)
        b_hi = max(lo + y.size - 1 for lo, y in spans)
        pad = max(20, (b_hi - b_lo) // 25)
        b_lo, b_hi = max(0, b_lo - pad), min(self.grid.size - 1, b_hi + pad)
        own = self.own_in(comps, b_lo, b_hi)
        forms = self.own_in(joined, b_lo, b_hi)
        obs = np.asarray(self.observed[b_lo:b_hi + 1], dtype=float)
        others = np.clip(np.asarray(self.model[b_lo:b_hi + 1], dtype=float) - own - forms, 0.0,
                         None)
        bins = np.arange(b_lo, b_hi + 1, dtype=float)
        k = int(np.ceil(bins.size / SPAN_POINTS))
        if k > 1:
            n = int(np.ceil(bins.size / k)) * k

            def pooled(a, how):
                a = np.pad(a, (0, n - a.size), mode="edge").reshape(-1, k)
                return how(a, axis=1)

            bins = pooled(bins, np.mean)
            obs, own, forms, others = (pooled(a, np.max) for a in (obs, own, forms, others))
        windows, strongest = [], []
        for kind, evs in (("own", evidence), ("joined", joined_evidence)):
            for ev in evs:
                for st in ev["charge_states"]:
                    if st["window_mz"] is None:
                        continue
                    windows.append((st["z"], *st["window_mz"], kind))
                    if kind == "own":
                        strongest.append((st["fitted_all_forms"], st["z"], st))
        zooms = []
        for _v, z, st in sorted(strongest, key=lambda s: -s[0])[:ZOOM_STATES]:
            g = st["grid"]
            w0, w1 = g["bins"]
            p = max(4, (w1 - w0) // 2)
            z0, z1 = max(0, w0 - p), min(self.grid.size - 1, w1 + p)
            zb = np.arange(z0, z1 + 1)

            def share(v, total=g["observed"]):
                return None if v is None or not total > 0 else float(v) / total

            dropped = (None if self.dropped_model is None
                       else np.asarray(self.dropped_model[z0:z1 + 1], dtype=float))
            zooms.append({"z": z, "window_mz": st["window_mz"], "fitted": st["fitted_all_forms"],
                          "fit_residual": share(g["fit_residual"]),
                          "unexplained": share(g["unexplained"]),
                          "overshoot": share(g["overshoot"]),
                          "dropped": share(g["dropped_model"]),
                          "mz": self.grid.bin_to_mz(zb),
                          "observed": np.asarray(self.observed[z0:z1 + 1], dtype=float),
                          "own": self.own_in(comps, z0, z1),
                          "joined": self.own_in(joined, z0, z1) if joined else None,
                          "model": np.asarray(self.model[z0:z1 + 1], dtype=float),
                          "dropped_model": (dropped if dropped is not None and dropped.any()
                                            else None)})
        return {"mz": self.grid.bin_to_mz(bins), "observed": np.array(obs, dtype=np.float32),
                "own": np.array(own, dtype=np.float32),
                "joined": np.array(forms, dtype=np.float32) if joined else None,
                "others": np.array(others, dtype=np.float32),
                "windows": sorted(windows), "zooms": zooms}


def _component_name(record: dict) -> str:
    """A component as the texts name it: its id and the fit pass that found it (a later pass
    numbers its components on from the number the earlier passes kept, so an id can recur in a
    region; the pass and the id name one)."""
    found = record.get("pass")
    return f"{record['id']}" + ("" if found is None else f" (fit pass {found})")


# ------------------------------------------------------------------------------ the trail
@dataclass
class _Trail:
    """A species' pieces as they were linked in their regions, and how they were joined."""

    species: Any
    pieces: list[dict]
    comps: list  # each piece's components (the region's Component objects)
    traces: list  # each piece's (time, intensity) as linked
    merges: list[dict] = field(default_factory=list)
    primary: int = 0  # the piece whose mass the species carries


class Recorder:
    """Collects the provenance of :func:`lcmsdeconv.process.process_run` as it runs (only with
    ``provenance=True``): :meth:`region` after each region is deconvolved and linked,
    :meth:`finish` with the merge steps' log once the species are final."""

    def __init__(self, view, predictor, instrument, params, aliases=None, per_scan: bool = True):
        t0 = time.perf_counter()
        self.view = view
        #: whether species are read scan by scan (deconvolution.quantify_frames), or once per
        #: region from its summed spectrum, a reading no single scan holds
        self.per_scan = per_scan
        self.instrument = instrument
        self.snr = float(params.snr)
        self.aliases = aliases
        self.estimator, self.model_file = describe_estimator(predictor)
        self.regions: dict[tuple[int, int], dict] = {}
        self.trails: dict[int, _Trail] = {}
        #: each region's grid view and fitted models, for its species' figures (until finish)
        self.contexts: dict[tuple[int, int], _Region] = {}
        self._scans: dict[int, dict[float, str]] = {}
        self.seconds = time.perf_counter() - t0

    def region(self, region, shared, summed, fr, keep, species=(), members=()) -> None:
        """Record one region: its deconvolution, and each species linked in it (its piece)."""
        t0 = time.perf_counter()
        key = (int(region.polarity), int(region.id))
        keep = keep or {}
        grid = keep.get("grid") or fr.meta.get("grid")
        passes = keep.get("passes", [])
        pass_of = {id(c): i for i, new in enumerate(passes) for c in new}
        self.regions[key] = {
            "shared": [bool(shared[0]), bool(shared[1])],
            "deconvolution": {
                "engine": "grid",
                "components": len(fr.components),
                "passes": sum(1 for p in passes if p),
                "compound_class": fr.components[0].compound_class if fr.components else None,
                "adducts": list(fr.meta.get("adducts", ())),
                "noise_sigma": _num(fr.noise_sigma),
                "residual_fraction": _num(fr.residual_fraction),
                "dropped": dropped_records(keep.get("dropped"), keep.get("parts"), pass_of),
                "grid": None if grid is None else {
                    "mz_min": float(grid.mz_min), "mz_max": float(grid.mz_max),
                    "step": float(grid.step), "bins": int(grid.size)},
            },
        }
        if species and "observed" in keep and grid is not None:
            ctx = _Region(grid, keep, summed, fr, self.instrument, self.snr)
            by_part = {(c.id, c.mass): c for c in fr.components}
            parts = {id(s): p for s, p in members or ()}
            for s in species:
                comps = [by_part[p] for p in parts.get(id(s), ()) if p in by_part]
                evidence = [ctx.component(c) for c in comps]
                t = np.asarray(s.time, dtype=float)
                piece = {
                    "region": {"id": key[1], "polarity": key[0]},
                    "species_id": int(s.id), "mass": float(s.mass), "rt_apex": _num(s.rt_apex),
                    "rt_range": [_num(t.min()), _num(t.max())] if t.size else None,
                    "scans": int(t.size),
                    "amount": {"base": float(s.base_intensity),
                               "all_forms": float(s.total_intensity)},
                    "charges": piece_charges(s),
                    "counted": True, "joined_by": None,
                    "evidence": {
                        "kind": "grid_components",
                        "mass_rule": ("component" if len(comps) == 1
                                      else "intensity-weighted mean of the components"),
                        "components": evidence},
                }
                self.trails[id(s)] = _Trail(s, [piece], [comps],
                                            [(t.copy(), np.asarray(s.intensity, dtype=float).copy())])
            ctx.release()
            self.contexts[key] = ctx
        self.seconds += time.perf_counter() - t0

    def _merge(self, log) -> None:
        """Follow the merge steps' log (``(step, kept, joined, detail)``, in the order taken):
        the joined species' pieces become the kept one's."""
        for seq, (step, kept, joined, detail) in enumerate(log):
            tk, tj = self.trails.get(id(kept)), self.trails.pop(id(joined), None)
            if tk is None or tj is None:
                continue
            off = len(tk.pieces)
            counted = step not in _NOT_COUNTED
            for p in tj.pieces:
                if p["joined_by"] is None:
                    p["joined_by"] = step
                if not counted:
                    p["counted"] = False
            for m in tj.merges:
                tk.merges.append(m | {"kept_piece": m["kept_piece"] + off,
                                      "joined_pieces": [i + off for i in m["joined_pieces"]]})
            # the kept species' own piece is the first of its trail, wherever its mass is from
            tk.merges.append({
                "seq": seq, "step": step, "kept_piece": 0,
                "joined_pieces": list(range(off, off + len(tj.pieces))),
                "joined": {"species_id": int(joined.id), "mass": float(joined.mass),
                           "rt_apex": _num(joined.rt_apex),
                           "amount": {"base": float(joined.base_intensity),
                                      "all_forms": float(joined.total_intensity)}},
                "counted": counted, **detail})
            if detail.get("mass_from_piece"):
                tk.primary = off + tj.primary
            tk.pieces += tj.pieces
            tk.comps += tj.comps
            tk.traces += tj.traces

    def _scan_ids(self, polarity: int) -> dict[float, str]:
        """The ID of the scan at each retention time of the analysed scans of ``polarity``."""
        if polarity not in self._scans:
            ids: dict[float, str] = {}
            for f in self.view.frames(polarity):
                ids.setdefault(float(f.rt), f.scan_id)
            self._scans[polarity] = ids
        return self._scans[polarity]

    def finish(self, result, merges=None) -> Provenance:
        """The provenance of ``result``'s species, their merges followed."""
        t0 = time.perf_counter()
        self._merge(merges or ())
        records, figures = [], []
        for s in result.species:
            tr = self.trails.get(id(s))
            records.append(self._species(s, tr))
            figures.append(self._figure(s, tr))
        self.contexts.clear()
        self.seconds += time.perf_counter() - t0
        prov = Provenance(self.estimator, self.regions, records, figures,
                          {"collect_s": self.seconds}, self.model_file)
        if self.aliases:
            prov.alias_texts(self.aliases)
        return prov

    def _species(self, s, tr: _Trail | None) -> dict:
        t = np.asarray(s.time, dtype=float)
        y = np.asarray(s.intensity, dtype=float)
        nz = np.flatnonzero(y > 0)
        tot = sum(v for v in s.charges.values() if v > 0)
        ids = self._scan_ids(s.polarity)
        pieces = tr.pieces if tr is not None else []
        regions: dict[tuple[int, int], dict] = {}
        for i, p in enumerate(pieces):
            r = regions.setdefault((p["region"]["polarity"], p["region"]["id"]),
                                   {**p["region"], "pieces": [], "counted": False})
            r["pieces"].append(i)
            r["counted"] = r["counted"] or bool(p["counted"])
        primary = tr.primary if tr is not None else None
        merges = sorted(tr.merges, key=lambda m: m["seq"]) if tr is not None else []
        if primary is None or not pieces:
            how = None
        else:
            p = pieces[primary]
            comps = p["evidence"]["components"]
            how = (f"the mass of component {_component_name(comps[0])} of region "
                   f"{p['region']['id']}'s summed spectrum" if len(comps) == 1 else
                   "the intensity-weighted mean, over region "
                   f"{p['region']['id']}'s scans, of the masses of components "
                   + ", ".join(_component_name(c) for c in comps))
            if p["joined_by"] == "boundary_piece":
                how += ("; taken from a boundary piece, whose mass lies nearer the one the "
                        "isotopes put the molecule at")
        return {
            "id": int(s.id), "name": s.name, "mass": float(s.mass), "mass_type": s.mass_type,
            "polarity": int(s.polarity), "compound_class": s.compound_class,
            "rt_apex": _num(s.rt_apex),
            "rt_range": [_num(t.min()), _num(t.max())] if t.size else None,
            "rt_signal": [_num(t[nz[0]]), _num(t[nz[-1]])] if nz.size else None,
            "amount": {"base": float(s.base_intensity), "all_forms": float(s.total_intensity),
                       "max": float(s.max_intensity)},
            "adduct_fractions": {k: float(v) for k, v in s.adduct_fractions().items()},
            "charges": [{"z": int(z), "share": float(v) / tot, "value": float(v)}
                        for z, v in sorted(s.charges.items()) if v > 0 and tot > 0],
            "adduct_form_of": s.adduct_form_of, "adduct_form": s.adduct_form,
            "flags": list(s.flags), "annotations": list(s.annotations),
            "regions": list(regions.values()),
            "mass_from": {"piece": primary, "how": how},
            "pieces": pieces,
            "merges": [{k: v for k, v in m.items() if k != "seq"} for m in merges],
            "eic": {"per_scan": self.per_scan,
                    "scan_ids": [ids.get(float(v)) if self.per_scan else None for v in t],
                    "rt": _floats(t), "intensity": _floats(y),
                    "forms": {k: _floats(v) for k, v in s.adduct_intensity.items()}},
            "figure": None,
        }

    def _figure(self, s, tr: _Trail | None) -> dict | None:
        """What the species' figure draws: the summed spectrum of the region its mass is from,
        with the components of its piece there and those of its other counted pieces of that
        region, and its trace with each piece's."""
        if tr is None or not tr.pieces:
            return None
        p = tr.pieces[tr.primary]
        key = (p["region"]["polarity"], p["region"]["id"])
        ctx, comps = self.contexts.get(key), tr.comps[tr.primary]
        if ctx is None or not comps:
            return None
        seen = {id(c) for c in comps}
        joined, joined_evidence, joined_pieces = [], [], []
        for i, q in enumerate(tr.pieces):
            if (i == tr.primary or not q["counted"]
                    or (q["region"]["polarity"], q["region"]["id"]) != key):
                continue
            for c, ev in zip(tr.comps[i] or (), q["evidence"]["components"]):
                if id(c) not in seen:
                    seen.add(id(c))
                    joined.append(c)
                    joined_evidence.append(ev)
                    if i not in joined_pieces:
                        joined_pieces.append(i)
        spectrum = ctx.figure(comps, p["evidence"]["components"], joined, joined_evidence)
        ctx.forget()
        if spectrum is None:
            return None
        base = s.adduct_intensity.get("base")
        merged = sum(len(ev["fitted_parts"]) for ev in p["evidence"]["components"]
                     if ev.get("fitted_parts"))
        return {"spectrum": spectrum, "region": p["region"], "joined_pieces": joined_pieces,
                "fitted_parts": merged,
                "time": np.asarray(s.time, dtype=float).copy(),
                "intensity": np.asarray(s.intensity, dtype=float).copy(),
                "base": None if base is None else np.asarray(base, dtype=float).copy(),
                "pieces": [(i, q["region"], q["species_id"], q["mass"], q["rt_apex"],
                            q["counted"], *tr.traces[i]) for i, q in enumerate(tr.pieces)]}


# ------------------------------------------------------------------------------ the result
@dataclass
class Provenance:
    """Where each species of a processed run was read from (``ProcessResult.provenance``).

    ``species`` holds one JSON-ready record per species, in the order of ``result.species``
    (largest base-form amount first), ``figures`` what each one's evidence figure draws;
    :meth:`to_dict` puts them together with the run's regions as ``provenance.json`` holds them.
    """

    estimator: dict
    regions: dict
    species: list[dict]
    figures: list
    timings: dict
    model_file: str | None = None

    def alias_texts(self, aliases) -> None:
        """Show NEW for OLD in the texts taken from outside the run and the method, which come
        aliased with them: the charge model's file name (as given on the command line or in the
        method). Applied to the name as read, so applying it again changes nothing."""
        if aliases and self.model_file is not None:
            self.estimator["file"] = aliases.apply(self.model_file)

    def to_dict(self, result, figures: dict[int, str] | None = None,
                max_figures: int = MAX_FIGURES, results_sha256: str | None = None) -> dict:
        """provenance.json: the run, its regions and its species (``figures``: the figure file
        of each species drawn, by its position in ``species``; ``results_sha256``: that of the
        results.json written with it)."""
        from . import __version__

        figures = figures or {}
        head: dict[str, Any] = {"schema": SCHEMA, "version": SCHEMA_VERSION,
                                "lcmsdeconv": __version__, "run": result.run_name,
                                "method": result.method.name}
        if getattr(result, "alias_pairs", 0):
            head["alias_pairs"] = result.alias_pairs
        head["results"] = {"file": "results.json", "sha256": results_sha256}
        by_region: dict[tuple[int, int], list[int]] = {}
        for rec in self.species:
            for p in rec["pieces"]:
                key = (p["region"]["polarity"], p["region"]["id"])
                if p["counted"] and rec["id"] not in by_region.setdefault(key, []):
                    by_region[key].append(rec["id"])
        regions, outline = [], {}
        for r in result.regions:
            key = (int(r.polarity), int(r.id))
            extra = self.regions.get(key, {})
            shared = extra.get("shared", [False, False])
            # what a species' record says of each of its regions (the scan IDs: its trace's)
            outline[key] = {"id": int(r.id), "polarity": int(r.polarity), "start": float(r.start),
                            "end": float(r.end), "scans": len(r.frames),
                            "first_scan": r.frames[0].scan_id if r.frames else None,
                            "last_scan": r.frames[-1].scan_id if r.frames else None,
                            "shared_boundaries": {"start": shared[0], "end": shared[1]}}
            regions.append({
                "id": int(r.id), "polarity": int(r.polarity), "start": float(r.start),
                "end": float(r.end),
                "scans": {"n": len(r.frames), "ids": [f.scan_id for f in r.frames],
                          "rt": ([float(r.frames[0].rt), float(r.frames[-1].rt)]
                                 if r.frames else None)},
                "shared_boundaries": {"start": shared[0], "end": shared[1]},
                "species": by_region.get(key, []),
                "deconvolution": extra.get("deconvolution"),
            })
        species = [rec | {"figure": figures.get(i),
                          "regions": [outline.get((q["polarity"], q["id"]),
                                                  {"id": q["id"], "polarity": q["polarity"]})
                                      | {"pieces": q["pieces"], "counted": q["counted"]}
                                      for q in rec["regions"]]}
                   for i, rec in enumerate(self.species)]
        instrument = result.method.instrument
        lock = result.lock_mass or {}
        mz = "m/z"
        if lock.get("applied_ppm") is not None:
            mz = ("m/z after the lock-mass correction (lock_mass.applied_ppm): the m/z the file "
                  "holds = m/z / (1 + applied_ppm * 1e-6)")
        return head | {
            "units": {
                "rt": "min", "mass": "Da", "mz": mz,
                "observed": "the region's scans' own points within the window, all scans "
                            "together: their number, summed intensity (the instrument's counts "
                            "added over the scans), intensity-weighted mean m/z and the scans "
                            "with signal there; max: the tallest point of the region-summed "
                            "spectrum there",
                "grid": "the summed spectrum resampled on the ln(m/z) grid the fit reads "
                        "(intensity integrated over each bin, divided by its width), summed "
                        "over the window's bins; fitted intensities are template areas on that "
                        "grid",
                "eic": "deconvolved intensity per scan"},
            "estimator": self.estimator,
            "instrument": instrument.to_dict() if instrument is not None else None,
            "rt_window": result.rt_window,
            "lock_mass": result.lock_mass,
            "merge_steps": MERGE_STEPS,
            "drop_steps": DROP_STEPS,
            "regions": regions,
            "figures": {"directory": "evidence", "max": int(max_figures),
                        "drawn": len(figures), "species": len(self.species),
                        "rule": f"the {int(max_figures)} largest species by base-form amount"},
            "timings": dict(self.timings),
            "species": species,
        }


# ------------------------------------------------------------------------------ writing
def figure_name(record: dict) -> str:
    """An evidence figure's file name: the species' id and mass only."""
    return f"species-{record['id']}-{record['mass']:.2f}.png"


class _WeightNote(logging.Filter):
    """matplotlib's note that a font has no face of the weight asked for, so it draws with the
    face it has (the CJK fallback, say): a log line on stderr that reads like an error."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("findfont: Failed to find font weight")


@contextlib.contextmanager
def _quiet_font_weights():
    """While the figures are drawn, leave out :class:`_WeightNote`'s notes."""
    log = logging.getLogger("matplotlib.font_manager")
    note = _WeightNote()
    log.addFilter(note)
    try:
        yield
    finally:
        log.removeFilter(note)


def write_provenance(result, outdir: str | Path, max_figures: int = MAX_FIGURES, aliases=None,
                     results_sha256: str | None = None) -> list[Path]:
    """Write ``provenance.json``, ``provenance.html`` and the evidence figures of
    ``result.provenance`` (the ``max_figures`` largest species by base-form amount) into
    ``outdir``, removing the figures of ``evidence/`` that this call does not draw (an earlier
    run's); returns the files written (JSON, HTML, figures).

    ``aliases`` (:class:`lcmsdeconv.aliases.Aliases`) show NEW for OLD in the texts that do not
    come with the run and the method: the network's file name (for a run and a method aliased
    already, processed without ``aliases=``). ``results_sha256`` is that of the results.json the
    provenance goes with; by default that of the one :func:`lcmsdeconv.io.results.write_outputs`
    writes for ``result``."""
    from .io.results import results_sha256 as summary_sha256
    from .io.results import save_json
    from .report.html import _plot_fonts

    prov = getattr(result, "provenance", None)
    if prov is None:
        raise ValueError("the result holds no provenance: process the run with "
                         "process_run(..., provenance=True) (lcmsdeconv process --provenance)")
    if aliases:
        prov.alias_texts(aliases)
    if results_sha256 is None:
        results_sha256 = summary_sha256(result)
    t0 = time.perf_counter()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    drawn: dict[int, str] = {}
    images: list[Path] = []
    todo = [(i, rec) for i, rec in enumerate(prov.species[:max(0, int(max_figures))])
            if prov.figures[i] is not None]
    if todo:
        (outdir / "evidence").mkdir(exist_ok=True)
        names = [result.run_name, *(rec["name"] or "" for _i, rec in todo)]
        with _plot_fonts(result, extra=names), _quiet_font_weights():
            for i, rec in todo:
                path = outdir / "evidence" / figure_name(rec)
                _draw(result, rec, prov.figures[i], path)
                drawn[i] = f"evidence/{path.name}"
                images.append(path)
    _remove_stale_figures(outdir / "evidence", images)
    data = prov.to_dict(result, drawn, max_figures, results_sha256)
    data["timings"]["figures_s"] = time.perf_counter() - t0
    json_path = save_json(data, outdir / "provenance.json")
    html_path = outdir / "provenance.html"
    html_path.write_text(_index(data), encoding="utf-8")
    return [json_path, html_path, *images]


def _remove_stale_figures(folder: Path, images: list[Path]) -> None:
    """Remove the evidence figures in ``folder`` that are not ``images`` (an earlier run's,
    which no provenance.html written now links)."""
    if not folder.is_dir():
        return
    keep = {p.name for p in images}
    for p in folder.glob("species-*.png"):
        if p.name not in keep and p.is_file():
            with contextlib.suppress(OSError):
                p.unlink()


def _charges_text(charges: list[dict]) -> str:
    zs = sorted(c["z"] for c in charges)
    if not zs:
        return "-"
    return f"{zs[0]}" if len(zs) == 1 else f"{zs[0]}-{zs[-1]} ({len(zs)} states)"


#: what the index and the figures say of a region whose pieces are not counted
_NOT_COUNTED_TEXT = {"set_aside": "set aside", "other_polarity": "other polarity"}

#: what a figure says of a piece drawn with the species' own in its region's spectrum
_JOINED_TEXT = {"adduct_fold": "an adduct form folded in", "adjacent_pieces": "joined",
                "boundary_piece": "a boundary piece joined"}


def _region_label(region: dict, both: bool) -> str:
    """A region's id, with its polarity where the run's regions have both (ids restart in each
    polarity)."""
    return f"{region['id']}" + (("+" if region["polarity"] > 0 else "-") if both else "")


def _numbers(values) -> str:
    """Whole numbers as runs: 1-3, 5, 7-8."""
    out: list[str] = []
    vals = sorted(set(int(v) for v in values))
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[j] + 1:
            j += 1
        out.append(f"{vals[i]}" if j == i else f"{vals[i]}-{vals[j]}")
        i = j + 1
    return ", ".join(out)


def _joined_text(piece: dict) -> str:
    """How a piece came into its species, as a figure's legend says it: "" for the species' own,
    else how it was joined and, for one not counted, that it is not."""
    step = piece.get("joined_by")
    if step is None:
        return ""
    if not piece.get("counted", True):
        return f" ({_NOT_COUNTED_TEXT.get(step, 'joined')}: not counted)"
    return f" ({_JOINED_TEXT.get(step, 'joined')})"


def _regions_text(s: dict, both: bool) -> str:
    """A species' regions, those whose pieces are not counted marked with why."""
    out = []
    for r in s["regions"]:
        text = _region_label(r, both)
        if not r.get("counted", True):
            why = {s["pieces"][i]["joined_by"] for i in r.get("pieces", ())}
            text += " (" + ", ".join(sorted(_NOT_COUNTED_TEXT.get(w, "not counted")
                                            for w in why)) + ")"
        out.append(text)
    return ", ".join(out)


def _index(data: dict) -> str:
    """provenance.html: each species with its mass, RT range and charges, its figure linked."""
    def e(x) -> str:
        return escape(str(x), quote=True)

    def f(v, nd) -> str:
        return "-" if v is None else f"{v:,.{nd}f}"

    both = len({r["polarity"] for r in data["regions"]}) > 1
    rows = []
    for s in data["species"]:
        rt = s["rt_signal"] or s["rt_range"] or [None, None]
        regions = _regions_text(s, both)
        fig = (f'<a href="{e(s["figure"])}">figure</a>' if s["figure"] else "not drawn")
        mass = f(s["mass"], 2) + (" (mono)" if s["mass_type"] == "mono" else "")
        rows.append("<tr>" + "".join(f"<td{c}>{v}</td>" for c, v in (
            ("", e(s["id"])), (' class="l"', e(s["name"] or "-")), ("", mass),
            ("", f(s["rt_apex"], 3)), ("", f"{f(rt[0], 3)} - {f(rt[1], 3)}"),
            ("", e(regions or "-")), ("", e(_charges_text(s["charges"]))),
            ("", e(len(s["pieces"]))), ("", fig))) + "</tr>")
    est = data["estimator"]
    est_text = (est.get("file") if est.get("kind") == "network" else est.get("kind")) or "-"
    figs = data["figures"]
    sha = (data.get("results") or {}).get("sha256") or "-"
    left = ' class="l"'
    head = "".join(f"<th{left if h == 'Name' else ''}>{e(h)}</th>" for h in (
        "Species", "Name", "Mass (Da)", "RT apex (min)", "RT range (min)", "Regions", "Charges",
        "Pieces", "Evidence"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Provenance: {e(data["run"])}</title>
<style>
:root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --mut:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8e8; --bg:#16181a; --mut:#9aa0a6; --line:#333; }} }}
body {{ font: 14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); margin:0; padding:24px 28px; }}
h1 {{ font-size:20px; margin:0 0 6px; }} p {{ color:var(--mut); margin:4px 0; }}
code {{ font-size:12px; overflow-wrap:anywhere; }}
table {{ border-collapse:collapse; margin-top:14px; font-size:13px; }}
th,td {{ text-align:right; padding:4px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th.l,td.l {{ text-align:left; }} th {{ color:var(--mut); }}
</style></head><body>
<h1>Provenance: {e(data["run"])}</h1>
<p>Method {e(data["method"])} &middot; charge estimator {e(est_text)} &middot; lcmsdeconv {e(data["lcmsdeconv"])} &middot; <a href="provenance.json">provenance.json</a></p>
<p>Written with the results.json whose SHA-256 is <code>{e(sha)}</code>: where the results.json beside this page has another, this page describes other results.</p>
<p>{e(figs["species"])} species; evidence figures for {e(figs["drawn"])} ({e(figs["rule"])}). RT range: the first and last scan with a reading. Regions: those of the species' pieces; a region whose pieces are not counted in its amount is marked with why.</p>
<table><thead><tr>{head}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
</body></html>
"""


# ------------------------------------------------------------------------------ figures
def _percent(v) -> str:
    return "-" if v is None else f"{100 * v:.0f} %"


def _draw(result, rec: dict, data: dict, path: Path) -> None:
    """One species' evidence figure: the region-summed spectrum over its charge states (the
    windows shaded and numbered, its fitted templates, its other counted pieces' and the other
    components'), its strongest charge states close up, and its trace over the scans with its
    apex."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    spec = data["spectrum"]
    zooms = spec["zooms"]
    fig = Figure(figsize=(10.0, 8.4), dpi=90)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, max(1, len(zooms)), height_ratios=[1.2, 1.0, 1.0], hspace=0.78,
                          wspace=0.28, top=0.88, bottom=0.07, left=0.08, right=0.98)
    both = len({int(r.polarity) for r in result.regions}) > 1
    region = data["region"]
    reg = next((r for r in result.regions
                if int(r.id) == region["id"] and int(r.polarity) == region["polarity"]), None)
    blue, grey, orange, purple, red = "#1b6ec2", "#555555", "#d97706", "#8e44ad", "#c0392b"

    ax = fig.add_subplot(gs[0, :])
    ax.plot(spec["mz"], spec["observed"], color=grey, lw=0.6,
            label="region-summed spectrum (grid view)")
    ax.plot(spec["mz"], spec["others"], color=orange, lw=0.6, alpha=0.8,
            label="other components' fitted templates")
    if spec.get("joined") is not None:
        joined = data.get("joined_pieces", ())
        ax.plot(spec["mz"], spec["joined"], color=purple, lw=0.8,
                label="its other counted pieces of this region "
                      f"(piece{'s' if len(joined) != 1 else ''} {_numbers(joined)})")
    merged = data.get("fitted_parts") or 0
    ax.plot(spec["mz"], spec["own"], color=blue, lw=0.9,
            label="this species' fitted templates"
                  + (f" ({merged} parts the fit merged, each where fitted)" if merged else ""))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    # every window shaded; numbered where the number has room (the lowest m/z first)
    span = float(spec["mz"][-1] - spec["mz"][0]) if len(spec["mz"]) > 1 else 1.0
    for kind, colour in (("own", blue), ("joined", purple)):
        windows = sorted((w for w in spec["windows"] if w[3] == kind), key=lambda w: w[1])
        if not windows:
            continue
        ax.add_collection(PolyCollection([[(lo, 0.0), (lo, 1.0), (hi, 1.0), (hi, 0.0)]
                                          for _z, lo, hi, _k in windows], transform=trans,
                                         facecolors=colour, alpha=0.13 if kind == "own" else 0.1,
                                         linewidths=0), autolim=False)
        last = -np.inf
        for z, lo, hi, _k in windows:
            centre = 0.5 * (lo + hi)
            if centre - last >= 0.022 * span:
                ax.text(centre, 1.01 if kind == "own" else 1.07, str(z), transform=trans,
                        ha="center", va="bottom", fontsize=6.5, color=colour)
                last = centre
    where = (f"region {_region_label(region, both)}"
             + (f", {reg.start:.2f}-{reg.end:.2f} min, {len(reg.frames)} scans"
                if reg is not None else ""))
    ax.set_title(f"Summed spectrum of {where}: charge states shaded and numbered",
                 fontsize=9, pad=18 if spec.get("joined") is not None else 12, parse_math=False)
    ax.set_xlabel("m/z", fontsize=8)
    ax.set_ylabel("intensity", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", frameon=False)

    for j, zm in enumerate(zooms):
        axz = fig.add_subplot(gs[1, j])
        axz.axvspan(*zm["window_mz"], color=blue, alpha=0.13, lw=0)
        axz.plot(zm["mz"], zm["observed"], color=grey, lw=0.8)
        axz.plot(zm["mz"], zm["model"], color=orange, lw=0.7, ls="--")
        if zm.get("dropped_model") is not None:
            axz.plot(zm["mz"], zm["dropped_model"], color=red, lw=0.8, ls=":",
                     label="fitted, then dropped")
            axz.legend(fontsize=5.5, loc="upper right", frameon=False)
        if zm.get("joined") is not None and np.any(zm["joined"] > 0):
            axz.plot(zm["mz"], zm["joined"], color=purple, lw=0.8)
        axz.plot(zm["mz"], zm["own"], color=blue, lw=1.0)
        lines = [f"z = {zm['z']}, fit residual {_percent(zm['fit_residual'])}",
                 f"unexplained {_percent(zm['unexplained'])}, "
                 f"overshoot {_percent(zm['overshoot'])}"]
        if zm.get("dropped") and zm["dropped"] >= 0.01:
            lines.append(f"dropped components {_percent(zm['dropped'])}")
        axz.set_title("\n".join(lines), fontsize=7, parse_math=False)
        axz.tick_params(labelsize=6.5)
        axz.set_xlabel("m/z", fontsize=7)

    axe = fig.add_subplot(gs[2, :])
    t, y = data["time"], data["intensity"]
    regions = {(int(r.id), int(r.polarity)): r for r in result.regions}
    for q in rec["regions"]:
        r = regions.get((q["id"], q["polarity"]))
        if r is not None:
            axe.axvspan(r.start, r.end, color=grey, alpha=0.07, lw=0)
            for b in (r.start, r.end):
                axe.axvline(b, color=grey, lw=0.5, ls=":")
    if len(data["pieces"]) > 1:
        # every piece's trace: each in a colour of its own and named in the legend, or, of a
        # species of more pieces than colours, the first ones so and the rest grey under one
        # entry (provenance.json lists them all)
        pieces = data["pieces"]
        k = len(pieces) if len(pieces) <= len(PIECE_COLOURS) else len(PIECE_COLOURS) - 1
        named, rest = pieces[:k], pieces[k:]
        for (i, reg_i, sid, mass, apex_i, counted, tp, yp), colour in zip(named, PIECE_COLOURS):
            axe.plot(tp, yp, lw=0.8, ls=":" if counted else "-.", color=colour,
                     label=f"piece {i}: species {sid}, region {_region_label(reg_i, both)}, "
                           f"{mass:,.2f} Da"
                           + ("" if apex_i is None else f", apex {apex_i:.2f} min")
                           + _joined_text(rec["pieces"][i]))
        for j, (_i, _reg, _sid, _mass, _apex, counted, tp, yp) in enumerate(rest):
            axe.plot(tp, yp, lw=0.7, ls=":" if counted else "-.", color="#999999",
                     label=(f"pieces {_numbers(p[0] for p in rest)}: grey, listed in "
                            "provenance.json") if j == 0 else None)
    axe.plot(t, y, color="black", lw=1.0, marker=".", ms=3, label="all forms")
    if data["base"] is not None and len(rec["adduct_fractions"]) > 1:
        axe.plot(t, data["base"], color=blue, lw=0.8, label="base form")
    apex = rec["rt_apex"]
    if apex is not None:
        axe.axvline(apex, color=red, lw=0.9)
        axe.text(apex, 1.0, f" apex {apex:.3f} min", transform=blended_transform_factory(
            axe.transData, axe.transAxes), color=red, fontsize=7, va="top")
    axe.set_xlabel("retention time (min)", fontsize=8)
    axe.set_ylabel("deconvolved intensity", fontsize=8)
    axe.set_title("Trace over the scans (regions shaded)", fontsize=9)
    axe.tick_params(labelsize=7)
    many = len(axe.get_legend_handles_labels()[1]) > 6  # set closer, to keep within the panel
    axe.legend(fontsize=6 if many else 6.5, labelspacing=0.25 if many else 0.5, loc="upper right",
               frameon=False)

    zs = [c["z"] for c in rec["charges"]]
    name = f" ({rec['name']})" if rec["name"] else ""
    n = len(rec["pieces"])
    title = (f"{result.run_name}: species {rec['id']}{name}, {rec['mass']:,.2f} Da\n"
             f"{rec['mass_type']} mass, {'positive' if rec['polarity'] > 0 else 'negative'} "
             "mode, apex " + ("-" if apex is None else f"{apex:.3f} min") + ", charges "
             + (f"{min(zs)}-{max(zs)}" if zs else "-") + f", {n} piece{'s' if n != 1 else ''}")
    fig.suptitle(title, fontsize=10, parse_math=False)
    # the title, the only text drawn that holds names (the run's, the species'), is kept as
    # text in the file too: the leak check of a later run with name aliases reads it
    # (lcmsdeconv.aliases), which the pixels would not let it
    fig.savefig(path, dpi=90, metadata={"Title": title})
