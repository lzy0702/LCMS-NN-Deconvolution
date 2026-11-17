"""Diagnostics that locate where the deconvolution chain loses species.

These tools answer, cheaply and without training anything, the questions that decide where
effort should go:

* **Charge oracles.** :func:`oracle_predictor` assigns every grid bin the true charge of the ion
  under it. Run through the normal decoder, it measures the ceiling of everything downstream of
  the charge network: if a perfect charge map still misses a species, a better network cannot
  recover it. :func:`confused_oracle_predictor` corrupts that map at a chosen rate with the
  errors a network actually makes (harmonics ``2z`` and ``z/2``, neighbours ``z +/- 1``), which
  gives the charge accuracy a network must reach before the decoder stops producing false
  species.
* **Loss attribution.** :func:`attribute_losses` follows each true species through the stages
  recorded by ``process_run(..., trace=[])`` and reports the first stage where it disappeared.
* **False-species attribution.** :func:`classify_false_species` names what each unmatched
  reported species is: a charge harmonic, an adduct-shifted copy, a mass error or unexplained.
* **One-to-one scoring.** :func:`match_species` pairs reported and true species so that a
  reported species can be credited to at most one true species.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chem.adducts import AdductState
from .deconv.oracle import OraclePredictor
from .synth.render import ComponentInstance, Scene, build_sticks
from .synth.spec import Sticks

#: Order of the stages recorded by the pipeline trace, from first to last.
STAGES = ("votes", "picked", "gated", "harmonics", "adduct_dedupe", "detected", "fit",
          "post_merge", "components", "species")

TIERS = [(">=10%", 0.10, 1.01), ("1-10%", 0.01, 0.10), ("0.1-1%", 0.001, 0.01),
         ("0.01-0.1%", 0.0001, 0.001)]


# ------------------------------------------------------------------ truth helpers
def truth_components(truth: dict) -> list[dict]:
    """Flatten a ``RunTruth.to_dict()`` into one record per true species.

    ``rel`` is the species' share of its chromatographic peak, which is how impurity tiers are
    defined throughout the package.
    """
    out = []
    for pi, peak in enumerate(truth.get("peaks", [])):
        total = sum(m["rel"] for m in peak["members"]) or 1.0
        for mem in peak["members"]:
            rec = {"mass": float(mem["mass"]), "rel": float(mem["rel"]) / total,
                   "rt": float(peak["rt"]), "name": mem["name"], "kind": mem["kind"], "peak": pi}
            for key in ("peak_counts", "support_counts"):
                if key in mem:
                    rec[key] = float(mem[key])
            out.append(rec)
    return out


def tier_of(rel: float) -> str | None:
    return next((n for n, lo, hi in TIERS if lo <= rel < hi), None)


def mass_tolerance(mass: float, tol_ppm: float = 100.0, tol_da: float = 0.5) -> float:
    return max(mass * tol_ppm * 1e-6, tol_da)


def truth_sticks(truth, polarity: int) -> Sticks:
    """Isotopologue sticks of every true species of a synthetic run (``RunTruth`` object)."""
    comps = []
    for p in truth.peaks:
        adducts = getattr(p, "_adducts", {}) or {}
        for idx, (comp, rel) in enumerate(p.members):
            cd = p.charge_dists.get(idx, {})
            if not cd:
                continue
            ad = adducts.get(idx) or {z: {AdductState(): 1.0} for z in cd}
            comps.append(ComponentInstance(comp, p.intensity * rel, cd, ad))
    if not comps:
        return Sticks.empty()
    return build_sticks(Scene(comps, polarity))


# ------------------------------------------------------------------ charge oracles
def oracle_predictor(truth, polarity: int) -> OraclePredictor:
    """Perfect per-bin charge assignment for a synthetic run."""
    return OraclePredictor(truth_sticks(truth, polarity), None)


def confuse_charges(sticks: Sticks, accuracy: float, rng: np.random.Generator,
                    kernel: str = "mixed") -> Sticks:
    """Relabel whole charge-state envelopes with the errors a charge network makes.

    Each (species, charge) envelope keeps its charge with probability ``accuracy``; otherwise it
    is relabelled by ``kernel``: ``harmonic`` (``2z`` or ``z/2``), ``neighbour`` (``z +/- 1``) or
    ``mixed`` (``2z``, ``z/2`` with 0.3 each and ``z +/- 1`` with 0.2 each). An odd charge has no
    ``z/2``, so that draw becomes a neighbour error instead.
    """
    z = sticks.z.copy()
    keys = np.stack([sticks.comp_id, sticks.z], axis=1)
    families, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    new_z = families[:, 1].copy()
    for f, (_cid, zf) in enumerate(families):
        if rng.random() < accuracy:
            continue
        zf = int(zf)
        if kernel == "harmonic":
            choice = "double" if rng.random() < 0.5 else "half"
        elif kernel == "neighbour":
            choice = "up" if rng.random() < 0.5 else "down"
        else:
            r = rng.random()
            choice = "double" if r < 0.3 else "half" if r < 0.6 else "up" if r < 0.8 else "down"
        if choice == "half" and (zf % 2 or zf < 2):
            choice = "up" if rng.random() < 0.5 else "down"
        if choice == "down" and zf <= 1:
            choice = "up"
        new_z[f] = {"double": 2 * zf, "half": zf // 2, "up": zf + 1, "down": zf - 1}[choice]
    z = new_z[inverse]
    return Sticks(sticks.mz, sticks.intensity, sticks.comp_id, z.astype(sticks.z.dtype),
                  sticks.adduct_mass)


def confused_oracle_predictor(truth, polarity: int, accuracy: float, seed: int = 0,
                              kernel: str = "mixed") -> OraclePredictor:
    """Oracle whose charge map is wrong for a fraction ``1 - accuracy`` of envelopes."""
    sticks = confuse_charges(truth_sticks(truth, polarity), accuracy,
                             np.random.default_rng(seed), kernel)
    return OraclePredictor(sticks, None)


# ------------------------------------------------------------------ scoring
@dataclass
class Match:
    truth_index: int
    species_index: int
    error_da: float


def match_species(species: list[dict], truths: list[dict], tol_ppm: float = 100.0,
                  tol_da: float = 0.5, rt_tol: float = 0.5) -> list[Match]:
    """Pair reported and true species one-to-one.

    True species are served in order of abundance, each taking the closest still-unmatched
    reported species within tolerance. Serving the closest pair first instead would hand a main
    component's report to an unresolvable impurity one dalton away (a deamidated form of a
    75 kDa protein, say) and score the main component as missed.
    """
    order = sorted(range(len(truths)), key=lambda i: -float(truths[i].get("rel", 1.0)))
    used: set[int] = set()
    out = []
    for ti in order:
        t = truths[ti]
        tol = mass_tolerance(t["mass"], tol_ppm, tol_da)
        best, best_err = None, np.inf
        for si, s in enumerate(species):
            if si in used:
                continue
            err = abs(float(s["mass"]) - t["mass"])
            if err > tol or err >= best_err:
                continue
            if rt_tol and abs(float(s.get("rt_apex", t["rt"])) - t["rt"]) > rt_tol:
                continue
            best, best_err = si, err
        if best is not None:
            used.add(best)
            out.append(Match(ti, best, float(best_err)))
    return out


def classify_false_species(species: list[dict], truths: list[dict], matched: set[int],
                           adduct_deltas: dict[str, float] | None = None,
                           tol_ppm: float = 100.0, tol_da: float = 0.5) -> list[dict]:
    """Name what each unmatched reported species most likely is.

    Categories, tested in this order: ``harmonic`` (n*M or M/n of a true species, from a charge
    assigned n-fold too high or too low), ``adduct`` (a true species shifted by up to three
    adducts, i.e. reported at an adducted rather than base mass), ``mass_error`` (within 0.5 %
    of a true species but outside tolerance) and ``unexplained``.
    """
    deltas = sorted({abs(d) for d in (adduct_deltas or {}).values() if abs(d) > 0.5})
    out = []
    for si, s in enumerate(species):
        if si in matched:
            continue
        m = float(s["mass"])
        kind, ref = "unexplained", None
        for t in truths:
            M = t["mass"]
            tol = max(mass_tolerance(M, 500.0, 2.0), 2.0)
            if any(abs(m - n * M) <= n * tol or abs(m - M / n) <= tol for n in range(2, 7)):
                kind, ref = "harmonic", t
                break
        if kind == "unexplained" and deltas:
            for t in truths:
                M = t["mass"]
                hit = False
                for d in deltas:
                    for k in (1, 2, 3):
                        if abs(abs(m - M) - k * d) <= max(tol_da, 0.6):
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    kind, ref = "adduct", t
                    break
        if kind == "unexplained":
            for t in truths:
                if abs(m - t["mass"]) <= 5e-3 * t["mass"]:
                    kind, ref = "mass_error", t
                    break
        out.append({"mass": m, "kind": kind,
                    "relative_to": None if ref is None else ref["name"],
                    "rt_apex": s.get("rt_apex")})
    return out


# ------------------------------------------------------------------ loss attribution
def _region_for(trace: list[dict], rt: float, slack: float = 0.05) -> dict | None:
    """The deconvolved region whose retention-time window holds ``rt`` (None if no region does).

    A species whose chromatographic apex falls in no region was never deconvolved at all, which
    is a different failure from losing it inside a region and is reported as ``no_region``.
    """
    inside = [r for r in trace if r["start"] - slack <= rt <= r["end"] + slack]
    if not inside:
        return None
    return min(inside, key=lambda r: abs(0.5 * (r["start"] + r["end"]) - rt))


def _present(masses, M: float, tol: float) -> bool:
    return any(abs(float(m) - M) <= tol for m in masses)


def attribute_losses(trace: list[dict], truth: dict, species: list[dict],
                     tol_ppm: float = 100.0, tol_da: float = 0.5, rt_tol: float = 0.5) -> list[dict]:
    """First stage at which each true species disappeared from the chain.

    ``trace`` is the list filled by ``process_run(..., trace=[])``. The returned records carry
    ``lost_at`` (one of :data:`STAGES`, ``"no_region"`` or ``None`` when the species was
    reported) and, when the fit rejected it, the fit's reason.
    """
    truths = truth_components(truth)
    matches = {m.truth_index: m for m in match_species(species, truths, tol_ppm, tol_da, rt_tol)}
    out = []
    for ti, t in enumerate(truths):
        M = t["mass"]
        tol = mass_tolerance(M, tol_ppm, tol_da)
        rec = {"name": t["name"], "mass": M, "rel": t["rel"], "tier": tier_of(t["rel"]),
               "kind": t["kind"], "lost_at": None, "reason": None,
               "support_counts": t.get("support_counts")}
        region = _region_for(trace, t["rt"])
        if ti in matches:
            out.append(rec)
            continue
        if region is None:
            rec["lost_at"] = "no_region"
            out.append(rec)
            continue
        stages = region["stages"] or []
        # votes: histogram weight within tolerance, relative to the strongest vote
        vote_frac = 0.0
        best = 0.0
        for st in stages:
            if st["stage"] != "votes" or len(st["weight"]) == 0:
                continue
            best = max(best, float(np.max(st["weight"])))
            lm = st["log_mass"]
            sel = np.abs(np.exp(lm) - M) <= tol
            vote_frac = max(vote_frac, float(st["weight"][sel].sum()))
        rec["vote_weight"] = vote_frac / best if best > 0 else 0.0
        lost = "votes" if vote_frac <= 0 else None
        if lost is None:
            for name in ("picked", "gated", "harmonics", "adduct_dedupe", "detected"):
                if not any(_present(st["masses"], M, tol) for st in stages if st["stage"] == name):
                    lost = name
                    break
        if lost is None:
            fits = [st for st in stages if st["stage"] == "fit" and abs(st["candidate"] - M) <= tol]
            accepted = [f for f in fits if f["outcome"] == "accepted"]
            if not accepted:
                lost = "fit"
                rec["reason"] = fits[0]["outcome"] if fits else "not_fitted"
            elif not any(f["mass"] is not None and abs(f["mass"] - M) <= tol for f in accepted):
                lost = "fit"
                rec["reason"] = "mass_moved"
                rec["fit_mass"] = accepted[0]["mass"]
        if lost is None and not any(_present(st["masses"], M, tol)
                                    for st in stages if st["stage"] == "post_merge"):
            lost = "post_merge"
        if lost is None and not any(_present(st["masses"], M, tol)
                                    for st in stages if st["stage"] == "components"):
            lost = "components"
        if lost is None:
            lost = "species"
        rec["lost_at"] = lost
        out.append(rec)
    return out


def summarize_losses(records: list[dict], min_counts: float | None = None) -> dict[str, dict[str, int]]:
    """Counts of ``lost_at`` per abundance tier (``found`` for reported species).

    With ``min_counts``, only true species whose second charge state carries at least that many
    expected ion counts are counted (see :data:`lcmsdeconv.evaluate.DETECTABLE_COUNTS`).
    """
    out: dict[str, dict[str, int]] = {}
    for r in records:
        if min_counts is not None and (r.get("support_counts") is not None
                                       and r["support_counts"] < min_counts):
            continue
        tier = r["tier"] or "<0.01%"
        key = r["lost_at"] or "found"
        if key == "fit" and r.get("reason"):
            key = f"fit:{r['reason']}"
        out.setdefault(tier, {})
        out[tier][key] = out[tier].get(key, 0) + 1
    return out
