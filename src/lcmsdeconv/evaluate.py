"""Score processing output against synthetic ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TIERS = [(">=10%", 0.10, 1.01), ("1-10%", 0.01, 0.10), ("0.1-1%", 0.001, 0.01),
         ("0.01-0.1%", 0.0001, 0.001)]

#: A true species whose second-strongest charge state carries fewer expected ion counts than
#: this (summed over its chromatographic peak) cannot be confirmed by two charge states above
#: shot noise, whatever the software does. Recall is reported over all species and over the
#: detectable ones.
DETECTABLE_COUNTS = 50.0


def _truth_components(truth: dict) -> list[dict]:
    from .diagnostics import truth_components

    return truth_components(truth)


def score_species(species: list[dict], truth: dict, tol_ppm: float = 100.0,
                  tol_da: float = 0.5, rt_tol: float = 0.5,
                  detectable_counts: float = DETECTABLE_COUNTS) -> dict:
    """Recall, precision and mass error by abundance tier.

    Reported and true species are paired one-to-one (closest relative mass error first), so a
    reported species is credited to at most one true species. ``recall`` counts every true
    species of the tier; ``recall_detectable`` only those with at least ``detectable_counts``
    expected ion counts in their second-strongest charge state, when the truth records it.
    """
    from .diagnostics import match_species

    tcomps = _truth_components(truth)
    if not tcomps:
        return {}
    found = list(species)
    matches = match_species(found, tcomps, tol_ppm, tol_da, rt_tol)
    by_truth = {m.truth_index: m for m in matches}
    per_tier: dict[str, dict] = {name: {"n_true": 0, "n_found": 0, "n_det": 0, "n_det_found": 0,
                                        "ppm": []} for name, _, _ in TIERS}
    errors_ppm: list[float] = []
    for ti, tc in enumerate(tcomps):
        tier = next((n for n, lo, hi in TIERS if lo <= tc["rel"] < hi), None)
        if tier is None:
            continue
        d = per_tier[tier]
        d["n_true"] += 1
        detectable = tc.get("support_counts", float("inf")) >= detectable_counts
        d["n_det"] += int(detectable)
        m = by_truth.get(ti)
        if m is None:
            continue
        d["n_found"] += 1
        d["n_det_found"] += int(detectable)
        ppm = (float(found[m.species_index]["mass"]) - tc["mass"]) / tc["mass"] * 1e6
        d["ppm"].append(ppm)
        errors_ppm.append(ppm)

    tiers_out = {}
    for name, d in per_tier.items():
        if d["n_true"] == 0:
            continue
        tiers_out[name] = {
            "n_true": d["n_true"],
            "recall": d["n_found"] / d["n_true"],
            "n_detectable": d["n_det"],
            "recall_detectable": d["n_det_found"] / d["n_det"] if d["n_det"] else None,
            "median_abs_ppm": float(np.median(np.abs(d["ppm"]))) if d["ppm"] else None,
        }
    n_pred = len(found)
    return {
        "n_truth_components": len(tcomps),
        "n_predicted": n_pred,
        "n_matched": len(matches),
        "precision": len(matches) / n_pred if n_pred else 0.0,
        "median_abs_ppm": float(np.median(np.abs(errors_ppm))) if errors_ppm else None,
        "p95_abs_ppm": float(np.percentile(np.abs(errors_ppm), 95)) if errors_ppm else None,
        "tiers": tiers_out,
    }


def evaluate_directory(results_dir: Path, truth_path: Path | None = None,
                       method: str | None = None, model: str | None = None) -> dict:
    """Score an existing results directory, or process a synth directory then score it."""
    results_dir = Path(results_dir)
    res_file = results_dir / "results.json"
    if not res_file.exists():
        mzml = next(iter(results_dir.glob("*.mzML")), None)
        if mzml is None:
            raise FileNotFoundError(f"No results.json or mzML in {results_dir}")
        from .core.method import Method
        from .io.mzml import read_mzml
        from .process import process_run

        run = read_mzml(mzml)
        result = process_run(run, Method.load(method), model_path=model)
        summary = result.summary()
    else:
        summary = json.loads(res_file.read_text())
    if truth_path is None:
        cand = results_dir / "truth.json"
        truth_path = cand if cand.exists() else None
    out = {"n_species": summary.get("n_species", 0)}
    if truth_path is not None:
        truth = json.loads(Path(truth_path).read_text())
        out["scores"] = score_species(summary.get("species", []), truth)
    return out
