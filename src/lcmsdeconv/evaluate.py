"""Score processing output against synthetic ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TIERS = [(">=10%", 0.10, 1.01), ("1-10%", 0.01, 0.10), ("0.1-1%", 0.001, 0.01),
         ("0.01-0.1%", 0.0001, 0.001)]


def _truth_components(truth: dict) -> list[dict]:
    out = []
    for peak in truth.get("peaks", []):
        total = sum(mem["rel"] for mem in peak["members"])
        for mem in peak["members"]:
            out.append({"mass": mem["mass"], "rel": mem["rel"] / total if total else 0.0,
                        "rt": peak["rt"], "name": mem["name"], "kind": mem["kind"]})
    return out


def score_species(species: list[dict], truth: dict, tol_ppm: float = 100.0,
                  tol_da: float = 0.5, rt_tol: float = 0.5) -> dict:
    """Recall, precision and mass error by abundance tier."""
    tcomps = _truth_components(truth)
    if not tcomps:
        return {}
    found = list(species)
    matched_pred: set[int] = set()
    per_tier: dict[str, dict] = {name: {"n_true": 0, "n_found": 0, "ppm": []} for name, _, _ in TIERS}
    errors_ppm: list[float] = []

    for tc in tcomps:
        tier = next((n for n, lo, hi in TIERS if lo <= tc["rel"] < hi), None)
        if tier is None:
            continue
        per_tier[tier]["n_true"] += 1
        tol = max(tc["mass"] * tol_ppm * 1e-6, tol_da)
        best, best_err = None, np.inf
        for i, s in enumerate(found):
            if abs(s["mass"] - tc["mass"]) > tol:
                continue
            if rt_tol and abs(s.get("rt_apex", tc["rt"]) - tc["rt"]) > rt_tol:
                continue
            err = abs(s["mass"] - tc["mass"])
            if err < best_err:
                best, best_err = i, err
        if best is not None:
            matched_pred.add(best)
            per_tier[tier]["n_found"] += 1
            ppm = (found[best]["mass"] - tc["mass"]) / tc["mass"] * 1e6
            per_tier[tier]["ppm"].append(ppm)
            errors_ppm.append(ppm)

    tiers_out = {}
    for name, d in per_tier.items():
        if d["n_true"] == 0:
            continue
        tiers_out[name] = {
            "n_true": d["n_true"],
            "recall": d["n_found"] / d["n_true"],
            "median_abs_ppm": float(np.median(np.abs(d["ppm"]))) if d["ppm"] else None,
        }
    n_pred = len(found)
    return {
        "n_truth_components": len(tcomps),
        "n_predicted": n_pred,
        "precision": len(matched_pred) / n_pred if n_pred else 0.0,
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
