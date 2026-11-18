"""Benchmark the full chain on synthetic runs with known truth.

Each run is scored one-to-one against its truth, every missed true species is traced to the
stage where it was lost, and every unmatched reported species is classified (charge harmonic,
adduct-shifted copy, mass error, unexplained). Results are pooled per preset and written as
Markdown and, optionally, JSON.

The charge assignment can be swapped to measure the rest of the chain in isolation:

    --predictor comb            deterministic estimator (the default without a model)
    --predictor comb-ladder     the estimator without resolved-isotope charge evidence
    --predictor oracle          perfect per-bin charges: the ceiling of everything downstream
    --predictor confused:0.9    oracle with 10 % of charge envelopes relabelled 2z, z/2, z+/-1
    --predictor model.onnx      a trained network

and the generator can be switched between its default (realistic adduct load, impurities
sharing their parent's charge envelope), ``legacy`` (per-charge adduct propensity up to 0.15
and independent impurity envelopes, as earlier versions generated) and ``heavy`` (a per-ion
adduct load up to 1.0 of every type, as a stress test).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lcmsdeconv.chem.adducts import AdductLibrary
from lcmsdeconv.core.method import Method
from lcmsdeconv.diagnostics import (
    attribute_losses,
    classify_false_species,
    confused_oracle_predictor,
    match_species,
    oracle_predictor,
    summarize_losses,
    truth_components,
)
from lcmsdeconv.evaluate import DETECTABLE_COUNTS, TIERS, score_species
from lcmsdeconv.process import process_run
from lcmsdeconv.synth.chromatography import generate_run
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig

PRESETS = {
    "protein_tof": (dict(classes=[ClassConfig("peptide", (8000.0, 80000.0))], mode="rplc",
                         polarity=1, instrument_kind="tof",
                         resolution_range=(20000.0, 40000.0)), "rplc_pos_protein"),
    "oligo_iprp": (dict(classes=[ClassConfig("ps_rna", (5000.0, 20000.0))], mode="iprp",
                        polarity=-1, instrument_kind="tof",
                        resolution_range=(20000.0, 40000.0)), "iprp_neg_oligo"),
    "polymer": (dict(classes=[ClassConfig("peg", (1500.0, 12000.0))], mode="polymer",
                     polarity=1, instrument_kind="tof",
                     resolution_range=(20000.0, 40000.0)), "polymer_pos"),
    # resolved isotopes at the resolving power of the real Orbitrap runs (140 000 at m/z 200,
    # the reference m/z of InstrumentModel for Orbitraps), which the decoding method assumes;
    # "_lowres" keeps the 30 000-63 000 this preset first rendered. Neither is in the default
    # preset list
    "oligo_orbitrap": (dict(classes=[ClassConfig("ps_rna", (3000.0, 9000.0))], mode="iprp",
                            polarity=-1, instrument_kind="orbitrap",
                            resolution_range=(120000.0, 160000.0)), "orbitrap_iprp_neg_oligo"),
    "oligo_orbitrap_lowres": (dict(classes=[ClassConfig("ps_rna", (3000.0, 9000.0))], mode="iprp",
                                   polarity=-1, instrument_kind="orbitrap",
                                   resolution_range=(30000.0, 63000.0)), "orbitrap_iprp_neg_oligo"),
}
DEFAULT_PRESETS = ("protein_tof", "oligo_iprp", "polymer")

GENERATORS = {
    "default": {},
    "legacy": {"adduct_max_lambda": 0.15, "impurity_charge_from_parent": False},
    "heavy": {"adduct_max_per_ion": 1.0},
}


def make_predictor(spec: str, truth, polarity: int, seed: int):
    """(predictor, model_path) for ``process_run``."""
    if spec == "comb-ladder":  # charge-ladder comb only, without resolved-isotope evidence
        from lcmsdeconv.deconv.classical import CombPredictor

        return CombPredictor(z_max=60), None
    if spec == "oracle":
        return oracle_predictor(truth, polarity), None
    if spec.startswith("confused:"):
        return confused_oracle_predictor(truth, polarity, float(spec.split(":", 1)[1]), seed=seed), None
    return None, spec


def run_one(preset: str, seed: int, args) -> dict:
    kw, method_name = PRESETS[preset]
    method = Method.load(method_name)
    if args.single_charge != "auto":
        method.deconvolution.single_charge = args.single_charge == "on"
    cfg = SynthConfig(**kw, **GENERATORS[args.generator])
    rng = np.random.default_rng(seed)
    run, truth = generate_run(cfg, rng, n_peaks=args.peaks, rt_range=(0.5, 0.5 + args.minutes),
                              scan_rate_hz=args.rate)
    predictor, model_path = make_predictor(args.predictor, truth, cfg.polarity, seed)
    trace: list = []
    t0 = time.perf_counter()
    result = process_run(run, method, predictor=predictor, model_path=model_path, trace=trace)
    elapsed = time.perf_counter() - t0
    truth_d = truth.to_dict()
    species = [s.to_dict() for s in result.species]
    scores = score_species(species, truth_d)
    truths = truth_components(truth_d)
    matched = {m.species_index for m in match_species(species, truths)}
    library = AdductLibrary.from_mode(method.deconvolution.adducts.mode, cfg.polarity)
    false = classify_false_species(species, truths, matched, library.deltas())
    losses = attribute_losses(trace, truth_d, species)
    return {"preset": preset, "seed": seed, "seconds": elapsed, "n_species": len(species),
            **scores, "false_species": dict(Counter(f["kind"] for f in false)),
            "losses": losses}


def pool(rows: list[dict]) -> dict:
    """Pool counts over runs (recall per tier, precision, false-species kinds, loss stages)."""
    tiers: dict[str, dict[str, int]] = {}
    for r in rows:
        for name, t in r.get("tiers", {}).items():
            d = tiers.setdefault(name, {"n_true": 0, "found": 0, "n_det": 0, "det_found": 0})
            d["n_true"] += t["n_true"]
            d["found"] += round(t["recall"] * t["n_true"])
            d["n_det"] += t["n_detectable"]
            if t["recall_detectable"] is not None:
                d["det_found"] += round(t["recall_detectable"] * t["n_detectable"])
    n_pred = sum(r.get("n_predicted", 0) for r in rows)
    n_match = sum(r.get("n_matched", 0) for r in rows)
    false: Counter = Counter()
    for r in rows:
        false.update(r.get("false_species", {}))
    losses = [rec for r in rows for rec in r["losses"]]
    ppm = [r["median_abs_ppm"] for r in rows if r.get("median_abs_ppm") is not None]
    return {"tiers": tiers, "n_predicted": n_pred, "n_matched": n_match,
            "precision": n_match / n_pred if n_pred else 0.0,
            "median_of_median_ppm": float(np.median(ppm)) if ppm else None,
            "false_species": dict(false),
            "losses_detectable": summarize_losses(losses, DETECTABLE_COUNTS),
            "seconds": sum(r["seconds"] for r in rows)}


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.0f} % ({a}/{b})" if b else "-"


def markdown(pooled: dict[str, dict], rows: list[dict], args) -> str:
    lines = ["# Synthetic benchmark", "",
             f"`scripts/benchmark_synth.py --runs {args.runs} --predictor {args.predictor} "
             f"--generator {args.generator}`: {args.runs} runs per preset, {args.peaks} "
             f"chromatographic peaks per run, each a compound with its impurities.", "",
             "A reported species matches a true one when its mass agrees within 100 ppm or "
             "0.5 Da, whichever is larger, and its retention time within half a minute; pairing "
             "is one-to-one. *Detectable* species are those whose second-strongest charge state "
             f"carries at least {DETECTABLE_COUNTS:.0f} expected ion counts over the "
             "chromatographic peak; below that, two charge states cannot be confirmed above "
             "shot noise by any method.", "",
             "## Pooled over runs", "",
             "| Preset | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | 0.01-0.1 % | Median ppm | Seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for preset, p in pooled.items():
        cells = []
        for name, _, _ in TIERS:
            t = p["tiers"].get(name)
            cells.append(_pct(t["det_found"], t["n_det"]) if t else "-")
        ppm = p["median_of_median_ppm"]
        lines.append(f"| {preset} | {_pct(p['n_matched'], p['n_predicted'])} | "
                     + " | ".join(cells)
                     + f" | {'-' if ppm is None else f'{ppm:.1f}'} | {p['seconds']:.0f} |")
    lines += ["", "Recall columns count detectable species only; all true species, including "
              "undetectable ones:", "",
              "| Preset | Main (>=10 %) | 1-10 % | 0.1-1 % | 0.01-0.1 % |",
              "| --- | --- | --- | --- | --- |"]
    for preset, p in pooled.items():
        cells = []
        for name, _, _ in TIERS:
            t = p["tiers"].get(name)
            cells.append(_pct(t["found"], t["n_true"]) if t else "-")
        lines.append(f"| {preset} | " + " | ".join(cells) + " |")
    lines += ["", "## What the unmatched reported species are", "",
              "| Preset | Harmonic | Adduct-shifted | Mass error | Unexplained |",
              "| --- | --- | --- | --- | --- |"]
    for preset, p in pooled.items():
        f = p["false_species"]
        lines.append(f"| {preset} | {f.get('harmonic', 0)} | {f.get('adduct', 0)} | "
                     f"{f.get('mass_error', 0)} | {f.get('unexplained', 0)} |")
    lines += ["", "## Where detectable species were lost", "",
              "Stage at which each missed detectable species left the chain "
              "(`lcmsdeconv.diagnostics.attribute_losses`).", "",
              "| Preset | Tier | Outcome counts |", "| --- | --- | --- |"]
    for preset, p in pooled.items():
        for tier, counts in sorted(p["losses_detectable"].items()):
            text = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            lines.append(f"| {preset} | {tier} | {text} |")
    lines += ["", "## Per run", "",
              "| Preset | Seed | Reported | Matched | Precision | Seconds |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['preset']} | {r['seed']} | {r['n_species']} | {r.get('n_matched', 0)} | "
                     f"{100 * r.get('precision', 0):.0f} % | {r['seconds']:.1f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3, help="runs per preset")
    ap.add_argument("--seed0", type=int, default=1000, help="seed of the first run")
    ap.add_argument("--peaks", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--predictor", default="comb",
                    help="comb | oracle | confused:<accuracy> | path to an ONNX model")
    ap.add_argument("--model", default=None, help="alias for --predictor <path>")
    ap.add_argument("--generator", default="default", choices=sorted(GENERATORS))
    ap.add_argument("--presets", default=",".join(DEFAULT_PRESETS),
                    help=f"comma-separated, from {', '.join(PRESETS)}")
    ap.add_argument("--single-charge", default="auto", choices=("auto", "on", "off"),
                    help="accept species seen at one charge state with resolved isotopes "
                         "(auto: the method's setting)")
    ap.add_argument("--out", default=None, type=Path, help="Markdown report")
    ap.add_argument("--json", default=None, type=Path, help="full per-run results")
    args = ap.parse_args()
    if args.model:
        args.predictor = args.model

    rows = []
    pooled = {}
    for preset in args.presets.split(","):
        prow = []
        for i in range(args.runs):
            r = run_one(preset, args.seed0 + i, args)
            prow.append(r)
            brief = {k: r.get(k) for k in ("preset", "seed", "n_species", "n_matched", "precision")}
            brief["recall_detectable"] = {k: v["recall_detectable"] for k, v in r.get("tiers", {}).items()}
            brief["false_species"] = r["false_species"]
            print(json.dumps(brief, default=float), flush=True)
        rows += prow
        pooled[preset] = pool(prow)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown(pooled, rows, args), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"pooled": pooled, "rows": rows}, default=float))


if __name__ == "__main__":
    main()
