"""Per-bin charge reading on the known species' envelopes of the real Thermo runs.

For every chromatographic region of a run that holds a species of scripts/validate_oligo_runs.py's
truth table, the region's spectrum is built exactly as process_run builds the input of every
charge estimator (Region.summed, then spectrum_to_grid with the region's scans), every
estimator is run on it as the pipeline runs it (whole spectrum; networks in their own windows),
and the species' envelopes are located and labelled from mass and isotope spacing (envlib).
Writes one JSON record per accepted envelope, with its properties and each estimator's weighted
right/total sums, plus one record per region. Estimators: "comb" (the pipeline's comb with its
isotope reader), "comb, ladder only" and every --model.

    python bench/tools/real_env.py RAW... [--model LABEL=NET.onnx ...] [--threads N]
        [--providers auto|dml|cuda|cpu] --out real_env.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)

from tools import common  # noqa: E402

common.preset_threads()  # before numpy: --threads caps its BLAS / OpenMP / numba pools

METHOD = "orbitrap_iprp_neg_oligo"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", type=Path, help="Thermo .raw files")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--providers", default="auto", choices=("auto", "dml", "cuda", "cpu"))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    models = common.parse_models(args.model)
    common.cap_threads(args.threads)

    import validate_oligo_runs as vor

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.deconv.classical import CombPredictor, make_predictor
    from lcmsdeconv.features.regions import find_regions
    from lcmsdeconv.io.thermo import read_thermo_raw
    from lcmsdeconv.nn.grid import spectrum_to_grid
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.process import _deconv_params
    from tools import envlib

    method = Method.load(METHOD)
    params = _deconv_params(method)
    inst = method.instrument
    zmax = min(int(method.deconvolution.charge_range[1]), 60)
    estimators = {"comb": make_predictor("comb", z_max=zmax, instrument=inst),
                  "comb, ladder only": CombPredictor(z_max=zmax, instrument=None)}
    prov = common.providers(args.providers)
    for label, path in models.items():
        # a network with isotope-charge channels needs the method's resolving power
        estimators[label] = ChargePredictor(path, providers=prov, instrument=inst)
    out = {"envelopes": [], "regions": [], "estimators": list(estimators),
           "providers": {k: e.session.get_providers() for k, e in estimators.items()
                         if hasattr(e, "session")}}
    for path in args.raw:
        key = vor.run_key(path)
        t0 = time.time()
        run = read_thermo_raw(path)
        pol = -1
        regions = find_regions(run, method.regions.source, pol, method.regions.window_min,
                               method.regions.margin_min, method.regions.min_frames,
                               integration=method.integration_for("tic"),
                               min_snr=getattr(method.regions, "min_snr", 10.0),
                               min_rel_height=getattr(method.regions, "min_rel_height", 0.002))
        truth = vor.TRUTH[key]
        home: dict[int, list] = {}
        for name, mass, rt, kind in truth:
            inside = [i for i, r in enumerate(regions) if r.start - 0.05 <= rt <= r.end + 0.05]
            if not inside:
                out["envelopes"].append({"run": key, "species": name, "kind": kind, "rt": rt,
                                         "no_region": True})
                continue
            i = min(inside, key=lambda i, rt=rt: abs(0.5 * (regions[i].start + regions[i].end)
                                                     - rt))
            home.setdefault(i, []).append((name, mass, rt, kind))
        print(f"{key}: read in {time.time() - t0:.0f} s, {len(regions)} regions, "
              f"{len(home)} with truth species", flush=True)
        for i, members in sorted(home.items()):
            r = regions[i]
            t1 = time.time()
            summed = r.summed(run)
            grid, y, sigma = spectrum_to_grid(summed.mz, summed.intensity, pol, params.grid_step,
                                              params.grid_mz_max, summed.is_profile, inst,
                                              frames=summed.meta.get("frames"))
            preds = {}
            secs = {}
            for name, est in estimators.items():
                t2 = time.time()
                preds[name] = est.predict_grid(grid, y, sigma).top1_z.copy()
                secs[name] = time.time() - t2
            # every truth species eluting in the region is labelled (unscored ones too)
            species = []
            for name, mass, rt, kind in truth:
                if r.start - 0.1 <= rt <= r.end + 0.1:
                    masses = mass if isinstance(mass, tuple) else (mass,)
                    species.append((name, tuple(float(m) for m in masses),
                                    {"run": key, "kind": kind, "rt": rt,
                                     "home": any(nm == name for nm, *_ in members)}))
            envs, labels = envlib.measure(grid, y, sigma, species)
            envlib.score(envs, y, sigma, preds)
            frames = summed.meta.get("frames") or []
            _act, count, level = envlib.active_map(y, sigma)
            out["regions"].append({
                "run": key, "region": i, "start": float(r.start), "end": float(r.end),
                "scans": len(frames), "grid_bins": int(y.size), "mz_lo": float(grid.mz_min),
                "mz_hi": float(grid.mz_max), "sigma": float(sigma),
                "zero_frac": float((y <= 0).mean()), "level_bins": int((level > 0.5).sum()),
                "window_level_bins_max": int(count.max()),
                "labelled_bins": int((labels > 0).sum()), "clash_bins": int((labels < 0).sum()),
                "seconds": secs})
            for e in envs:
                e["region"] = i
                e["n_scans"] = len(frames)
                e.pop("_bins")
                out["envelopes"].append(e)
            print(f"  region {i} {r.start:.2f}-{r.end:.2f} ({len(frames)} scans): "
                  f"{len(envs)} envelopes of {len(species)} species, "
                  f"{int((labels > 0).sum())} labelled bins, {time.time() - t1:.0f} s", flush=True)
        del run
    common.write_json(args.out, out)


if __name__ == "__main__":
    main()
