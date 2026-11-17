"""Validate the pipeline on the published IP-RPLC / IP-HILIC oligonucleotide runs.

The four Thermo runs (``IPRPLC_NC_mix``, ``IPRPLC_GalNAc_conjugated_mix``, ``IPHILIC_NC_mix``,
``IPHILIC_GalNAc_conjugated_mix``) are those of Tutis et al., "Ion-Pairing Hydrophilic
Interaction Chromatography for Impurity Profiling of Therapeutic Phosphorothioated
Oligonucleotides", Anal. Chem. 2025, 97, 15717-15726 (https://doi.org/10.1021/acs.analchem.5c01407).
A GalNAc-conjugated, fully phosphorothioated 16-mer (FLP) was spiked with 2 % each of its n-1,
n+1, PS-to-PO (PO), deaminated (DA) and non-conjugated (NC) forms; the NC product was spiked
with 2 % of its deaminated form (DA-NC) and carries its own synthesis impurities.

Expected species are the average masses of the paper's Table 1 (or, for its tentative
assignments of NC impurities, the masses it reports), at the retention times where extracted-ion
chromatograms of the species peak at two or more charge states together. Species that co-elute
with a stronger one less than 1.5 Da away (DA with FLP, DA-NC with NC under IP-RPLC) cannot be
told apart by mass and are listed but not scored.

Every reported species that matches no expected one is checked the same model-free way: it is
"EIC-confirmed" when the extracted-ion chromatograms of its two strongest charge states peak
together within 0.25 min of its reported apex. This is a consistency check, not truth - a false
charge ladder built from co-eluting background can pass it - but every species here is lighter
than 8 kDa, where it is informative.

    python scripts/validate_oligo_runs.py IPRPLC_NC_mix.raw IPHILIC_NC_mix.raw ... [--single-charge]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lcmsdeconv.core.method import Method
from lcmsdeconv.io.thermo import read_thermo_raw
from lcmsdeconv.process import process_run

MP = 1.00727646688
MASS_TOL = 1.0  # Da; reported average masses of known species fall within 0.2 Da of theory
RT_TOL = 0.35  # min

#: (name, average mass in Da, retention time in min, kind); kind "paper" = named in the paper,
#: "eic" = not named there but a joint multi-charge EIC peak, "unscored" = unresolvable by mass
TRUTH = {
    "IPHILIC_GalNAc": [
        ("FLP", 6793.98, 16.40, "paper"), ("n-1", 6448.71, 15.96, "paper"),
        ("DA", 6794.96, 15.96, "paper"), ("PO", 6777.92, 16.98, "paper"),
        ("n+1", 7139.25, 16.98, "paper"), ("NC", 5162.25, 13.74, "paper")],
    "IPHILIC_NC": [
        ("NC", 5162.25, 13.41, "paper"), ("DA-NC", 5163.23, 12.85, "paper"),
        ("PO-NC", 5146.19, 14.11, "paper"), ("n-1 (paper: 4832.56)", 4832.56, 13.07, "paper"),
        ("n-2 (paper: 4503.53)", 4503.53, 12.60, "paper"),
        ("NC - dG", 4816.98, 12.48, "eic"), ("NC - T", 4841.99, 13.34, "eic"),
        ("NC - MeC", 4842.98, 12.83, "eic"), ("4512.6", 4512.60, 12.87, "eic"),
        ("2559.2", 2559.16, 8.63, "eic")],
    "IPRPLC_GalNAc": [
        ("FLP", 6793.98, 19.21, "paper"), ("PO", 6777.92, 19.21, "paper"),
        ("n+1", 7139.25, 20.34, "paper"), ("n-1", 6448.71, 18.71, "paper"),
        ("NC", 5162.25, 20.22, "paper"), ("DA (co-elutes with FLP)", 6794.96, 19.21, "unscored")],
    "IPRPLC_NC": [
        ("NC", 5162.25, 18.21, "paper"), ("PO-NC", 5146.19, 18.13, "paper"),
        ("n-1 (paper: 4832.56)", 4832.56, 17.50, "paper"),
        ("n-2 (paper: 4503.53)", 4503.53, 16.55, "paper"),
        ("DA-NC (co-elutes with NC)", 5163.23, 18.21, "unscored"),
        ("NC - dG", 4816.98, 17.54, "eic"), ("NC - T", 4841.99, 17.50, "eic"),
        ("NC - MeC", 4842.98, 17.78, "eic"), ("4512.6", 4512.60, 16.38, "eic"),
        ("2559.2", 2559.16, 8.87, "eic")],
}
#: mass differences that make a report a form of a found species rather than a new one
FORMS = {"+Na": 21.98194, "+K": 37.95588, "+201.4 (unassigned)": 201.4}


def run_key(path: Path) -> str:
    name = path.name.upper()
    lc = "IPHILIC" if "HILIC" in name else "IPRPLC"
    return f"{lc}_{'GalNAc' if 'GALNAC' in name else 'NC'}"


def joint_eic_peak(run, mass: float, rt: float) -> bool:
    """Two strongest charge states of ``mass`` peak together within 0.25 min of ``rt``."""
    frames = run.frames()
    times = np.array([f.rt for f in frames])
    near = np.abs(times - rt) <= 0.25
    if not near.any():
        return False
    zs = [z for z in range(1, 80) if 400 <= (mass - z * MP) / z <= 2500]
    traces = {}
    for z in zs:
        lo, hi = (mass - z * MP - 1.2) / z, (mass - z * MP + 1.2) / z
        traces[z] = np.array([float(f.intensity[np.searchsorted(f.mz, lo):
                                                np.searchsorted(f.mz, hi)].sum()) for f in frames])
    top = sorted(zs, key=lambda z: -traces[z][near].max())[:2]
    if len(top) < 2 or min(traces[z].max() for z in top) <= 0:
        return False
    a, b = (traces[z] / traces[z].max() for z in top)
    return bool(np.minimum(a, b)[near].max() >= 0.5)


def score(key: str, species: list, run) -> dict:
    truth = TRUTH[key]
    used: set[int] = set()
    rows = []
    for name, mass, rt, kind in sorted(truth, key=lambda t: t[3] != "paper"):
        tol = 1.5 if "paper:" in name else MASS_TOL
        hits = [i for i, s in enumerate(species) if i not in used
                and abs(s.rt_apex - rt) <= RT_TOL and abs(s.mass - mass) <= tol]
        best = min(hits, key=lambda i: abs(species[i].mass - mass)) if hits else None
        if best is not None and kind != "unscored":
            used.add(best)
        rows.append((name, mass, rt, kind, None if best is None else species[best]))
    found = [(s.mass, s.rt_apex) for *_, s in rows if s is not None]
    others = []
    for i, s in enumerate(species):
        if i in used:
            continue
        why = None
        for m, t in found:
            if abs(s.mass - m) <= MASS_TOL:
                why = f"same molecule as {m:.2f} (other time)"
            for label, d in FORMS.items():
                if abs(s.mass - m - d) <= MASS_TOL and abs(s.rt_apex - t) <= RT_TOL:
                    why = f"{label} form of {m:.2f}"
        if why is None:
            why = "EIC-confirmed" if joint_eic_peak(run, s.mass, s.rt_apex) else "unconfirmed"
        others.append((s, why))
    return {"rows": rows, "others": others}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--method", default="orbitrap_iprp_neg_oligo")
    ap.add_argument("--single-charge", action="store_true",
                    help="accept species seen at one charge state with resolved isotopes")
    args = ap.parse_args()
    totals = {"paper": [0, 0], "eic": [0, 0]}
    kinds: dict[str, int] = {}
    n_reported = 0
    for path in args.raw:
        key = run_key(path)
        method = Method.load(args.method)
        method.deconvolution.single_charge = args.single_charge
        run = read_thermo_raw(path)
        species = process_run(run, method).species
        top = max((s.total_intensity for s in species), default=1.0)
        n_reported += len(species)
        sc = score(key, species, run)
        print(f"== {key} ({path.name}): {len(species)} species reported")
        for name, mass, rt, kind, s in sc["rows"]:
            if kind in totals:
                totals[kind][1] += 1
                totals[kind][0] += s is not None
            got = "-" if s is None else (f"{s.mass:.2f} ({s.mass - mass:+.2f} Da) at {s.rt_apex:.2f} min, "
                                         f"{100 * s.total_intensity / top:.2f} %")
            print(f"   {kind:8s} {name:26s} {mass:9.2f} @ {rt:5.2f}: {got}")
        for s, why in sc["others"]:
            k = why.split(" form")[0] if " form" in why else why.split(" (")[0]
            kinds[k] = kinds.get(k, 0) + 1
            print(f"   other    {s.mass:9.2f} @ {s.rt_apex:5.2f} "
                  f"({100 * s.total_intensity / top:.3f} %): {why}")
    print(f"\npaper species found {totals['paper'][0]}/{totals['paper'][1]}, EIC-located species "
          f"{totals['eic'][0]}/{totals['eic'][1]}; {n_reported} reported, of the others: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
