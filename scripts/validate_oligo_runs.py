"""Validate the pipeline on the published IP-RPLC / IP-HILIC oligonucleotide runs.

The four Thermo runs (``IPRPLC_NC_mix``, ``IPRPLC_GalNAc_conjugated_mix``, ``IPHILIC_NC_mix``,
``IPHILIC_GalNAc_conjugated_mix``) are those of Tutis et al., "Ion-Pairing Hydrophilic
Interaction Chromatography for Impurity Profiling of Therapeutic Phosphorothioated
Oligonucleotides", Anal. Chem. 2025, 97, 15717-15726 (https://doi.org/10.1021/acs.analchem.5c01407).
A GalNAc-conjugated, fully phosphorothioated 16-mer (FLP) was spiked with 2 % each of its n-1,
n+1, PS-to-PO (PO), deaminated (DA) and non-conjugated (NC) forms; the NC product was spiked
with 2 % of its deaminated form (DA-NC) and carries its own synthesis impurities.

Expected species: the average masses of the paper's Table 1 (confirmed from the sequences of
its Supporting Information, Table S1), at the retention times of the SI's Figures S15-S21 or,
where those give none, where extracted-ion chromatograms of the species peak at two or more
charge states together. For the NC product, SI Figure S19 lists 17 synthesis impurities with
their retention times under both chromatographies; their masses are given there as the neutral
mass of an extracted isotope peak, which is converted to an average mass with an isotope model of
the NC composition (C160H205N60O78P15S15); as the figure does not say which isotopologue was
extracted, the two most abundant readings are both accepted (for NC itself the first reproduces
5161.56 -> 5162.25 Da). Two of the figure's labels are corrected: the z = 2 masses
were computed with three protons (1.007 Da too high), and the label of m/z 1281.4753 (3847.45 Da
at z = 3) repeats another row's mass. Species that co-elute with a stronger one less than
1.5 Da away (DA with FLP, DA-NC with NC under IP-RPLC) cannot be told apart by mass and are
listed but not scored.

Every reported species that matches no expected one is first compared with the species found:
a form of one (+Na, +K, their combinations, or the unassigned +201.4 Da form), a charge harmonic
of one (half, a third, double or triple its mass at the same time), or the same molecule at
another time (within 0.3 Da). The rest are checked the model-free way: "EIC-confirmed" when two
of their charge states have extracted-ion chromatograms with a peak within 0.25 min of the
reported apex, standing out from their surroundings (three times the median within 1.5 min),
and co-vary (r >= 0.7 within 0.6 min). This is a consistency check, not truth - a false charge
ladder built from co-eluting background can pass it, and a species whose strongest charge
states share their m/z windows with other signal can fail it - but every species here is
lighter than 8 kDa, where it is informative.

    python scripts/validate_oligo_runs.py IPRPLC_NC_mix.raw IPHILIC_NC_mix.raw ... \
        [--single-charge] [--model charge_net.onnx]
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

#: elemental composition per dalton of the NC product (C160H205N60O78P15S15, 5162.25 Da)
_NC = {"C": 160, "H": 205, "N": 60, "O": 78, "S": 15}


def peak_to_averages(peak_mass: float) -> tuple[float, float]:
    """Average masses of an NC-like oligonucleotide whose extracted isotope peak is given.

    Heavy-isotope counts are Poisson (13C, 2H, 15N, 17O and 33S add one nominal mass unit, 18O
    and 34S two). Figure S19 does not say which isotopologue each extracted m/z is, and near
    2.9 or 4.5 kDa the two most abundant ones differ by a few percent, so the analysts' choice
    could be either: both readings are returned, most abundant first. For NC itself the first
    reproduces 5161.56 -> 5162.25 Da.
    """
    from math import exp, lgamma, log

    f = peak_mass / 5162.25
    n = {k: v * f for k, v in _NC.items()}
    plus1 = [(n["C"] * 0.0107, 1.0033548), (n["H"] * 0.000115, 1.0062767),
             (n["N"] * 0.00364, 0.9970349), (n["O"] * 0.00038, 1.0042169),
             (n["S"] * 0.0075, 0.9993878)]
    plus2 = [(n["O"] * 0.00205, 2.0042449), (n["S"] * 0.0425, 1.9957959)]
    lam1, lam2 = sum(x for x, _ in plus1), sum(x for x, _ in plus2)
    mean_shift = sum(x * d for x, d in plus1 + plus2)
    per_unit = mean_shift / (lam1 + 2 * lam2)

    def pois(k: int, lam: float) -> float:
        return exp(k * log(lam) - lam - lgamma(k + 1)) if lam > 0 else float(k == 0)

    prob = [sum(pois(k - 2 * j, lam1) * pois(j, lam2) for j in range(k // 2 + 1))
            for k in range(30)]
    first, second = sorted(range(30), key=lambda k: -prob[k])[:2]
    return (peak_mass + mean_shift - first * per_unit, peak_mass + mean_shift - second * per_unit)


def _si(name: str, mz: float, z: int, rt: float) -> tuple:
    """A species of SI Figure S19, given by its extracted isotope peak (m/z, charge)."""
    return (name, tuple(round(m, 2) for m in peak_to_averages(z * mz + z * MP)), rt, "si")


#: (name, average mass in Da, retention time in min, kind); kind "paper" = named in the paper,
#: "si" = listed in its Supporting Information, "unscored" = unresolvable by mass
#: (Fig. S19 rows: m/z, charge, IP-HILIC and IP-RPLC retention times)
_S19 = [("NC - dG (4816.6)", 1604.5063, 3, 12.48, 17.54),
        ("4512.6", 1503.1786, 3, 12.87, 16.38), ("3527.5", 1174.8136, 3, 10.56, 13.15),
        ("3847.4", 1281.4753, 3, 10.68, 14.59), ("3206.4", 1067.7945, 3, 10.56, 11.48),
        ("3207.4", 1068.1222, 3, 9.54, 12.14), ("2887.4", 961.4484, 3, 9.50, 10.42),
        ("2558.3", 1278.1569, 2, 8.59, 8.87), ("2228.3", 1113.1503, 2, 7.96, 7.22),
        ("1908.3", 953.1282, 2, 8.04, 5.52), ("1579.2", 788.6150, 2, 7.69, 4.14),
        ("1883.3", 940.6252, 2, 6.27, 5.56), ("1563.2", 780.61, 2, 6.35, 4.14),
        ("1234.2", 616.0963, 2, 5.72, 3.12)]
#: the paper's n-1 ("4832.56 Da (-dG)") and n-2 ("4503.53 Da (-2x dG)") are 329.3 and 658.6 Da
#: below NC: one or two dA phosphorothioates (or dG phosphodiesters), whose average masses
#: by composition are these (the extracted peaks are the third isotopologue of each)
_N1 = 4832.98
_N2 = 4503.71
TRUTH = {
    "IPHILIC_GalNAc": [
        ("FLP", 6793.98, 16.40, "paper"), ("n-1", 6448.71, 15.94, "paper"),
        ("DA", 6794.96, 15.94, "paper"), ("PO", 6777.92, 16.98, "paper"),
        ("n+1", 7139.25, 16.98, "paper"), ("NC", 5162.25, 13.72, "paper")],
    "IPHILIC_NC": [
        ("NC", 5162.25, 13.41, "paper"), ("DA-NC", 5163.23, 12.75, "paper"),
        ("PO-NC", 5146.21, 14.11, "paper"), ("n-1 (-329.3)", round(_N1, 2), 13.03, "paper"),
        ("n-2 (-658.6)", round(_N2, 2), 12.60, "paper"),
        ("4842 (Fig. S21)", 4842.5, 12.75, "si")]
    + [_si(n, mz, z, rt_h) for n, mz, z, rt_h, _rt_r in _S19],
    "IPRPLC_GalNAc": [
        ("FLP", 6793.98, 19.21, "paper"), ("PO", 6777.92, 19.21, "paper"),
        ("n+1", 7139.25, 20.34, "paper"), ("n-1", 6448.71, 18.71, "paper"),
        ("NC", 5162.25, 20.22, "paper"), ("DA (co-elutes with FLP)", 6794.96, 19.21, "unscored")],
    "IPRPLC_NC": [
        ("NC", 5162.25, 18.21, "paper"), ("PO-NC", 5146.21, 18.09, "paper"),
        ("n-1 (-329.3)", round(_N1, 2), 17.54, "paper"),
        ("n-2 (-658.6)", round(_N2, 2), 16.55, "paper"),
        ("DA-NC (co-elutes with NC)", 5163.23, 18.21, "unscored")]
    + [_si(n, mz, z, rt_r) for n, mz, z, _rt_h, rt_r in _S19],
}
#: mass differences that make a report a form of a found species rather than a new one
FORMS = {"+Na": 21.98194, "+K": 37.95588, "+2Na": 43.96388, "+Na+K": 59.93782,
         "+3Na": 65.94582, "+2K": 75.91176, "+201.4 (unassigned)": 201.4}
#: the same molecule reported at another retention time agrees to well within this (Da)
SAME_TOL = 0.3


def run_key(path: Path) -> str:
    name = path.name.upper()
    lc = "IPHILIC" if "HILIC" in name else "IPRPLC"
    return f"{lc}_{'GalNAc' if 'GALNAC' in name else 'NC'}"


#: the extracted-ion check (minutes): a charge state counts when its trace peaks (is highest
#: within EIC_LOCAL) within EIC_APEX of the reported apex, at EIC_CONTRAST times the trace's
#: median within EIC_WINDOW or more; the two strongest such charge states must co-vary
#: (Pearson r >= EIC_R within EIC_COVARY), as the charge states of one molecule do
EIC_WINDOW, EIC_APEX, EIC_LOCAL, EIC_CONTRAST, EIC_COVARY, EIC_R = 1.5, 0.25, 0.1, 3.0, 0.6, 0.7


def joint_eic_peak(run, mass: float, rt: float) -> bool:
    """Two charge states of ``mass`` (negative mode) elute together, with a peak at ``rt``.

    Each charge state's extracted-ion chromatogram (m/z 400-2500, +-1.2 Da) is judged against
    its own surroundings: it must have a peak within EIC_APEX of ``rt`` standing EIC_CONTRAST
    times above its median within EIC_WINDOW. Neither the strongest signal its m/z window
    carries elsewhere in the run (the injection peak, the column wash) nor an isomer eluting
    a few tenths of a minute away counts against it. The two strongest charge states that pass
    must co-vary. On the four runs this confirms 47 of the 51 tabulated species at their
    tabulated times and 4 of 240 random (mass, time) pairs.
    """
    frames = run.frames()
    times = np.array([f.rt for f in frames])
    idx = np.flatnonzero(np.abs(times - rt) <= EIC_WINDOW)
    if idx.size < 3:
        return False
    tt = times[idx]
    near = np.flatnonzero(np.abs(tt - rt) <= EIC_APEX)
    if near.size == 0:
        return False
    passing = []
    for z in range(1, 80):
        if not 400 <= (mass - z * MP) / z <= 2500:
            continue
        lo, hi = (mass - z * MP - 1.2) / z, (mass - z * MP + 1.2) / z
        trace = np.array([float(frames[i].intensity[np.searchsorted(frames[i].mz, lo):
                                                    np.searchsorted(frames[i].mz, hi)].sum())
                          for i in idx])
        j = near[int(np.argmax(trace[near]))]
        if (trace[j] > 0 and trace[j] >= trace[np.abs(tt - tt[j]) <= EIC_LOCAL].max()
                and trace[j] >= EIC_CONTRAST * np.median(trace)):
            passing.append((trace[j], z, trace))
    if len(passing) < 2:
        return False
    (_, _, a), (_, _, b) = sorted(passing, key=lambda p: -p[0])[:2]
    cov = np.abs(tt - rt) <= EIC_COVARY
    if cov.sum() < 3 or a[cov].std() == 0 or b[cov].std() == 0:
        return False
    return bool(np.corrcoef(a[cov], b[cov])[0, 1] >= EIC_R)


def score(key: str, species: list, confirm) -> dict:
    """Match ``species`` (objects with ``mass`` and ``rt_apex``) to the truth of run ``key``.

    ``confirm(mass, rt)`` decides whether an unmatched report is EIC-confirmed.
    """
    truth = TRUTH[key]
    used: set[int] = set()
    rows = []
    for name, mass, rt, kind in sorted(truth, key=lambda t: t[3] != "paper"):
        cands = mass if isinstance(mass, tuple) else (mass,)

        def dist(sp, cands=cands):
            return min(abs(sp.mass - c) for c in cands)
        hits = [i for i, s in enumerate(species) if i not in used
                and abs(s.rt_apex - rt) <= RT_TOL and dist(s) <= MASS_TOL]
        best = min(hits, key=lambda i: dist(species[i])) if hits else None
        if best is not None and kind != "unscored":
            used.add(best)
        rows.append((name, mass if best is None else min(cands, key=lambda c: abs(
            species[best].mass - c)), rt, kind, None if best is None else species[best]))
    found = [(s.mass, s.rt_apex) for *_, s in rows if s is not None]
    rows = [(n, m[0] if isinstance(m, tuple) else m, r, k, s) for n, m, r, k, s in rows]
    others = []
    for i, s in enumerate(species):
        if i in used:
            continue
        why = None
        for m, t in found:
            if abs(s.mass - m) <= SAME_TOL:
                why = f"same molecule as {m:.2f} (other time)"
            for label, d in FORMS.items():
                if abs(s.mass - m - d) <= MASS_TOL and abs(s.rt_apex - t) <= RT_TOL:
                    why = f"{label} form of {m:.2f}"
            # a charge harmonic: the found species' envelopes read at z/k or k*z; it passes the
            # extracted-ion check because its ions are the found species' own
            for k in (2, 3):
                if abs(s.rt_apex - t) <= RT_TOL and (abs(s.mass - m / k) <= MASS_TOL
                                                      or abs(s.mass - m * k) <= k * MASS_TOL):
                    why = f"harmonic of {m:.2f}"
        if why is None:
            why = "EIC-confirmed" if confirm(s.mass, s.rt_apex) else "unconfirmed"
        others.append((s, why))
    return {"rows": rows, "others": others}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--method", default="orbitrap_iprp_neg_oligo")
    ap.add_argument("--single-charge", action="store_true",
                    help="accept species seen at one charge state with resolved isotopes")
    ap.add_argument("--model", default=None,
                    help="charge network (.onnx) to use instead of the comb estimator")
    args = ap.parse_args()
    totals = {"paper": [0, 0], "si": [0, 0]}
    kinds: dict[str, int] = {}
    n_reported = 0
    for path in args.raw:
        key = run_key(path)
        method = Method.load(args.method)
        method.deconvolution.single_charge = args.single_charge
        run = read_thermo_raw(path)
        species = process_run(run, method, model_path=args.model).species
        top = max((s.total_intensity for s in species), default=1.0)
        n_reported += len(species)
        sc = score(key, species, lambda m, t, run=run: joint_eic_peak(run, m, t))
        print(f"== {key} ({path.name}): {len(species)} species reported")
        for name, mass, rt, kind, s in sc["rows"]:
            if kind in totals:
                totals[kind][1] += 1
                totals[kind][0] += s is not None
            got = "-" if s is None else (f"{s.mass:.2f} ({s.mass - mass:+.2f} Da) at {s.rt_apex:.2f} min, "
                                         f"{100 * s.total_intensity / top:.2f} %")
            print(f"   {kind:8s} {name:26s} {mass:9.2f} @ {rt:5.2f}: {got}")
        for s, why in sc["others"]:
            k = (why.split(" form")[0] if " form" in why
                 else "harmonic" if why.startswith("harmonic")
                 else "same molecule" if why.startswith("same molecule") else why)
            kinds[k] = kinds.get(k, 0) + 1
            print(f"   other    {s.mass:9.2f} @ {s.rt_apex:5.2f} "
                  f"({100 * s.total_intensity / top:.3f} %): {why}")
    print(f"\npaper species found {totals['paper'][0]}/{totals['paper'][1]}, SI-listed species "
          f"{totals['si'][0]}/{totals['si'][1]}; {n_reported} reported, of the others: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
