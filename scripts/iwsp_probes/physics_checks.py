"""Numerical checks behind the primitives and circuits of docs/iwsp_transformer.md (4.2, 4.5).

    idioms     the MLP comparator: relu(K(a-b)) - relu(K(a-b) - 1) against 1 - relu(1 - relu(K(a-b)))
               over differences from 1e-13 to 25, and the select idiom on random pairs
    pointer    the pointer head's score in float64 and float32, in the dot-product and the distance
               form, at integer targets on the 91 525-bin grid; its leakage bounds
    interp     the sink-free softmax interpolation head: weights 1 - f and f on the bracketing keys,
               with positions relative to the left key and with absolute grid positions
    isobaric   FLP and NC against their + 0.984 Da forms (DA, DA-NC) as z = 4 grid templates: cosine
               and Gram condition number
    consensus  how far one Gaussian mean-shift step moves an isotopologue token towards its
               envelope's centre, for the decoder's 0.6 Da kernel and an envelope-wide one

    python scripts/iwsp_probes/physics_checks.py idioms
"""

from __future__ import annotations

import argparse

import numpy as np

MP = 1.00727646688


def relu(x):
    return np.maximum(x, 0.0)


def cmd_idioms(args) -> None:
    diffs = (1e-13, 1e-9, 1e-3, 0.5, 5.0, 25.0)
    print("a - b:", " ".join(f"{d:g}" for d in diffs))
    for K in (1e12, 1e16, 1e20, 1e100):
        old = [float(relu(K * d) - relu(K * d - 1.0)) for d in diffs]
        new = [float(1.0 - relu(1.0 - relu(K * d))) for d in diffs]
        print(f"K = {K:g}: relu(Kd) - relu(Kd - 1) -> {old}; 1 - relu(1 - relu(Kd)) -> {new}")
    print("1 - relu(1 - relu(K d)) at d = 0 and d = -1 (K = 1e100):",
          [float(1.0 - relu(1.0 - relu(1e100 * d))) for d in (0.0, -1.0)])
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 100, 10**6), rng.uniform(0, 100, 10**6)
    f = rng.integers(0, 2, 10**6).astype(float)
    M = 1e6
    sel = relu(a - M * (1 - f)) + relu(b - M * f)
    print(f"select relu(a - M(1-f)) + relu(b - M f) for 0 <= a, b < M, M = 1e6: largest error over "
          f"1e6 random pairs {np.max(np.abs(sel - np.where(f > 0, a, b))):.3g}")


def cmd_pointer(args) -> None:
    beta = 400.0
    rng = np.random.default_rng(1)
    q = rng.integers(0, 91525, 200000).astype(float)
    p = q + rng.integers(-2, 3, q.size)
    exact = -beta * (q - p) ** 2
    for dt in (np.float64, np.float32):
        qq, pp, b = q.astype(dt), p.astype(dt), dt(beta)
        dot = (2 * b * qq) * pp - b * pp * pp - b * qq * qq  # [2 beta q, -beta].[p, p^2] - beta q^2
        dist = -b * (qq - pp) ** 2
        err_dot = np.max(np.abs(dot.astype(float) - exact))
        err_dist = np.max(np.abs(dist.astype(float) - exact))
        hit = exact == 0
        wrong = np.sum((dot.astype(float) > -beta / 4) != hit)
        print(f"{np.dtype(dt).name}: dot-product form error up to {err_dot:.3g} "
              f"(hit or miss on the wrong side of the sink in {wrong} of {q.size}); distance form "
              f"error up to {err_dist:.3g}")
    print(f"beta = {beta:g}, sink -beta/4: a hit beats the sink by {beta / 4:g}, a key one bin off "
          f"loses to it by {3 * beta / 4:g}; with the target empty, keys one and two bins off leak "
          f"at most 2 x 5 (e^(-3 beta/4) + e^(-15 beta/4)) = "
          f"{10 * (np.exp(-0.75 * beta) + np.exp(-3.75 * beta)):.3g} (s <= 5), below the 1e-100 floor "
          f"for beta > {4 / 3 * np.log(10 * 1e100):.1f}")


def cmd_interp(args) -> None:
    beta = 400.0
    rng = np.random.default_rng(2)
    worst_rel = worst_abs = 0.0
    keys = np.arange(-3, 5, dtype=float)
    for _ in range(20000):
        f = float(rng.uniform(1e-6, 1 - 1e-6))
        i0 = float(rng.integers(3, 91520))
        c = 0.5 + np.log(f / (1 - f)) / (2 * beta)
        for base, slot in ((0.0, "rel"), (i0, "abs")):
            s = -beta * ((base + keys) - (base + c)) ** 2
            w = np.exp(s - s.max())
            w /= w.sum()
            err = max(abs(w[3] - (1 - f)), abs(w[4] - f))
            if slot == "rel":
                worst_rel = max(worst_rel, err)
            else:
                worst_abs = max(worst_abs, err)
    print(f"beta = {beta:g}, 0 < f < 1, keys i0-3 .. i0+4, centre i0 + 1/2 + ln(f/(1-f))/(2 beta): "
          f"largest weight error {worst_rel:.2g} with positions relative to i0, {worst_abs:.2g} "
          f"with absolute grid positions (20 000 random targets)")
    print("at f = 0 (the comb clamps a read below the grid's first bin to it) the centre is "
          "undefined: the head is then a pointer at i0")


def _templates(mass: float, z: int, resolution: float, delta: float):
    from lcmsdeconv.chem.classes import class_isotope_pattern
    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.nn.grid import LogMzGrid

    pat = class_isotope_pattern(mass, "ps_rna")
    m = np.asarray(pat.masses, float)
    a = np.asarray(pat.abundances, float)
    inst = InstrumentModel("orbitrap", resolution)
    g = LogMzGrid(400.0, 2500.0, 2e-5, -1)
    t1 = g.render_centroids((m - z * MP) / z, a, inst)
    t2 = g.render_centroids((m + delta - z * MP) / z, a, inst)
    return t1, t2


def cmd_isobaric(args) -> None:
    for name, mass in (("FLP", 6793.98), ("NC", 5162.25)):
        for res in (140000.0, 157000.0):
            t1, t2 = _templates(mass, 4, res, 0.984)
            cos = float(t1 @ t2 / np.linalg.norm(t1) / np.linalg.norm(t2))
            print(f"Orbitrap {res:,.0f} at m/z 200: {name} and {name} + 0.984 Da at z = 4 on the "
                  f"grid: cosine {cos:.3f}, Gram condition number {(1 + cos) / (1 - cos):.1f}")


def cmd_consensus(args) -> None:
    from lcmsdeconv.chem.classes import class_isotope_pattern

    for name, mass in (("FLP", 6793.98), ("NC", 5162.25)):
        pat = class_isotope_pattern(mass, "ps_rna")
        m = np.asarray(pat.masses, float)
        a = np.asarray(pat.abundances, float)
        a = a / a.sum()
        mu = float((m * a).sum())
        sd = float(np.sqrt(((m - mu) ** 2 * a).sum()))
        inst = mass / (140000.0 * (200.0 / 1700.0) ** 0.5) / 2.354820045
        for label, h in (("0.6 Da", 0.6), ("envelope-wide", float(np.hypot(sd, inst)))):
            step = h * h / (sd * sd + h * h)
            print(f"{name}: envelope sd {sd:.2f} Da; {label} kernel ({h:.2f} Da): one step moves a "
                  f"token {step:.1%} of the way to the centre, three steps {1 - (1 - step) ** 3:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=("idioms", "pointer", "interp", "isobaric", "consensus"))
    args = ap.parse_args()
    {"idioms": cmd_idioms, "pointer": cmd_pointer, "interp": cmd_interp,
     "isobaric": cmd_isobaric, "consensus": cmd_consensus}[args.check](args)


if __name__ == "__main__":
    main()
