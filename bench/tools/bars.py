"""The go/no-go table of the charge networks against the bars, beside the comb (same code).

Reads only a bench results folder, the Thermo inputs (bench/data/thermo: fovvar_rows.json, the
networks' visible-rung table of the real envelopes, and a0.json, arm A0's validation scores that
the synthetic validation bars are set from) and, optionally, each network's training console log
(the validation scores at its saved best epoch). Every piece shows 'pending' while its output is
missing; a value computed on part of the Thermo runs names them. While a piece is missing (the
bars suite's outputs, and the comb's and each network's records on the --expect-runs) the table
goes to bars_partial.md instead of OUT (an older OUT is removed) and the exit code is 3, so that a
rerun builds it again.

    python bench/tools/bars.py --results RESULTS [--model LABEL=NET.onnx ...]
        [--console-log LABEL=TRAIN.console.log ...] [--expect-runs HNC,RNC,RGA,HGA]
        [--data bench/data/thermo] --out bars.md

Columns: the comb and each --model label in order (default: every network label found in the
run records). RESULTS layout (the thermo and bars suites write it):
    thermo/runs/*.json          vrun.py records, one per Thermo run and predictor ("comb" or label)
    bars/blindspot.md           blindspot.py (lone envelopes)
    bars/real_env.json          real_env.py (real envelopes)
    bars/synth_<FC>_<CM>.json   synth_env.py --frames-config FC --charge-model CM
    bars/classify_arms.json     classify_arms.py (network-only unconfirmed reports)
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
from tools import common  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "thermo"
RUNS = ("HNC", "RNC", "RGA", "HGA")
LONG = {"RNC": "IPRPLC_NC", "HNC": "IPHILIC_NC", "RGA": "IPRPLC_GalNAc", "HGA": "IPHILIC_GalNAc"}
SYNTH = (("default", "real"), ("default", "default"), ("v3-real", "real"),
         ("v3-whole", "default"))
LONE_MASSES = ("4512", "1884")  # the masses the lone-envelope bar counts


def fmt(x, d=2, missing="pending"):
    return missing if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- the pieces
def lone_envelopes(path: Path):
    """(lone, extra) from blindspot.md: per estimator [right, cases, lone answers at 4512.8 (or at
    the extra mass), lone answers at 1884.5] for 4512.8 and 1884.5 Da together, and per other
    mass apart; None when the table is missing."""
    try:
        text = path.read_text()
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    if len(lines) < 2:
        return None
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    lone: dict = {}
    extra: dict = {}
    for ln in lines[2:]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        m = re.match(r"z=(\d+)", c[1]) if len(c) > 1 else None
        if not m:
            continue
        want = int(m.group(1))
        mass = c[0]
        tgt = lone if mass.startswith(LONE_MASSES) else extra.setdefault(mass, {})
        for name, cell in zip(head[2:], c[2:]):
            first = cell.split()[0] if cell.split() else ""
            got = int(first) if first.lstrip("-").isdigit() else None
            r = tgt.setdefault(name, [0, 0, [], []])
            r[0] += int(got == want)
            r[1] += 1
            if "+" not in c[1]:
                (r[2] if mass.startswith("4512") or tgt is not lone else r[3]).append(got)
    return lone, extra


def real_envelopes(path: Path, vis: dict):
    """({set name: count}, {estimator: {set name: mean share read right}}, envelopes without a
    visible-rung row) from real_env.json; None when it is missing."""
    d = load_json(path)
    if d is None:
        return None
    envs = [e for e in d["envelopes"] if not e.get("no_region") and e.get("home")
            and e["kind"] in ("paper", "si")]
    z2 = [e for e in envs if e["z"] >= 2 and e["w_int"] > 0]
    z1 = [e for e in envs if e["z"] == 1 and e["w_int"] > 0]

    def rung(e, variant):
        row = vis.get((e["run"], e["species"], e["z"]))
        return None if row is None else row.get(variant)

    unknown = sum(rung(e, "V0") is None for e in z2)
    sets = {"all z>=2": z2,
            "V0-rungless": [e for e in z2 if rung(e, "V0") == 0],
            "rung S/N>=100": [e for e in z2 if (rung(e, "V0") or 0) >= 100],
            "IPRPLC_GalNAc z>=2": [e for e in z2 if e["run"] == "IPRPLC_GalNAc"],
            "z=1": z1,
            "rungless everywhere": [e for e in z2 if rung(e, "V4") == 0],
            "species >= 5 kDa z>=2": [e for e in z2 if e.get("mass", 0) >= 5000]}
    real = {est: {k: mean([e[f"acc_int:{est}"] / e["w_int"] for e in v]) for k, v in sets.items()}
            for est in d["estimators"]}
    runs = sorted({e["run"] for e in envs})
    return {k: len(v) for k, v in sets.items()}, real, unknown, runs


def run_records(folder: Path) -> dict:
    """{(run key, label): vrun.py record} of the records in ``folder``."""
    short = {long: rk for rk, long in LONG.items()}
    out = {}
    for p in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        d = load_json(p)
        if isinstance(d, dict) and "label" in d and d.get("key") in short:
            out[(short[d["key"]], d["label"])] = d
    return out


def four_runs(records: dict, label: str):
    """Sums over the Thermo runs ``label`` has records of; None without any."""
    ds = [records[(rk, label)] for rk in RUNS if (rk, label) in records]
    if not ds:
        return None
    tg = {n: f is not None for d in ds for n, f in d["targets"].items()}
    unconf = [d["unconfirmed"] for d in ds]
    return {"paper": (sum(d["paper"][0] for d in ds), sum(d["paper"][1] for d in ds)),
            "si": (sum(d["si"][0] for d in ds), sum(d["si"][1] for d in ds)),
            "traced": (sum(tg.values()), len(tg)),
            "missed": [n for n, f in tg.items() if not f],
            "unconfirmed": None if None in unconf else sum(unconf),
            "keys": [rk for rk in RUNS if (rk, label) in records],
            "per_run": {d["key"]: (d["paper"], d["si"], d["unconfirmed"], d["n_reported"])
                        for d in ds}}


def best_epoch(path: Path | None):
    """Validation scores at the saved best epoch of a training console log: charge accuracy,
    band z = 6-15, and z >= 2 accuracy (log-SNR weighted) on time-of-flight and Orbitrap frames;
    None when the log is missing or names no saved epoch."""
    if path is None:
        return None
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    pat = re.compile(r"epoch (\d+): charge_acc_given_ion ema ([\d.]+) live ([\d.]+).*bands (\{.*\})")
    best = None
    for i, ln in enumerate(txt):
        m = pat.match(ln)
        if not m:
            continue
        cur = {"epoch": int(m.group(1)), "acc": max(float(m.group(2)), float(m.group(3)))}
        try:
            cur["z6_15"] = ast.literal_eval(m.group(4)).get("z=6-15")
        except (ValueError, SyntaxError):
            cur["z6_15"] = None
        saved = False
        for nxt in txt[i + 1:i + 12]:
            if pat.match(nxt):
                break
            for key, name in (("tof", "time-of-flight"), ("orbi", "Orbitrap")):
                if nxt.strip().startswith(name):
                    k = re.search(r"network ([\d.]+) / ([\d.]+)", nxt)
                    if k:
                        cur[key] = float(k.group(1))
            saved = saved or nxt.strip().startswith("saved ")
        if saved:
            best = cur
    return best


# ---------------------------------------------------------------- the table
def build(results: Path, labels: list[str] | None, logs: dict[str, Path], data: Path) -> str:
    out: list[str] = []
    say = out.append
    records = run_records(results / "thermo" / "runs")
    if not labels:
        labels = sorted({lab for _rk, lab in records if lab != "comb"})
    cols = ["comb", *labels]
    bars = results / "bars"
    say(f"# Charge networks against the bars\n\nColumns: {', '.join(cols)}.\n")

    # lone envelopes: the bar counts 4512.8 Da (z = 2-6) and 1884.5 Da (z = 2-4), lone and with
    # z+1 at 0.25; other masses are reported apart
    lone_extra = lone_envelopes(bars / "blindspot.md")
    lone, extra = lone_extra if lone_extra else ({}, {})
    say("## Lone envelopes\n")
    if lone_extra is None:
        say("pending\n")
    else:
        say("Right / cases; answers for the lone 4512.8 Da at z = 2..6:\n")
        for name, (k, n, seq, _s) in lone.items():
            say(f"- {name}: {k}/{n}; {seq}")
        for mass, d in extra.items():
            say(f"- {mass} Da (lone answers at its z = ...; right / cases with z+1): "
                + "; ".join(f"{name} {r[2]} {r[0]}/{r[1]}" for name, r in d.items()))
        say("")

    # real envelopes
    vis = {}
    for r in load_json(data / "fovvar_rows.json") or []:
        vis[(r["run"], r["species"], r["z"])] = r["vis"]
    real_res = real_envelopes(bars / "real_env.json", vis)
    real = real_res[1] if real_res else {}
    say("## Real envelopes\n")
    if real_res is None:
        say("pending\n")
    else:
        counts, _real, unknown, env_runs = real_res
        say(f"Intensity-weighted share read right, each envelope the same (runs: "
            f"{', '.join(env_runs)}):\n")
        say("| estimator | " + " | ".join(f"{k} ({n})" for k, n in counts.items()) + " |")
        say("| --- |" + " --- |" * len(counts))
        for est, vals in real.items():
            say(f"| {est} | " + " | ".join(fmt(x, missing="-") for x in vals.values()) + " |")
        if unknown:
            say(f"\n{unknown} envelope(s) of z >= 2 have no visible-rung row "
                f"(fovvar_rows.json) and count in no rung set.")
        say("")

    # the Thermo runs
    runs = {c: four_runs(records, c) for c in cols}
    say("## Thermo runs\n")
    for c, r in runs.items():
        if r is None:
            say(f"- {c}: pending")
            continue
        say(f"- {c} ({', '.join(r['keys'])}): paper {r['paper'][0]}/{r['paper'][1]}, SI "
            f"{r['si'][0]}/{r['si'][1]}, traced {r['traced'][0]}/{r['traced'][1]} (missed: "
            f"{', '.join(r['missed']) or '-'}), unconfirmed {r['unconfirmed']}; per run (paper, "
            f"SI, unconfirmed, reported): {r['per_run']}")
    say("")

    # network-only unconfirmed reports
    cl = load_json(bars / "classify_arms.json")
    klass: dict = {}
    cl_runs: dict = {}
    if cl is not None:
        for r in cl["reports"]:
            klass.setdefault(r["net"], collections.Counter())[r["primary"]] += 1
        cl_runs = cl.get("runs", {})
        for lab, keys in cl_runs.items():
            if keys:
                klass.setdefault(lab, collections.Counter())
    say("## Network-only unconfirmed reports by primary class\n")
    if cl is None:
        say("pending\n")
    else:
        for net, c in klass.items():
            say(f"- {net} ({', '.join(cl_runs.get(net, [])) or '?'}): {sum(c.values())}: "
                f"{dict(c)}")
        say("")

    # synthetic envelopes (z >= 2)
    syn = {}
    say("## Synthetic envelopes, z >= 2\n")
    for fc, cm in SYNTH:
        d = load_json(bars / f"synth_{fc}_{cm}.json")
        if d is None:
            say(f"- {fc} frames, {cm} charges: pending")
            continue
        envs = [e for e in d["envelopes"] if e["z"] >= 2 and e["w_int"] > 0]
        syn[(fc, cm)] = {est: mean([e[f"acc_int:{est}"] / e["w_int"] for e in envs])
                         for est in d["estimators"]}
        say(f"- {fc} frames, {cm} charges ({len(d['frames'])} frames, {len(envs)} envelopes): "
            + ", ".join(f"{k} {fmt(x, 3, '-')}" for k, x in syn[(fc, cm)].items()))
    say("")

    # training validation at the saved best epoch
    a0 = load_json(data / "a0.json") or {}
    val = {lab: best_epoch(logs.get(lab)) for lab in labels}
    say("## Training validation at the saved best epoch\n")
    say(f"- A0 (the bars' reference): {a0 or 'missing'}")
    for lab in labels:
        say(f"- {lab}: " + (str(val[lab]) if val[lab] else
                            ("no console log given" if lab not in logs else "no saved epoch read")))
    say("")

    # the bar table
    say("## Bars\n")
    say("| Check | Bar | " + " | ".join(cols) + " |")
    say("| --- | --- |" + " --- |" * len(cols))

    def row(check, bar, vals):
        say(f"| {check} | {bar} | " + " | ".join(vals) + " |")

    n_lone = max((v[1] for v in lone.values()), default=0)
    row(f"Lone envelopes right ({n_lone or 'pending'})",
        ">= 16/20 (A0/A1 table: 15 cases), 4512.8 Da follows z = 2-6",
        [f"{lone[c][0]}/{lone[c][1]}; 4512.8: {lone[c][2]}" if c in lone else "pending"
         for c in cols])
    for k, bar in (("V0-rungless", ">= 0.85"), ("all z>=2", ">= 0.88"), ("rung S/N>=100", ">= 0.90"),
                   ("IPRPLC_GalNAc z>=2", ">= 0.90"), ("z=1", ">= 0.80"),
                   ("species >= 5 kDa z>=2", "(info)")):
        row(f"Real envelopes, {k}", bar,
            [fmt(real[c][k], missing="-") if c in real else "pending" for c in cols])

    comb = runs.get("comb")
    part = "" if comb is None or len(comb["keys"]) == len(RUNS) else f" ({', '.join(comb['keys'])})"

    def cov(c):
        r = runs[c]
        return "" if len(r["keys"]) == len(RUNS) else f" ({', '.join(r['keys'])})"

    def rr(c, f):
        return f"{runs[c][f][0]}/{runs[c][f][1]}{cov(c)}" if runs.get(c) else "pending"

    if comb is not None:
        p, s, t = comb["paper"][1], comb["si"][1], comb["traced"][1]
        bar_p, bar_s, bar_t = f"{p}/{p}{part}", (f">= {s - 1}/{s}{part}" if s else "-"), \
            (f"{t}/{t}{part}" if t else "-")
    else:
        bar_p, bar_s, bar_t = "all", "all but one", "all"
    row("Thermo runs: paper", bar_p, [rr(c, "paper") for c in cols])
    row("Thermo runs: SI", bar_s, [rr(c, "si") for c in cols])
    row("Thermo runs: traced species", bar_t, [rr(c, "traced") for c in cols])
    cap = (f"<= {int(round(comb['unconfirmed'] * 1.1))}{part}"
           if comb is not None and comb["unconfirmed"] is not None else "<= comb + 10 %")

    def unconfirmed(c):
        r = runs.get(c)
        if r is None:
            return "pending"
        if c == "comb" or len(r["keys"]) == len(RUNS):
            return f"{r['unconfirmed']}{cov(c)}"
        # on part of the runs: the comb's count on the same runs beside it
        same = [records[(rk, "comb")]["unconfirmed"] for rk in r["keys"] if (rk, "comb") in records]
        beside = (f"; comb {sum(same)}" if len(same) == len(r["keys"]) and None not in same
                  else "")
        return f"{r['unconfirmed']} ({', '.join(r['keys'])}{beside})"

    row("Thermo runs: unconfirmed", cap, [unconfirmed(c) for c in cols])

    def single(lab):
        if lab not in klass:
            return "pending"
        n = sum(k for c_, k in klass[lab].items() if c_.startswith("singly"))
        keys = cl_runs.get(lab, [])
        return f"{n}" + ("" if len(keys) == len(RUNS) else f" ({', '.join(keys) or 'no run'})")

    row("Network-only singly charged background", "<= 3", ["-"] + [single(lab) for lab in labels])
    for key, name in (("tof", "Synthetic TOF z>=2 (validation)"),
                      ("orbi", "Synthetic Orbitrap z>=2 (validation)"),
                      ("z6_15", "Synthetic z = 6-15 (validation)")):
        bar = f">= A0 - 0.02 = {a0[key] - 0.02:.3f}" if key in a0 else "A0 - 0.02"
        row(name, bar, ["-"] + [fmt(val[lab].get(key), 3) if val.get(lab) and
                                val[lab].get(key) is not None else
                                ("no log" if lab not in logs else "pending") for lab in labels])
    for (fc, cm), name, bar in ((("default", "real"),
                                 "Real-like synthetic z>=2 (default frames, real charges)", ">= 0.85"),
                                (("v3-real", "real"),
                                 "Real-like synthetic z>=2 (v3 frames at real resolving power)",
                                 "(info)"),
                                (("default", "default"),
                                 "Synthetic z>=2 (default frames, default charges)", "(info)"),
                                (("v3-whole", "default"),
                                 "Synthetic z>=2 (v3 whole frames, default charges)", "(info)")):
        row(name, bar, [fmt(syn[(fc, cm)].get(c), missing="-")
                        if (fc, cm) in syn and c in syn[(fc, cm)] else "pending" for c in cols])
    return "\n".join(out) + "\n"


def pending(results: Path, labels: list[str], expect_runs: list[str]) -> list[str]:
    """The pieces of the table still missing: the bars suite's outputs, and the run records of
    the comb and of each network on the Thermo runs ``expect_runs``."""
    bars = results / "bars"
    need = [bars / "blindspot.md", *(bars / f"synth_{fc}_{cm}.json" for fc, cm in SYNTH)]
    if expect_runs:
        need.append(bars / "real_env.json")
        if labels:
            need.append(bars / "classify_arms.json")
    missing = [p.name for p in need if not (p.is_file() and p.stat().st_size > 0)]
    records = run_records(results / "thermo" / "runs")
    missing += [f"run record {rk} {lab}" for rk in expect_runs for lab in ["comb", *labels]
                if (rk, lab) not in records]
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, type=Path, help="the bench results folder")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                    help="the networks' columns, in order (the paths are not read)")
    ap.add_argument("--console-log", action="append", default=[], metavar="LABEL=PATH",
                    help="a network's training console log")
    ap.add_argument("--data", type=Path, default=DATA, help="bench/data/thermo")
    ap.add_argument("--expect-runs", default=",".join(RUNS),
                    help="Thermo run keys whose run records the table needs (comma list, empty "
                         "for none). While any piece is missing the table is written to OUT's "
                         "_partial file instead and the exit code is 3, so that a rerun of the "
                         "bench builds it again")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    expect = [k for k in args.expect_runs.split(",") if k]
    if set(expect) - set(RUNS):
        ap.error(f"--expect-runs takes {', '.join(RUNS)}")
    labels = list(common.parse_models(args.model))
    try:
        logs = {k: Path(p) for k, p in common.parse_models(args.console_log).items()}
    except ValueError as exc:
        ap.error(str(exc).replace("--model", "--console-log"))
    text = build(args.results, labels, logs, args.data)
    missing = pending(args.results, labels, expect)
    partial = args.out.with_name(f"{args.out.stem}_partial{args.out.suffix}")
    if missing:
        common.write_atomic(partial, text)
        args.out.unlink(missing_ok=True)  # an earlier complete table is not this one
        print(f"bars table incomplete, {len(missing)} piece(s) pending ({', '.join(missing)}): "
              f"written to {partial}", file=sys.stderr, flush=True)
        return 3
    common.write_atomic(args.out, text)
    partial.unlink(missing_ok=True)
    print(f"bars table written: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
