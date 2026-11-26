"""Bars suite: the charge networks against the go/no-go bars, beside the comb.

Jobs (outputs in ``<results>/bars/``): real_env.py on the Thermo runs found (real envelopes),
blindspot.py at 4512.8, 1884.5, 3206.4 and 6011.5 Da (lone envelopes), synth_env.py on four
(frames, charge model) pairs (synthetic envelopes), classify_arms.py on the thermo suite's run
records once its jobs are done (network-only unconfirmed reports), and last bars.py, which writes
bars.md from all of these, the networks' training console logs and bench/data/thermo. The table
is read from the results folder only, so it can be rebuilt at any time (summarize does, from what
exists, while bars.md is missing).
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from tools import common  # noqa: E402

from suites import thermo  # noqa: E402

NAME = "bars"
TOOLS = BENCH / "tools"
MASSES = ("4512.8", "1884.5", "3206.4", "6011.5")
#: the (frames, charge model) pairs of the original evaluation (run_eval.sh)
SYNTH = (("default", "real"), ("default", "default"), ("v3-real", "real"), ("v3-whole", "default"))
SYNTH_FRAMES = 32
#: peak memory (GB) of a job: (with one network, per further network). Measured (peak RSS,
#: onnxruntime on the processor, 2 threads, 4-core Linux): real_env on HGA 1.68 with one network
#: (the largest run adds about 0.25); blindspot 1.60 with one network, 2.42 with two; synth_env
#: 2 frames 2.26 with one network (default frames), 2.75 with two (v3 frames), and 5.2-6.1 with
#: five over 32 frames in the original evaluation, where two at once were killed in 7 GB;
#: classify_arms on HGA 1.68 with one network. With room for more threads, larger runs, a
#: comb-view network and a GPU provider's host-side buffers.
MEM = {"real_env": (2.5, 0.9), "blindspot": (2.0, 0.9), "synth_env": (3.0, 0.9),
       "classify_arms": (2.5, 0.9)}
#: wall time there (s): real_env 10 (HGA, one network; the four runs with five networks took
#: 988 in the original evaluation), blindspot 19 for one mass and one network (34 with two),
#: synth_env 11-12 for 2 frames (270-415 for 32 frames with five networks), classify_arms 8
#: (HGA, one network; 253 for the four runs and two networks)


def out_dir(cfg: dict) -> Path:
    return Path(cfg["results_dir"]) / "bars"


def _mem(kind: str, n_nets: int) -> float:
    base, per = MEM[kind]
    return round(base + per * max(n_nets - 1, 0), 1)


def jobs(cfg: dict) -> list[dict]:
    smoke = bool(cfg.get("smoke"))
    nets = thermo.networks(cfg)
    model_args = [a for m in nets for a in ("--model", f"{m['label']}={m['onnx']}")]
    threads = int(cfg.get("threads_per_job", 4))
    common_args = ["--threads", str(threads), "--providers", str(cfg.get("providers", "auto"))]
    out, logs = out_dir(cfg), out_dir(cfg) / "logs"
    raws = thermo.runs(cfg)
    # what each output is made from: these jobs read every network at once, so an output made
    # with other networks (or other Thermo runs) is redone, not kept
    nets_fp = thermo.fingerprints(nets)
    jobs_: list[dict] = []

    def job(name, cmd, dest, kind, inputs, after=()):
        jobs_.append({"name": f"bars/{name}", "cmd": ["{python}", *cmd], "out": str(dest),
                      "log": str(logs / f"{name}.log"), "threads": threads,
                      "mem_gb": _mem(kind, len(nets)), "after": list(after),
                      "inputs": {"networks": nets_fp, **inputs}})

    if raws:
        job("real_env", [str(TOOLS / "real_env.py"), *map(str, raws.values()), *model_args,
                         *common_args, "--out", str(out / "real_env.json")],
            out / "real_env.json", "real_env", {"runs": list(raws)})
    masses = MASSES[:1] if smoke else MASSES
    job("blindspot", [str(TOOLS / "blindspot.py"), "--mass", *masses, *model_args, *common_args,
                      "--out", str(out / "blindspot.md")], out / "blindspot.md", "blindspot",
        {"masses": list(masses)})
    frames = 2 if smoke else SYNTH_FRAMES
    for fc, cm in SYNTH:
        dest = out / f"synth_{fc}_{cm}.json"
        job(f"synth_{fc}_{cm}", [str(TOOLS / "synth_env.py"), "--frames", str(frames),
                                 "--frames-config", fc, "--charge-model", cm, *model_args,
                                 *common_args, "--out", str(dest)], dest, "synth_env",
            {"frames": frames})
    vruns = [j["name"] for j in thermo.jobs(cfg)] if "thermo" in cfg.get("suites", [NAME]) else []
    if nets and raws:
        job("classify_arms", [str(TOOLS / "classify_arms.py"), "--runs", str(thermo.run_dir(cfg)),
                              "--thermo-dir", str(cfg["thermo_dir"]), *model_args, *common_args,
                              "--out", str(out / "classify_arms.json")],
            out / "classify_arms.json", "classify_arms", {"runs": list(raws)}, after=vruns)
    logs_args = [a for m in nets if m.get("console_log")
                 for a in ("--console-log", f"{m['label']}={m['console_log']}")]
    jobs_.append({"name": "bars/table",
                  "cmd": ["{python}", str(TOOLS / "bars.py"), "--results", str(cfg["results_dir"]),
                          *model_args, *logs_args, "--expect-runs", ",".join(raws),
                          "--out", str(out / "bars.md")],
                  "out": str(out / "bars.md"), "log": str(logs / "table.log"), "threads": 1,
                  "mem_gb": 0.5, "after": [j["name"] for j in jobs_] + vruns,
                  "inputs": {"columns": [m["label"] for m in nets], "runs": list(raws),
                             "console_logs": [m.get("console_log") for m in nets]}})
    return jobs_


def summarize(cfg: dict) -> str:
    """bars.md when the table job has written it; else the table from the outputs that exist
    (its headings one level down, under the summary's)."""
    path = out_dir(cfg) / "bars.md"
    try:
        return common.demote_headings(path.read_text(encoding="utf-8"))
    except OSError:
        pass
    try:
        from tools import bars

        nets = thermo.networks(cfg)
        logs = {m["label"]: Path(m["console_log"]) for m in nets if m.get("console_log")}
        text = bars.build(Path(cfg["results_dir"]), [m["label"] for m in nets], logs, bars.DATA)
        return ("(bars.md not written yet: the table from the outputs that exist)\n\n"
                + common.demote_headings(text))
    except Exception as exc:  # the summary must not fail on partial outputs
        have = sorted(p.name for p in out_dir(cfg).glob("*") if p.is_file())
        return (f"## Bars\n\nbars.md is pending ({type(exc).__name__}: {exc}); outputs so far: "
                f"{', '.join(have) or 'none'}\n")
