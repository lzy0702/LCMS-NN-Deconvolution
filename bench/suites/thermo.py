"""Thermo suite: the pipeline on each Thermo run found, with the comb and with each network.

One job per (run, predictor): bench/tools/vrun.py with the orbitrap_iprp_neg_oligo method, the
run's target species (data/thermo/targets_*.json) and the extracted-ion check, writing
``<results>/thermo/runs/<RUN>_<label>.json`` (scored against the truth of
scripts/validate_oligo_runs.py: the paper's and the SI's species found, the reports nothing
confirms). The bars suite classifies and tabulates these records.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from tools import common  # noqa: E402

NAME = "thermo"
TOOLS = BENCH / "tools"
DATA = BENCH / "data" / "thermo"
#: the runs in the order their jobs are listed (the largest first)
ORDER = ("RGA", "RNC", "HNC", "HGA")
TARGETS = {"HNC": "targets_HNC.json", "RNC": "targets_RNC.json", "RGA": "targets_none.json",
           "HGA": "targets_none.json"}
#: peak memory (GB) of one vrun.py job, comb or network. Measured (peak RSS, onnxruntime on the
#: processor, 2 threads, 4-core Linux): the comb 0.26 (HGA), 0.30 (HNC), 0.35 (RNC) and 0.48
#: (RGA); a window-view network 1.69 (HGA), 1.73 (HNC), 1.77 (RNC) and 1.84 (RGA). With room
#: for more threads, a comb-view network and a GPU provider's host-side buffers.
MEM_GB = {"comb": 1.0, "net": 2.5}
#: wall time of those runs there (s): the comb 8 (HGA), 25 (HNC), 28 (RNC), 20 (RGA); the
#: window-view network 36 (HGA), 87 (HNC), 126 (RNC), 79 (RGA)


def slug(label: str) -> str:
    """``label`` as a file-name part."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "net"


def networks(cfg: dict) -> list[dict]:
    """cfg's networks with unique labels and file tags: [{"label", "onnx", "console_log",
    "tag", "sha256"}]. A label "comb" (the baseline's) or one seen before gets a suffix."""
    out: list[dict] = []
    labels: set[str] = {"comb"}
    tags: set[str] = {"comb"}
    for m in cfg.get("models") or []:
        label = str(m["label"])
        base, n = label, 1
        while label.lower() in {x.lower() for x in labels}:
            n += 1
            label = f"{base}_{n}"
        tag, n = slug(label), 1
        while tag.lower() in {x.lower() for x in tags}:
            n += 1
            tag = f"{slug(label)}_{n}"
        labels.add(label)
        tags.add(tag)
        out.append({"label": label, "onnx": str(m["onnx"]), "console_log": m.get("console_log"),
                    "tag": tag, "sha256": m.get("sha256")})
    return out


def fingerprints(nets: list[dict]) -> list[list]:
    """[label, sha256] of each network, sorted: a job's "inputs" (its output is redone when a
    network is added, removed or replaced)."""
    return sorted([m["label"], m.get("sha256")] for m in nets)


def runs(cfg: dict) -> dict[str, Path]:
    """The Thermo runs found under cfg["thermo_dir"] by key, in ORDER (HGA alone in smoke)."""
    found = common.find_thermo_raws(cfg["thermo_dir"]) if cfg.get("thermo_dir") else {}
    keys = [k for k in ORDER if k in found]
    if cfg.get("smoke") and keys:
        keys = ["HGA"] if "HGA" in keys else keys[-1:]
    return {k: found[k] for k in keys}


def run_dir(cfg: dict) -> Path:
    return Path(cfg["results_dir"]) / "thermo" / "runs"


def jobs(cfg: dict) -> list[dict]:
    out_dir = run_dir(cfg)
    log_dir = Path(cfg["results_dir"]) / "thermo" / "logs"
    threads = int(cfg.get("threads_per_job", 4))
    preds = [("comb", None, "comb", None),
             *((m["label"], m["onnx"], m["tag"], m["sha256"]) for m in networks(cfg))]
    out = []
    for rk, raw in runs(cfg).items():
        for label, onnx, tag, sha in preds:
            dest = out_dir / f"{rk}_{tag}.json"
            cmd = ["{python}", str(TOOLS / "vrun.py"), str(raw),
                   "--targets", str(DATA / TARGETS[rk]), "--confirm",
                   "--threads", str(threads), "--providers", str(cfg.get("providers", "auto")),
                   "--out", str(dest)]
            if onnx is not None:
                cmd += ["--model", f"{label}={onnx}"]
            # the label too: the record holds it and the bars tools find the record by it, so
            # a network relabelled onto the same file (a label differing in case on Windows, or
            # in characters slug() drops) is evaluated again
            out.append({"name": f"thermo/vrun/{rk}/{tag}", "cmd": cmd, "out": str(dest),
                        "log": str(log_dir / f"vrun_{rk}_{tag}.log"), "threads": threads,
                        "mem_gb": MEM_GB["comb" if onnx is None else "net"], "after": [],
                        "inputs": None if onnx is None else {"network": sha, "label": label}})
    return out


def summarize(cfg: dict) -> str:
    found = runs(cfg)
    lines = ["## Thermo runs (vrun.py)", ""]
    if not found:
        return "\n".join(lines + [f"No Thermo .raw files found under {cfg.get('thermo_dir')}.", ""])
    lines += ["| run | predictor | paper | SI | targets found | reported | unconfirmed | "
              "pipeline s |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    preds = [("comb", "comb"), *((m["label"], m["tag"]) for m in networks(cfg))]
    for rk in found:
        for label, tag in preds:
            path = run_dir(cfg) / f"{rk}_{tag}.json"
            try:
                d = json.loads(path.read_text())
            except (OSError, ValueError):
                lines.append(f"| {rk} | {label} | pending | | | | | |")
                continue
            tg = d.get("targets") or {}
            found_t = f"{sum(f is not None for f in tg.values())}/{len(tg)}" if tg else "-"
            lines.append(f"| {rk} | {label} | {d['paper'][0]}/{d['paper'][1]} | "
                         f"{d['si'][0]}/{d['si'][1]} | {found_t} | {d['n_reported']} | "
                         f"{d['unconfirmed']} | {d['run_s']:.0f} |")
    return "\n".join(lines + [""])
