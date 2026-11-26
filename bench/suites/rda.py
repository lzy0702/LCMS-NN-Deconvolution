"""Suite ``rda``: the Waters RDa runs of a confidential sample, one job per variant x run x
predictor (the comb and each network) and one comparison per variant after its runs.

Everything about the sample lives in ``bench/private/confidential_sample_1/``, which ships with
the package only and is never committed: ``sample.json`` (display name, formula, base
composition, type), ``suite.json`` (the runs and the variants of the method), ``smoke.json`` (the
runs of a smoke test), ``run_rda.py`` (one run, one variant, one predictor), ``compare.py`` (one
variant, every predictor with results), the reference the comparison reads, and ``data/`` (the
runs, unless ``cfg["rda_dir"]`` names another folder). Without that folder the suite has no
jobs. Nothing about the sample is written here.

Outputs under ``<results>/rda/``: ``<variant>/<predictor>/<run>/results.json`` (with
``timing.json``), ``<variant>/summary.md`` and ``summary.json``, ``content/summary.md`` and
``summary.json`` where ``suite.json`` names a content scorer (``"content"``: the variant it scores,
the script and its reference, all in the private folder), and ``logs/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from tools import common  # noqa: E402

from suites import thermo  # noqa: E402  (the networks' labels and file tags)

NAME = "rda"
SAMPLE = "confidential_sample_1"
#: peak memory (GB) of one run job: measured here (CPU provider, 2 threads; the smallest runs
#: and the largest) at 0.31-0.38 GB for the comb, 1.75-1.82 GB for a full-size window-view
#: network and 2.2 GB for a comb-view one, with room for more threads and a GPU provider's
#: host-side buffers
MEM_GB_COMB = 1.0
MEM_GB_NET = 3.5


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _setup(cfg: dict) -> tuple[Path, dict, list[str]] | None:
    """The private folder, its suite.json and the runs to do; None without the folder."""
    priv = BENCH / "private" / SAMPLE
    if not (priv / "suite.json").is_file():
        return None
    suite = _load(priv / "suite.json")
    runs = _load(priv / "smoke.json")["runs"] if cfg.get("smoke") else suite["runs"]
    return priv, suite, runs


def _data(cfg: dict, priv: Path) -> Path:
    return Path(cfg.get("rda_dir") or priv / "data")


def _compare_args(cfg: dict, priv: Path, variant: str, runs: list[str]) -> list[str]:
    root = Path(cfg["results_dir"]) / NAME / variant
    args = [str(priv / "compare.py"), "--results", str(root), "--variant", variant,
            "--out", str(root / "summary.md")]
    return args + (["--runs", *runs] if cfg.get("smoke") else [])


def _content_args(cfg: dict, priv: Path, spec: dict) -> list[str]:
    results = Path(cfg["results_dir"]) / NAME
    return [str(priv / spec["script"]), "--results", str(results / spec["variant"]),
            "--reference", str(priv / spec["reference"]),
            "--out", str(results / "content" / "summary.md")]


def jobs(cfg: dict) -> list[dict]:
    setup = _setup(cfg)
    if setup is None:
        return []
    priv, suite, runs = setup
    data = _data(cfg, priv)
    logs = Path(cfg["results_dir"]) / NAME / "logs"
    threads = int(cfg.get("threads_per_job", 4))
    preds = [("comb", None, "comb", None), *((m["label"], m["onnx"], m["tag"], m["sha256"])
                                             for m in thermo.networks(cfg))]
    out: list[dict] = []
    run_jobs: dict[str, list[str]] = {}
    for variant in suite["variants"]:
        root = Path(cfg["results_dir"]) / NAME / variant
        names = []
        for run in runs:
            raw = data / f"{run}.mzMLb"
            if not raw.is_file():
                continue  # summarize() names the missing inputs
            for label, onnx, tag, sha in preds:
                dest = root / tag / run / "results.json"
                cmd = ["{python}", str(priv / "run_rda.py"), str(raw), "--variant", variant,
                       "--providers", str(cfg.get("providers", "auto")),
                       "--threads", str(threads), "--out", str(dest)]
                if onnx is not None:
                    cmd += ["--model", f"{label}={onnx}"]
                name = f"{NAME}/{variant}/{tag}/{run}"
                out.append({"name": name, "cmd": cmd, "out": str(dest),
                            "log": str(logs / f"{variant}_{tag}_{run}.log"), "threads": threads,
                            "mem_gb": MEM_GB_COMB if onnx is None else MEM_GB_NET, "after": [],
                            "inputs": None if onnx is None else {"network": sha}})
                names.append(name)
        run_jobs[variant] = names
        if names:
            out.append({"name": f"{NAME}/{variant}/compare",
                        "cmd": ["{python}", *_compare_args(cfg, priv, variant, runs)],
                        "out": str(root / "summary.md"), "log": str(logs / f"{variant}_compare.log"),
                        "threads": 1, "mem_gb": 0.5, "after": names})
    spec = suite.get("content")
    if spec and run_jobs.get(spec["variant"]):
        out.append({"name": f"{NAME}/content", "cmd": ["{python}", *_content_args(cfg, priv, spec)],
                    "out": str(Path(cfg["results_dir"]) / NAME / "content" / "summary.md"),
                    "log": str(logs / "content.log"), "threads": 1, "mem_gb": 0.5,
                    "after": run_jobs[spec["variant"]]})
    return out


def summarize(cfg: dict) -> str:
    """Each variant's comparison table, a column per predictor ("pending" without results).

    The comparison is redone here first (it only reads the results), so that the table holds
    every predictor with results, also after a rerun that added networks or finished runs that
    had failed while the comparison job's own output was already there."""
    setup = _setup(cfg)
    if setup is None:
        return f"## {NAME}\n\nNot run: this package has no bench/private/{SAMPLE}/.\n"
    priv, suite, runs = setup
    sample = _load(priv / "sample.json")
    comp = " ".join(f"{b}{n}" for b, n in sample["base_composition"].items() if n)
    lines = [f"## {sample['display_name']}", "",
             f"{sample['type']}, {sample['formula']}, base composition {comp}; "
             f"{len(runs)} runs{' (smoke)' if cfg.get('smoke') else ''}.", ""]
    data = _data(cfg, priv)
    missing = [r for r in runs if not (data / f"{r}.mzMLb").is_file()]
    if missing:
        lines += [f"Not found in {data}: {', '.join(r + '.mzMLb' for r in missing)}.", ""]
    results = Path(cfg["results_dir"])
    for variant, spec in suite["variants"].items():
        root = results / NAME / variant
        lines += [f"### Variant {variant}", "", spec.get("description", ""), ""]
        if any(root.glob("*/*/results.json")):
            done = subprocess.run([sys.executable, *_compare_args(cfg, priv, variant, runs)],
                                  capture_output=True, text=True, timeout=600)
            if done.returncode:
                err = (done.stderr.strip().splitlines() or ["no message"])[-1]
                lines += [f"The comparison failed: {err}", ""]
        if (root / "summary.json").is_file():
            lines += [_load(root / "summary.json")["summary_md"], "",
                      f"Per run: {(root / 'summary.md').relative_to(results).as_posix()}", ""]
        else:
            lines += ["pending", ""]
    spec = suite.get("content")
    if spec:
        lines += [f"### All content (variant {spec['variant']})", "", spec.get("description", ""), ""]
        if any((results / NAME / spec["variant"]).glob("*/*/results.json")):
            done = subprocess.run([sys.executable, *_content_args(cfg, priv, spec)],
                                  capture_output=True, text=True, timeout=600)
            if done.returncode:
                err = (done.stderr.strip().splitlines() or ["no message"])[-1]
                lines += [f"The content scoring failed: {err}", ""]
        md = results / NAME / "content" / "summary.md"
        lines += [common.demote_headings(md.read_text(encoding="utf-8"), 3) if md.is_file()
                  else "pending", ""]
    return "\n".join(lines)
