"""Measure this computer for the developers: training steps on the GPU and the synthesis of
training spectra on the processor, each experiment in a process of its own, packed into one zip
file to send back.

    gpu_training\\profile.bat                 (Windows: the setup where needed, then this)
    bash gpu_training/profile.sh             (Linux)
    python gpu_training/profile_machine.py [--only gpu|cpu|e2e] [--minutes N] [--smoke]

About 45-50 minutes on a computer like the one the package was made for (24 logical
processors, an RTX 4080 SUPER); --minutes caps it (default DEFAULT_MINUTES): an experiment that
would not fit is skipped and recorded as such, except the environment, the GPU's batches and the
short training run, which always run. Keep the computer otherwise idle while it runs. It
changes no setting of the computer: priorities, affinities and power throttling are set for
its own processes only, which end with each experiment.

1. The environment: the processor and each logical processor's kind (performance or
   efficient), Windows and its power scheme and power mode, the memory, the GPU and its driver,
   PyTorch, nvidia-smi, and setup's record of this computer (gpu_training/machine.json).
2. The GPU (both views of the full LadderNet as train_all trains them, on train_all's recipe,
   4 spectra a step, under the 14 GB cap, on the runs' first training batches): the step time
   of each GPU speedup alone, all four, setup's choice and other head chunkings; the step split
   into its phases; the CUDA kernels and operators that take the time; memory peaks and
   allocator retries over 40 batches with several allocator settings; nvidia-smi during a
   steady minute; and for information bf16 autocast and 8 spectra a step.
3. The processor: the speed of each logical processor alone; power throttling and priority on
   one core of each kind; 20, 22 and 23 workers at once through PyTorch's DataLoader as training
   uses it, pinned or not, opted out of power throttling or not, and through an
   order-preserving dynamic dispatcher (profkit/dispatch.py); where an item's time goes.
4. A short real training run with what was found best (two epochs of 400 steps).
5. Last, what is for information or least likely to matter: bf16 autocast, 8 spectra a step,
   nvidia-smi on the comb view, three more allocator settings.

Everything goes to runs/profile_<date-time>/ (results.json with one record per experiment,
summary.txt, each experiment's log, CSV) and is packed into runs/profile_upload_<date-time>.zip:
JSON, logs and CSV only, the package folder's path replaced by <package> and the home folder's
by ~. Large intermediate files (the batches, about 1.5 GB) are deleted at the end.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu_training"))

from profkit import common  # noqa: E402
from profkit.common import THREAD_VARS, Unavailable, read_json, write_json  # noqa: E402

RUNS = ROOT / "runs"
WINDOWS = sys.platform == "win32"
#: the time budget: the plan's estimate is about 42 minutes on the computer the package was made
#: for, and this leaves room for a slower one (an experiment that would not fit is skipped)
DEFAULT_MINUTES = 60.0
GROUPS = ("gpu", "cpu", "e2e")
#: what the upload holds: files of these kinds, each at most MAX_FILE_MB
PACK_SUFFIXES = (".json", ".jsonl", ".log", ".txt", ".csv", ".gz")
MAX_FILE_MB = 25.0
#: the end of setup.log packed with the results
SETUP_LOG_TAIL = 300_000
#: an experiment's estimated seconds on the computer the package was made for, and its timeout
#: (the end-to-end check: the workers' start, two epochs of 1600 spectra at 25-40 a second,
#: their validation)
COSTS = {"env": (15, 300), "prepare": (45, 1200), "steps": (22, 900), "phases": (20, 900),
         "profile": (35, 1200), "memory": (22, 900), "smi": (75, 900), "autocast": (35, 1200),
         "batch8": (30, 1200), "cores": (110, 2400), "throttle": (50, 1800),
         "cprofile": (30, 1200), "loader": (65, 900), "e2e": (270, 1800)}
#: seconds a stopped experiment's process is given to end before the runner goes on without it
STOP_WAIT = 30.0
#: PYTORCH_CUDA_ALLOC_CONF of each memory variant (None: the default allocator)
ALLOC_VARIANTS = {"default": None, "gc08": "garbage_collection_threshold:0.8",
                  "split512": "max_split_size_mb:512",
                  "gc08_split512": "garbage_collection_threshold:0.8,max_split_size_mb:512",
                  "expandable": "expandable_segments:True",
                  "async": "backend:cudaMallocAsync"}


class Console:
    """What the runner prints, also written to runner.log."""

    def __init__(self, path: Path):
        self.path = path

    def say(self, text: str = "") -> None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:  # a console that cannot show a character
            print(text.encode("ascii", "replace").decode("ascii"), flush=True)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:  # held by another program: the console has shown it
            pass


# ------------------------------------------------------------------ the plan
def entry(eid: str, group: str, title: str, fn: str, kind: str, params: dict | None = None,
          smoke: bool = False, **more) -> dict:
    est, timeout = COSTS[kind]
    return {"id": eid, "group": group, "title": title, "fn": fn, "kind": kind,
            "params": params or {}, "estimate": max(3, est // 4) if smoke else est,
            "timeout": timeout, "needs": [], **more}


def setup_choices(views=("window", "comb")) -> dict:
    """What training uses on this computer, from setup's machine.json as train_all reads it:
    each view's GPU speedups and head chunk, the data workers, the RAM budget."""
    from train_all import BATCH, data_rate, head_chunk_for, speedups_for

    machine = read_json(ROOT / "gpu_training" / "machine.json") or {}
    gpu_gb = float(machine.get("gpu_memory_gb") or 14.0)
    workers = machine.get("workers")
    rate = data_rate(machine, int(workers)) if workers else None
    return {"machine": machine, "workers": workers,
            "views": {v: {"speedups": speedups_for(machine, v),
                          "chunk": head_chunk_for(machine, v, gpu_gb, BATCH, rate)}
                      for v in views}}


def gpu_plan(smoke: bool, choices: dict) -> tuple[list, list]:
    """The GPU experiments: those that always run, and those run last, most useful first: bf16
    autocast and 8 spectra a step (for information), nvidia-smi on the comb view, then the
    allocator settings least likely to matter."""
    from lcmsdeconv.nn.train import GPU_SPEEDUPS

    sz = ({"size": "small", "crop": 16384} if smoke else {})
    few = ({"synced": 4, "free": 2, "warm": 1, "cap_warm": 1, "cap_timed": 1} if smoke else {})
    main = [entry("gpu.prepare", "gpu", "the runs' first 40 training batches, the at-cap batch "
                  "and the weights", "profkit.gpu:prepare", "prepare",
                  {"batches": 4 if smoke else 40, **({"workers": 2} if smoke else {}), **sz},
                  smoke, essential=True)]
    later = []
    needs = ["gpu.prepare"]
    for view in ("window", "comb"):
        choice = choices["views"][view]
        chunk, greedy = choice["chunk"], list(choice["speedups"])
        configs = [("none", []), *((s, [s]) for s in GPU_SPEEDUPS), ("all", list(GPU_SPEEDUPS))]
        if greedy and greedy not in [c for _n, c in configs]:
            configs.append(("greedy", greedy))
        how = "all charges at once" if chunk is None else f"{chunk} charges at a time"
        for name, sp in configs:
            main.append(entry(f"gpu.steps.{view}.{name}", "gpu",
                              f"steps, {view} view, {'+'.join(sp) or 'no speedups'}, {how}",
                              "profkit.gpu:steps", "steps",
                              {"view": view, "speedups": sp, "chunk": chunk, "label": name,
                               **few, **sz}, smoke, needs=needs, view=view))
        greedy_name = next((n for n, c in configs if c == greedy), "greedy")
        for c in ([None, 16] + ([8, 32] if view == "window" else [])):
            if c == chunk:
                continue  # measured above with setup's choice
            tag = "unchunked" if c is None else f"chunk{c}"
            main.append(entry(f"gpu.steps.{view}.{greedy_name}.{tag}", "gpu",
                              f"steps, {view} view, {'+'.join(greedy) or 'no speedups'}, "
                              f"{'all charges at once' if c is None else f'{c} at a time'}",
                              "profkit.gpu:steps", "steps",
                              {"view": view, "speedups": greedy, "chunk": c,
                               "label": f"{greedy_name}.{tag}", **few, **sz},
                              smoke, needs=needs, view=view))
    best = {"label": "best"}
    for view in ("window", "comb"):
        main.append(entry(f"gpu.phases.{view}", "gpu", f"phases of the best step, {view} view",
                          "profkit.gpu:phases", "phases",
                          {"view": view, **best, **({"steps_n": 4, "warm": 1} if smoke else {}),
                           **sz}, smoke, needs=needs, resolve={"best": view}))
        chunk = choices["views"][view]["chunk"]
        main.append(entry(f"gpu.profile.{view}.baseline", "gpu",
                          f"profile, {view} view, no speedups", "profkit.gpu:profile", "profile",
                          {"view": view, "speedups": [], "chunk": chunk, "label": "none",
                           **({"steps_n": 1, "warm": 1} if smoke else {}), **sz}, smoke,
                          needs=needs))
        main.append(entry(f"gpu.profile.{view}.best", "gpu", f"profile, {view} view, the best",
                          "profkit.gpu:profile", "profile",
                          {"view": view, **best, **({"steps_n": 1, "warm": 1} if smoke else {}),
                           **sz}, smoke, needs=needs, resolve={"best": view}))
    last_memory = []  # combined settings, expandable segments (which Windows may lack), async
    for view, variants in (("window", ("default", "gc08", "split512")),
                           ("comb", ("default", "gc08", "split512")),
                           ("window", ("gc08_split512", "expandable", "async"))):
        for variant in variants:
            conf = ALLOC_VARIANTS[variant]
            e = entry(f"gpu.memory.{view}.{variant}", "gpu",
                      f"memory over 40 batches, {view} view, {conf or 'default allocator'}",
                      "profkit.gpu:memory", "memory",
                      {"view": view, **best, "variant": variant,
                       "batches": 4 if smoke else 40, **sz}, smoke, needs=needs,
                      resolve={"best": view},
                      env={"PYTORCH_CUDA_ALLOC_CONF": conf} if conf else {"PYTORCH_CUDA_ALLOC_CONF": None})
            (last_memory if variant in ("gc08_split512", "expandable", "async")
             else main).append(e)
    main.append(entry("gpu.smi.window", "gpu", "a steady minute with nvidia-smi, window view",
                      "profkit.gpu:smi_loop", "smi",
                      {"view": "window", **best, "seconds": 5.0 if smoke else 60.0, **sz},
                      smoke, needs=needs, resolve={"best": "window"}))
    for view in ("window", "comb"):
        later.append(entry(f"gpu.autocast.{view}", "gpu",
                           f"bf16 autocast (information), {view} view", "profkit.gpu:autocast",
                           "autocast", {"view": view, **best,
                                        **({"steps_n": 2, "warm": 1} if smoke else {}), **sz},
                           smoke, needs=needs, resolve={"best": view}))
        later.append(entry(f"gpu.batch8.{view}", "gpu", f"8 spectra a step (information), {view} "
                           f"view", "profkit.gpu:batch8", "batch8",
                           {"view": view, **best, **({"steps_n": 2, "warm": 1} if smoke else {}),
                            **sz}, smoke, needs=needs, resolve={"best": view}))
    e = entry("gpu.smi.comb", "gpu", "half a steady minute with nvidia-smi, comb view",
              "profkit.gpu:smi_loop", "smi",
              {"view": "comb", **best, "seconds": 5.0 if smoke else 30.0, **sz}, smoke,
              needs=needs, resolve={"best": "comb"})
    e["estimate"] = e["estimate"] // 2
    later.append(e)
    return main, later + last_memory


def worker_counts(logical: int, smoke: bool) -> list[int]:
    """The numbers of data workers measured: all logical processors but four, two and one (20,
    22 and 23 of 24); in a smoke run all but two and one."""
    counts = [logical - 2, logical - 1] if smoke else [logical - 4, logical - 2, logical - 1]
    return list(dict.fromkeys(max(1, n) for n in counts))


def cpu_plan(smoke: bool, logical: int, windows: bool) -> list:
    crop = {"crop": 16384} if smoke else {}
    plan = [entry("cpu.cores", "cpu", "items per second of each logical processor alone",
                  "profkit.cpu:core_map", "cores", {"items": 2 if smoke else 12, **crop}, smoke),
            entry("cpu.throttle", "cpu", "power throttling and priority, one core of each kind",
                  "profkit.cpu:throttle", "throttle",
                  {"items": 2 if smoke else 6, "reps": 1 if smoke else 2, **crop}, smoke)]
    variants = [("dataloader", False, False), ("dispatcher", False, False),
                ("dataloader", True, False), ("dispatcher", True, False)]
    if windows:
        variants.append(("dataloader", True, True))
    for n in worker_counts(logical, smoke):
        for kind, pinned, opt_out in variants:
            name = kind + ("_pinned" if pinned else "") + ("_optout" if opt_out else "")
            words = (f"{kind}{', pinned' if pinned else ''}"
                     f"{', power throttling opted out' if opt_out else ''}")
            plan.append(entry(f"cpu.loader.n{n}.{name}", "cpu", f"{n} workers, {words}",
                              "profkit.cpu:loader", "loader",
                              {"kind": kind, "workers": n, "pinned": pinned, "opt_out": opt_out,
                               **({"seconds": 4.0, "warm_rounds": 1} if smoke else {}), **crop},
                              smoke, workers=n, variant=name))
    plan.append(entry("cpu.cprofile", "cpu", "where an item's time goes (cProfile), one core of "
                      "each kind", "profkit.cpu:cprofile_items", "cprofile",
                      {"items": 2 if smoke else 6, **crop}, smoke))
    return plan


def e2e_plan(smoke: bool, logical: int, choices: dict) -> list:
    """The short real training run, essential: it runs whatever the budget, as the check that
    shows whether the data or the GPU sets training's pace."""
    allowed = [n for n in worker_counts(logical, smoke)][:2]
    fallback = int(choices.get("workers") or allowed[-1])
    params = {"view": "window", "label": "best", "workers": fallback}
    if smoke:
        params.update(size="small", epochs=1, train_len=16, val_len=2, minutes=5.0)
    return [entry("e2e.train", "e2e", "a short real training run with what was found best",
                  "profkit.e2e:train_check", "e2e", params, smoke, essential=True,
                  resolve={"best": "window", "workers": allowed})]


def build_plan(groups, env: dict, smoke: bool, only_ids=None) -> list:
    """Every experiment to run after ``env``, in order: the GPU's, the processor's, the
    end-to-end check, then the GPU's for information and the allocator settings least likely to
    matter (the first skipped should time run out)."""
    common.add_paths()
    choices = setup_choices()
    logical = int((env.get("cpu") or {}).get("logical") or os.cpu_count() or 2)
    windows = (env.get("os") or {}).get("build") is not None or WINDOWS
    gpu_main, gpu_later = gpu_plan(smoke, choices) if "gpu" in groups else ([], [])
    plan = gpu_main
    if "cpu" in groups:
        plan += cpu_plan(smoke, logical, windows)
    if "e2e" in groups:
        plan += e2e_plan(smoke, logical, choices)
    plan += gpu_later
    for e in plan:
        e["fallback"] = choices["views"].get(e.get("view") or (e.get("resolve") or {}).get("best"))
    if only_ids:
        keep = {e["id"] for e in plan if any(e["id"].startswith(p) for p in only_ids)}
        for e in plan:
            if e["id"] in keep:
                keep.update(e.get("needs") or [])
        plan = [e for e in plan if e["id"] in keep]
    return plan


# ------------------------------------------------------------------ what was found best
def best_config(records: list, view: str, fallback: dict | None = None) -> dict:
    """The fastest configuration measured for ``view`` (median wall time of a step on the
    real batches) among those whose losses and gradients agree with the step without speedups
    and whose at-cap step fitted the GPU; setup's choice (``fallback``) if none was measured."""
    best = None
    for r in records:
        if not r["id"].startswith(f"gpu.steps.{view}.") or r.get("outcome") != "ok":
            continue
        res = r.get("result") or {}
        ms = res.get("median_ms")
        agrees = (res.get("agreement") or {}).get("same") is not False
        fits = "error" not in (res.get("cap") or {})
        if ms and agrees and fits and (best is None or ms < best[0]):
            best = (ms, r)
    if best is None:
        fb = fallback or {}
        return {"speedups": list(fb.get("speedups") or []), "chunk": fb.get("chunk"),
                "label": "setup's choice (nothing measured)", "median_ms": None}
    cfg = best[1]["config"]
    return {"speedups": list(cfg.get("speedups") or []), "chunk": cfg.get("chunk"),
            "label": f"best: {best[1]['id'].split('.', 3)[-1]}", "median_ms": best[0]}


def best_workers(records: list, allowed: list, fallback: int) -> int:
    """The number of workers among ``allowed`` whose DataLoader (as training uses it) delivered
    the most; the fewer where two are within 2 %."""
    rates = {}
    for r in records:
        if (r.get("outcome") == "ok" and r["id"].startswith("cpu.loader.")
                and r.get("variant") == "dataloader" and r.get("workers") in allowed):
            rates[r["workers"]] = (r.get("result") or {}).get("items_per_second") or 0.0
    if not rates:
        return fallback
    top = max(rates.values())
    return min(n for n, v in rates.items() if v >= 0.98 * top)


# ------------------------------------------------------------------ running the experiments
class Runner:
    """Runs plan entries one at a time, each in a process of its own with a timeout, and keeps
    their records (results.json after each)."""

    def __init__(self, run_dir: Path, console: Console, smoke: bool, budget_s: float,
                 header: dict):
        self.run_dir = run_dir
        self.console = console
        self.smoke = smoke
        self.budget_s = budget_s
        self.header = header
        self.records: list = []
        self.started = time.time()
        self.interrupted = False
        self.est_done = self.real_done = 0.0
        self.save_error: str | None = None  # why results.json could not be written last time
        for sub in ("raw", "logs", "work/plan"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # the scale of the estimates on this computer, from the experiments done so far
    def scale(self) -> float:
        if self.est_done < 60:
            return 1.0
        return min(3.0, max(0.5, self.real_done / self.est_done))

    def save(self, extra: dict | None = None, final: bool = False) -> Path | None:
        """results.json with every record so far, written after each experiment. Never stops
        the profile: where the file is held (open in another program, for example) it is
        written after a later experiment, and if the last save (``final``) fails too, the
        records go to results_final.json instead, which is packed as well. Returns the file
        written, None if none was."""
        data = {**self.header, **(extra or {}), "experiments": self.records}
        try:
            write_json(self.run_dir / "results.json", data, attempts=20 if final else 4)
            self.save_error = None
            return self.run_dir / "results.json"
        except OSError as exc:
            first = self.save_error is None
            self.save_error = f"{type(exc).__name__}: {exc}"
        if not final:
            if first:
                self.console.say(f"    results.json could not be written ({self.save_error[:160]})"
                                 f"; it is written again after the next experiment")
            return None
        alt = self.run_dir / "results_final.json"
        try:
            write_json(alt, {**data, "results_json_error": self.save_error})
        except OSError as exc:
            self.console.say(f"The results could not be written ({exc}): the logs are packed")
            return None
        self.console.say(f"results.json could not be written ({self.save_error[:160]}): the "
                         f"results are in {alt.name}")
        return alt

    def record(self, e: dict, outcome: str, error: str | None = None, **more) -> dict:
        rec = {"id": e["id"], "group": e["group"], "title": e["title"], "fn": e["fn"],
               "config": e["params"], "outcome": outcome, "error": error,
               "estimate_s": e["estimate"], "timeout_s": e["timeout"], **more}
        for key in ("workers", "variant", "view"):
            if key in e:
                rec[key] = e[key]
        if e.get("env"):
            rec["env"] = e["env"]
        self.records.append(rec)
        self.save()
        return rec

    def outcome_of(self, eid: str) -> dict | None:
        return next((r for r in self.records if r["id"] == eid), None)

    def resolve(self, e: dict) -> None:
        """Fill in what depends on earlier results: the best configuration of a view, the best
        number of workers."""
        how = e.get("resolve") or {}
        if "best" in how:
            b = best_config(self.records, how["best"], e.get("fallback"))
            e["params"].update(speedups=b["speedups"], chunk=b["chunk"], label=b["label"])
        if "workers" in how:
            e["params"]["workers"] = best_workers(self.records, how["workers"],
                                                  e["params"].get("workers"))

    def run(self, e: dict, k: int, n: int) -> dict:
        rec = self.start(e, k, n)
        if rec["outcome"] in ("skipped", "unavailable") and rec.get("seconds") is None:
            self.console.say(f"[{k}/{n}] {e['id']}: {rec['outcome']} ({rec['error'][:110]})")
        return rec

    def start(self, e: dict, k: int, n: int) -> dict:
        for need in e.get("needs") or []:
            got = self.outcome_of(need)
            if got is None or got["outcome"] != "ok":
                why = "it was not run" if got is None else f"it was {got['outcome']}"
                if got and got.get("error"):
                    why += f": {str(got['error']).strip().splitlines()[-1][:200]}"
                outcome = "unavailable" if got and got["outcome"] == "unavailable" else "skipped"
                return self.record(e, outcome, f"needs {need}, and {why}")
        if self.interrupted:
            return self.record(e, "skipped", "the profile was interrupted")
        elapsed = time.time() - self.started
        if not e.get("essential") and elapsed + e["estimate"] * self.scale() > self.budget_s:
            return self.record(e, "skipped", f"the time budget ({self.budget_s / 60:.0f} min) "
                                             f"would be exceeded")
        self.resolve(e)
        left = self.remaining_estimate(k)
        self.console.say(f"[{k}/{n}] {e['title']} ({e['id']}; about {e['estimate'] * self.scale():.0f}"
                         f" s; {elapsed / 60:.1f} min done, about {left / 60:.0f} min to go)")
        rec = self.launch(e)
        self.est_done += e["estimate"]
        self.real_done += rec.get("seconds") or 0.0
        self.console.say(f"    {rec['outcome']} in {rec.get('seconds', 0):.0f} s"
                         + (f": {brief(rec)}" if brief(rec) else ""))
        return rec

    def remaining_estimate(self, k: int) -> float:
        rest = self.plan[k - 1:] if hasattr(self, "plan") else []
        return self.scale() * sum(x["estimate"] for x in rest
                                  if self.outcome_of(x["id"]) is None)

    def launch(self, e: dict) -> dict:
        """Start the experiment's process, echo its [say] lines, stop it at its timeout (with
        every process it started), and record what it left."""
        eid = e["id"]
        write_json(self.run_dir / "work" / "plan" / f"{eid}.json", e)
        log = self.run_dir / "logs" / f"{eid}.log"
        raw = self.run_dir / "raw" / f"{eid}.json"
        cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--child", eid,
               "--run-dir", str(self.run_dir)]
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        for var in THREAD_VARS:
            env.setdefault(var, "1")  # one thread per process, as training's workers
        if self.smoke:
            env[common.SMOKE_ENV] = "1"
        for k, v in (e.get("env") or {}).items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
              else {"start_new_session": True})
        t0 = time.time()
        outcome = error = None
        code = None
        with open(log, "wb") as fh:
            fh.write(f"=== {common.now_text()} {eid}\n".encode())
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    **kw)
            pos = fh.tell()
            try:
                while True:
                    code = proc.poll()
                    pos = self.echo(log, pos)
                    if code is not None:
                        break
                    if time.time() - t0 > e["timeout"]:
                        common.kill_tree(proc.pid)
                        outcome, error = "timeout", f"no result within {e['timeout']} s"
                        try:
                            proc.wait(timeout=STOP_WAIT)
                        except subprocess.TimeoutExpired:  # stuck in a driver call, for one
                            error += (f"; its process did not end within {STOP_WAIT:g} s of "
                                      f"being stopped and was left behind")
                        break
                    time.sleep(0.25)
            except KeyboardInterrupt:
                common.kill_tree(proc.pid)
                self.interrupted = True
                outcome, error = "interrupted", "Ctrl+C"
        seconds = time.time() - t0
        got = read_json(raw) if raw.exists() else None
        tail = self.tail(log)
        if outcome is None:
            if got is not None:
                outcome, error = got.get("outcome", "error"), got.get("error")
            else:
                outcome = "crashed"
                error = f"exit code {code} without a result; the end of its log:\n{tail}"
        more = {"seconds": round(seconds, 1), "exit_code": code, "log": f"logs/{eid}.log",
                "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))}
        warns = self.warnings(log)
        if warns:
            more["log_warnings"] = warns
        if got is not None:
            more["result"] = got.get("result")
            more["process"] = got.get("process")
        return self.record(e, outcome, error, **more)

    def echo(self, log: Path, pos: int) -> int:
        """Show the experiment's [say] lines written since ``pos``; returns the new position."""
        try:
            with open(log, "rb") as fh:
                fh.seek(pos)
                data = fh.read()
        except OSError:
            return pos
        end = data.rfind(b"\n")
        if end < 0:
            return pos
        for line in data[:end].decode("utf-8", errors="replace").splitlines():
            if line.startswith("[say] "):
                self.console.say("    " + line[6:])
        return pos + end + 1

    @staticmethod
    def warnings(log: Path, most: int = 10) -> list[str]:
        """The distinct warning lines of an experiment's log (an allocator setting this PyTorch
        does not support says so when CUDA starts, early in the log)."""
        try:
            text = log.read_bytes()[:4_000_000].decode("utf-8", errors="replace")
        except OSError:
            return []
        lines = [ln.strip() for ln in text.splitlines() if "warn" in ln.lower()]
        return list(dict.fromkeys(ln[:300] for ln in lines))[:most]

    @staticmethod
    def tail(log: Path, n: int = 4000) -> str:
        try:
            data = log.read_bytes()[-n:]
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace")

    def run_all(self, plan: list) -> None:
        self.plan = plan
        for k, e in enumerate(plan, start=1):
            try:
                self.run(e, k, len(plan))
            except Exception:  # a fault of the runner itself is recorded; the next one runs
                self.runner_failed(e, traceback.format_exc())

    def runner_failed(self, e: dict, text: str) -> None:
        """Record that the runner itself failed on ``e`` (its own record, if it has one yet,
        keeps its outcome and gains the error)."""
        last = text.strip().splitlines()[-1] if text.strip() else "?"
        self.console.say(f"    the runner failed on {e.get('id')}: {last[:200]}")
        try:
            rec = self.outcome_of(e.get("id"))
            if rec is None:
                self.record(e, "error", f"the runner failed:\n{text[-4000:]}")
            else:
                rec["runner_error"] = text[-4000:]
                self.save()
        except Exception:  # nothing more can be kept: the console has said it
            pass


def child_main(eid: str, run_dir: Path) -> int:
    """One experiment in this process (the runner started it): its result, or the error that
    stopped it, goes to raw/<id>.json."""
    common.add_paths()
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    e = read_json(run_dir / "work" / "plan" / f"{eid}.json") or {"id": eid, "fn": "?"}
    ctx = common.Context(run_dir, e)
    t0 = time.time()
    rc = 0
    try:
        module, name = e["fn"].split(":")
        fn = getattr(importlib.import_module(module), name)
        rec = {"outcome": "ok", "result": fn(ctx, **(e.get("params") or {}))}
    except Unavailable as exc:
        rec = {"outcome": "unavailable", "error": str(exc)}
    except BaseException:  # an error of any kind is the record (KeyboardInterrupt too)
        rec = {"outcome": "error", "error": traceback.format_exc()[-6000:]}
        traceback.print_exc()
        rc = 1
    rec["seconds"] = round(time.time() - t0, 2)
    rec["process"] = process_info()
    write_json(run_dir / "raw" / f"{eid}.json", rec)
    return rc


def process_info() -> dict:
    """The experiment process's settings, for its record."""
    from profkit import cpusets

    out = {"python": sys.version.split()[0], "affinity": cpusets.current_affinity(),
           "priority": cpusets.priority_now(),
           "env": {k: os.environ.get(k) for k in (*THREAD_VARS, "PYTORCH_CUDA_ALLOC_CONF")}}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        out["torch_threads"] = torch.get_num_threads()
    return out


# ------------------------------------------------------------------ the summary
def _ms(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def brief(rec: dict) -> str:
    """One line of an experiment's result for the console."""
    res = rec.get("result") or {}
    fn = rec.get("fn", "")
    if rec.get("outcome") != "ok":
        err = str(rec.get("error") or "").strip().splitlines()
        return err[-1][:200] if err else ""
    try:
        if fn.endswith(":steps"):
            r = res["real"]["wall_ms"]
            cap = res.get("cap") or {}
            agree = res.get("agreement") or {}
            same = "" if agree.get("same") is None else (
                "; the same loss and gradients" if agree["same"] else f"; NOT the same: "
                                                                      f"{agree.get('why')}")
            return (f"{r['median']:.1f} ms a step (p10 {r['p10']:.1f}, p90 {r['p90']:.1f}), "
                    f"free-running {res['free_running']['ms_per_step']:.1f}, at the cap "
                    + (f"{cap['wall_ms']['median']:.1f}" if "wall_ms" in cap else
                       cap.get("error", "-")[:60]) + same)
        if fn.endswith(":phases"):
            return (f"GPU {res['step_gpu_ms']['median']:.1f} ms a step, host busy "
                    f"{res['host_busy_ms']['median']:.1f}, nonzero wait "
                    f"{res['nonzero_wait_ms']['median']:.1f}: {res['verdict'].split(':')[0]}")
        if fn.endswith(":profile"):
            k = (res.get("kernels") or [{}])[0]
            busy = res.get("gpu_busy_share")
            return (f"{res['kernels_per_step']:.0f} kernels a step, {res['kernel_ms_per_step']:.1f}"
                    f" ms of kernels"
                    + (f" in a step of {res['ms_per_step']:.1f} ms ({100 * busy:.0f} % busy)"
                       if busy is not None and res.get("ms_per_step") else "")
                    + f"; top: {k.get('name', '-')[:60]}"
                    + ("; the trace was not written" if res.get("trace_error") else ""))
        if fn.endswith(":memory"):
            return (f"{res.get('peak_reserved_gb', 0):.2f} GB reserved at most, "
                    f"{res.get('alloc_retries', 0)} retries, {res.get('ooms', 0)} out of memory, "
                    f"step {_ms(res['step_ms'].get('median'))} ms")
        if fn.endswith(":smi_loop"):
            s = res.get("nvidia_smi") or {}
            util = (s.get("utilization.gpu") or {}).get("mean")
            return (f"{res['ms_per_step']:.1f} ms a step, {res['spectra_per_second']:.1f} "
                    f"spectra/s; GPU utilisation {util if util is not None else '-'} %")
        if fn.endswith(":autocast"):
            return "; ".join(f"{m} {_ms((v.get('wall_ms') or {}).get('median'))} ms"
                             for m, v in res["modes"].items())
        if fn.endswith(":batch8"):
            return "; ".join(f"{b}: {v.get('spectra_per_second', v.get('error'))} spectra/s"
                             for b, v in res["batches"].items())
        if fn.endswith(":core_map"):
            return "; ".join(f"{k}: {v['median']:.2f} items/s" for k, v in res["by_kind"].items())
        if fn.endswith(":throttle"):
            words = {"opt_out_gain_normal": "opt-out at normal priority",
                     "opt_out_gain_below_normal": "opt-out at below normal",
                     "below_normal_change_default": "below normal",
                     "below_normal_change_opt_out": "below normal, opted out"}
            return "; ".join(
                f"{k} core {v['cpu']}: " + ", ".join(f"{words.get(e, e)} {100 * x:+.1f} %"
                                                   for e, x in v["effects"].items())
                for k, v in res["cores"].items())
        if fn.endswith(":loader"):
            w = res.get("consumer_wait_ms") or {}
            return (f"{res['items_per_second']:.1f} items/s; a batch waited for "
                    f"{_ms(w.get('median'))} ms (p90 {_ms(w.get('p90'))})")
        if fn.endswith(":cprofile_items"):
            return "; ".join(f"{k} {v['wall_s']:.1f} s" for k, v in res["profiles"].items())
        if fn.endswith(":train_check"):
            last = (res.get("epochs") or [{}])[-1]
            return last.get("line", f"exit code {res.get('exit_code')}")[:200]
        if fn.endswith(":environment"):
            c = res.get("cpu") or {}
            return f"{c.get('name')}; {c.get('counts')}; CUDA {res['torch'].get('cuda')}"
        if fn.endswith(":prepare"):
            return f"{res['config']['batches']} batches in {res['seconds_making']:.0f} s"
    except (KeyError, TypeError, IndexError, AttributeError, ValueError):
        return ""
    return ""


def summary_lines(records: list, env: dict, header: dict) -> list[str]:
    """The human summary: the computer, each view's GPU steps, the processor, the end-to-end
    check, and what failed."""
    lines = []
    by_id = {r["id"]: r for r in records}
    counts: dict = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    lines.append(f"Profile of {header['started']} to {header.get('finished', '?')}"
                 f" ({header.get('minutes', 0):.1f} min): {len(records)} experiments, "
                 + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    try:
        cpu, tor, mem = env.get("cpu") or {}, env.get("torch") or {}, env.get("memory") or {}
        osr = env.get("os") or {}
        rel = osr.get("release") or {}
        scheme = (osr.get("power_scheme") or {})
        mode = ((osr.get("power_mode") or {}).get("effective") or {}).get("known_as")
        lines.append(f"Computer: {cpu.get('name')}; {cpu.get('logical')} logical processors "
                     f"{cpu.get('counts')} ({cpu.get('source')}); {mem.get('total_gb')} GB RAM")
        lines.append(f"  {osr.get('platform')}" + (f", {rel.get('DisplayVersion')}" if rel else "")
                     + (f"; power scheme {scheme.get('name') or scheme.get('known_as')}"
                        if scheme else "") + (f"; power mode {mode}" if mode else ""))
        gpus = (env.get("nvidia_smi") or {}).get("gpus") or [{}]
        name = tor.get("gpu") or gpus[0].get("name")
        gpu = (f"GPU {name}, driver {gpus[0].get('driver_version')}" if name else
               "no GPU (" + str((env.get("nvidia_smi") or {}).get("why", "")) + ")")
        lines.append(f"  {gpu}; PyTorch {tor.get('version')} (CUDA {tor.get('cuda_build')}, "
                     f"available: {tor.get('cuda')})")
    except Exception as exc:  # the summary never stops the packing
        lines.append(f"(the computer's summary failed: {exc})")
    for view in ("window", "comb"):
        steps = [r for r in records if r["id"].startswith(f"gpu.steps.{view}.")]
        if not steps:
            continue
        if not any(r["outcome"] == "ok" for r in steps):
            gpu_runs = [r for r in records if r["id"].startswith("gpu.") and r["id"].count(view)]
            lines.append(f"GPU, {view} view: not measured ({len(gpu_runs)} experiments "
                         f"{steps[0]['outcome']}: {brief(steps[0])[:160]})")
            continue
        lines.append(f"GPU, {view} view: ms a step on the runs' first 8 batches (median [p10-p90]"
                     f"; free-running; at the active-bin cap)")
        base = next((r for r in steps if r["id"] == f"gpu.steps.{view}.none"
                     and r["outcome"] == "ok"), None)
        base_ms = base and base["result"].get("median_ms")
        for r in steps:
            res = r.get("result") or {}
            name = r["id"].split(".", 3)[-1]
            if r["outcome"] != "ok":
                lines.append(f"  {name:28s} {r['outcome']}: {brief(r)[:80]}")
                continue
            w = res["real"]["wall_ms"]
            cap = res.get("cap") or {}
            gain = (f" {100 * (1 - w['median'] / base_ms):+5.1f} %" if base_ms else "")
            agree = res.get("agreement") or {}
            flag = "" if agree.get("same") is not False else "  (NOT the same results)"
            lines.append(f"  {name:28s} {w['median']:7.1f} [{w['p10']:.1f}-{w['p90']:.1f}]{gain}"
                         f"; {res['free_running']['ms_per_step']:.1f}; "
                         f"{_ms((cap.get('wall_ms') or {}).get('median'))}"
                         f"; {res.get('memory', {}).get('peak_reserved_gb', '-')} GB{flag}")
        b = best_config(records, view)
        lines.append(f"  best: {b['label']} ({'+'.join(b['speedups']) or 'no speedups'}, "
                     f"{'all charges at once' if b['chunk'] is None else b['chunk']})")
        ph = by_id.get(f"gpu.phases.{view}")
        if ph and ph["outcome"] == "ok":
            res = ph["result"]
            parts = [f"{p} {v['gpu_ms'].get('median', 0):.1f}/{v['host_ms'].get('median', 0):.1f}"
                     for p, v in res["phases"].items()]
            lines.append(f"  phases, ms GPU/host: {', '.join(parts)}; nonzero wait "
                         f"{res['nonzero_wait_ms'].get('median', 0):.1f}; {res['verdict']}")
        for tag in ("baseline", "best"):
            pr = by_id.get(f"gpu.profile.{view}.{tag}")
            if pr and pr["outcome"] == "ok":
                ks = pr["result"].get("kernels") or []
                top = ", ".join(f"{100 * k['share']:.0f} % {k['name'][:40]}" for k in ks[:3])
                busy = pr["result"].get("gpu_busy_share")
                busy = "" if busy is None else (f", the GPU busy {100 * busy:.0f} % of a step of "
                                                f"{pr['result'].get('ms_per_step')} ms")
                lines.append(f"  kernels ({tag}): {pr['result']['kernels_per_step']:.0f} a step"
                             f"{busy}; {top}")
        mems = [r for r in records if r["id"].startswith(f"gpu.memory.{view}.")]
        if mems:
            lines.append("  memory: " + "; ".join(
                f"{r['id'].rsplit('.', 1)[-1]} "
                + (f"{r['result'].get('peak_reserved_gb', 0):.1f} GB, "
                   f"{r['result'].get('alloc_retries', 0)} retries, "
                   f"{r['result'].get('ooms', 0)} OOM" if r["outcome"] == "ok" else r["outcome"])
                for r in mems))
        sm = by_id.get(f"gpu.smi.{view}")
        if sm and sm["outcome"] == "ok":
            s = sm["result"].get("nvidia_smi") or {}

            def mean(key, s=s):
                return (s.get(key) or {}).get("mean")

            lines.append(f"  nvidia-smi over {sm['result']['seconds']:.0f} s: "
                         f"{sm['result']['spectra_per_second']:.1f} spectra/s; utilisation "
                         f"{mean('utilization.gpu')} %, SM {mean('clocks.sm')} MHz, "
                         f"{mean('power.draw')} W, {mean('temperature.gpu')} C; reasons "
                         f"{s.get('clock_event_reasons')}")
        for name in ("autocast", "batch8"):
            r = by_id.get(f"gpu.{name}.{view}")
            if r:
                lines.append(f"  {name} (information): {brief(r)}")
    cores = by_id.get("cpu.cores")
    if cores:
        lines.append(f"Processor, one worker per core: {brief(cores)}")
    thr = by_id.get("cpu.throttle")
    if thr:
        lines.append(f"  power throttling and priority: {brief(thr)}")
    loaders = [r for r in records if r["id"].startswith("cpu.loader.")]
    if loaders:
        variants = list(dict.fromkeys(r.get("variant") for r in loaders))
        lines.append("  items/s delivered (90th percentile of the wait for a batch, ms): "
                     + " | ".join(variants))
        for n in dict.fromkeys(r.get("workers") for r in loaders):
            cells = []
            for v in variants:
                r = next((x for x in loaders if x.get("workers") == n and x.get("variant") == v),
                         None)
                if r is None:
                    cells.append("-")
                elif r["outcome"] != "ok":
                    cells.append(r["outcome"])
                else:
                    w = r["result"].get("consumer_wait_ms") or {}
                    cells.append(f"{r['result']['items_per_second']:.1f} ({_ms(w.get('p90'))})")
            lines.append(f"    {n:3d} workers: " + " | ".join(cells))
    cp = by_id.get("cpu.cprofile")
    if cp and cp["outcome"] == "ok":
        for kind, prof in cp["result"]["profiles"].items():
            top = ", ".join(f"{r['function'].split('(')[-1].rstrip(')')} "
                            f"{100 * r['own_share']:.0f} %" for r in prof["by_own"][:4])
            lines.append(f"  cProfile, {kind} core: {top}")
    e2e = by_id.get("e2e.train")
    if e2e:
        res = e2e.get("result") or {}
        cfg = res.get("config") or e2e.get("config") or {}
        sp = "+".join(cfg.get("speedups") or []) or "no speedups"
        lines.append(f"End-to-end ({cfg.get('view')} view, {sp}, {cfg.get('workers')} workers): "
                     f"{e2e['outcome']}" + (f": {brief(e2e)}" if e2e["outcome"] != "ok" else ""))
        for ep in res.get("epochs") or []:
            p = ep.get("pace") or {}
            st = ep.get("steady") or {}
            steady = (f"; {st['spectra_per_second']} spectra/s from step {st['from_step']} on"
                      if st.get("spectra_per_second") else "")
            lines.append(f"  epoch {ep.get('epoch')}: {ep.get('spectra_per_second_epoch')} "
                         f"spectra/s, {100 * float(p.get('data_wait_share', 0)):.0f} % waiting "
                         f"for data, {100 * float(p.get('gpu_busy_share', 0)):.0f} % GPU busy "
                         f"({p.get('gpu_step_ms')} ms a step), {p.get('alloc_retries')} retries"
                         f"{steady}")
    bad = [r for r in records if r["outcome"] in ("error", "timeout", "crashed")]
    for r in bad:
        err = str(r.get("error") or "").strip().splitlines()
        lines.append(f"FAILED {r['id']} ({r['outcome']}): {err[-1][:160] if err else ''}")
    return lines


# ------------------------------------------------------------------ packing
def sanitizer():
    """Replace, in text, the package folder's path by <package> and the home folder's by ~ (as
    written and as JSON writes them, with doubled backslashes)."""
    pairs = []
    for path, name in ((ROOT, "<package>"), (Path.home(), "~")):
        s = str(path)
        if len(s) < 4:
            continue
        for v in {s, s.replace("\\", "/"), s.replace("\\", "\\\\"), json.dumps(s)[1:-1]}:
            pairs.append((v, name))
    pairs.sort(key=lambda p: -len(p[0]))  # the package folder before the home folder above it

    def clean(text: str) -> str:
        for old, new in pairs:
            text = text.replace(old, new)
        return text

    return clean


def pack(run_dir: Path, zip_path: Path) -> dict:
    """The upload: the run folder's JSON, logs, text, CSV and compressed traces (never the work
    folder, nothing over MAX_FILE_MB), setup's record of this computer and the end of setup.log,
    paths made anonymous; returns what was packed and left out."""
    clean = sanitizer()
    packed, left = [], []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(run_dir).as_posix()
            if rel.startswith("work/"):
                left.append({"file": rel, "why": "intermediate"})
                continue
            if f.suffix not in PACK_SUFFIXES:
                left.append({"file": rel, "why": "not a kind the upload holds"})
                continue
            size = f.stat().st_size
            if size > MAX_FILE_MB * 2**20:
                left.append({"file": rel, "why": f"{size / 2**20:.0f} MB"})
                continue
            arc = f"{run_dir.name}/{rel}"
            if f.suffix == ".gz":
                zf.write(f, arc)
            else:
                zf.writestr(arc, clean(f.read_text(encoding="utf-8", errors="replace")))
            packed.append(rel)
        for f in (ROOT / "gpu_training" / "machine.json", ROOT / "gpu_training" / "setup_done.json"):
            if f.exists():
                zf.writestr(f"{run_dir.name}/setup/{f.name}",
                            clean(f.read_text(encoding="utf-8", errors="replace")))
                packed.append(f"setup/{f.name}")
        log = ROOT / "gpu_training" / "setup.log"
        if log.exists():
            data = log.read_bytes()[-SETUP_LOG_TAIL:].decode("utf-8", errors="replace")
            zf.writestr(f"{run_dir.name}/setup/setup_log_tail.txt", clean(data))
            packed.append("setup/setup_log_tail.txt")
        zf.writestr(f"{run_dir.name}/manifest.json",
                    json.dumps({"packed": packed, "left_out": left}, indent=1))
    return {"packed": len(packed), "left_out": left, "zip_mb": round(zip_path.stat().st_size / 2**20, 2)}


# ------------------------------------------------------------------ main
def env_entry(smoke: bool) -> dict:
    return entry("env", "env", "the environment: processor, Windows, memory, GPU, PyTorch",
                 "profkit.environment:environment", "env", {}, smoke, essential=True)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=None,
                    help="run only these groups, comma-separated: gpu, cpu, e2e (default: all)")
    ap.add_argument("--minutes", type=float, default=DEFAULT_MINUTES,
                    help=f"the time budget in minutes (default {DEFAULT_MINUTES:g}): experiments "
                         f"that would not fit are skipped and recorded as such")
    ap.add_argument("--smoke", action="store_true",
                    help=f"tiny sizes everywhere, to check that it runs (or {common.SMOKE_ENV}=1)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the intermediate files (batches, weights) in the run folder")
    ap.add_argument("--experiments", default=None, help=argparse.SUPPRESS)  # id prefixes
    ap.add_argument("--runs-dir", type=Path, default=None, help=argparse.SUPPRESS)  # tests
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return child_main(args.child, args.run_dir)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    groups = set(GROUPS) if not args.only else {g.strip() for g in args.only.split(",") if g.strip()}
    unknown = groups - set(GROUPS)
    if unknown:
        ap.error(f"unknown group {', '.join(sorted(unknown))}: choose from {', '.join(GROUPS)}")
    smoke = args.smoke or common.smoke()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = Path(args.runs_dir or RUNS)
    run_dir = runs / f"profile_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console = Console(run_dir / "runner.log")
    header = {"profile_version": 2, "started": common.now_text(), "options": {
        "only": sorted(groups), "minutes": args.minutes, "smoke": smoke,
        "experiments": args.experiments},
        "setup_exit_code": os.environ.get("LCMSDECONV_SETUP_RC")}
    console.say("=" * 78)
    console.say("Profiling this computer for the developers" + (" (smoke size)" if smoke else ""))
    console.say(f"  results: {run_dir}")
    console.say("  Keep the computer otherwise idle. Ctrl+C stops it and still packs what is done.")
    console.say("=" * 78)
    runner = Runner(run_dir, console, smoke, 60.0 * args.minutes, header)
    awake = keep_awake()
    try:
        return profile(args, groups, smoke, stamp, runs, run_dir, console, header, runner)
    finally:
        if awake is not None:
            awake.stop()


def keep_awake():
    """Keep Windows from going to sleep while the profile runs, as training does (the screen
    may still switch off); None where that is not possible."""
    common.add_paths()
    try:
        from lcmsdeconv.nn.train import _KeepAwake

        awake = _KeepAwake()
        awake.start()
        return awake
    except Exception:  # a convenience only
        return None


def profile(args, groups, smoke, stamp, runs, run_dir, console, header, runner) -> int:
    env_rec = runner.launch(env_entry(smoke))
    console.say(f"    environment: {env_rec['outcome']}" + (f": {brief(env_rec)}" if brief(env_rec)
                                                             else ""))
    env = env_rec.get("result") or {}
    header["environment"] = env
    try:
        plan = build_plan(groups, env, smoke,
                          [p.strip() for p in args.experiments.split(",")]
                          if args.experiments else None)
    except Exception:  # without a plan only the environment is packed
        console.say("The plan of experiments could not be made:\n" + traceback.format_exc())
        plan = []
    total = sum(e["estimate"] for e in plan)
    console.say(f"{len(plan)} experiments, each in a process of its own: about {total / 60:.0f} "
                f"min (at most {args.minutes:g}; what would not fit is skipped)")
    try:
        runner.run_all(plan)
    except KeyboardInterrupt:
        runner.interrupted = True
        console.say("Interrupted: packing what is done.")
    finished = time.time()
    header.update(finished=common.now_text(), minutes=round((finished - runner.started) / 60, 2))
    try:
        lines = summary_lines(runner.records, env, header)
    except Exception:  # the summary is a convenience: the records are packed regardless
        lines = ["(the summary failed:)", traceback.format_exc()]
    try:
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:  # held by another program: the console shows the summary below
        console.say(f"summary.txt could not be written: {exc}")
    runner.save({"finished": header["finished"], "minutes": header["minutes"]}, final=True)
    zip_path = runs / f"profile_upload_{stamp}.zip"
    try:
        info = pack(run_dir, zip_path)
    except Exception:  # the run folder holds everything, to be zipped by hand
        info = None
        console.say("The zip file could not be made:\n" + traceback.format_exc()[-1500:])
    if not args.keep_work:
        import shutil

        shutil.rmtree(run_dir / "work", ignore_errors=True)
    console.say("")
    for line in lines:
        console.say(line)
    console.say("")
    if info is not None:
        console.say(f"Send this file to the developers: {zip_path} ({info['zip_mb']:.1f} MB)")
    else:
        console.say(f"Zip the folder {run_dir} and send it to the developers instead")
    # an experiment that failed is a result like any other: the profile itself succeeded
    return 130 if runner.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
