"""Measure this computer for training, within the given memory limits.

Runs real training steps of the full-size networks of both views (the whole-spectrum recipe
with isotope-charge channels, at the recipe's batch size, every spectrum at the active-bin cap:
the most memory a step can need) on the GPU under the GPU memory cap, scoring the charges in
chunks and all at once (train_all.py chooses between the two from these measurements), then
starts data workers the way training does on the recipe's spectra and measures how much memory
each one takes and how many spectra per second they deliver. From that it chooses the number of
workers that keeps the training process and its workers within the RAM budget, and writes
everything to gpu_training/machine.json, which train_all.py reads.

    python gpu_training/probe.py [--ram-gb 16] [--gpu-memory-gb 14] [--workers N]

setup.py runs it; run it again on its own to re-measure (for example after changing the limits).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "gpu_training"))

#: memory the training process holds for its 128 validation spectra (about 7 MB each)
VALIDATION_GB = 0.9


def process_gb(proc) -> float:
    """RAM of one process as the task manager shows it (its private working set)."""
    from lcmsdeconv.nn.machine import process_ram_gb

    return process_ram_gb(proc)


def commit_gb(proc) -> float:
    """Memory Windows has promised the process (private bytes), reserved or used; only for the
    record: it is not RAM, but RAM plus the paging file must hold it."""
    return getattr(proc.memory_info(), "private", 0) / 2**30


def recipe_mixture(measured: bool = True) -> list:
    """The training spectra of train_all's runs, as training configures them (singly charged
    ions labelled, isotope-charge channels, and the comb view's measured channel)."""
    from train_charge_net import mixture_v3

    mix = mixture_v3()
    for _w, c in mix:
        c.singly_charged = "keep"
        c.iso_channels = True
        c.measured_channel = measured
    return mix


def gpu_view(view: str, batch_size: int, head_chunk: int | None) -> dict:
    """Peak GPU memory and time of a training step of the full network of ``view`` on spectra of
    the recipe, every one at the active-bin cap (distinct levels above the active level on all
    its grid bins)."""
    import torch

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.synth.dataset import FrameDataset

    ds = FrameDataset(recipe_mixture(), batch_size, base_seed=777)
    items = [ds[i] for i in range(batch_size)]
    names = ("features", "topk_z", "topk_w", "heat")
    batch = {k: torch.stack([it[j] for it in items]) for j, k in enumerate(names)}
    x = batch["features"]
    on_grid = x[:, -1] > 0.5
    level = 0.6 + 0.3 * torch.rand(x[:, 0].shape, generator=torch.Generator().manual_seed(0))
    x[:, 0] = torch.where(on_grid, torch.maximum(x[:, 0], level), x[:, 0])
    if view == "window":
        x = x[:, :-1]  # the window view has no measured channel
    batch["features"] = x.contiguous()
    batch = {k: v.cuda() for k, v in batch.items()}
    model = build_model("full", z_max=100, in_ch=x.shape[1], arch="ladder", window=131072,
                        iso=True, view=view, head_chunk=head_chunk).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ChargeLoss()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(6):
        torch.cuda.synchronize()
        t = time.time()
        loss = loss_fn(model(batch["features"]), batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    out = {"batch": batch_size, "peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
           "allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "step_seconds": round(statistics.median(times[2:]), 4)}
    del model, opt, batch, x
    torch.cuda.empty_cache()
    return out


def gpu_steps(gpu_memory_gb: float, batch_size: int, head_chunk: int | None) -> dict:
    """Both views at ``batch_size`` (halved until a step fits under the cap) in chunks of
    ``head_chunk`` charges, then at the batch that fit with all charges at once
    (``"unchunked"``)."""
    import torch

    from lcmsdeconv.nn.train import _cap_gpu_memory, _warm_up

    _cap_gpu_memory(torch, gpu_memory_gb)
    _warm_up(recipe_mixture())
    views = {}
    for view in ("window", "comb"):
        b = batch_size
        while b >= 1:
            try:
                views[view] = gpu_view(view, b, head_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  the {view} view does not fit at {b} spectra per step under the cap",
                      flush=True)
                b //= 2
        if view not in views:
            views[view] = {"batch": batch_size, "peak_gb": None, "step_seconds": None,
                           "error": "out of GPU memory even at one spectrum per step"}
            continue
        whole = None
        try:
            whole = gpu_view(view, views[view]["batch"], None)
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        views[view]["unchunked"] = whole or {
            "batch": views[view]["batch"], "peak_gb": None, "step_seconds": None,
            "error": "out of GPU memory"}
    props = torch.cuda.get_device_properties(0)
    return {"gpu": props.name, "gpu_total_gb": round(props.total_memory / 2**30, 1),
            "gpu_views": views, "torch": torch.__version__, "torch_cuda": torch.version.cuda}


def workers_run(n: int, rounds: int = 4, batch_size: int = 4) -> dict:
    """Start ``n`` workers as training does on the recipe's spectra and measure their speed and
    memory. Speed is timed from the moment every worker has delivered its first batch, so that
    start-up does not count."""
    import psutil
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import FrameDataset, mp_context, worker_init

    ds = FrameDataset(recipe_mixture(), n * rounds * batch_size, base_seed=4242)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=n, worker_init_fn=worker_init,
                        **mp_context(n))
    me = psutil.Process()
    started = time.time()
    warm = None
    spectra = 0
    per_worker = commit = total_workers = 0.0
    for k, (x, *_rest) in enumerate(loader):
        if k == n - 1:
            warm = time.time()  # every worker is up and has delivered a batch
        elif k >= n:
            spectra += x.shape[0]
        if k % max(1, n // 2) == 0:
            sizes = []
            for p in me.children(recursive=True):
                try:
                    sizes.append((process_gb(p), commit_gb(p)))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if sizes:
                per_worker = max(per_worker, max(a for a, _b in sizes))
                commit = max(commit, max(b for _a, b in sizes))
                total_workers = max(total_workers, sum(a for a, _b in sizes))
    elapsed = time.time() - (warm or started)
    return {"workers": n, "startup_seconds": round((warm or started) - started, 1),
            "spectra_per_second": round(spectra / max(elapsed, 1e-9), 1),
            "worker_gb": round(per_worker, 2), "worker_commit_gb": round(commit, 2),
            "workers_total_gb": round(total_workers, 2)}


def main() -> int:
    from train_all import BATCH, DEFAULT_RUNS, HEAD_CHUNK, RUN_TABLE, SPECTRA, head_chunk_for

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ram-gb", type=float, default=16.0,
                    help="RAM budget of the training process and its workers together")
    ap.add_argument("--gpu-memory-gb", type=float, default=14.0,
                    help="GPU memory the training process may use, CUDA context included")
    ap.add_argument("--workers", type=int, default=None, help="measure with this many workers")
    ap.add_argument("--batch-size", type=int, default=BATCH,
                    help=f"spectra per step to measure the GPU at (default {BATCH}, the recipe's)")
    ap.add_argument("--out", type=Path, default=ROOT / "gpu_training" / "machine.json")
    ap.add_argument("--no-gpu", action="store_true", help=argparse.SUPPRESS)  # tests only
    args = ap.parse_args()

    import psutil
    import torch

    from lcmsdeconv.nn.machine import MARGIN_GB, auto_workers, cpu_cores, single_threaded_workers

    if args.no_gpu:
        gpu = {"gpu": None, "gpu_views": {}}
    elif not torch.cuda.is_available():
        print("CUDA is not available to PyTorch: run gpu_training/setup.py first.")
        return 1
    else:
        print(f"Measuring training steps of both views on the GPU ({args.batch_size} spectra per "
              f"step, memory capped at {args.gpu_memory_gb:g} GB) ...", flush=True)
        gpu = gpu_steps(args.gpu_memory_gb, args.batch_size, HEAD_CHUNK)
    # the training process: what this one holds now (PyTorch, the CUDA context) and the
    # validation spectra it keeps in memory
    main_gb = process_gb(psutil.Process()) + VALIDATION_GB
    chunks = {view: head_chunk_for(gpu, view, args.gpu_memory_gb, args.batch_size)
              for view in gpu["gpu_views"]}
    for view, m in gpu["gpu_views"].items():
        if m.get("peak_gb"):
            print(f"  {gpu['gpu']}, {view} view: {m['step_seconds'] * 1000:.0f} ms per training "
                  f"step of {m['batch']} spectra, {m['peak_gb']:.1f} GB of GPU memory at the "
                  f"peak, scoring {HEAD_CHUNK} charges at a time", flush=True)
            whole = m.get("unchunked") or {}
            if whole.get("peak_gb"):
                print(f"    all charges at once: {whole['step_seconds'] * 1000:.0f} ms, "
                      f"{whole['peak_gb']:.1f} GB", flush=True)
            else:
                print(f"    all charges at once: {whole.get('error', 'not measured')}", flush=True)
            print(f"    training will score "
                  f"{'all charges at once' if chunks[view] is None else f'{HEAD_CHUNK} at a time'}",
                  flush=True)
        else:
            print(f"  {view} view: {m.get('error')}", flush=True)

    single_threaded_workers()
    cores = cpu_cores()
    n = args.workers or auto_workers(args.ram_gb, main_gb=max(main_gb, 1.0))
    run = None
    while n >= 1:
        print(f"Measuring {n} data workers on the recipe's spectra ...", flush=True)
        try:
            run = workers_run(n)
            break
        except (OSError, RuntimeError) as exc:  # e.g. Windows error 1455: paging file
            print(f"  {n} workers failed to run ({exc}); trying {n // 2}", flush=True)
            n //= 2
    if run is None:
        print("No data worker could be started.")
        return 1
    print(f"  {run['workers']} workers: {run['spectra_per_second']:.0f} spectra/s after "
          f"{run['startup_seconds']:.0f} s to start; {run['worker_gb']:.2f} GB of RAM each at most "
          f"({run['workers_total_gb']:.1f} GB together; {run['worker_commit_gb']:.1f} GB committed "
          f"each)", flush=True)
    worker_gb = max(run["worker_gb"], 0.3)
    fit = int((args.ram_gb - main_gb - MARGIN_GB) // worker_gb)
    by_cpu = cores - 2 if cores <= 8 else cores - 4
    chosen = args.workers or max(1, min(by_cpu, fit))
    rate = run["spectra_per_second"] * chosen / max(run["workers"], 1)
    # train_all.py's default runs, each 20 x 6000 spectra; 10 % more for validation. The data
    # workers and the GPU work at the same time, so the slower of the two sets the pace
    hours, paces = 0.0, {}
    for name in DEFAULT_RUNS:
        view = RUN_TABLE[name]["view"]
        m = gpu["gpu_views"].get(view) or {}
        if m and chunks.get(view, HEAD_CHUNK) is None:
            m = m["unchunked"]  # the step train_all will take
        pace = rate
        if m.get("step_seconds"):
            pace = min(rate, m["batch"] / m["step_seconds"])
        paces[view] = pace
        hours += 1.1 * SPECTRA / max(pace, 1e-6) / 3600
    info = {"recipe": "v3", "workers": chosen, "ram_budget_gb": args.ram_gb,
            "gpu_memory_gb": args.gpu_memory_gb, "gpu_batch": args.batch_size,
            "head_chunk": HEAD_CHUNK, "head_chunks": chunks, "main_gb": round(main_gb, 2), "worker_gb": round(worker_gb, 2),
            "expected_ram_gb": round(main_gb + chosen * worker_gb, 1),
            "v3_spectra_per_second": round(rate, 1), "v3_workers": chosen,
            "expected_hours": round(hours, 1), "cores": cores, "measured": [run], **gpu}
    args.out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    gpu_rates = {}
    for v, m in gpu["gpu_views"].items():
        if m and chunks.get(v, HEAD_CHUNK) is None:
            m = m["unchunked"]
        if m.get("step_seconds"):
            gpu_rates[v] = m["batch"] / m["step_seconds"]
    if gpu_rates:
        pace_text = "; ".join(f"the GPU could take {r:.0f} spectra/s in the {v} view"
                              for v, r in gpu_rates.items())
        slower = ("the workers set the pace" if all(r >= rate for r in gpu_rates.values())
                  else "the GPU sets the pace")
        pace_text = f" ({pace_text}, so {slower})"
    else:
        pace_text = " (the GPU was not measured)"
    print(f"Chosen: {chosen} data workers (about {info['expected_ram_gb']:.1f} GB of RAM with "
          f"the training process, budget {args.ram_gb:g} GB; {cores} cores). Expected "
          f"{rate:.0f} spectra/s from the workers{pace_text}: about {hours:.1f} h for the two "
          f"runs. Written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
