"""Measure this computer for training, within the given memory limits.

Runs real training steps of the full-size network on the GPU under the GPU memory cap, then
starts data workers the way training does and measures how much memory each one takes and how
many spectra per second they deliver. From that it chooses the number of workers that keeps the
training process and its workers within the RAM budget, and writes everything to
gpu_training/machine.json, which train_all.py reads.

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


def process_gb(proc) -> float:
    """RAM of one process as the task manager shows it (its private working set)."""
    from lcmsdeconv.nn.machine import process_ram_gb

    return process_ram_gb(proc)


def commit_gb(proc) -> float:
    """Memory Windows has promised the process (private bytes), reserved or used; only for the
    record: it is not RAM, but RAM plus the paging file must hold it."""
    return getattr(proc.memory_info(), "private", 0) / 2**30


def gpu_step(gpu_memory_gb: float, batch_size: int = 8) -> dict:
    """Peak GPU memory and time of a training step of the full network on real spectra."""
    import torch
    from train_charge_net import mixture

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import _cap_gpu_memory, _warm_up
    from lcmsdeconv.synth.dataset import FrameDataset

    _cap_gpu_memory(torch, gpu_memory_gb)
    mix = mixture(background_ions=200)
    for _w, c in mix:
        c.singly_charged = "keep"
    _warm_up(mix)
    ds = FrameDataset(mix, batch_size, base_seed=777)
    items = [ds[i] for i in range(batch_size)]
    names = ("features", "topk_z", "topk_w", "heat")
    batch = {k: torch.stack([it[j] for it in items]).cuda() for j, k in enumerate(names)}
    model = build_model("full", z_max=100, in_ch=4, arch="ladder").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ChargeLoss()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(8):
        torch.cuda.synchronize()
        t = time.time()
        loss = loss_fn(model(batch["features"]), batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    props = torch.cuda.get_device_properties(0)
    return {"gpu": props.name, "gpu_total_gb": round(props.total_memory / 2**30, 1),
            "gpu_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "step_seconds": round(statistics.median(times[3:]), 4),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda}


def workers_run(n: int, stage: int, rounds: int = 4, batch_size: int = 8) -> dict:
    """Start ``n`` workers as training does on the spectra of ``stage`` (1, or 2 with dense
    background) and measure their speed and memory. Speed is timed from the moment every
    worker has delivered its first batch, so that start-up does not count."""
    import psutil
    from torch.utils.data import DataLoader
    from train_charge_net import mixture

    from lcmsdeconv.synth.dataset import FrameDataset, mp_context, worker_init

    mix = mixture(background_ions=200 if stage == 2 else None)
    for _w, c in mix:
        c.singly_charged = "keep" if stage == 2 else "ignore"
    ds = FrameDataset(mix, n * rounds * batch_size, base_seed=4242 + stage)
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
    return {"stage": stage, "workers": n,
            "startup_seconds": round((warm or started) - started, 1),
            "spectra_per_second": round(spectra / max(elapsed, 1e-9), 1),
            "worker_gb": round(per_worker, 2), "worker_commit_gb": round(commit, 2),
            "workers_total_gb": round(total_workers, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ram-gb", type=float, default=16.0,
                    help="RAM budget of the training process and its workers together")
    ap.add_argument("--gpu-memory-gb", type=float, default=14.0,
                    help="GPU memory the training process may use, CUDA context included")
    ap.add_argument("--workers", type=int, default=None, help="measure with this many workers")
    ap.add_argument("--out", type=Path, default=ROOT / "gpu_training" / "machine.json")
    ap.add_argument("--no-gpu", action="store_true", help=argparse.SUPPRESS)  # tests only
    args = ap.parse_args()

    import psutil
    import torch

    from lcmsdeconv.nn.machine import MARGIN_GB, auto_workers, cpu_cores, single_threaded_workers

    if args.no_gpu:
        gpu = {"gpu": None, "step_seconds": 0.05}
    elif not torch.cuda.is_available():
        print("CUDA is not available to PyTorch: run gpu_training/setup.py first.")
        return 1
    else:
        print(f"Measuring a training step on the GPU (memory capped at {args.gpu_memory_gb:g} "
              f"GB) ...", flush=True)
        gpu = gpu_step(args.gpu_memory_gb)
    main_gb = process_gb(psutil.Process())
    if gpu["gpu"]:
        print(f"  {gpu['gpu']}: {gpu['step_seconds'] * 1000:.0f} ms per training step of 8 "
              f"spectra, {gpu['gpu_peak_gb']:.1f} GB of GPU memory at the peak; training process "
              f"{main_gb:.1f} GB of RAM", flush=True)

    single_threaded_workers()
    cores = cpu_cores()
    n = args.workers or auto_workers(args.ram_gb, main_gb=max(main_gb, 1.0))
    runs = {}
    for stage in (1, 2):
        while n >= 1:
            print(f"Measuring {n} data workers on stage-{stage} spectra ...", flush=True)
            try:
                runs[stage] = workers_run(n, stage)
                break
            except (OSError, RuntimeError) as exc:  # e.g. Windows error 1455: paging file
                print(f"  {n} workers failed to run ({exc}); trying {n // 2}", flush=True)
                n //= 2
        if stage not in runs:
            print("No data worker could be started.")
            return 1
        r = runs[stage]
        print(f"  {r['workers']} workers: {r['spectra_per_second']:.0f} spectra/s after "
              f"{r['startup_seconds']:.0f} s to start; {r['worker_gb']:.2f} GB of RAM each at "
              f"most ({r['workers_total_gb']:.1f} GB together; {r['worker_commit_gb']:.1f} GB "
              f"committed each)", flush=True)
    worker_gb = max(max(r["worker_gb"] for r in runs.values()), 0.3)
    fit = int((args.ram_gb - main_gb - MARGIN_GB) // worker_gb)
    by_cpu = cores - 2 if cores <= 8 else cores - 4
    chosen = args.workers or max(1, min(by_cpu, fit))
    rates = {s: r["spectra_per_second"] * chosen / max(r["workers"], 1) for s, r in runs.items()}
    # train_all.py: stage 1 is 40 x 6000 spectra, stage 2 16 x 6000; 10 % for validation
    hours = 1.1 * (240_000 / max(rates[1], 1e-6) + 96_000 / max(rates[2], 1e-6)) / 3600
    gpu_rate = 8 / max(gpu["step_seconds"], 1e-6)
    info = {"workers": chosen, "ram_budget_gb": args.ram_gb, "gpu_memory_gb": args.gpu_memory_gb,
            "main_gb": round(main_gb, 2), "worker_gb": round(worker_gb, 2),
            "expected_ram_gb": round(main_gb + chosen * worker_gb, 1),
            "spectra_per_second": round(min(rates.values()), 1),
            "spectra_per_second_stage1": round(rates[1], 1),
            "spectra_per_second_stage2": round(rates[2], 1),
            "expected_hours": round(hours, 1), "gpu_spectra_per_second": round(gpu_rate, 1),
            "cores": cores, "measured": list(runs.values()), **gpu}
    args.out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Chosen: {chosen} data workers (about {info['expected_ram_gb']:.1f} GB of RAM with "
          f"the training process, budget {args.ram_gb:g} GB; {cores} cores). Expected "
          f"{rates[1]:.0f} spectra/s in stage 1 and {rates[2]:.0f} in stage 2, about "
          f"{hours:.1f} h of training in all; the GPU could take {gpu_rate:.0f} spectra/s, so "
          f"the workers set the pace. Written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
