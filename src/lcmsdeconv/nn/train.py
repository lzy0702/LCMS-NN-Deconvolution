"""Training loop for the charge network (CPU proof runs and GPU full runs)."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrainConfig:
    model_size: str = "proof"
    z_max: int = 100
    epochs: int = 3
    train_len: int = 2000
    val_len: int = 200
    batch_size: int = 4
    num_workers: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    heat_weight: float = 0.2
    ema_decay: float = 0.99
    grad_clip: float = 5.0
    base_seed: int = 0
    device: str = "auto"
    out: str = "models/charge_unet_proof.pt"
    max_seconds: float | None = None
    log_every: int = 20
    threads: int | None = None
    #: select the saved checkpoint by validation charge accuracy (``"accuracy"``) rather than
    #: by validation loss: a loss can improve while the model drifts toward a constant answer
    select_by: str = "accuracy"
    #: per-charge comb evidence channels added to the four base input features
    evidence_channels: int = 0
    #: add that evidence straight to the charge logits (see ChargeUNet)
    comb_prior: bool = False
    #: "unet" (ChargeUNet) or "ladder" (LadderNet)
    arch: str = "unet"
    #: singly charged ions in the training data: "keep", "drop" or "ignore" (see SynthConfig)
    singly_charged: str = "keep"
    #: draw new frames every epoch (seeds advance by train_len) instead of repeating them
    fresh_each_epoch: bool = False
    #: keep generated frames in memory (CPU runs that repeat a small pool)
    cache_frames: bool = False
    #: checkpoint to start from (same architecture and size), e.g. a previous stage
    init: str | None = None
    #: continue from ``<out>.resume.pt`` (written every ``save_every_minutes``, at the end of
    #: every epoch, and when the run is paused with Ctrl+C or a file named PAUSE next to ``out``)
    resume: bool = False
    #: start afresh even though ``<out>.resume.pt`` exists (it is overwritten); without this
    #: or ``resume`` such a run refuses to start rather than overwrite a paused run's state
    restart: bool = False
    save_every_minutes: float = 10.0
    #: cap on this process's GPU memory (GB, CUDA context included); None leaves PyTorch's
    #: default, the whole card
    max_gpu_memory_gb: float | None = None
    #: cap on the memory of this process and its data workers together (GB); when it is
    #: exceeded the run pauses, saving its state, so that it can resume with fewer workers
    max_ram_gb: float | None = None
    #: remove PAUSE files left over at the start; gpu_training/train_all.py clears them itself
    #: and turns this off, so that a pause asked for while a stage starts up is not lost
    clear_pause_files: bool = True


def _device(name: str):
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _batch_dict(sample, device):
    x, topk_z, topk_w, heat = sample
    return {
        "features": x.to(device),
        "topk_z": topk_z.to(device),
        "topk_w": topk_w.to(device),
        "heat": heat.to(device),
    }


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        import torch

        with torch.no_grad():
            for s, m in zip(self.shadow.parameters(), model.parameters()):
                s.mul_(self.decay).add_(m, alpha=1 - self.decay)
            for s, m in zip(self.shadow.buffers(), model.buffers()):
                s.copy_(m)


def train(config: TrainConfig, synth_config=None):
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    from ..synth.config import SynthConfig
    from ..synth.dataset import make_dataloaders
    from .losses import ChargeLoss
    from .model import build_model

    synth_config = synth_config or SynthConfig(z_max=config.z_max)
    for sc in ([c for _w, c in synth_config] if isinstance(synth_config, (list, tuple))
               else [synth_config]):
        sc.evidence_channels = config.evidence_channels
        sc.singly_charged = config.singly_charged
    in_ch = 4 + config.evidence_channels
    if config.threads:
        torch.set_num_threads(int(config.threads))
    device = _device(config.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available, and this run is set to train on the GPU only. Check with "
            "'.venv\\Scripts\\python gpu_training\\check_gpu.py' (Linux: .venv/bin/python "
            "gpu_training/check_gpu.py); to reinstall PyTorch for CUDA, run "
            "'gpu_training\\setup_windows.bat --reinstall-torch'.")
    if device.startswith("cuda") and config.max_gpu_memory_gb:
        _cap_gpu_memory(torch, config.max_gpu_memory_gb)
    model = build_model(config.model_size, z_max=config.z_max, in_ch=in_ch,
                        comb_prior=config.comb_prior, arch=config.arch).to(device)
    if config.init:
        state = torch.load(config.init, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state)
        print(f"started from {config.init}")
    print(f"model {config.arch}/{config.model_size}: {model.num_params()/1e6:.3f}M params "
          f"on {device}; singly charged ions: {config.singly_charged}")

    if config.num_workers > 0:
        _warm_up(synth_config)
    tl, vl = make_dataloaders(synth_config, config.train_len, config.val_len,
                              batch_size=config.batch_size, num_workers=config.num_workers,
                              base_seed=config.base_seed, cache=config.cache_frames,
                              pin_memory=device.startswith("cuda"))
    steps_per_epoch = len(tl)
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = max(1, steps_per_epoch) * config.epochs
    sched = OneCycleLR(opt, max_lr=config.lr, total_steps=steps, pct_start=0.2)
    loss_fn = ChargeLoss(heat_weight=config.heat_weight,
                         ignore_charges=(1,) if config.singly_charged == "ignore" else ())
    ema = EMA(model, config.ema_decay)
    # the quantity checkpoints are selected by: accuracy on ion bins (z >= 2 only when singly
    # charged ions are not trained on), the way inference reads charges
    metric = "charge_acc_given_ion" if config.singly_charged == "keep" else "charge_acc_given_ion_z2"

    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log.jsonl")
    resume_path = out_path.with_suffix(".resume.pt")

    def pause_files() -> list[Path]:
        # Windows Explorer names a new text file "PAUSE.txt"; either name, any case, counts
        return [f for f in out_path.parent.iterdir()
                if f.is_file() and f.name.lower() in ("pause", "pause.txt")]

    history: list = []
    best_state = None
    best_loss = float("inf")
    best_key = None
    epoch0, step0 = 0, 0
    run_sum, run_n = 0.0, 0
    comb = None
    if resume_path.exists() and not (config.resume or config.restart):
        raise SystemExit(
            f"{resume_path} holds the state of a paused or interrupted run. Add --resume to "
            f"continue it, or --restart to discard it and start again (this overwrites "
            f"{out_path} too).")
    synth_key = synth_fingerprint(synth_config)
    if config.resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device, weights_only=False)
        _check_same_run(st.get("config", {}), asdict(config))
        if st.get("synth") != synth_key:
            why = ("was written by an earlier version of this package, which did not record its "
                   "synthetic spectra (and whose training mixture differed)"
                   if st.get("synth") is None else
                   "was written by a run that trained on other synthetic spectra")
            raise SystemExit(
                f"cannot resume: {resume_path} {why}; continuing would mix the two. Finish it "
                f"with the version that started it, or start again with --restart.")
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        history, best_state, best_loss = st["history"], st["best_state"], st["best_loss"]
        best_key = tuple(st["best_key"]) if st["best_key"] is not None else None
        epoch0, step0, comb = st["epoch"], st["step"], st["comb"]
        run_sum, run_n = st.get("running", (0.0, 0))
        if epoch0 >= config.epochs:
            print(f"{resume_path}: this run already finished all {config.epochs} epochs")
        else:
            print(f"resumed from {resume_path}: epoch {epoch0}, step {step0} of "
                  f"{steps_per_epoch}")
    elif config.resume:
        print(f"nothing to resume at {resume_path}; starting from the beginning")
    if config.clear_pause_files:
        for f in pause_files():
            f.unlink()
            print(f"removed {f} left from the last pause")

    if comb is None:
        # Before spending hours: the true labels must beat every constant answer by a clear
        # margin on real batches, or the optimiser will settle on the constant.
        from .checks import degeneracy_report

        # from the validation spectra, which are in memory: iterating the training loader here
        # would start its persistent workers while the comb's workers below are running too
        first = next(iter(vl), None)
        if first is not None:
            rep = degeneracy_report(_batch_dict(first, "cpu"), loss_fn)
            print(f"loss of constant predictors relative to the truth: {rep['margins']}")
            _log(log_path, {"degeneracy": rep})
    if comb is None or "by_kind" not in comb:
        # the bar to clear: the comb as the pipeline runs it (whole spectrum, isotope-spacing
        # charges on Orbitrap spectra) on the validation frames, scored as a network is
        print("scoring the comb estimator, as the pipeline runs it, on the validation frames ...",
              flush=True)
        comb = comb_baseline(vl, num_workers=config.num_workers)
        _log(log_path, {"comb_baseline": {k: v for k, v in comb.items() if k != "kinds"}})
    bands = {k: round(v, 3) for k, v in comb["bands_given_ion"].items()}
    print(f"comb estimator ({comb['estimator']}): {metric} {comb[metric]:.3f} bands {bands}")
    for line in by_kind_lines(None, comb.get("by_kind")):
        print(line)

    def save_state(epoch: int, step: int, stop_reason: str | None = None) -> None:
        state = {"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                 "best_state": best_state, "best_key": best_key, "best_loss": best_loss,
                 "epoch": epoch, "step": step, "comb": comb, "config": asdict(config),
                 "running": (run_sum, run_n), "synth": synth_key, "stop_reason": stop_reason}
        _atomic_save(state, resume_path)
        _write_status(out_path, epoch, step, config.epochs, stop_reason)

    # Ctrl+C asks for a pause: the current step finishes, the state is saved, the run ends.
    # A second Ctrl+C aborts at once. A file named PAUSE next to the output does the same.
    import signal
    import threading

    pause = {"requested": False}

    def on_interrupt(signum, frame):
        if pause["requested"]:
            raise KeyboardInterrupt
        pause["requested"] = True
        print("\n  pause requested: saving after this step (Ctrl+C again aborts)", flush=True)

    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.signal(signal.SIGINT, on_interrupt)

    def paused() -> bool:
        return pause["requested"] or bool(pause_files())

    def stop_here(epoch: int, step: int, why: str, reason: str = "paused") -> tuple:
        save_state(epoch, step, reason)
        print(f"{why}: state saved to {resume_path}; run the same command with --resume to "
              f"continue (epoch {epoch}, step {step})", flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    session_start = time.time()
    last_save = time.time()
    ram = _RamWatch(config.max_ram_gb)
    pace = {"steps": 0, "seconds": 0.0}  # training steps done in this session, and their time
    awake = _KeepAwake()
    awake.start()
    try:
        for epoch in range(epoch0, config.epochs):
            model.train()
            if epoch != epoch0:
                run_sum, run_n = 0.0, 0
            first_step = step0 if epoch == epoch0 else 0
            # new spectra every epoch: epoch e reads items e * train_len ... (EpochSampler)
            tl.sampler.epoch = epoch if config.fresh_each_epoch else 0
            tl.sampler.start = first_step * config.batch_size
            epoch_start = time.time()
            i = first_step - 1
            try:
                for i, sample in enumerate(tl, start=first_step):
                    batch = _batch_dict(sample, device)
                    out = model(batch["features"])
                    losses = loss_fn(out, batch)
                    loss_val = float(losses["loss"].detach())
                    opt.zero_grad()
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    opt.step()
                    if sched.last_epoch < steps - 1:
                        sched.step()
                    ema.update(model)
                    run_sum += loss_val
                    run_n += 1
                    if i % config.log_every == 0:
                        print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss {loss_val:.4f} "
                              f"ce {float(losses['ce']):.4f} heat {float(losses['heat']):.4f}",
                              flush=True)
                    if paused():
                        return stop_here(epoch, i + 1, "paused")
                    if config.max_seconds and (time.time() - session_start) > config.max_seconds:
                        return stop_here(epoch, i + 1, "time budget reached", "time")
                    if ram.over():
                        return stop_here(epoch, i + 1,
                                         f"memory budget of {config.max_ram_gb:g} GB reached "
                                         f"({ram.used_gb():.1f} GB in use)", "memory")
                    if time.time() - last_save > 60.0 * config.save_every_minutes:
                        save_state(epoch, i + 1)
                        last_save = time.time()
            except (RuntimeError, OSError):
                # Ctrl+C also reaches the data-loading workers, which then exit; on Windows the
                # interrupted wait can surface as an OSError (InterruptedError) instead
                if pause["requested"]:
                    return stop_here(epoch, i + 1, "paused")
                raise
            pace["steps"] += i + 1 - first_step
            pace["seconds"] += time.time() - epoch_start
            try:
                val_ema = evaluate(ema.shadow, vl, loss_fn, device, kinds=comb.get("kinds"))
                val_live = evaluate(model, vl, loss_fn, device, kinds=comb.get("kinds"))
            except (RuntimeError, OSError):
                # interrupted while validating: the epoch's training is done, so resuming
                # starts with this validation
                if pause["requested"]:
                    return stop_here(epoch, steps_per_epoch, "paused")
                raise

            def key(v):
                if config.select_by == "accuracy":
                    # accuracy on ion bins: what inference uses (the top charges among the ion
                    # classes); an argmax that includes the background class splits the votes
                    return (v[metric], -v["val_loss"])
                return (-v["val_loss"],)

            best = "ema" if key(val_ema) >= key(val_live) else "live"
            val = val_ema if best == "ema" else val_live
            history.append({"epoch": epoch, "train_loss": run_sum / max(1, run_n),
                            "ema": val_ema, "live": val_live, "best": best,
                            "seconds": time.time() - session_start})
            _log(log_path, history[-1])
            bands = {k: round(v, 3) for k, v in val["bands_given_ion"].items()}
            print(f"epoch {epoch}: {metric} ema {val_ema[metric]:.3f} live "
                  f"{val_live[metric]:.3f} (comb {comb[metric]:.3f}) -> keeping {best}; "
                  f"bands {bands}", flush=True)
            for line in by_kind_lines(val.get("by_kind"), comb.get("by_kind")):
                print(line, flush=True)
            print("  " + _pace_line(pace, epoch, epoch_start, config, steps_per_epoch, ram,
                                    device), flush=True)
            if best_state is None or key(val) > best_key:
                best_key = key(val)
                best_loss = val["val_loss"]
                best_state = copy.deepcopy((ema.shadow if best == "ema" else model).state_dict())
                _save(out_path, best_state, config, in_ch, val, comb, metric)
                print(f"  saved {out_path} ({metric} {val[metric]:.3f})")
            run_sum, run_n = 0.0, 0
            save_state(epoch + 1, 0, "finished" if epoch + 1 >= config.epochs else None)
            last_save = time.time()
            if paused() and epoch + 1 < config.epochs:
                return stop_here(epoch + 1, 0, "paused")
    finally:
        awake.stop()
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)

    if best_state is None:
        best_state = model.state_dict()
        _save(out_path, best_state, config, in_ch, None, comb, metric)
    print(f"finished; best checkpoint {out_path} (val_loss {best_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


def _warm_up(synth_config) -> None:
    """Generate one spectrum of every configuration in this process before the data workers
    start, so that the compiled numba functions are cached once instead of being compiled by
    every worker at the same time."""
    import numpy as _np

    from ..synth.frames import generate_frame

    configs = ([c for _w, c in synth_config] if isinstance(synth_config, (list, tuple))
               else [synth_config])
    for cfg in configs:
        generate_frame(cfg, _np.random.default_rng(0))


def _write_status(out_path: Path, epoch: int, step: int, epochs: int,
                  stop_reason: str | None) -> None:
    """``<out>.status.json``: where the run stands, for gpu_training/train_all.py (which then
    does not need to load the resume state, and PyTorch with it)."""
    import json
    import os

    path = out_path.with_suffix(".status.json")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"epoch": epoch, "step": step, "epochs": epochs,
                               "stop_reason": stop_reason}) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def _cap_gpu_memory(torch, max_gb: float, context_gb: float = 0.75) -> None:
    """Keep this process's GPU memory below ``max_gb``: PyTorch's caching allocator gets
    ``max_gb`` less an allowance for the CUDA context, and raises out-of-memory rather than
    spill into shared system memory, which Windows drivers otherwise do."""
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, max(0.05, (max_gb - context_gb) * 2**30 / total))
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"GPU memory capped at {max_gb:g} GB of {total / 2**30:.1f} GB "
          f"({frac * total / 2**30:.1f} GB for tensors, the rest for the CUDA context)")


class _KeepAwake:
    """Keeps Windows from going to sleep while a run trains (the screen may still switch off).
    Elsewhere it does nothing."""

    def __init__(self):
        self.active = False

    def start(self) -> None:
        import sys

        if sys.platform != "win32":
            return
        try:
            import ctypes

            # a private, typed handle on kernel32: the shared ctypes.windll one has untyped
            # prototypes that other libraries may rely on
            self.fn = ctypes.WinDLL("kernel32").SetThreadExecutionState
            self.fn.argtypes = [ctypes.c_uint32]
            self.fn.restype = ctypes.c_uint32
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            self.active = self.fn(0x80000000 | 0x00000001) != 0
        except Exception:  # pragma: no cover - only a convenience
            self.active = False

    def stop(self) -> None:
        if not self.active:
            return
        try:
            self.fn(0x80000000)  # ES_CONTINUOUS alone: sleep is allowed again
        except Exception:  # pragma: no cover
            pass
        self.active = False


class _RamWatch:
    """RAM of this process and its data workers together, checked at most every 15 s: the sum
    of their unique sets (on Windows, private working sets), as the task manager shows them
    (:func:`lcmsdeconv.nn.machine.process_ram_gb`)."""

    def __init__(self, limit_gb: float | None, every: float = 15.0):
        self.limit = limit_gb
        self.every = every
        self.last = 0.0
        self.proc = None
        try:
            import psutil

            self.proc = psutil.Process()
        except ImportError:
            if limit_gb:
                print("psutil is not installed, so the memory budget cannot be watched "
                      "(pip install psutil)")

    def used_gb(self) -> float:
        if self.proc is None:
            return 0.0
        import psutil

        from .machine import process_ram_gb

        total = 0.0
        for p in [self.proc, *self.proc.children(recursive=True)]:
            try:
                total += process_ram_gb(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def over(self) -> bool:
        if not self.limit or self.proc is None or time.time() - self.last < self.every:
            return False
        self.last = time.time()
        return self.used_gb() > self.limit


def _pace_line(pace: dict, epoch: int, epoch_start: float, config, steps_per_epoch: int,
               ram: _RamWatch, device: str) -> str:
    """Duration and speed of the epoch, memory in use, and the time left."""
    import torch

    took = time.time() - epoch_start
    per_step = pace["seconds"] / max(1, pace["steps"])
    left = per_step * steps_per_epoch * (config.epochs - epoch - 1)
    parts = [f"epoch took {took / 60:.1f} min",
             f"{config.batch_size / max(per_step, 1e-9):.0f} spectra/s"]
    if ram.proc is not None:
        parts.append(f"RAM {ram.used_gb():.1f} GB" + (f" of {ram.limit:g}" if ram.limit else ""))
    if device.startswith("cuda"):
        parts.append(f"GPU {torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved")
    if config.epochs - epoch - 1 > 0:
        parts.append(f"about {left / 3600:.1f} h to go" if left >= 3600
                     else f"about {left / 60:.0f} min to go")
    return "; ".join(parts)


def synth_fingerprint(synth_config) -> str:
    """A short digest of the synthetic-data settings (the whole mixture), kept in the resume
    state so that a run is never continued on other spectra than it started with."""
    import hashlib

    return hashlib.sha256(repr(synth_config).encode()).hexdigest()[:16]


def _check_same_run(saved: dict, now: dict) -> None:
    """Refuse to resume with settings that change the model, the schedule or the frames."""
    keys = ("arch", "model_size", "singly_charged", "epochs", "train_len", "batch_size", "lr",
            "base_seed", "z_max", "evidence_channels", "comb_prior", "fresh_each_epoch",
            "weight_decay", "heat_weight")
    diff = {k: (saved.get(k), now.get(k)) for k in keys if saved.get(k) != now.get(k)}
    if diff:
        raise SystemExit(
            "cannot resume: these settings differ from the paused run's (saved, now): "
            f"{diff}. Resume with the original command plus --resume, or use --restart.")


def _atomic_save(obj, path: Path) -> None:
    """Write ``obj`` to ``path`` so that a crash or power cut leaves the old file or the new
    one, never a truncated one: write a temporary file, flush it to disk, then rename."""
    import os

    import torch

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    # on Windows the rename fails while another program (a virus scanner, a backup or sync
    # tool, Explorer's preview) briefly holds either file open; such holds last moments
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def _save(path: Path, state, config, in_ch: int, val, comb, metric: str) -> None:
    from .grid import DEFAULT_STEP

    _atomic_save({"state_dict": state, "config": asdict(config),
                  "synth_config_z_max": config.z_max, "model_size": config.model_size,
                  "z_max": config.z_max, "in_ch": in_ch,
                  "evidence_channels": config.evidence_channels,
                  "comb_prior": config.comb_prior, "arch": config.arch,
                  "singly_charged": config.singly_charged, "selection_metric": metric,
                  "grid_step": DEFAULT_STEP, "window": 32768,
                  "val": val, "comb_baseline": comb}, path)


def comb_baseline(loader, num_workers: int | None = None) -> dict:
    """Charge metrics of the comb estimator, as the pipeline runs it, on ``loader``'s frames.

    Every validation frame is generated again with its whole spectrum, which CombPredictor reads
    as ``process_run`` does: all of it at once (a crop would hide the far neighbours of low
    charge states), with isotope-spacing charges on Orbitrap and FT-ICR spectra
    (:mod:`lcmsdeconv.nn.baseline`). Its charges inside each frame's crop are scored exactly as
    :func:`evaluate` scores a network, batch by batch, so the two numbers compare. A frame
    rendered onto its crop alone falls back to the comb's answer in feature channel 3, which
    ``estimator`` then says. Also returns every frame's kind of spectrum (``kinds``, in loader
    order) and per-kind accuracies with both weightings (``by_kind``, see
    :class:`~lcmsdeconv.nn.checks.ChargeScoreboard`).
    """
    import torch
    from torch.utils.data import DataLoader

    from ..synth.dataset import CombBarFrames, mp_context, worker_init
    from .checks import ChargeScoreboard, charge_metrics

    source = getattr(loader.dataset, "frames", loader.dataset)  # CachedFrames keeps it
    workers = loader.num_workers if num_workers is None else num_workers
    frames = DataLoader(CombBarFrames(source), batch_size=loader.batch_size,
                        num_workers=workers, shuffle=False,
                        worker_init_fn=worker_init if workers else None,
                        **mp_context(workers))
    sums: dict[str, float] = {}
    bands: dict[str, list[float]] = {}
    kinds: list[str] = []
    hows: set[str] = set()
    board = ChargeScoreboard()
    nb = 0
    for f0, true, zc, kind, how in frames:
        n, length = zc.shape
        zc = zc.long().clamp(0, 100)
        logits = torch.full((n, 101, length), -10.0)
        logits.scatter_(1, zc[:, None, :], 10.0)
        # where the comb has no answer it reads no charge: the ion-class argmax would otherwise
        # land on class 1 and score singly charged bins as correct
        logits[:, 100, :] = torch.where(zc == 0, torch.zeros_like(logits[:, 100, :]),
                                        logits[:, 100, :])
        m = charge_metrics(logits, {"features": f0, "topk_z": true})
        for k in ("weighted_accuracy", "charge_accuracy_given_ion", "charge_accuracy_given_ion_z2"):
            sums[k] = sums.get(k, 0.0) + m[k]
        for k, v in m["bands_given_ion"].items():
            bands.setdefault(k, []).append(v)
        for i in range(n):
            board.add(kind[i], "comb", zc[i].numpy(), true[i, :, 0].numpy(), f0[i, 0].numpy())
        kinds += list(kind)
        hows |= set(how)
        nb += 1
    nb = max(nb, 1)
    estimator = {"pipeline": "whole spectrum, isotope charges on Orbitrap",
                 "channel 3": "channel-3 answer inside the crop"}
    return {"charge_acc": sums.get("weighted_accuracy", 0.0) / nb,
            "charge_acc_given_ion": sums.get("charge_accuracy_given_ion", 0.0) / nb,
            "charge_acc_given_ion_z2": sums.get("charge_accuracy_given_ion_z2", 0.0) / nb,
            "bands_given_ion": {k: float(np.mean(v)) for k, v in bands.items()},
            "estimator": "; ".join(estimator[h] for h in sorted(hows)) or "no frames",
            "kinds": kinds, "by_kind": board.summary()}


def by_kind_lines(net: dict | None, comb: dict | None) -> list[str]:
    """Lines setting a network's per-kind accuracy at z >= 2 beside the comb's, both weighted
    by log-SNR and by intensity (``by_kind`` of :func:`evaluate` and :func:`comb_baseline`)."""
    from .baseline import KIND_NAMES

    def cell(board: dict | None, kind: str, est: str) -> str | None:
        d = (board or {}).get(kind, {}).get(est)
        if not d:
            return None
        a, b = d.get("log-SNR", {}).get("z>=2"), d.get("intensity", {}).get("z>=2")
        if a is None or b is None:
            return None
        return f"{a:.3f} / {b:.3f}"

    lines = []
    for kind, name in KIND_NAMES.items():
        parts = [f"{label} {c}" for label, c in (("network", cell(net, kind, "network")),
                                                 ("comb", cell(comb, kind, "comb"))) if c]
        if parts:
            lines.append(f"    {name:15s} " + "   ".join(parts))
    if lines:
        lines.insert(0, "  charge of z >= 2 ion bins read right (log-SNR / intensity weighted):")
    return lines


def _log(path: Path, record: dict) -> None:
    import json

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def evaluate(model, loader, loss_fn, device, kinds: list[str] | None = None) -> dict:
    """Validation loss and charge metrics (see :func:`lcmsdeconv.nn.checks.charge_metrics`).

    ``charge_acc`` is the log-SNR-weighted top-1 accuracy over ion-bearing bins with the
    background class competing; ``charge_acc_given_ion`` takes the top charge among the ion
    classes only, as inference does (``..._z2``: on bins of charge two or more), and is the
    quantity checkpoints are selected by. With ``kinds`` (each frame's kind of spectrum, in
    loader order) ``by_kind`` adds per-kind accuracies weighted by log-SNR and by intensity.
    """
    import torch

    from .checks import ChargeScoreboard, charge_metrics

    model.eval()
    tot = 0.0
    nb = 0
    sums: dict[str, float] = {}
    bands: dict[str, list[float]] = {}
    never = 0
    board = ChargeScoreboard() if kinds else None
    pos = 0
    with torch.no_grad():
        for sample in loader:
            batch = _batch_dict(sample, device)
            out = model(batch["features"])
            losses = loss_fn(out, batch)
            tot += float(losses["loss"])
            nb += 1
            if board is not None:
                pred = (out["charge_logits"][:, 1:, :].argmax(1) + 1).cpu().numpy()
                true = batch["topk_z"][:, :, 0].cpu().numpy()
                f0 = batch["features"][:, 0, :].cpu().numpy()
                for i in range(pred.shape[0]):
                    if pos + i < len(kinds):
                        board.add(kinds[pos + i], "network", pred[i], true[i], f0[i])
                pos += pred.shape[0]
            m = charge_metrics(out["charge_logits"], batch)
            for k in ("weighted_accuracy", "charge_accuracy_given_ion",
                      "charge_accuracy_given_ion_z2", "harmonic_error",
                      "neighbour_error", "missed_as_background", "background_called_ion"):
                sums[k] = sums.get(k, 0.0) + m[k]
            for k, v in m["bands_given_ion"].items():
                bands.setdefault(k, []).append(v)
            never = max(never, m["classes_never_predicted"])
    nb = max(1, nb)
    res = {"val_loss": tot / nb, "charge_acc": sums.get("weighted_accuracy", 0.0) / nb,
           "charge_acc_given_ion": sums.get("charge_accuracy_given_ion", 0.0) / nb,
           "charge_acc_given_ion_z2": sums.get("charge_accuracy_given_ion_z2", 0.0) / nb}
    for k in ("harmonic_error", "neighbour_error", "missed_as_background", "background_called_ion"):
        res[k] = sums.get(k, 0.0) / nb
    res["bands_given_ion"] = {k: float(np.mean(v)) for k, v in bands.items()}
    res["classes_never_predicted"] = never
    if board is not None:
        res["by_kind"] = board.summary()
    return res
