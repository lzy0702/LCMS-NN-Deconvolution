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
    model = build_model(config.model_size, z_max=config.z_max, in_ch=in_ch,
                        comb_prior=config.comb_prior, arch=config.arch).to(device)
    if config.init:
        state = torch.load(config.init, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state)
        print(f"started from {config.init}")
    print(f"model {config.arch}/{config.model_size}: {model.num_params()/1e6:.3f}M params "
          f"on {device}; singly charged ions: {config.singly_charged}")

    tl, vl = make_dataloaders(synth_config, config.train_len, config.val_len,
                              batch_size=config.batch_size, num_workers=config.num_workers,
                              base_seed=config.base_seed, cache=config.cache_frames)
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = max(1, len(tl)) * config.epochs
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
    if config.resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device, weights_only=False)
        _check_same_run(st.get("config", {}), asdict(config))
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
            print(f"resumed from {resume_path}: epoch {epoch0}, step {step0} of {len(tl)}")
    elif config.resume:
        print(f"nothing to resume at {resume_path}; starting from the beginning")
    for f in pause_files():
        f.unlink()
        print(f"removed {f} left from the last pause")

    if comb is None:
        # Before spending hours: the true labels must beat every constant answer by a clear
        # margin on real batches, or the optimiser will settle on the constant.
        from .checks import degeneracy_report

        first = next(iter(tl), None)
        if first is not None:
            rep = degeneracy_report(_batch_dict(first, "cpu"), loss_fn)
            print(f"loss of constant predictors relative to the truth: {rep['margins']}")
            _log(log_path, {"degeneracy": rep})
        # the bar to clear: the comb estimator's own charge reading on the validation frames
        comb = comb_baseline(vl)
        print(f"comb estimator on the validation frames: {metric} {comb[metric]:.3f} "
              f"bands {comb['bands_given_ion']}")
        _log(log_path, {"comb_baseline": comb})

    def save_state(epoch: int, step: int) -> None:
        state = {"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                 "best_state": best_state, "best_key": best_key, "best_loss": best_loss,
                 "epoch": epoch, "step": step, "comb": comb, "config": asdict(config),
                 "running": (run_sum, run_n)}
        _atomic_save(state, resume_path)

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

    def stop_here(epoch: int, step: int, why: str) -> tuple:
        save_state(epoch, step)
        print(f"{why}: state saved to {resume_path}; run the same command with --resume to "
              f"continue (epoch {epoch}, step {step})", flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    session_start = time.time()
    last_save = time.time()
    try:
        for epoch in range(epoch0, config.epochs):
            model.train()
            if epoch != epoch0:
                run_sum, run_n = 0.0, 0
            if config.fresh_each_epoch:
                tl.dataset.base_seed = config.base_seed + epoch * config.train_len
            first_step = step0 if epoch == epoch0 else 0
            loader = tl if first_step == 0 else _loader_from(tl, first_step * config.batch_size)
            i = first_step - 1
            try:
                for i, sample in enumerate(loader, start=first_step):
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
                        print(f"  epoch {epoch} step {i}/{len(tl)} loss {loss_val:.4f} "
                              f"ce {float(losses['ce']):.4f} heat {float(losses['heat']):.4f}",
                              flush=True)
                    if paused():
                        return stop_here(epoch, i + 1, "paused")
                    if config.max_seconds and (time.time() - session_start) > config.max_seconds:
                        return stop_here(epoch, i + 1, "time budget reached")
                    if time.time() - last_save > 60.0 * config.save_every_minutes:
                        save_state(epoch, i + 1)
                        last_save = time.time()
            except (RuntimeError, OSError):
                # Ctrl+C also reaches the data-loading workers, which then exit; on Windows the
                # interrupted wait can surface as an OSError (InterruptedError) instead
                if pause["requested"]:
                    return stop_here(epoch, i + 1, "paused")
                raise
            try:
                val_ema = evaluate(ema.shadow, vl, loss_fn, device)
                val_live = evaluate(model, vl, loss_fn, device)
            except (RuntimeError, OSError):
                # interrupted while validating: the epoch's training is done, so resuming
                # starts with this validation
                if pause["requested"]:
                    return stop_here(epoch, len(tl), "paused")
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
            if best_state is None or key(val) > best_key:
                best_key = key(val)
                best_loss = val["val_loss"]
                best_state = copy.deepcopy((ema.shadow if best == "ema" else model).state_dict())
                _save(out_path, best_state, config, in_ch, val, comb, metric)
                print(f"  saved {out_path} ({metric} {val[metric]:.3f})")
            run_sum, run_n = 0.0, 0
            save_state(epoch + 1, 0)
            last_save = time.time()
            if paused() and epoch + 1 < config.epochs:
                return stop_here(epoch + 1, 0, "paused")
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)

    if best_state is None:
        best_state = model.state_dict()
        _save(out_path, best_state, config, in_ch, None, comb, metric)
    print(f"finished; best checkpoint {out_path} (val_loss {best_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


def _loader_from(loader, start: int):
    """``loader`` over its dataset from item ``start`` on (resuming inside an epoch)."""
    from torch.utils.data import DataLoader, Subset

    ds = Subset(loader.dataset, range(start, len(loader.dataset)))
    return DataLoader(ds, batch_size=loader.batch_size, num_workers=loader.num_workers,
                      shuffle=False, drop_last=True)


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
    os.replace(tmp, path)


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


def comb_baseline(loader) -> dict:
    """Charge metrics of the comb's answer in feature channel 3 on ``loader``.

    This is the ladder comb without the isotope-spacing evidence the pipeline adds for Orbitrap
    spectra, so on those it understates the pipeline's comb; scripts/eval_charge_bins.py scores
    a trained model against the pipeline's estimator itself.
    """
    import torch

    from .checks import charge_metrics

    sums: dict[str, float] = {}
    bands: dict[str, list[float]] = {}
    nb = 0
    for sample in loader:
        batch = _batch_dict(sample, "cpu")
        feats = batch["features"]
        n, _c, length = feats.shape
        zc = torch.round(feats[:, 3, :] * 64).long().clamp(0, 100)
        logits = torch.full((n, 101, length), -10.0)
        logits.scatter_(1, zc[:, None, :], 10.0)
        # where the comb has no answer (channel 3 = 0) it reads no charge: the ion-class argmax
        # would otherwise land on class 1 and score singly charged bins as correct
        logits[:, 100, :] = torch.where(zc == 0, torch.full_like(zc, 0, dtype=logits.dtype),
                                        logits[:, 100, :])
        m = charge_metrics(logits, batch)
        for k in ("weighted_accuracy", "charge_accuracy_given_ion", "charge_accuracy_given_ion_z2"):
            sums[k] = sums.get(k, 0.0) + m[k]
        for k, v in m["bands_given_ion"].items():
            bands.setdefault(k, []).append(v)
        nb += 1
    nb = max(nb, 1)
    return {"charge_acc": sums.get("weighted_accuracy", 0.0) / nb,
            "charge_acc_given_ion": sums.get("charge_accuracy_given_ion", 0.0) / nb,
            "charge_acc_given_ion_z2": sums.get("charge_accuracy_given_ion_z2", 0.0) / nb,
            "bands_given_ion": {k: float(np.mean(v)) for k, v in bands.items()}}


def _log(path: Path, record: dict) -> None:
    import json

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def evaluate(model, loader, loss_fn, device) -> dict:
    """Validation loss and charge metrics (see :func:`lcmsdeconv.nn.checks.charge_metrics`).

    ``charge_acc`` is the intensity-weighted top-1 accuracy over ion-bearing bins with the
    background class competing; ``charge_acc_given_ion`` takes the top charge among the ion
    classes only, as inference does, and is the quantity checkpoints are selected by.
    """
    import torch

    from .checks import charge_metrics

    model.eval()
    tot = 0.0
    nb = 0
    sums: dict[str, float] = {}
    bands: dict[str, list[float]] = {}
    never = 0
    with torch.no_grad():
        for sample in loader:
            batch = _batch_dict(sample, device)
            out = model(batch["features"])
            losses = loss_fn(out, batch)
            tot += float(losses["loss"])
            nb += 1
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
    return res
