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
    in_ch = 4 + config.evidence_channels
    if config.threads:
        torch.set_num_threads(int(config.threads))
    device = _device(config.device)
    model = build_model(config.model_size, z_max=config.z_max, in_ch=in_ch).to(device)
    print(f"model {config.model_size}: {model.num_params()/1e6:.2f}M params on {device}")

    tl, vl = make_dataloaders(synth_config, config.train_len, config.val_len,
                              batch_size=config.batch_size, num_workers=config.num_workers,
                              base_seed=config.base_seed)
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = max(1, len(tl)) * config.epochs
    sched = OneCycleLR(opt, max_lr=config.lr, total_steps=steps, pct_start=0.2)
    loss_fn = ChargeLoss(heat_weight=config.heat_weight)
    ema = EMA(model, config.ema_decay)

    history = []
    best_state = None
    best_loss = float("inf")
    best_key = None
    start = time.time()
    stop = False
    log_path = Path(config.out).with_suffix(".log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Before spending hours: the true labels must beat every constant answer by a clear margin
    # on real batches, or the optimiser will settle on the constant.
    from .checks import degeneracy_report

    first = next(iter(tl), None)
    if first is not None:
        rep = degeneracy_report(_batch_dict(first, "cpu"), loss_fn)
        print(f"loss of constant predictors relative to the truth: {rep['margins']}")
        _log(log_path, {"degeneracy": rep})
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for i, sample in enumerate(tl):
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
            running += loss_val
            if i % config.log_every == 0:
                print(f"  epoch {epoch} step {i}/{len(tl)} loss {loss_val:.4f} "
                      f"ce {float(losses['ce']):.4f} heat {float(losses['heat']):.4f}")
            if config.max_seconds and (time.time() - start) > config.max_seconds:
                print("  reached time budget; stopping")
                stop = True
                break
        val_ema = evaluate(ema.shadow, vl, loss_fn, device)
        val_live = evaluate(model, vl, loss_fn, device)

        def key(v):
            if config.select_by == "accuracy":
                return (v["charge_acc"], -v["val_loss"])
            return (-v["val_loss"],)

        best = "ema" if key(val_ema) >= key(val_live) else "live"
        val = val_ema if best == "ema" else val_live
        history.append({"epoch": epoch, "train_loss": running / max(1, i + 1),
                        "ema": val_ema, "live": val_live, "best": best,
                        "seconds": time.time() - start})
        _log(log_path, history[-1])
        print(f"epoch {epoch}: ema={val_ema} live={val_live} -> keeping {best}")
        if best_state is None or key(val) > best_key:
            best_key = key(val)
            best_loss = val["val_loss"]
            best_state = copy.deepcopy((ema.shadow if best == "ema" else model).state_dict())
        if stop:
            break

    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    torch.save({"state_dict": best_state, "config": asdict(config),
                "synth_config_z_max": config.z_max, "model_size": config.model_size,
                "z_max": config.z_max, "in_ch": in_ch,
                "evidence_channels": config.evidence_channels}, out_path)
    print(f"saved {out_path} (best val_loss {best_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


def _log(path: Path, record: dict) -> None:
    import json

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def evaluate(model, loader, loss_fn, device) -> dict:
    """Validation loss and charge metrics (see :func:`lcmsdeconv.nn.checks.charge_metrics`).

    ``charge_acc`` is the intensity-weighted top-1 charge accuracy over ion-bearing bins, the
    quantity checkpoints are selected by.
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
            for k in ("weighted_accuracy", "harmonic_error", "neighbour_error",
                      "missed_as_background", "background_called_ion"):
                sums[k] = sums.get(k, 0.0) + m[k]
            for k, v in m["bands"].items():
                bands.setdefault(k, []).append(v)
            never = max(never, m["classes_never_predicted"])
    nb = max(1, nb)
    res = {"val_loss": tot / nb, "charge_acc": sums.get("weighted_accuracy", 0.0) / nb}
    for k in ("harmonic_error", "neighbour_error", "missed_as_background", "background_called_ion"):
        res[k] = sums.get(k, 0.0) / nb
    res["bands"] = {k: float(np.mean(v)) for k, v in bands.items()}
    res["classes_never_predicted"] = never
    return res
