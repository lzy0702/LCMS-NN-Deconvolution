"""Training loop for the charge network (CPU proof runs and GPU full runs)."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path


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
    device = _device(config.device)
    model = build_model(config.model_size, z_max=config.z_max).to(device)
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
    start = time.time()
    stop = False
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
        best = "ema" if val_ema["val_loss"] <= val_live["val_loss"] else "live"
        val = val_ema if best == "ema" else val_live
        history.append({"epoch": epoch, "train_loss": running / max(1, i + 1),
                        "ema": val_ema, "live": val_live, "best": best})
        print(f"epoch {epoch}: ema={val_ema} live={val_live} -> keeping {best}")
        if best_state is None or val["val_loss"] < best_loss:
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
                "z_max": config.z_max}, out_path)
    print(f"saved {out_path} (best val_loss {best_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, loss_fn, device) -> dict:
    import torch

    model.eval()
    tot = 0.0
    charge_acc = 0.0
    n = 0
    nb = 0
    with torch.no_grad():
        for sample in loader:
            batch = _batch_dict(sample, device)
            out = model(batch["features"])
            losses = loss_fn(out, batch)
            tot += float(losses["loss"])
            nb += 1
            # intensity-weighted top-1 charge accuracy
            pred = out["charge_logits"].argmax(1)  # [N, L]
            true = batch["topk_z"][:, :, 0]  # dominant class
            w = batch["features"][:, 0, :]
            mask = w > 0.5
            charge_acc += float(((pred == true) & mask).sum())
            n += float(mask.sum())
    return {"val_loss": tot / max(1, nb), "charge_acc": charge_acc / max(1.0, n)}
