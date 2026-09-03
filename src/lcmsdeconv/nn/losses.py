"""Losses for the charge network: soft charge cross-entropy and CenterNet focal heatmap."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_charge_ce(
    charge_logits: torch.Tensor,  # [N, C, L]
    topk_z: torch.Tensor,  # [N, L, k] int64
    topk_w: torch.Tensor,  # [N, L, k] float
    weight_bins: torch.Tensor,  # [N, L]
) -> torch.Tensor:
    logp = F.log_softmax(charge_logits, dim=1)  # [N, C, L]
    logp = logp.permute(0, 2, 1)  # [N, L, C]
    gathered = torch.gather(logp, 2, topk_z.clamp(min=0))  # [N, L, k]
    per_bin = -(topk_w * gathered).sum(dim=2)  # [N, L]
    w = weight_bins
    denom = w.sum() + 1e-8
    return (per_bin * w).sum() / denom


def centernet_focal(apex_logit: torch.Tensor, heat: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet focal loss, normalized per positive.

    Crops that contain no peak at all have no positives; normalizing those by 1 would make the
    negative term explode with the crop length, so they are normalized per bin instead.
    """
    p = torch.sigmoid(apex_logit).squeeze(1).clamp(1e-4, 1 - 1e-4)  # [N, L]
    pos = (heat >= 1.0 - 1e-6).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - heat) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n_pos = pos.sum()
    if float(n_pos) < 1.0:
        return neg_loss.sum() / max(heat.numel(), 1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


class ChargeLoss(nn.Module):
    def __init__(self, heat_weight: float = 0.1, weight_floor: float = 0.02, weight_cap: float = 4.0):
        super().__init__()
        self.heat_weight = heat_weight
        self.weight_floor = weight_floor
        self.weight_cap = weight_cap

    def forward(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["charge_logits"]
        # bin weights from the log-SNR feature channel 0
        ch0 = batch["features"][:, 0, :]
        w = self.weight_floor + ch0.clamp(0, self.weight_cap)
        ce = soft_charge_ce(logits, batch["topk_z"], batch["topk_w"], w)
        hl = centernet_focal(outputs["apex_logit"], batch["heat"])
        total = ce + self.heat_weight * hl
        return {"loss": total, "ce": ce.detach(), "heat": hl.detach()}
