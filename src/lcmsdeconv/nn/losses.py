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


def bin_weights(features: torch.Tensor, topk_z: torch.Tensor, topk_w: torch.Tensor,
                floor: float = 0.02, cap: float = 4.0, background_scale: float = 0.1) -> torch.Tensor:
    """Per-bin loss weights: log-SNR, scaled by how much of the bin is ion signal.

    The log-SNR feature alone gives a noise bin a weight of 0.3-1, and noise bins outnumber ion
    bins several times over, so background ended up carrying 70-95 % of the loss and the
    cheapest way to lower it was to call everything background. Scaling the SNR term by
    ``background_scale + (1 - background_scale) * ion_share`` keeps background bins in the loss
    (they still teach what is not an ion) without letting them dominate it.
    """
    ch0 = features[:, 0, :]
    ion_share = (topk_w * (topk_z > 0).to(topk_w.dtype)).sum(dim=2).clamp(0.0, 1.0)
    scale = background_scale + (1.0 - background_scale) * ion_share
    return floor + ch0.clamp(0, cap) * scale


class ChargeLoss(nn.Module):
    """Soft charge cross-entropy plus the apex heatmap loss.

    ``ignore_charges`` removes bins carrying those charges from the charge loss in proportion
    to their share: with ``(1,)``, singly charged ions are neither rewarded nor penalised, so the
    network spends nothing on them while still seeing them in its input.
    """

    def __init__(self, heat_weight: float = 0.1, weight_floor: float = 0.02, weight_cap: float = 4.0,
                 background_scale: float = 0.1, ignore_charges: tuple[int, ...] = ()):
        super().__init__()
        self.heat_weight = heat_weight
        self.weight_floor = weight_floor
        self.weight_cap = weight_cap
        self.background_scale = background_scale
        self.ignore_charges = tuple(int(z) for z in ignore_charges)

    def forward(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["charge_logits"]
        topk_w = batch["topk_w"]
        w = bin_weights(batch["features"], batch["topk_z"], topk_w, self.weight_floor,
                        self.weight_cap, self.background_scale)
        if self.ignore_charges:
            ignored = torch.zeros_like(topk_w, dtype=torch.bool)
            for z in self.ignore_charges:
                ignored = ignored | (batch["topk_z"] == z)
            share = (topk_w * ignored.to(topk_w.dtype)).sum(dim=2).clamp(0.0, 1.0)
            # the bin counts in proportion to what is not ignored, and the ignored charges leave
            # the target: a bin half singly charged and half z = 5 teaches z = 5 at half weight
            # and nothing about z = 1
            w = w * (1.0 - share)
            topk_w = torch.where(ignored, torch.zeros_like(topk_w), topk_w)
        ce = soft_charge_ce(logits, batch["topk_z"], topk_w, w)
        hl = centernet_focal(outputs["apex_logit"], batch["heat"])
        total = ce + self.heat_weight * hl
        return {"loss": total, "ce": ce.detach(), "heat": hl.detach()}
