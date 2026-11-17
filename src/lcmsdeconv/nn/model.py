"""1D U-Net for per-bin charge-state assignment on the log(m/z) grid.

Outputs a per-bin charge-share distribution over classes 0..Zmax (0 = non-ion) and an apex
heatmap. Only ONNX-friendly ops are used (Conv1d, GroupNorm, SiLU, nearest upsample), so the
trained model exports cleanly and runs on onnxruntime (CPU / iGPU / NPU) with no PyTorch at
deployment time.
"""

from __future__ import annotations

import torch
from torch import nn

from .constants import DOWNSAMPLE  # noqa: F401  (re-exported for callers)


def _gn(ch: int) -> nn.GroupNorm:
    groups = 1
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 7, dilation: int = 1):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.norm = _gn(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int = 7, dilation: int = 1):
        super().__init__()
        self.b1 = ConvBlock(ch, ch, k, dilation)
        self.b2 = ConvBlock(ch, ch, k, dilation)

    def forward(self, x):
        return x + self.b2(self.b1(x))


class ChargeUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        z_max: int = 100,
        channels: tuple[int, ...] = (16, 32, 48, 64, 96, 128, 160),
        kernel: int = 7,
        bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        comb_prior: bool = False,
    ):
        super().__init__()
        self.z_max = z_max
        self.n_classes = z_max + 1
        #: with per-charge comb evidence among the inputs (channels 4..), add it straight to the
        #: charge logits (learnable scale) and start the network's own head at zero, so the model
        #: begins at the comb's answer and learns corrections instead of rediscovering it
        self.n_evidence = max(0, min(in_ch - 4, z_max)) if comb_prior else 0
        self.stem = ConvBlock(in_ch, channels[0], kernel)
        self.down = nn.ModuleList()
        self.down_res = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_res.append(ResBlock(channels[i], kernel))
            self.down.append(nn.Conv1d(channels[i], channels[i + 1], kernel, stride=2, padding=kernel // 2))
        cb = channels[-1]
        self.bottleneck = nn.Sequential(*[ResBlock(cb, kernel, d) for d in bottleneck_dilations])
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.up_res = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up.append(nn.Conv1d(channels[i], channels[i - 1], kernel, padding=kernel // 2))
            self.up_conv.append(ConvBlock(channels[i - 1] * 2, channels[i - 1], kernel))
            self.up_res.append(ResBlock(channels[i - 1], kernel))
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.charge_head = nn.Conv1d(channels[0], self.n_classes, 1)
        self.apex_head = nn.Conv1d(channels[0], 1, 1)
        if self.n_evidence:
            nn.init.zeros_(self.charge_head.weight)
            nn.init.zeros_(self.charge_head.bias)
            self.prior_scale = nn.Parameter(torch.tensor(8.0))
            self.prior_background = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        inp = x
        x = self.stem(x)
        skips = []
        for res, down in zip(self.down_res, self.down):
            x = res(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, conv, res, skip in zip(self.up, self.up_conv, self.up_res, reversed(skips)):
            x = up(self.upsample(x))
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., : skip.shape[-1]]
            x = conv(torch.cat([x, skip], dim=1))
            x = res(x)
        logits = self.charge_head(x)
        if self.n_evidence:
            ev = inp[:, 4:4 + self.n_evidence, :]
            background = self.prior_background * (1.0 - inp[:, 0:1, :] / 5.0)
            rest = self.n_classes - 1 - self.n_evidence
            parts = [background, self.prior_scale * ev]
            if rest > 0:
                parts.append(torch.zeros_like(logits[:, :rest, :]))
            logits = logits + torch.cat(parts, dim=1)
        return {"charge_logits": logits, "apex_logit": self.apex_head(x)}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(size: str = "proof", z_max: int = 100, in_ch: int = 4,
                comb_prior: bool = False) -> ChargeUNet:
    if size == "proof":
        return ChargeUNet(in_ch, z_max, channels=(8, 16, 24, 32, 48, 64, 80), comb_prior=comb_prior)
    if size == "full":
        return ChargeUNet(in_ch, z_max, channels=(16, 32, 48, 64, 96, 128, 160), comb_prior=comb_prior)
    if size == "small":
        return ChargeUNet(in_ch, z_max, channels=(8, 16, 24, 32, 48, 64), comb_prior=comb_prior)
    raise ValueError(f"Unknown model size {size!r}")
