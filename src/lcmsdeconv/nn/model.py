"""Networks for per-bin charge-state assignment on the log(m/z) grid.

``ChargeUNet`` is a 1D U-Net; ``LadderNet`` reads every candidate charge's ladder at its exact
offsets on the grid (see its docstring).

Outputs a per-bin charge-share distribution over classes 0..Zmax (0 = non-ion) and an apex
heatmap. Only ONNX-friendly ops are used (Conv1d, GroupNorm, SiLU, nearest upsample), so the
trained model exports cleanly and runs on onnxruntime (CPU / iGPU / NPU) with no PyTorch at
deployment time.
"""

from __future__ import annotations

import torch
from torch import nn

from .constants import (
    DOWNSAMPLE,  # noqa: F401  (re-exported for callers)
    LADDER_ACTIVE_TOP,
    LADDER_WINDOW,
    ladder_active_top,
)


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


def _shift(t: torch.Tensor, s: int) -> torch.Tensor:
    """``y[..., i] = t[..., i + s]``, zero where ``i + s`` falls outside (static ``s``)."""
    if s == 0:
        return t
    length = t.shape[-1]
    if abs(s) >= length:
        return torch.zeros_like(t)
    if s > 0:
        return nn.functional.pad(t[..., s:], (0, s))
    return nn.functional.pad(t[..., :s], (-s, 0))


class LadderNet(nn.Module):
    """Charge assignment by reading every candidate charge's ladder at its exact offsets.

    On the grid u = ln(m/z -/+ m_p) the charge states of one species lie at u = ln M - ln z, so
    if a bin at u carries charge z, the same species' charge z + k sits at u + ln(z / (z + k)),
    whatever its mass. A convolutional network has to discover these offsets - thousands of bins
    long and different for every z - from data; here they are given. For each charge z up to
    ``z_ladder`` the network reads a learned local description of the spectrum (peak shape,
    isotope spacing) at the bin itself, at its neighbours z +/- 1..3, and at the two positions
    that only a reading at 2z would fill (u + ln(2z / (2z +/- 1)), halfway between the z ladder's
    rungs), and a small network shared across charges, with a learned bias per charge, scores
    the reading. The comb estimator's own score - the bin's log-SNR times the mean of its
    neighbours' - is added with a learnable weight and the learned part starts at zero, so an
    untrained model assigns charges as the comb does and training learns corrections. A
    depthwise convolution over neighbouring bins, initialised to the identity, lets the bins of
    one peak agree.

    Ladders are read only at the ``active_top`` strongest bins of a window whose log-SNR feature
    also exceeds ``active_level`` (ions occupy a few percent of the bins; without a vendor
    threshold most of the rest is noise above any fixed level); elsewhere every charge gets one
    learned constant logit. Training uses a sparse gather over those bins; ``dense = True`` computes the same
    function with static shifts over the whole window, which is what is exported to ONNX.
    Charges above ``z_ladder`` get a constant low logit. The offsets are fixed for one grid
    step and window length, which inference uses throughout; an offset is kept only if it is
    shorter than the window, so a window of 131 072 bins reads z = 1 <-> 2, 3, 4 (34 657, 54 931
    and 69 315 bins), which a 32 768-bin window never holds.

    With ``iso`` the input has two more channels (4 and 5): the charge read from resolved
    isotope spacing at each bin (0 = none; empty on time-of-flight spectra) and the strength of
    that reading (:func:`lcmsdeconv.nn.features.isotope_channels`). For each ladder charge z the
    scorer then
    also reads ``oh = relu(1 - |zc - z|)`` (1 exactly when the isotopes read z), ``oh * strength``
    and ``(zc >= 0.5) * (1 - oh)`` (the isotopes read another charge), and the charge-z logit
    gets ``iso_scale * oh * clip((strength - 0.2) * 10, 0, 1)``, with ``iso_scale`` learnable and
    starting at 10, above the comb prior's ceiling of ``prior_scale`` (8): the untrained network
    reads charges as the comb with isotope charges does, the isotope charge where the spacing is
    resolved and its neighbours are strong, the ladder elsewhere. Without ``iso`` the network is
    exactly the one trained before these channels existed.
    """

    def __init__(self, in_ch: int = 4, z_max: int = 100, z_ladder: int = 64, step: float = 2e-5,
                 width: int = 16, proj: int = 4, hidden: int = 16, ks: tuple[int, ...] = (1, 2, 3),
                 window: int = LADDER_WINDOW, active_level: float = 0.5,
                 active_top: int = LADDER_ACTIVE_TOP, iso: bool = False):
        super().__init__()
        import math

        self.z_max = z_max
        self.n_classes = z_max + 1
        self.z_ladder = min(z_ladder, z_max)
        self.in_ch = in_ch
        self.window = window
        self.active_level = active_level
        self.active_top = active_top
        self.iso = bool(iso)
        if self.iso and in_ch < 6:
            raise ValueError("a LadderNet reading isotope channels needs six input channels "
                             f"(the four base features, then charge and strength), not {in_ch}")
        self.dense = False

        def shift(z_from: float, z_to: float) -> int | None:
            if z_to < 1:
                return None
            sh = int(round(math.log(z_from / z_to) / step))
            return sh if abs(sh) < window else None

        offsets, comb, comb_n = [], [], []
        for z in range(1, self.z_ladder + 1):
            o: list[int | None] = [0]
            c: list[int] = []
            n_defined = 0
            for k in ks:
                for zt in (z + k, z - k):
                    sh = shift(z, zt)
                    o.append(sh)
                    if zt >= 1 and sh != 0:
                        # the comb counts every neighbour charge that exists; one outside the
                        # window contributes zero but still counts (lcmsdeconv.nn.features)
                        n_defined += 1
                        if sh is not None:
                            c.append(sh)
            o.append(shift(2 * z, 2 * z + 1))
            o.append(shift(2 * z, 2 * z - 1))
            offsets.append(o)
            comb.append(c)
            comb_n.append(n_defined)
        self.offsets = offsets  # per charge: self, z +/- k, two 2z-only positions (None = absent)
        self.comb_offsets = comb
        self.comb_count = comb_n
        self.n_pos = len(offsets[0])
        big = 10 * window  # sentinel: outside any window
        self._absent = big
        #: the longest offset read (the sparse path pads its input by this much)
        self.reach = max([abs(v) for o in offsets for v in o if v is not None]
                         + [abs(v) for c in comb for v in c] + [1])
        self.register_buffer("off_t", torch.tensor([[big if v is None else v for v in o]
                                                    for o in offsets], dtype=torch.long),
                             persistent=False)
        n_comb = max(1, max(len(c) for c in comb))
        self.register_buffer("comb_t", torch.tensor([c + [big] * (n_comb - len(c)) for c in comb],
                                                    dtype=torch.long), persistent=False)
        self.register_buffer("comb_den", torch.tensor([max(n, 1) for n in comb_n],
                                                      dtype=torch.float32), persistent=False)

        self.encoder = nn.Sequential(ConvBlock(2, width, 9), ConvBlock(width, width, 9, 2),
                                     ConvBlock(width, width, 9, 4), ConvBlock(width, width, 9, 8))
        self.proj = nn.Conv1d(width, proj, 1)
        # with isotope channels three more inputs per charge follow the ladder's J x P features
        self.mlp1 = nn.Linear(self.n_pos * proj + (3 if self.iso else 0), hidden)
        self.z_bias = nn.Parameter(torch.zeros(self.z_ladder, hidden))
        self.mlp2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mlp2.weight)
        nn.init.zeros_(self.mlp2.bias)
        self.inactive = nn.Parameter(torch.tensor(0.0))
        self.background = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.background.weight)
        nn.init.zeros_(self.background.bias)
        self.prior_scale = nn.Parameter(torch.tensor(8.0))
        self.prior_background = nn.Parameter(torch.tensor(2.0))
        n = self.z_ladder + 1
        self.smooth = nn.Conv1d(n, n, 5, padding=2, groups=n, bias=False)
        with torch.no_grad():
            self.smooth.weight.zero_()
            self.smooth.weight[:, 0, 2] = 1.0
        self.apex_head = nn.Conv1d(width, 1, 1)
        # start the apex heatmap near p = 0.01 everywhere, as for CenterNet: at p = 0.5 the
        # focal loss's negative term over 32 768 bins swamps the charge loss at first
        nn.init.zeros_(self.apex_head.weight)
        nn.init.constant_(self.apex_head.bias, -4.6)
        if self.iso:
            # above the ladder prior's ceiling (prior_scale x 1): a strong isotope reading wins
            self.iso_scale = nn.Parameter(torch.tensor(10.0))
            self.register_buffer("z_ids", torch.arange(1, self.z_ladder + 1, dtype=torch.float32),
                                 persistent=False)

    # ------------------------------------------------------------------ ladder scores
    def _sparse(self, q: torch.Tensor, s0: torch.Tensor, active: torch.Tensor,
                iso: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Ladder logits ``[N, z_ladder, L]``: scored at active bins, constant elsewhere.
        ``iso``: the isotope charge and strength ``[N, L]`` (networks built with ``iso``)."""
        n_b, n_p, length = q.shape
        out = self.inactive.expand(n_b, self.z_ladder, length).clone()
        b_idx, i_idx = active.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            return out
        # every position outside the window reads zero: the window is padded with zeros by the
        # longest offset on the left and by one window more on the right, where the absent
        # positions (sentinel offsets) are sent, so no index needs clamping or masking
        m = self.reach
        lp = 2 * length + 2 * m
        start = b_idx * lp + i_idx + m  # [n]: each active bin in the padded rows
        qf = nn.functional.pad(q, (m, length + m)).permute(0, 2, 1).reshape(n_b * lp, n_p)
        off = torch.where(self.off_t == self._absent, length + m, self.off_t)
        g = qf[start[:, None, None] + off[None]]  # [n, Z, J, P]
        feats = g.reshape(g.shape[0], self.z_ladder, -1)
        if iso is None:
            hid = self.mlp1(feats) + self.z_bias[None]
        else:
            zc = iso[0][b_idx, i_idx][:, None]  # [n, 1]
            st = iso[1][b_idx, i_idx][:, None]
            oh = torch.relu(1.0 - torch.abs(zc - self.z_ids[None]))  # [n, Z]
            other = (zc >= 0.5).to(q.dtype) * (1.0 - oh)
            # mlp1 over [ladder features, oh, oh * st, other], its last three inputs applied
            # apart so that the ladder features are not copied to append them
            jp = feats.shape[-1]
            w1 = self.mlp1.weight
            extra = torch.stack([oh, oh * st, other], dim=-1)  # [n, Z, 3]
            hid = (nn.functional.linear(feats, w1[:, :jp], self.mlp1.bias)
                   + nn.functional.linear(extra, w1[:, jp:]) + self.z_bias[None])
        del g, feats
        learned = self.mlp2(nn.functional.silu(hid)).squeeze(-1)  # [n, Z]
        sf = nn.functional.pad(s0, (m, length + m)).reshape(n_b * lp)
        off_c = torch.where(self.comb_t == self._absent, length + m, self.comb_t)
        nb = sf[start[:, None, None] + off_c[None]]  # [n, Z, C]
        centre = sf[start]
        comb = centre[:, None] * nb.sum(-1) / self.comb_den[None]
        val = learned + self.prior_scale * comb
        if iso is not None:
            val = val + self.iso_scale * oh * torch.clamp((st - 0.2) * 10.0, 0.0, 1.0)
        out[b_idx, :, i_idx] = val
        return out

    def _dense(self, q: torch.Tensor, s0: torch.Tensor, active: torch.Tensor,
               iso: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """The same function with static shifts over every bin (ONNX export)."""
        w1 = self.mlp1.weight  # [H, J*P (+3)]
        n_p = q.shape[1]
        zero_q = torch.zeros_like(q)
        if iso is not None:
            zc, st = iso[0][:, None, :], iso[1][:, None, :]  # [N, 1, L]
            has = (zc >= 0.5).to(q.dtype)
            gate = torch.clamp((st - 0.2) * 10.0, 0.0, 1.0)
        rows = []
        for zi in range(self.z_ladder):
            parts = [zero_q if s is None else _shift(q, s) for s in self.offsets[zi]]
            if iso is not None:
                oh = torch.relu(1.0 - torch.abs(zc - float(zi + 1)))
                parts += [oh, oh * st, has * (1.0 - oh)]
            hid = nn.functional.conv1d(torch.cat(parts, dim=1), w1[:, :, None], self.mlp1.bias)
            hid = hid + self.z_bias[zi][None, :, None]
            learned = nn.functional.conv1d(nn.functional.silu(hid), self.mlp2.weight[:, :, None],
                                           self.mlp2.bias)
            nb = self.comb_offsets[zi]
            if nb:
                acc = _shift(s0, nb[0])
                for sh in nb[1:]:
                    acc = acc + _shift(s0, sh)
                comb = s0 * acc / float(self.comb_count[zi])
            else:
                comb = torch.zeros_like(s0)
            row = learned + self.prior_scale * comb
            if iso is not None:
                row = row + self.iso_scale * oh * gate
            rows.append(row)
        del n_p
        dense = torch.cat(rows, dim=1)
        return torch.where(active[:, None, :], dense, self.inactive.expand_as(dense))

    def forward(self, x):
        s0 = x[:, 0:1, :] / 5.0
        h = self.encoder(torch.cat([s0, x[:, 1:2, :]], dim=1))
        q = self.proj(h)
        level = x[:, 0, :]
        kth = torch.topk(level, min(self.active_top, level.shape[-1]), dim=-1).values[:, -1:]
        active = (level > self.active_level) & (level >= kth)
        iso = (x[:, 4, :], x[:, 5, :]) if self.iso else None
        ladder = (self._dense if self.dense else self._sparse)(q, s0, active, iso)
        background = self.background(h) + self.prior_background * (1.0 - s0)
        logits = self.smooth(torch.cat([background, ladder], dim=1))
        rest = self.n_classes - 1 - self.z_ladder
        if rest > 0:
            low = torch.zeros_like(background).repeat(1, rest, 1) - 10.0
            logits = torch.cat([logits, low], dim=1)
        return {"charge_logits": logits, "apex_logit": self.apex_head(h)}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


#: LadderNet sizes: (encoder width, projection channels, hidden units, highest charge read)
LADDER_SIZES = {"small": (8, 4, 12, 48), "proof": (16, 4, 16, 64), "full": (24, 8, 32, 80)}


def build_model(size: str = "proof", z_max: int = 100, in_ch: int = 4,
                comb_prior: bool = False, arch: str = "unet", window: int = LADDER_WINDOW,
                active_top: int | None = None, iso: bool = False):
    """A charge network. For ``arch="ladder"``: ``window`` is the number of grid bins it reads
    at once (its ladder keeps every offset shorter than that), ``active_top`` the number of
    strongest bins whose ladders it reads (default 6 144 per 32 768 bins of window) and ``iso``
    adds the isotope-charge input channels (``in_ch`` 6). The U-Net reads any length and has
    neither."""
    if arch == "ladder":
        if size not in LADDER_SIZES:
            raise ValueError(f"Unknown model size {size!r}")
        width, proj, hidden, z_ladder = LADDER_SIZES[size]
        if int(window) <= 0 or int(window) % DOWNSAMPLE:
            # the predictor pads its input to a multiple of DOWNSAMPLE, and an exported LadderNet
            # reads exactly its window: any other window gives an ONNX model it cannot run
            raise ValueError(f"LadderNet window must be a positive multiple of {DOWNSAMPLE} bins, "
                             f"not {window}")
        top = ladder_active_top(window) if active_top is None else int(active_top)
        return LadderNet(in_ch, z_max, z_ladder=z_ladder, width=width, proj=proj, hidden=hidden,
                         window=int(window), active_top=top, iso=iso)
    if arch != "unet":
        raise ValueError(f"Unknown architecture {arch!r}")
    if iso:
        raise ValueError("isotope-charge channels are read by LadderNet only (arch='ladder')")
    if size == "proof":
        return ChargeUNet(in_ch, z_max, channels=(8, 16, 24, 32, 48, 64, 80), comb_prior=comb_prior)
    if size == "full":
        return ChargeUNet(in_ch, z_max, channels=(16, 32, 48, 64, 96, 128, 160), comb_prior=comb_prior)
    if size == "small":
        return ChargeUNet(in_ch, z_max, channels=(8, 16, 24, 32, 48, 64), comb_prior=comb_prior)
    raise ValueError(f"Unknown model size {size!r}")
