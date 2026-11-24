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


class ChannelNorm(nn.Module):
    """Layer normalisation over the channels at each position (``[N, C, L]``), with a learned
    scale and shift per channel. Unlike a GroupNorm, whose statistics run over the whole length,
    its output at a bin does not depend on how long the input is."""

    def __init__(self, ch: int, eps: float = 1e-5):
        super().__init__()
        self.ch = int(ch)  # a constant, not the input's shape: the ONNX export needs it static
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x):
        y = nn.functional.layer_norm(x.transpose(1, 2), (self.ch,), self.weight, self.bias,
                                     self.eps)
        return y.transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 7, dilation: int = 1, norm: str = "group"):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.norm = ChannelNorm(cout) if norm == "channel" else _gn(cout)
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


def _shift_any(t: torch.Tensor, s: int) -> torch.Tensor:
    """:func:`_shift` for an input of any length: padding first and slicing relative to the
    ends, so that a graph traced at one length (ONNX, dynamic length axis) is right at every
    other, including lengths below ``|s|`` (all zeros)."""
    if s == 0:
        return t
    if s > 0:
        return nn.functional.pad(t, (0, s))[..., s:]
    return nn.functional.pad(t, (-s, 0))[..., :s]


#: views of a LadderNet: "window", a fixed window (the inference grid is cut into windows of the
#: training length), and "comb", any length (the whole grid in one pass; see LadderNet)
VIEWS = ("window", "comb")


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

    With ``iso`` the input has isotope-charge channels from channel 4 on
    (:func:`lcmsdeconv.nn.features.isotope_channels`; empty on time-of-flight spectra): the charge
    ``zc`` read from resolved isotope spacing at each bin (0 = none), the strength ``st`` of that
    reading and, in the current layout (``iso_count`` 3), the bin's isotope limit ``zlim``, the
    highest charge the reader resolves there. For each ladder charge z the scorer then also
    reads ``oh = relu(1 - |zc - z|)`` (1 exactly when the isotopes read z), ``oh * st``,
    ``(zc >= 0.5) * (1 - oh)`` (the isotopes read another charge) and ``clamp(zlim - z + 1, 0,
    1)`` (z is resolvable here), and the charge-z logit gets ``iso_scale * oh * clip((st - 0.2)
    * 10, 0, 1) * gate``, with ``iso_scale`` learnable and starting at 10, above the comb prior's
    ceiling of ``prior_scale`` (8), and ``gate = clamp(zlim - 2 zc + 1, 0, 1)``: 1 where twice
    the read charge is still resolvable, so that the reading cannot be the half of a charge the
    reader does not resolve (it answers such an envelope with z / 2 at full strength). The
    untrained network reads charges as the comb with isotope charges does where the gate is
    open, the ladder elsewhere. Networks of the first layout (``iso_count`` 2, in_ch 6) read
    neither the limit nor the gate; without ``iso`` the network is exactly the one trained
    before these channels existed.

    ``view = "comb"`` is the same ladder for an input of any length (multiple of 64 for the
    predictor), read whole in one pass: the encoder normalises each position over its channels
    (:class:`ChannelNorm`) instead of over the whole length, the last input channel says which
    bins are measured (1 on the grid, 0 on padding), every ladder offset is kept, and a tooth
    whose position lies outside the input or on an unmeasured bin reads a learned vector
    ``e_unmeasured`` (zero at the start) instead of the encoder's projection, while the comb
    prior reads 0 there. Only measured bins can be active; the cap stays ``active_top``.
    ``head_chunk`` (training only) scores that many charges at a time and recomputes them in the
    backward pass (``torch.utils.checkpoint``): the same outputs, a fraction of the memory.
    ``head_gather = "index_select"`` (training only, a plain attribute) reads the ladder rows
    with ``torch.index_select``: the same outputs, gradients equal up to the order of summation.
    """

    def __init__(self, in_ch: int = 4, z_max: int = 100, z_ladder: int = 64, step: float = 2e-5,
                 width: int = 16, proj: int = 4, hidden: int = 16, ks: tuple[int, ...] = (1, 2, 3),
                 window: int = LADDER_WINDOW, active_level: float = 0.5,
                 active_top: int = LADDER_ACTIVE_TOP, iso: bool = False,
                 iso_count: int | None = None, view: str = "window",
                 head_chunk: int | None = None):
        super().__init__()
        import math

        if view not in VIEWS:
            raise ValueError(f"LadderNet view must be one of {VIEWS}, not {view!r}")
        self.view = view
        comb_view = view == "comb"
        self.z_max = z_max
        self.n_classes = z_max + 1
        self.z_ladder = min(z_ladder, z_max)
        self.in_ch = in_ch
        self.window = window
        self.active_level = active_level
        self.active_top = active_top
        self.iso = bool(iso)
        if self.iso and iso_count is None:
            # recorded with every network of the current layout; a network of the first layout
            # (two channels) is the one with six inputs
            iso_count = in_ch - 5 if comb_view else min(in_ch - 4, 3)
        #: isotope channels read from channel 4 on: 2 (charge, strength) or 3 (and the limit)
        self.iso_count = int(iso_count) if self.iso else 0
        if self.iso and self.iso_count not in (2, 3):
            raise ValueError(f"a LadderNet reads 2 or 3 isotope channels, not {self.iso_count}")
        need = 4 + self.iso_count + (1 if comb_view else 0)
        if (in_ch != need) if comb_view else (in_ch < need):
            what = ("the four base features" + (f", {self.iso_count} isotope channels" if self.iso
                                                else "") + (" and the measured channel"
                                                            if comb_view else ""))
            raise ValueError(f"this LadderNet reads {need} input channels ({what}), not {in_ch}")
        #: charges scored at a time in training (None: all at once)
        self.head_chunk = head_chunk
        #: how the sparse path's head reads each active bin's ladder rows: "index" (advanced
        #: indexing) or "index_select" (``torch.index_select`` with int32 indices, whose backward
        #: pass on CUDA adds into the rows directly instead of sorting the indices first). The
        #: same values; a training setting (the GPU speedup of that name), saved with nothing
        self.head_gather = "index"
        self.dense = False

        def shift(z_from: float, z_to: float) -> int | None:
            if z_to < 1:
                return None
            sh = int(round(math.log(z_from / z_to) / step))
            return sh if comb_view or abs(sh) < window else None

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
        self.n_proj = proj
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

        norm = "channel" if comb_view else "group"
        self.encoder = nn.Sequential(ConvBlock(2, width, 9, norm=norm),
                                     ConvBlock(width, width, 9, 2, norm=norm),
                                     ConvBlock(width, width, 9, 4, norm=norm),
                                     ConvBlock(width, width, 9, 8, norm=norm))
        self.proj = nn.Conv1d(width, proj, 1)
        # with isotope channels three (first layout) or four more inputs per charge follow the
        # ladder's J x P features
        self.mlp1 = nn.Linear(self.n_pos * proj + (self.iso_count + 1 if self.iso else 0), hidden)
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
        if comb_view:
            # what a tooth reads outside the input or on an unmeasured bin
            self.e_unmeasured = nn.Parameter(torch.zeros(proj))
            self.register_buffer("present_t", torch.tensor(
                [[0.0 if v is None else 1.0 for v in o] for o in offsets]), persistent=False)

    # ------------------------------------------------------------------ ladder scores
    def _tooth_bias(self) -> torch.Tensor:
        """Comb view: what ``e_unmeasured`` read at every tooth that exists adds to the first
        hidden layer, per charge ``[z_ladder, hidden]``. The teeth themselves read the encoder's
        projection less ``e_unmeasured`` (zero where unmeasured or outside), so a tooth reads
        the projection where measured and ``e_unmeasured`` elsewhere."""
        jp = self.n_pos * self.n_proj
        w = self.mlp1.weight[:, :jp].reshape(-1, self.n_pos, self.n_proj)
        we = (w * self.e_unmeasured).sum(-1)  # [H, J]
        return self.present_t @ we.t()

    def _iso_terms(self, zc, st, zl, z_ids):
        """Per bin and charge: the scorer's isotope inputs ``[..., Z, 3 or 4]`` and ``oh``."""
        oh = torch.relu(1.0 - torch.abs(zc - z_ids))
        extras = [oh, oh * st, (zc >= 0.5).to(zc.dtype) * (1.0 - oh)]
        if zl is not None:
            extras.append(torch.clamp(zl - z_ids + 1.0, 0.0, 1.0))
        return torch.stack(extras, dim=-1), oh

    def _head_rows(self, z0: int, z1: int, qf, start, off, zc, st, zl):
        """The learned scores of charges ``z0 + 1 .. z1`` at the active bins ``[n, z1 - z0]``."""
        idx = start[:, None, None] + off[z0:z1][None]  # [n, Zc, J]: rows of qf
        if self.head_gather == "index_select":
            small = qf.shape[0] <= torch.iinfo(torch.int32).max
            flat = idx.reshape(-1).to(torch.int32) if small else idx.reshape(-1)
            g = torch.index_select(qf, 0, flat).reshape(*idx.shape, qf.shape[1])
        else:
            g = qf[idx]  # [n, Zc, J, P]
        feats = g.reshape(g.shape[0], z1 - z0, -1)
        bias = self.z_bias[z0:z1][None]
        if self.view == "comb":
            bias = bias + self._tooth_bias()[z0:z1][None]
        if zc is None:
            hid = self.mlp1(feats) + bias
        else:
            # mlp1 over [ladder features, isotope inputs], the latter applied apart so that the
            # ladder features are not copied to append them
            extra, _oh = self._iso_terms(zc, st, zl, self.z_ids[z0:z1][None])
            jp = feats.shape[-1]
            w1 = self.mlp1.weight
            hid = (nn.functional.linear(feats, w1[:, :jp], self.mlp1.bias)
                   + nn.functional.linear(extra, w1[:, jp:]) + bias)
        del g, feats
        return self.mlp2(nn.functional.silu(hid)).squeeze(-1)

    def _sparse(self, q: torch.Tensor, s0: torch.Tensor, active: torch.Tensor,
                iso: tuple[torch.Tensor, ...] | None = None,
                meas: torch.Tensor | None = None) -> torch.Tensor:
        """Ladder logits ``[N, z_ladder, L]``: scored at active bins, constant elsewhere.
        ``iso``: the isotope channels ``[N, L]`` each (networks built with ``iso``); ``meas``:
        the measured channel ``[N, 1, L]`` (comb view)."""
        n_b, n_p, length = q.shape
        out = self.inactive.expand(n_b, self.z_ladder, length).clone()
        b_idx, i_idx = active.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            return out
        if meas is not None:
            # comb view: the teeth read q - e (zero where unmeasured, and outside), and e comes
            # back through the head's bias (_tooth_bias) at every tooth that exists
            q = (q - self.e_unmeasured[None, :, None]) * meas
            s0 = s0 * meas
        # every position outside the window reads zero: the window is padded with zeros by the
        # longest offset on the left and by one window more on the right, where the absent
        # positions (sentinel offsets) are sent, so no index needs clamping or masking
        m = self.reach
        lp = 2 * length + 2 * m
        start = b_idx * lp + i_idx + m  # [n]: each active bin in the padded rows
        qf = nn.functional.pad(q, (m, length + m)).permute(0, 2, 1).reshape(n_b * lp, n_p)
        off = torch.where(self.off_t == self._absent, length + m, self.off_t)
        zc = st = zl = None
        if iso is not None:
            zc = iso[0][b_idx, i_idx][:, None]  # [n, 1]
            st = iso[1][b_idx, i_idx][:, None]
            zl = iso[2][b_idx, i_idx][:, None] if len(iso) > 2 else None
        chunk = int(self.head_chunk or self.z_ladder)
        if chunk >= self.z_ladder:
            learned = self._head_rows(0, self.z_ladder, qf, start, off, zc, st, zl)  # [n, Z]
        else:
            from torch.utils.checkpoint import checkpoint

            rows = []
            for z0 in range(0, self.z_ladder, chunk):
                args = (z0, min(self.z_ladder, z0 + chunk), qf, start, off, zc, st, zl)
                rows.append(checkpoint(self._head_rows, *args, use_reentrant=False)
                            if torch.is_grad_enabled() else self._head_rows(*args))
            learned = torch.cat(rows, dim=1)
        sf = nn.functional.pad(s0, (m, length + m)).reshape(n_b * lp)
        off_c = torch.where(self.comb_t == self._absent, length + m, self.comb_t)
        nb = sf[start[:, None, None] + off_c[None]]  # [n, Z, C]
        centre = sf[start]
        comb = centre[:, None] * nb.sum(-1) / self.comb_den[None]
        val = learned + self.prior_scale * comb
        if iso is not None:
            oh = torch.relu(1.0 - torch.abs(zc - self.z_ids[None]))  # [n, Z]
            prior = self.iso_scale * oh * torch.clamp((st - 0.2) * 10.0, 0.0, 1.0)
            if zl is not None:
                prior = prior * torch.clamp(zl - 2.0 * zc + 1.0, 0.0, 1.0)
            val = val + prior
        out[b_idx, :, i_idx] = val
        return out

    def _dense(self, q: torch.Tensor, s0: torch.Tensor, active: torch.Tensor,
               iso: tuple[torch.Tensor, ...] | None = None,
               meas: torch.Tensor | None = None) -> torch.Tensor:
        """The same function with static shifts over every bin (ONNX export). In the comb view
        the shifts pad and slice relative to the ends, so the graph holds at any length."""
        w1 = self.mlp1.weight  # [H, J*P (+3 or 4)]
        bias = self.z_bias
        sh = _shift
        if meas is not None:
            q = (q - self.e_unmeasured[None, :, None]) * meas
            s0 = s0 * meas
            bias = bias + self._tooth_bias()
            sh = _shift_any
        zero_q = torch.zeros_like(q)
        if iso is not None:
            zc, st = iso[0][:, None, :], iso[1][:, None, :]  # [N, 1, L]
            zl = iso[2][:, None, :] if len(iso) > 2 else None
            has = (zc >= 0.5).to(q.dtype)
            gate = torch.clamp((st - 0.2) * 10.0, 0.0, 1.0)
            if zl is not None:
                limit = torch.clamp(zl - 2.0 * zc + 1.0, 0.0, 1.0)
        rows = []
        for zi in range(self.z_ladder):
            parts = [zero_q if s is None else sh(q, s) for s in self.offsets[zi]]
            if iso is not None:
                oh = torch.relu(1.0 - torch.abs(zc - float(zi + 1)))
                parts += [oh, oh * st, has * (1.0 - oh)]
                if zl is not None:
                    parts.append(torch.clamp(zl - float(zi + 1) + 1.0, 0.0, 1.0))
            hid = nn.functional.conv1d(torch.cat(parts, dim=1), w1[:, :, None], self.mlp1.bias)
            hid = hid + bias[zi][None, :, None]
            learned = nn.functional.conv1d(nn.functional.silu(hid), self.mlp2.weight[:, :, None],
                                           self.mlp2.bias)
            nb = self.comb_offsets[zi]
            if nb:
                acc = sh(s0, nb[0])
                for s in nb[1:]:
                    acc = acc + sh(s0, s)
                comb = s0 * acc / float(self.comb_count[zi])
            else:
                comb = torch.zeros_like(s0)
            row = learned + self.prior_scale * comb
            if iso is not None:
                prior = self.iso_scale * oh * gate
                if zl is not None:
                    prior = prior * limit
                row = row + prior
            rows.append(row)
        dense = torch.cat(rows, dim=1)
        return torch.where(active[:, None, :], dense, self.inactive.expand_as(dense))

    def forward(self, x):
        s0 = x[:, 0:1, :] / 5.0
        h = self.encoder(torch.cat([s0, x[:, 1:2, :]], dim=1))
        q = self.proj(h)
        level = x[:, 0, :]
        meas = None
        if self.view == "comb":
            meas = x[:, self.in_ch - 1:self.in_ch, :]
            level = level * (meas[:, 0, :] > 0.5).to(level.dtype)  # unmeasured: never active
            # padded with zeros, below every level, the length always holds active_top: the
            # same k-th level at any length, and a static k for the exported graph
            kth = torch.topk(nn.functional.pad(level, (0, self.active_top)), self.active_top,
                             dim=-1).values[:, -1:]
        else:
            kth = torch.topk(level, min(self.active_top, level.shape[-1]), dim=-1).values[:, -1:]
        active = (level > self.active_level) & (level >= kth)
        iso = tuple(x[:, 4 + k, :] for k in range(self.iso_count)) if self.iso else None
        ladder = (self._dense if self.dense else self._sparse)(q, s0, active, iso, meas)
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
                active_top: int | None = None, iso: bool = False, iso_count: int | None = None,
                view: str = "window", head_chunk: int | None = None):
    """A charge network. For ``arch="ladder"``: ``window`` is the number of grid bins it reads
    at once (its ladder keeps every offset shorter than that; in the comb view, which reads any
    length, the training crop), ``active_top`` the number of strongest bins whose ladders it
    reads (default 6 144 per 32 768 bins of window), ``iso`` adds the isotope-charge input
    channels (``iso_count`` of them, 3 by default; ``in_ch`` 7, or 6 for the first layout's 2),
    ``view`` is "window" or "comb" (one more input channel, the measured bins; see LadderNet)
    and ``head_chunk`` the charges scored at a time in training. The U-Net reads any length and
    has none of these."""
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
                         window=int(window), active_top=top, iso=iso, iso_count=iso_count,
                         view=view, head_chunk=head_chunk)
    if arch != "unet":
        raise ValueError(f"Unknown architecture {arch!r}")
    if iso:
        raise ValueError("isotope-charge channels are read by LadderNet only (arch='ladder')")
    if view != "window":
        raise ValueError("the comb view is a LadderNet's (arch='ladder')")
    if size == "proof":
        return ChargeUNet(in_ch, z_max, channels=(8, 16, 24, 32, 48, 64, 80), comb_prior=comb_prior)
    if size == "full":
        return ChargeUNet(in_ch, z_max, channels=(16, 32, 48, 64, 96, 128, 160), comb_prior=comb_prior)
    if size == "small":
        return ChargeUNet(in_ch, z_max, channels=(8, 16, 24, 32, 48, 64), comb_prior=comb_prior)
    raise ValueError(f"Unknown model size {size!r}")
