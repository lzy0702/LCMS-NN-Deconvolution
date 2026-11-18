"""PyTorch dataset that synthesizes training frames on the fly."""

from __future__ import annotations

import numpy as np

from .frames import generate_frame

try:
    import torch
    from torch.utils.data import Dataset

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False
    Dataset = object  # type: ignore


class FrameDataset(Dataset):
    """Deterministic per-index synthetic frames (seed = base_seed + index).

    ``config`` is one :class:`SynthConfig` or a list of ``(weight, SynthConfig)`` pairs; with a
    list each sample draws its configuration (polarity, chromatography mode, compound classes)
    from the mixture, deterministically from its index.
    """

    def __init__(self, config, length: int, base_seed: int = 0, k: int = 3, cache: bool = False):
        self.config = config
        self.length = length
        self.base_seed = base_seed
        self.k = k
        self._cache: dict | None = {} if cache else None
        if isinstance(config, (list, tuple)):
            w = np.array([float(wc[0]) for wc in config])
            self._weights = w / w.sum()
            self._configs = [wc[1] for wc in config]
        else:
            self._weights = None
            self._configs = [config]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self._cache is not None:
            key = (self.base_seed, idx)
            if key not in self._cache:
                self._cache[key] = self._make(idx)
            return self._cache[key]
        return self._make(idx)

    def frame(self, idx: int):
        """Item ``idx`` as its configuration and whole :class:`FrameSample`, ``meta`` included."""
        rng = np.random.default_rng(self.base_seed + idx)
        cfg = self._configs[0] if self._weights is None else \
            self._configs[int(rng.choice(len(self._configs), p=self._weights))]
        return cfg, generate_frame(cfg, rng)

    def _make(self, idx: int):
        _cfg, fs = self.frame(idx)
        x = torch.from_numpy(fs.features)
        topk_z = torch.from_numpy(fs.topk_z.astype(np.int64))
        topk_w = torch.from_numpy(fs.topk_w.astype(np.float32))
        heat = torch.from_numpy(fs.heat.astype(np.float32))
        return x, topk_z, topk_w, heat


class CombBarFrames(Dataset):
    """The frames of a :class:`FrameDataset` with the charges the pipeline's comb reads on each.

    Items are (feature channel 0 ``[1, L]``, true dominant class ``[L, 1]``, comb charges
    ``[L]``, kind of spectrum, how the comb was read): what ``train.comb_baseline`` scores. The
    frames are generated again (they are deterministic per index) so that their whole spectra,
    which the training items leave out, are at hand.
    """

    def __init__(self, frames: FrameDataset):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        from ..nn.baseline import comb_charges, spectrum_kind

        cfg, fs = self.frames.frame(idx)
        zc, how = comb_charges(cfg, fs)
        return (torch.from_numpy(np.ascontiguousarray(fs.features[:1])),
                torch.from_numpy(fs.topk_z[:, :1].astype(np.int64)),
                torch.from_numpy(zc), spectrum_kind(fs.meta), how)


class EpochSampler:
    """Indices ``epoch * length + start ... epoch * length + length - 1``, in order.

    FrameDataset seeds item ``i`` with ``base_seed + i``, so epoch ``e`` of a run that draws new
    spectra every epoch reads items ``e * length ...``. The epoch travels with the indices, which
    the main process hands to the workers; worker processes can therefore live for the whole
    run (``persistent_workers``) instead of being started again every epoch, which on Windows,
    where every worker is a fresh process importing PyTorch, costs many seconds each time.
    ``start`` skips the first items of an epoch (resuming in the middle of one).
    """

    def __init__(self, length: int):
        self.length = int(length)
        self.epoch = 0
        self.start = 0

    def __iter__(self):
        base = self.epoch * self.length
        return iter(range(base + self.start, base + self.length))

    def __len__(self) -> int:
        return max(0, self.length - self.start)


class CachedFrames(Dataset):
    """The items of a :class:`FrameDataset` generated once, by ``num_workers`` processes, and
    then served from memory (the validation set, read twice every epoch). ``frames`` is the
    underlying dataset, whose spectra can be generated again with everything they carry."""

    def __init__(self, frames: FrameDataset, num_workers: int = 0, batch_size: int = 8):
        from torch.utils.data import DataLoader

        self.frames = frames
        self.items: list = []
        loader = DataLoader(frames, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, worker_init_fn=worker_init if num_workers else None,
                            **mp_context(num_workers))
        for x, topk_z, topk_w, heat in loader:
            for i in range(x.shape[0]):
                self.items.append((x[i].clone(), topk_z[i].to(torch.int16), topk_w[i].clone(),
                                   heat[i].clone()))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, topk_z, topk_w, heat = self.items[idx]
        return x, topk_z.to(torch.int64), topk_w, heat


def mp_context(num_workers: int) -> dict:
    """DataLoader keywords for the worker start method: the platform's own unless
    ``LCMSDECONV_MP_CONTEXT`` names one (``spawn`` makes Linux start workers the way Windows
    does, which the tests use)."""
    import os

    ctx = os.environ.get("LCMSDECONV_MP_CONTEXT")
    return {"multiprocessing_context": ctx} if ctx and num_workers > 0 else {}


def worker_init(_worker_id: int) -> None:
    """Run a data worker at lowered priority, so that the computer stays responsive while the
    workers generate spectra; when nothing else needs the processor they run at full speed."""
    import os
    import sys

    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            # typed prototypes: untyped, the 64-bit pseudo-handle of GetCurrentProcess is cut
            # to 32 bits and the call silently fails
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.SetPriorityClass.restype = wintypes.BOOL
            below_normal = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(5)
    except Exception:  # pragma: no cover - a priority is a courtesy, never a requirement
        pass


def make_dataloaders(config, train_len: int, val_len: int, batch_size: int = 8,
                     num_workers: int = 1, base_seed: int = 0, cache: bool = False,
                     pin_memory: bool = False):
    """Train and validation loaders.

    The training loader reads :class:`FrameDataset` items through an :class:`EpochSampler`
    (set ``tl.sampler.epoch`` for new spectra every epoch, ``tl.sampler.start`` to resume inside
    one), with persistent, low-priority worker processes. The validation set is generated once
    and kept in memory (:class:`CachedFrames`). With ``cache`` training frames are kept once
    generated too, which only pays with ``num_workers = 0`` (worker processes keep their own
    copies)."""
    from torch.utils.data import DataLoader

    if cache:
        num_workers = 0
    train = FrameDataset(config, train_len, base_seed=base_seed, cache=cache)
    val = FrameDataset(config, val_len, base_seed=10_000_000 + base_seed, cache=cache)
    workers = dict(num_workers=num_workers, persistent_workers=True, worker_init_fn=worker_init,
                   prefetch_factor=2, **mp_context(num_workers)) if num_workers > 0 \
        else dict(num_workers=0)
    tl = DataLoader(train, batch_size=batch_size, sampler=EpochSampler(train_len), drop_last=True,
                    pin_memory=pin_memory, **workers)
    vl = DataLoader(CachedFrames(val, num_workers, batch_size), batch_size=batch_size,
                    shuffle=False, num_workers=0, pin_memory=pin_memory)
    return tl, vl
