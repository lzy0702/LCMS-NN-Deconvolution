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

    def item_spec(self) -> list | None:
        """[(shape, dtype)] of the items' tensors (x, topk_z, topk_w, heat), read off item 0,
        when every configuration of the mixture makes items of one shape (the same crop and
        channels); None otherwise. The ordered loader (lcmsdeconv.synth.pool) sizes the shared
        slots it passes items through by it."""
        keys = {(c.crop_size, c.evidence_channels, c.iso_channels, c.measured_channel)
                for c in self._configs}
        if len(keys) != 1 or self.length < 1:
            return None
        return [(tuple(t.shape), t.dtype) for t in self._make(0)]

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
    """The items of a :class:`FrameDataset` generated once, by ``num_workers`` processes placed
    as ``placement`` says (or by the training's own workers, ``pool``: an
    :class:`lcmsdeconv.synth.pool.WorkerPool`), and then served from memory (the validation set,
    read twice every epoch). ``frames`` is the underlying dataset, whose spectra can be
    generated again with everything they carry."""

    def __init__(self, frames: FrameDataset, num_workers: int = 0, batch_size: int = 8,
                 placement=None, pool=None, window: int | None = None):
        from torch.utils.data import DataLoader

        self.frames = frames
        self.items: list = []
        if pool is not None:  # the same items, in the same batches
            from .pool import ordered_batches

            loader = ordered_batches(pool, "val", frames, len(frames), batch_size, window)
        else:
            loader = DataLoader(frames, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False,
                                worker_init_fn=worker_init_for(placement, num_workers),
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


def worker_init_for(placement, num_workers: int = 1):
    """The ``worker_init_fn`` of a loader with ``num_workers`` workers: None without workers;
    :func:`worker_init` without a placement (below-normal priority only, as always); with one
    (:class:`lcmsdeconv.nn.cpus.Placement`) the worker also takes the placement's priority and
    power throttling and binds itself to its core(s), worker id by worker id. A partial of a
    module-level function and a frozen dataclass: it pickles, so spawned workers (Windows) get
    it."""
    if not num_workers:
        return None
    if placement is None:
        return worker_init
    import functools

    from ..nn.cpus import worker_init as placed_worker_init

    return functools.partial(placed_worker_init, placement=placement)


class CountedItems(Dataset):
    """``dataset``'s items, unchanged, each counted in ``stats`` (an
    :class:`lcmsdeconv.synth.pool.ItemStats` in shared memory) by the DataLoader worker that
    made it and the logical processor it ran on: what each class of core delivers, which the
    training's pace record reports."""

    def __init__(self, dataset, stats):
        self.dataset = dataset
        self.stats = stats

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        import time

        from torch.utils.data import get_worker_info

        from ..nn.cpus import current_cpu

        t0, c0, cpu0 = time.perf_counter(), time.process_time(), current_cpu()
        item = self.dataset[idx]
        cpu1 = current_cpu()
        info = get_worker_info()
        self.stats.add(info.id if info is not None else 0, cpu1, time.perf_counter() - t0,
                       time.process_time() - c0,
                       cpu0 is not None and cpu1 is not None and cpu0 != cpu1)
        return item


def make_dataloaders(config, train_len: int, val_len: int, batch_size: int = 8,
                     num_workers: int = 1, base_seed: int = 0, cache: bool = False,
                     pin_memory: bool = False, loader: str | None = None, placement=None,
                     window: int | None = None):
    """Train and validation loaders.

    The training loader reads :class:`FrameDataset` items through an :class:`EpochSampler`
    (set ``tl.sampler.epoch`` for new spectra every epoch, ``tl.sampler.start`` to resume inside
    one), with persistent, low-priority worker processes. The validation set is generated once
    and kept in memory (:class:`CachedFrames`). With ``cache`` training frames are kept once
    generated too, which only pays with ``num_workers = 0`` (worker processes keep their own
    copies).

    ``loader`` (default "dataloader"; train() reads LCMSDECONV_LOADER first,
    :func:`lcmsdeconv.synth.pool.choose_loader`): "dataloader" is torch's DataLoader, which
    hands whole batches to its workers in turn; "ordered" is
    :class:`lcmsdeconv.synth.pool.OrderedBatches`, which hands single items to whichever worker
    is free, at most ``window`` items ahead (default :func:`lcmsdeconv.synth.pool.
    default_window`), on one pool of workers that also makes the validation spectra (and the
    comb's: ``tl.pool``; ``tl.close()`` stops it). Both give the same batches.

    ``placement`` (:func:`lcmsdeconv.nn.cpus.plan`) sets where each worker runs, its priority
    and power throttling (:func:`worker_init_for`); it changes where spectra are made, never
    what they are. ``tl.item_stats`` (:class:`lcmsdeconv.synth.pool.ItemStats`) counts what the
    workers made, where."""
    from torch.utils.data import DataLoader

    from .pool import LOADERS, ItemStats

    if cache:
        num_workers = 0
    loader = loader or "dataloader"
    if loader not in LOADERS:
        raise ValueError(f"loader must be one of {LOADERS}, not {loader!r}")
    train = FrameDataset(config, train_len, base_seed=base_seed, cache=cache)
    val = FrameDataset(config, val_len, base_seed=10_000_000 + base_seed, cache=cache)
    if num_workers > 0 and loader == "ordered":
        return _ordered_loaders(train, val, train_len, batch_size, num_workers, pin_memory,
                                placement, window)
    stats = ItemStats(num_workers, shared=True) if num_workers > 0 else None
    workers = dict(num_workers=num_workers, persistent_workers=True,
                   worker_init_fn=worker_init_for(placement, num_workers),
                   prefetch_factor=2, **mp_context(num_workers)) if num_workers > 0 \
        else dict(num_workers=0)
    tl = DataLoader(CountedItems(train, stats) if stats is not None else train,
                    batch_size=batch_size, sampler=EpochSampler(train_len), drop_last=True,
                    pin_memory=pin_memory, **workers)
    tl.item_stats = stats  # what the workers made, where (None without workers)
    vl = DataLoader(CachedFrames(val, num_workers, batch_size, placement=placement),
                    batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    return tl, vl


def _ordered_loaders(train, val, train_len: int, batch_size: int, num_workers: int,
                     pin_memory: bool, placement, window):
    """make_dataloaders' loaders on one pool of persistent workers (lcmsdeconv.synth.pool): the
    validation spectra, then the training batches, item by item to whichever worker is free, in
    exactly the DataLoader's order. Training items pass through shared slots when they all have
    one shape (FrameDataset.item_spec), else through torch's shared memory."""
    from torch.utils.data import DataLoader

    from .pool import OrderedBatches, SlotRing, WorkerPool, default_window, ring_slots

    window = int(window or default_window(num_workers, batch_size))
    spec = train.item_spec()
    ring = SlotRing(ring_slots(window, batch_size, num_workers), spec) if spec else None
    labels = [placement.label(w) for w in range(num_workers)] if placement is not None else None
    pool = WorkerPool(num_workers, init=worker_init_for(placement, num_workers), ring=ring,
                      labels=labels)
    try:
        tl = OrderedBatches(train, batch_size, EpochSampler(train_len), pool, window,
                            pin_memory=pin_memory)
        vl = DataLoader(CachedFrames(val, batch_size=batch_size, pool=pool, window=window),
                        batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=pin_memory)
    except BaseException:
        pool.close()
        raise
    return tl, vl
