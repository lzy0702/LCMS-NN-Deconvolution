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

    def _make(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        cfg = self._configs[0] if self._weights is None else \
            self._configs[int(rng.choice(len(self._configs), p=self._weights))]
        fs = generate_frame(cfg, rng)
        x = torch.from_numpy(fs.features)
        topk_z = torch.from_numpy(fs.topk_z.astype(np.int64))
        topk_w = torch.from_numpy(fs.topk_w.astype(np.float32))
        heat = torch.from_numpy(fs.heat.astype(np.float32))
        return x, topk_z, topk_w, heat


def make_dataloaders(config, train_len: int, val_len: int, batch_size: int = 8,
                     num_workers: int = 1, base_seed: int = 0, cache: bool = False):
    """Train and validation loaders. With ``cache`` frames are kept once generated, which only
    pays with ``num_workers = 0`` (worker processes keep their own copies)."""
    from torch.utils.data import DataLoader

    if cache:
        num_workers = 0
    train = FrameDataset(config, train_len, base_seed=base_seed, cache=cache)
    val = FrameDataset(config, val_len, base_seed=10_000_000 + base_seed, cache=cache)
    tl = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    vl = DataLoader(val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return tl, vl
