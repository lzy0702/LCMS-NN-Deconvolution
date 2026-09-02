"""PyTorch dataset that synthesizes training frames on the fly."""

from __future__ import annotations

import numpy as np

from .config import SynthConfig
from .frames import generate_frame

try:
    import torch
    from torch.utils.data import Dataset

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False
    Dataset = object  # type: ignore


class FrameDataset(Dataset):
    """Deterministic per-index synthetic frames (seed = base_seed + index)."""

    def __init__(self, config: SynthConfig, length: int, base_seed: int = 0, k: int = 3):
        self.config = config
        self.length = length
        self.base_seed = base_seed
        self.k = k

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        fs = generate_frame(self.config, rng)
        x = torch.from_numpy(fs.features)
        topk_z = torch.from_numpy(fs.topk_z.astype(np.int64))
        topk_w = torch.from_numpy(fs.topk_w.astype(np.float32))
        heat = torch.from_numpy(fs.heat.astype(np.float32))
        return x, topk_z, topk_w, heat


def make_dataloaders(config: SynthConfig, train_len: int, val_len: int, batch_size: int = 8,
                     num_workers: int = 1, base_seed: int = 0):
    from torch.utils.data import DataLoader

    train = FrameDataset(config, train_len, base_seed=base_seed)
    val = FrameDataset(config, val_len, base_seed=10_000_000 + base_seed)
    tl = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    vl = DataLoader(val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return tl, vl
