"""Persist processing results (JSON summary, HDF5 frame detail, CSV tables)."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_frames_h5(frame_results, path: str | Path) -> Path:
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        for i, fr in enumerate(frame_results):
            g = h.create_group(f"frame_{i:06d}")
            g.attrs["rt"] = fr.rt
            g.attrs["polarity"] = fr.polarity
            g.attrs["noise_sigma"] = fr.noise_sigma
            g.attrs["residual_fraction"] = fr.residual_fraction
            if fr.components:
                g.create_dataset("mass", data=np.array([c.mass for c in fr.components]))
                g.create_dataset("intensity", data=np.array([c.intensity for c in fr.components]))
    return path
