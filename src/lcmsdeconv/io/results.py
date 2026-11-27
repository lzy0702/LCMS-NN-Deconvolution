"""Persist processing results (JSON summary, HDF5 frame detail, CSV tables).

:func:`write_outputs` writes what ``lcmsdeconv process`` writes for a processed run.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
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


def results_sha256(result) -> str:
    """The SHA-256 of the ``results.json`` :func:`write_outputs` writes for ``result``: of the
    bytes :func:`save_json` writes for its summary (its text with the platform's line ends, in
    UTF-8)."""
    text = json.dumps(result.summary(), indent=2, default=_json_default)
    return hashlib.sha256(text.replace("\n", os.linesep).encode("utf-8")).hexdigest()


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


# ------------------------------------------------------------------ what `process` writes
def write_outputs(result, outdir: str | Path, report: bool = True,
                  plots: bool | None = None, provenance: bool | None = None,
                  aliases=None) -> list[Path]:
    """Write a processed run's outputs into ``outdir`` as ``lcmsdeconv process`` does:
    ``results.json`` (:meth:`~lcmsdeconv.process.ProcessResult.summary`), ``species.csv`` (where
    there are impurity rows), a ``peaks_<name>.csv`` per peak table (:func:`csv_names`) and, with
    ``report``, ``report.html`` (with plots unless ``plots`` is False; default: the method's
    ``report.plots``); then, with ``provenance``, ``provenance.json`` (which names the
    results.json written here by its SHA-256), ``provenance.html`` and the evidence figures under
    ``evidence/`` (:func:`lcmsdeconv.provenance.write_provenance`; default: where the result
    holds provenance, ``process_run(..., provenance=True)``). ``aliases``
    (:class:`lcmsdeconv.aliases.Aliases`) are applied to the provenance's texts that do not come
    with the run and the method (the network's file name), as for a run and a method aliased
    already; nothing else is changed by them. Returns the files written, in that order."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = [save_json(result.summary(), outdir / "results.json")]
    written += write_csvs(result, outdir)
    if report:
        from ..report.html import build_report

        if plots is None:
            plots = result.method.report.plots
        written.append(build_report(result, outdir / "report.html", plots=plots))
    if provenance is None:
        provenance = getattr(result, "provenance", None) is not None
    if provenance:
        from ..provenance import write_provenance

        sha = hashlib.sha256(written[0].read_bytes()).hexdigest()
        written += write_provenance(result, outdir, aliases=aliases, results_sha256=sha)
    return written


def write_csvs(result, outdir: str | Path) -> list[Path]:
    """species.csv (where there are impurity rows) and a peaks_<name>.csv per peak table;
    returns the files written."""
    import pandas as pd

    outdir = Path(outdir)
    written = []
    if result.impurities:
        rows = []
        for r in result.impurities:
            rows.append({k: v for k, v in r.items() if k not in ("adducts", "flags")}
                        | {"adducts": "; ".join(f"{k} {v*100:.2f}%" for k, v in r["adducts"].items()),
                           "flags": "; ".join(r["flags"])})
        pd.DataFrame(rows).to_csv(outdir / "species.csv", index=False)
        written.append(outdir / "species.csv")
    for name, safe in csv_names(list(result.peak_tables)).items():
        path = outdir / f"peaks_{safe}.csv"
        result.peak_tables[name].to_dataframe().to_csv(path, index=False)
        written.append(path)
    return written


#: the longest name part of a peak table's CSV, in UTF-8 bytes (a file name may hold 255)
CSV_PART_BYTES = 100


def csv_names(names: list[str]) -> dict[str, str]:
    """The file name part of each peak table's CSV (``peaks_<part>.csv``): the table's name in
    Unicode's composed form (NFC) with any character but letters, digits, '-', '_' and '.' made
    '_', a name longer than CSV_PART_BYTES cut short and ended with 8 hex digits of its hash.
    Where two names come out the same, also apart from case (one file on Windows and macOS), the
    first keeps it and the others get '_2', '_3', ... added, so no table overwrites another."""
    def key(part: str) -> str:
        return unicodedata.normalize("NFC", part.casefold())

    safe = {n: _csv_part(n) for n in names}
    taken = {key(s) for s in safe.values()}
    out: dict[str, str] = {}
    used: set[str] = set()
    for n in names:
        part = safe[n]
        if key(part) in used:
            k = 2
            while key(f"{part}_{k}") in taken:
                k += 1
            part = f"{part}_{k}"
            taken.add(key(part))
        used.add(key(part))
        out[n] = part
    return out


def _csv_part(name: str) -> str:
    part = "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in unicodedata.normalize("NFC", name))
    data = part.encode("utf-8")
    if len(data) <= CSV_PART_BYTES:
        return part
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return data[:CSV_PART_BYTES - 9].decode("utf-8", "ignore") + "_" + digest
