"""Streaming mzML reader built on pyteomics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.model import Chromatogram, Run, Spectrum


def _polarity(scan: dict) -> int:
    if "positive scan" in scan:
        return 1
    if "negative scan" in scan:
        return -1
    pol = scan.get("polarity")
    if isinstance(pol, str):
        return -1 if pol.lower().startswith("neg") else 1
    return 1


def _rt_minutes(scan: dict) -> float:
    sl = scan.get("scanList", {})
    scans = sl.get("scan", [{}])
    s0 = scans[0] if scans else {}
    t = s0.get("scan start time")
    if t is None:
        t = scan.get("scan start time", 0.0)
    unit = getattr(t, "unit_info", None)
    val = float(t) if t is not None else 0.0
    if unit and "second" in str(unit).lower():
        val /= 60.0
    return val


def read_mzml(path: str | Path, ms_level: int = 1, load_chromatograms: bool = True) -> Run:
    """Read an mzML file into a :class:`Run` (MS1 frames + chromatograms).

    UV/PDA traces stored as chromatograms are picked up; UV stored as spectra with a
    non-MS spectrum type is ignored here (rare in practice).
    """
    from pyteomics import mzml

    path = Path(path)
    spectra: list[Spectrum] = []
    with mzml.read(str(path)) as reader:
        for i, sc in enumerate(reader):
            if sc.get("ms level", 1) != ms_level and "m/z array" in sc:
                if sc.get("ms level", 1) != ms_level:
                    continue
            if "m/z array" not in sc:
                continue
            mz = np.asarray(sc["m/z array"], dtype=np.float64)
            it = np.asarray(sc["intensity array"], dtype=np.float64)
            order = np.argsort(mz)
            mz, it = mz[order], it[order]
            centroided = "centroid spectrum" in sc
            profile = "profile spectrum" in sc
            is_prof = True if profile else (False if centroided else None)
            spectra.append(
                Spectrum(
                    mz=mz,
                    intensity=it,
                    rt=_rt_minutes(sc),
                    polarity=_polarity(sc),
                    ms_level=sc.get("ms level", 1),
                    scan_id=str(sc.get("id", i)),
                    index=i,
                    is_profile=is_prof,
                )
            )
    chroms: dict[str, Chromatogram] = {}
    if load_chromatograms:
        chroms = _read_chromatograms(path)
    return Run(spectra=spectra, chromatograms=chroms, name=path.stem, source=str(path))


def _read_chromatograms(path: Path) -> dict[str, Chromatogram]:
    from pyteomics import mzml

    out: dict[str, Chromatogram] = {}
    try:
        with mzml.read(str(path)) as reader:
            get = getattr(reader, "iterfind", None)
            if get is None:
                return out
            for ch in reader.iterfind("chromatogram"):
                t = ch.get("time array")
                y = ch.get("intensity array")
                if t is None or y is None:
                    continue
                cid = str(ch.get("id", "chromatogram"))
                kind = "tic"
                low = cid.lower()
                if "sic" in low or "selected ion" in low or "eic" in low:
                    kind = "eic"
                if "absorption" in ch or "electromagnetic radiation" in ch or "uv" in low or "pda" in low or "dad" in low:
                    kind = "uv"
                if "total ion current chromatogram" in ch:
                    kind = "tic"
                unit = "counts"
                name = cid
                out[name] = Chromatogram(
                    np.asarray(t, dtype=float), np.asarray(y, dtype=float), name, kind, unit
                )
    except Exception:
        return out
    return out


def read_uv_csv(path: str | Path, name: str = "UV", time_col: int = 0, value_col: int = 1,
                time_unit: str = "min") -> Chromatogram:
    """Read a two-column CSV/TXT UV trace (time, absorbance)."""
    import pandas as pd

    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    try:
        df = pd.read_csv(path, sep=sep, engine="python", comment="#")
        t = df.iloc[:, time_col].to_numpy(dtype=float)
        y = df.iloc[:, value_col].to_numpy(dtype=float)
    except Exception:
        arr = np.loadtxt(path)
        t, y = arr[:, time_col], arr[:, value_col]
    if time_unit.startswith("s"):
        t = t / 60.0
    return Chromatogram(t, y, name, "uv", "AU")
