"""Read Thermo Fisher ``.raw`` files directly, through the open-source ``opentfraw`` parser.

``opentfraw`` (Apache-2.0, pure Rust with Python bindings) reads raw-file format versions 8 to
66 without Thermo's .NET libraries, so this works on Linux and macOS as well as Windows. It is an
optional dependency: ``pip install lcmsdeconv[thermo]``. Profile scans are read as profile data;
centroid-only scans as centroids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.model import Run, Spectrum


def _polarity(meta: dict) -> int:
    pol = meta.get("polarity")
    if isinstance(pol, str):
        return -1 if pol.lower().startswith(("neg", "-")) else 1
    if isinstance(pol, (int, float)):
        return -1 if pol < 0 else 1
    filt = str(meta.get("filter_string", ""))
    return -1 if " - " in filt.split("ESI")[0] or filt.startswith("FTMS -") else 1


def read_thermo_raw(path: str | Path, ms_level: int = 1, prefer_profile: bool = True) -> Run:
    """Read the MS1 scans of a Thermo ``.raw`` file into a :class:`Run`."""
    try:
        import opentfraw
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("Reading Thermo .raw files needs the 'opentfraw' package: "
                          "pip install lcmsdeconv[thermo] (or convert to mzML with msconvert)") from exc

    path = Path(path)
    raw = opentfraw.RawFile(str(path))
    spectra: list[Spectrum] = []
    for scan in range(raw.first_scan, raw.last_scan + 1):
        meta = raw.scan(scan)
        if int(meta.get("ms_level", 1) or 1) != ms_level:
            continue
        profile = prefer_profile and str(meta.get("scan_mode", "")).lower() == "profile"
        mz, inten = raw.profile(scan) if profile else raw.peaks(scan)
        mz = np.asarray(mz, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64)
        order = np.argsort(mz)
        spectra.append(Spectrum(
            mz=mz[order], intensity=inten[order].astype(np.float32),
            rt=float(meta.get("retention_time", 0.0)), polarity=_polarity(meta),
            ms_level=ms_level, scan_id=f"scan={scan}", index=len(spectra), is_profile=profile,
            meta={"filter": meta.get("filter_string", ""),
                  "low_mz": meta.get("low_mz"), "high_mz": meta.get("high_mz")},
        ))
    info = {"instrument": getattr(raw, "instrument_model", ""), "format": "thermo-raw"}
    return Run(spectra, {}, name=path.stem, source=str(path), meta=info)
