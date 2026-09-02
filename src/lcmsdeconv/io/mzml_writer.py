"""mzML writer (psims) used by the synthetic generator and result export."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..core.model import Run


def write_mzml(run: Run, path: str | Path, include_chromatograms: bool = True) -> Path:
    from psims.mzml import MzMLWriter

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MzMLWriter(open(path, "wb"), close=True) as out:
            out.controlled_vocabularies()
            out.file_description(file_contents=["MS1 spectrum"])
            out.software_list([{"id": "lcmsdeconv", "params": ["lcmsdeconv"]}])
            with out.run(id=run.name or "run"):
                n = len(run.spectra)
                with out.spectrum_list(count=n):
                    for s in run.spectra:
                        pol = "positive scan" if s.polarity >= 0 else "negative scan"
                        centroided = s.is_profile is False
                        params = ["MS1 spectrum", {"ms level": s.ms_level}, pol]
                        params.append("centroid spectrum" if centroided else "profile spectrum")
                        scan_no = s.index + 1 if s.index >= 0 else 1
                        out.write_spectrum(
                            s.mz.astype(np.float64),
                            s.intensity.astype(np.float32),
                            id=f"scan={scan_no}",
                            polarity=pol,
                            centroided=centroided,
                            scan_start_time=s.rt,
                            params=params,
                        )
                chroms = list(run.chromatograms.values()) if include_chromatograms else []
                if chroms:
                    with out.chromatogram_list(count=len(chroms)):
                        for c in chroms:
                            ctype = (
                                "electromagnetic radiation chromatogram"
                                if c.kind == "uv"
                                else "total ion current chromatogram"
                            )
                            out.write_chromatogram(
                                c.time.astype(np.float64),
                                c.intensity.astype(np.float32),
                                id=c.name,
                                chromatogram_type=ctype,
                                time_unit="minute",
                            )
    return path
