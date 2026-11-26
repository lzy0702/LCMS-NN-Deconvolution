"""Streaming mzML and mzMLb reader built on pyteomics."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from ..core.model import Chromatogram, Run, Spectrum

# a Waters native ID: "function=N process=P scan=S"
_WATERS_FUNCTION = re.compile(r"(?:^|\s)function=(\d+)(?=\s|$)")

#: a Waters function not read is the lockspray when its scans span at most this many m/z (their
#: declared scan window, else their data): the RDa scans 10 m/z round the lock mass, an MS function
#: hundreds or thousands
LOCKSPRAY_MAX_SPAN = 100.0

# the chromatogram types of the PSI-MS CV (MS:1000626 and its children) and the kind each is read
# as: the LC's own sensors, which are not detector traces, then optical detectors (uv), then ion
# currents
_CHROMATOGRAM_TYPES = {
    "pressure chromatogram": "pressure",
    "flow rate chromatogram": "flow",
    "temperature chromatogram": "temperature",
    "electromagnetic radiation chromatogram": "uv",
    "absorption chromatogram": "uv",
    "emission chromatogram": "uv",
    "total ion current chromatogram": "tic",
    "basepeak chromatogram": "bpc",
    "selected ion current chromatogram": "eic",
    "selected ion monitoring chromatogram": "eic",
    "selected reaction monitoring chromatogram": "eic",
    "consecutive reaction monitoring chromatogram": "eic",
    "precursor ion current chromatogram": "eic",
    "ion current chromatogram": "tic",
}
_SENSOR_KINDS = ("pressure", "flow", "temperature")
_SENSOR_NAMES = (
    ("pressure", re.compile(r"\bpressure\b")),
    ("flow", re.compile(r"\bflow\b")),
    ("temperature", re.compile(r"\btemp(?:erature)?\b")),
)
# a time array's unit (its UO accession) per minute
_PER_MINUTE = {"UO:0000028": 60000.0, "UO:0000010": 60.0, "UO:0000031": 1.0, "UO:0000032": 1 / 60}

# what the XML of an mzMLb file is checked for before it is parsed
_DTD_MARKUP = re.compile(rb"<!(?:DOCTYPE|ENTITY)", re.IGNORECASE)
_XML_ENCODING = re.compile(rb"""\bencoding\s*=\s*["']([^"']*)["']""")
_ASCII_ENCODINGS = frozenset(
    {"utf-8", "utf8", "us-ascii", "ascii", "iso-8859-1", "iso_8859-1", "latin-1", "latin1"})


class UnsafeXMLError(ValueError):
    """An mzMLb file's XML declares a DOCTYPE or an ENTITY, or hides its markup from the check."""


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


def _is_mzmlb(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".mzmlb"


def _check_mzmlb_xml(path: Path, block: int = 1 << 22) -> None:
    """Refuse an mzMLb file whose mzML document declares a DOCTYPE or an ENTITY.

    mzML has neither. A document type is where XML declares entities, which make a parser expand
    text (nested, exponentially) or read other files, so the XML of an mzMLb file, which may come
    from anywhere, is searched for both before any parser reads it. The search reads bytes, so
    the document must also be in an encoding whose markup is ASCII (UTF-8 or Latin-1, as mzML
    writers make it): one in UTF-16 or UTF-7 could hide its declarations from it.
    """
    import h5py

    with h5py.File(path, "r") as h5:
        xml = h5.get("mzML")
        if not isinstance(xml, h5py.Dataset) or xml.ndim != 1 or xml.dtype.itemsize != 1:
            raise ValueError(f"{path}: no mzML document (mzMLb keeps it as the byte dataset 'mzML')")
        n = int(xml.shape[0])
        head = np.asarray(xml[: min(n, 4096)]).tobytes()
        head = head[3:] if head.startswith(b"\xef\xbb\xbf") else head
        encoding = "utf-8"  # what XML without a declared encoding is read as
        if head.startswith(b"<?xml"):
            end = head.find(b"?>")
            found = _XML_ENCODING.search(head[:end]) if end >= 0 else None
            if end < 0:
                encoding = ""  # a declaration too long to read here could name any encoding
            elif found:
                encoding = found.group(1).decode("latin-1").lower()
        # UTF-16 and UTF-32 start with a byte order mark or a NUL byte next to the '<'
        if (encoding not in _ASCII_ENCODINGS or b"\x00" in head[:4]
                or not head.lstrip(b" \t\r\n").startswith(b"<")):
            raise UnsafeXMLError(f"{path}: its mzML document is not UTF-8 or Latin-1 XML; not read")
        tail = b""
        for start in range(0, n, block):
            text = tail + np.asarray(xml[start:start + block]).tobytes()
            if _DTD_MARKUP.search(text):
                raise UnsafeXMLError(f"{path}: its mzML document declares a DOCTYPE or an ENTITY, "
                                     "which mzML never does; not read")
            tail = text[-8:]


def _open_reader(path: Path):
    """pyteomics' reader of an mzML file, or of an mzMLb file once its XML has passed the check.

    The mzMLb reader leaves its data arrays as records that read and decode them when asked
    (:func:`_decoded`), so only the arrays kept are read from the HDF5 file. Each reader is
    given the PSI-MS vocabulary saved with the package (:mod:`lcmsdeconv.io.vocabularies`),
    through which a term the saved copy lacks is read rather than failing the file: without
    one, pyteomics 5 has psims fetch it from purl.obolibrary.org.
    """
    from .vocabularies import reader_vocabulary

    if _is_mzmlb(path):
        _check_mzmlb_xml(path)
        from pyteomics.mzmlb import MzMLb

        # MzMLb passes cv on to the mzML parser it reads the document with
        return MzMLb(str(path), decode_binary=False, cv=reader_vocabulary())
    from pyteomics import mzml

    # what mzml.read(path) makes, with the vocabulary: read() takes a cv but does not pass it on
    return mzml.MzML(str(path), read_schema=False, iterative=True, use_index=False, dtype=None,
                     huge_tree=False, decode_binary=True, cv=reader_vocabulary())


def _decoded(array) -> np.ndarray:
    """A data array, decoded if pyteomics left it as a record."""
    if isinstance(array, np.ndarray):
        return array
    decode = getattr(array, "decode", None)
    return decode() if callable(decode) else np.asarray(array)


def _waters_function(native_id) -> int | None:
    m = _WATERS_FUNCTION.search(str(native_id or ""))
    return int(m.group(1)) if m else None


def _declared_span(sc: dict) -> float | None:
    """The m/z span of a scan's declared scan window, None when it declares none."""
    try:
        window = sc["scanList"]["scan"][0]["scanWindowList"]["scanWindow"][0]
        span = float(window["scan window upper limit"]) - float(window["scan window lower limit"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return span if span >= 0 else None


def _lockspray(found: list, chosen: int) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """The scans of the Waters functions other than ``chosen`` that span at most
    LOCKSPRAY_MAX_SPAN m/z, as (rt in minutes, m/z, intensity) in time order. A function's span
    is the median of its scans' declared windows, else the data of its first scan, so a wide
    function (MSE's high-energy scans) is never decoded here."""
    by_function: dict[int, list] = {}
    for fn, i, span, item in found:
        if fn is not None and fn != chosen:
            by_function.setdefault(fn, []).append((i, span, item))
    kept: list[Spectrum] = []
    for scans in by_function.values():
        spans = [s for _i, s, _item in scans if s is not None]
        if spans:
            span = float(np.median(spans))
        else:
            i, _s, item = scans[0]
            first = item if isinstance(item, Spectrum) else _spectrum(item, i)
            span = float(first.mz[-1] - first.mz[0]) if first.mz.size else 0.0
        if span > LOCKSPRAY_MAX_SPAN:
            continue
        kept.extend(item if isinstance(item, Spectrum) else _spectrum(item, i)
                    for i, _s, item in scans)
    kept.sort(key=lambda s: s.rt)
    return [(float(s.rt), s.mz, s.intensity) for s in kept]


def _spectrum(sc: dict, position: int) -> Spectrum:
    mz = np.asarray(_decoded(sc["m/z array"]), dtype=np.float64)
    it = np.asarray(_decoded(sc["intensity array"]), dtype=np.float64)
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    centroided = "centroid spectrum" in sc
    profile = "profile spectrum" in sc
    is_prof = True if profile else (False if centroided else None)
    return Spectrum(
        mz=mz,
        intensity=it,
        rt=_rt_minutes(sc),
        polarity=_polarity(sc),
        ms_level=sc.get("ms level", 1),
        scan_id=str(sc.get("id", position)),
        index=position,
        is_profile=is_prof,
    )


def _choose_function(counts: dict[int, int], function: int | None, name: str,
                     ms_level: int) -> int | None:
    """The Waters function whose scans are kept; None when no scan carries a function number."""
    if function is not None:
        function = int(function)
        if function not in counts:
            seen = (f"its MS{ms_level} scans by function: {counts}" if counts
                    else "its scan IDs carry no Waters function numbers")
            raise ValueError(f"{name}: no MS{ms_level} scans of Waters function {function} ({seen})")
        return function
    if not counts:
        return None
    # the most scans; on a tie the lowest number (in MSE data the low-energy function)
    return min(counts, key=lambda n: (-counts[n], n))


def read_mzml(path: str | Path, ms_level: int = 1, load_chromatograms: bool = True,
              function: int | None = None) -> Run:
    """Read an mzML or mzMLb file into a :class:`Run` (MS frames of one level + chromatograms).

    A ``.mzMLb`` file (any case of the suffix; mzML in HDF5, as ProteoWizard writes it) is read
    with pyteomics' MzMLb once its XML has passed :func:`_check_mzmlb_xml`, and the arrays of
    the scans kept are the only ones read.

    Waters native IDs read ``function=N process=P scan=S``. A Waters run holds its MS
    function(s) and a lockspray reference function (a few scans of a narrow window round the
    lock mass) whose scans are interleaved with theirs and must not enter the run. When scans
    carry function numbers, those of one function are kept: ``function`` if given, else the
    function with the most scans of ``ms_level`` (on a tie the lowest number, in MSE data the
    low-energy function), and ``run.meta`` records ``{"waters_functions": {N: scans of ms_level,
    ...}, "function": N}``. A ``function`` the run has no scans of raises ValueError. Scans
    without a function number are kept as before. ``Spectrum.index`` counts the scans kept from
    0; ``scan_id`` is the native ID. The scans of the functions not read whose scan window is
    narrow (at most LOCKSPRAY_MAX_SPAN m/z: the lockspray, a few scans) are kept apart in
    ``run.meta["lockspray"]`` as (rt in minutes, m/z array, intensity array) in time order, for
    the lock-mass correction (:mod:`lcmsdeconv.io.lockmass`); the key is absent when there are
    none.

    UV/PDA traces stored as chromatograms are picked up; UV stored as spectra with a
    non-MS spectrum type is ignored here (rare in practice).
    """
    path = Path(path)
    lazy = _is_mzmlb(path)
    found: list[tuple[int | None, int, float | None, Spectrum | dict]] = []
    with _open_reader(path) as reader:
        for i, sc in enumerate(reader):
            if "m/z array" not in sc or sc.get("ms level", 1) != ms_level:
                continue
            # an mzMLb scan is decoded once it is known to be kept
            found.append((_waters_function(sc.get("id")), i, _declared_span(sc),
                          sc if lazy else _spectrum(sc, i)))
        counts = dict(sorted(Counter(fn for fn, _i, _w, _s in found if fn is not None).items()))
        chosen = _choose_function(counts, function, path.name, ms_level)
        spectra: list[Spectrum] = []
        for fn, i, _w, item in found:
            if fn is not None and fn != chosen:
                continue
            s = item if isinstance(item, Spectrum) else _spectrum(item, i)
            s.index = len(spectra)
            spectra.append(s)
        lockspray = _lockspray(found, chosen) if counts else []
    meta = {"waters_functions": counts, "function": chosen} if counts else {}
    if lockspray:
        meta["lockspray"] = lockspray
    chroms: dict[str, Chromatogram] = {}
    if load_chromatograms:
        chroms = _read_chromatograms(path)
    return Run(spectra=spectra, chromatograms=chroms, name=path.stem, source=str(path), meta=meta)


def _array_unit(record: dict, array: str) -> str | None:
    """The unit accession of a data array (pyteomics keeps it on the array's key)."""
    key = next((k for k in record if k == array), None)
    return getattr(key, "unit_accession", None)


def _chromatogram_kind(ch: dict, name: str) -> str:
    """tic, bpc, eic or uv (any optical detector), or pressure, flow or temperature.

    The chromatogram's type term decides; without one, its name. The last three are the LC's own
    sensors (pump pressure, flow rate, column temperature), not detector traces, also when only
    the name says so.
    """
    typed = next((kind for term, kind in _CHROMATOGRAM_TYPES.items() if term in ch), None)
    if typed in _SENSOR_KINDS:
        return typed
    low = name.lower()
    for kind, pattern in _SENSOR_NAMES:
        if pattern.search(low):
            return kind
    if typed is not None:
        return typed
    kind = "tic"
    if "sic" in low or "selected ion" in low or "eic" in low:
        kind = "eic"
    if "uv" in low or "pda" in low or "dad" in low:
        kind = "uv"
    return kind


def _read_chromatograms(path: Path) -> dict[str, Chromatogram]:
    """The chromatograms of an mzML or mzMLb file, times in minutes.

    A chromatogram that cannot be read ends the reading with those read so far, and a warning
    says so; an mzMLb file whose XML is refused raises :class:`UnsafeXMLError`.
    """
    path = Path(path)
    out: dict[str, Chromatogram] = {}
    try:
        with _open_reader(path) as reader:
            for ch in reader.iterfind("chromatogram"):
                t = ch.get("time array")
                y = ch.get("intensity array")
                if t is None or y is None:
                    continue
                name = str(ch.get("id", "chromatogram"))
                per_minute = _PER_MINUTE.get(_array_unit(ch, "time array"), 1.0)
                out[name] = Chromatogram(
                    np.asarray(_decoded(t), dtype=float) / per_minute,
                    np.array(_decoded(y), dtype=float), name, _chromatogram_kind(ch, name),
                    "counts",
                )
    except UnsafeXMLError:
        raise
    except Exception as exc:
        warnings.warn(f"{path.name}: its chromatograms after the first {len(out)} were not read "
                      f"({type(exc).__name__}: {exc})", stacklevel=2)
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
