"""Refresh the controlled vocabularies saved with lcmsdeconv (src/lcmsdeconv/data/cv).

lcmsdeconv reads and writes mzML with the saved copies of PSI-MS and UO and never fetches them at
run time (lcmsdeconv/io/vocabularies.py). This script, which nothing imports, replaces them:

    python scripts/refresh_vocabularies.py              # download, check and write both
    python scripts/refresh_vocabularies.py --from DIR   # psi-ms.obo and uo.obo downloaded to DIR

Each vocabulary is fetched from where its purl.obolibrary.org address leads (the OBO Foundry's
PURL configuration: config/ms.yml and config/uo.yml of OBOFoundry/purl.obolibrary.org) into a
temporary file and parsed with psims to check that it is an OBO vocabulary holding the terms
lcmsdeconv writes mzML with (REQUIRED). Only when both pass are both written, gzipped with no
time stamp and no file name, so the same download always gives the same bytes. The data-version,
date, licence lines and sha256 printed for each go into data/cv/README.md, the top-level
README.md and tests/test_vocabularies.py.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import tempfile
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "lcmsdeconv" / "data" / "cv"
#: file -> (the purl that mzML files and psims name the vocabulary by, where that purl leads)
SOURCES = {
    "psi-ms.obo": ("http://purl.obolibrary.org/obo/ms/psi-ms.obo",
                   "https://github.com/HUPO-PSI/psi-ms-CV/releases/latest/download/psi-ms.obo"),
    "uo.obo": ("http://purl.obolibrary.org/obo/uo.obo",
               "https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/"
               "master/uo.obo"),
}
#: file -> the terms lcmsdeconv's mzML writer and psims use from it, by accession and name
REQUIRED = {
    "psi-ms.obo": {
        "MS:1000016": "scan start time", "MS:1000579": "MS1 spectrum", "MS:1000511": "ms level",
        "MS:1000128": "profile spectrum", "MS:1000127": "centroid spectrum",
        "MS:1000129": "negative scan", "MS:1000130": "positive scan",
        "MS:1000235": "total ion current chromatogram", "MS:1000628": "basepeak chromatogram",
        "MS:1000627": "selected ion current chromatogram",
        "MS:1000811": "electromagnetic radiation chromatogram",
        "MS:1003019": "pressure chromatogram", "MS:1003020": "flow rate chromatogram",
        "MS:1002715": "temperature chromatogram", "MS:1000514": "m/z array",
        "MS:1000515": "intensity array", "MS:1000595": "time array",
        "MS:1000521": "32-bit float", "MS:1000523": "64-bit float",
        "MS:1000574": "zlib compression", "MS:1000576": "no compression",
        "MS:1000774": "multiple peak list nativeID format", "UO:0000031": "minute",
    },
    "uo.obo": {"UO:0000010": "second", "UO:0000031": "minute"},
}


def fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def header(data: bytes) -> list[str]:
    """The header lines of an OBO file (those before its first stanza)."""
    lines = []
    for line in data.decode("utf-8").splitlines():
        if line.startswith("["):
            break
        lines.append(line)
    return lines


def check(data: bytes, name: str) -> dict:
    """The data-version, date, licence lines and term count of an OBO vocabulary; exits when
    ``data`` is not one."""
    from psims.controlled_vocabulary.controlled_vocabulary import ControlledVocabulary

    lines = header(data)
    tags = {}
    for line in lines:
        tag, sep, value = line.partition(":")
        if sep:
            tags.setdefault(tag.strip(), value.strip())
    if "format-version" not in tags or "data-version" not in tags:
        raise SystemExit(f"{name}: no OBO header (format-version and data-version); not written")
    try:
        cv = ControlledVocabulary.from_obo(io.BytesIO(data))
    except ValueError as exc:  # psims refuses a vocabulary without terms
        raise SystemExit(f"{name}: {exc}; not written") from exc
    for accession, term_name in REQUIRED[name].items():
        try:
            term = cv[accession]
        except KeyError:
            raise SystemExit(f"{name}: no term {accession} ({term_name}); not written") from None
        if term.name != term_name or term.get("is_obsolete", False):
            raise SystemExit(f"{name}: {accession} is now {term.name!r}"
                             f"{' (obsolete)' if term.get('is_obsolete', False) else ''}, not "
                             f"{term_name!r}; not written")
    return {"data-version": tags["data-version"], "date": tags.get("date", "-"), "terms": len(cv),
            "licence": [line for line in lines if "licen" in line.lower()]}


def write(data: bytes, dest: Path) -> None:
    """``data`` gzipped to ``dest`` with no time stamp and no file name in the gzip header."""
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(data)
    os.replace(tmp, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the vocabularies saved with lcmsdeconv.")
    ap.add_argument("--from", dest="source", type=Path, default=None,
                    help="a directory holding psi-ms.obo and uo.obo, downloaded already")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checked = {}
    with tempfile.TemporaryDirectory() as tmp:
        # both fetched and checked before either is written, so a failure leaves both as they were
        for name, (_purl, url) in SOURCES.items():
            if args.source is None:
                path = Path(tmp) / name
                fetch(url, path)
            else:
                path = args.source / name
            data = path.read_bytes()
            checked[name] = (data, check(data, name), url if args.source is None else path)
    for name, (data, info, source) in checked.items():
        write(data, OUT / f"{name}.gz")
        print(f"{name}.gz  ({SOURCES[name][0]}, from {source})")
        print(f"  data-version {info['data-version']}, date {info['date']}, "
              f"{info['terms']} terms, {len(data)} bytes")
        print(f"  sha256 {hashlib.sha256(data).hexdigest()}")
        for line in info["licence"]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
