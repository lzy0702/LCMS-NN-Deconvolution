"""Download the public spectra used for validation (never for training).

Both sources are permissively licensed and reachable without authentication:
  * UniDec example data (modified BSD) - native mass spectra as two-column text.
  * ms_deisotope test data (Apache 2.0) - Orbitrap mzML with resolved isotopes.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FILES = {
    "unidec/BSA.txt":
        "https://raw.githubusercontent.com/michaelmarty/UniDec/master/unidec/bin/Example%20Data/BSA.txt",
    "unidec/ADH.txt":
        "https://raw.githubusercontent.com/michaelmarty/UniDec/master/unidec/bin/Example%20Data/ADH.txt",
    "unidec/ADHclean.txt":
        "https://raw.githubusercontent.com/michaelmarty/UniDec/master/unidec/bin/Example%20Data/ADHclean.txt",
    "unidec/GroEL.txt":
        "https://raw.githubusercontent.com/michaelmarty/UniDec/master/unidec/bin/Example%20Data/GroEL%20UniDec.txt",
    "ms_deisotope/three_test_scans.mzML":
        "https://raw.githubusercontent.com/mobiusklein/ms_deisotope/master/tests/test_data/three_test_scans.mzML",
    "ms_deisotope/small.mzML":
        "https://raw.githubusercontent.com/mobiusklein/ms_deisotope/master/tests/test_data/small.mzML",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/public", type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for rel, url in FILES.items():
        dest = args.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not args.force:
            print(f"have   {dest} ({dest.stat().st_size:,} bytes)")
            continue
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            print(f"got    {dest} ({dest.stat().st_size:,} bytes)")
        except Exception as exc:
            print(f"FAILED {dest}: {exc}")


if __name__ == "__main__":
    main()
