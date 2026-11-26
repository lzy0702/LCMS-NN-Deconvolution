"""Target species for vrun.py --targets: mass, apex and charge-state shares of named truth species
of scripts/validate_oligo_runs.py, as a run record found them (a vrun.py record, or any record
whose "species" carry mass, rt or rt_apex, and charges).

    python bench/tools/targets.py RECORD.json --names "3206.4|1883.3" [--key IPHILIC_NC]
        --out targets.json

--key is validate_oligo_runs' run key (default: the record's "key").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)

from tools import common  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record", type=Path)
    ap.add_argument("--names", required=True, help="truth species names, separated by |")
    ap.add_argument("--key", default=None, help="run key (default: the record's)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import validate_oligo_runs as v

    d = json.loads(args.record.read_text())
    key = args.key or d.get("key")
    if key not in v.TRUTH:
        ap.error(f"unknown run key {key!r}: one of {', '.join(v.TRUTH)}")
    names = args.names.split("|")
    sp = [SimpleNamespace(mass=s["mass"], rt_apex=s.get("rt_apex", s.get("rt")),
                          charges=s["charges"])
          for s in d["species"] if 1000.0 <= s["mass"] <= 10000.0]
    sc = v.score(key, sp, lambda m, t: False)
    res = []
    for name, mass, rt, _kind, s in sc["rows"]:
        if name not in names:
            continue
        if s is None:
            print("not found in record:", name)
            continue
        ch = {int(z): float(w) for z, w in s.charges.items()}
        tot = sum(ch.values())
        shares = {z: w / tot for z, w in sorted(ch.items())}
        res.append({"name": name, "truth_mass": mass, "truth_rt": rt, "mass": s.mass,
                    "rt": s.rt_apex, "shares": shares})
        print(f"{name:14s} truth {mass:8.2f}@{rt:5.2f}  rec {s.mass:8.2f}@{s.rt_apex:5.2f}  "
              + " ".join(f"z{z}:{w:.3f}" for z, w in shares.items() if w >= 0.005))
    common.write_json(args.out, res)


if __name__ == "__main__":
    main()
