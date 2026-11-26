"""Helpers shared by the bench tools: thread caps, onnxruntime providers, atomic outputs, inputs.

A tool starts with::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
    import _paths  # noqa: F401  (<repo>/src and <repo>/scripts on sys.path)
    from tools import common
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")

#: the four Thermo runs of Tutis et al. 2025 by key
THERMO_RUNS = {"RNC": "IPRPLC_NC_mix.raw", "RGA": "IPRPLC_GalNAc_conjugated_mix.raw",
               "HNC": "IPHILIC_NC_mix.raw", "HGA": "IPHILIC_GalNAc_conjugated_mix.raw"}


def thread_env(n: int) -> dict:
    """Environment entries capping BLAS / OpenMP / numba threads at ``n``."""
    return {k: str(int(n)) for k in THREAD_VARS}


def preset_threads(argv: list[str] | None = None) -> None:
    """Set the environment of :func:`thread_env` from a ``--threads N`` (or ``--threads=N``) in
    ``argv`` (default: ``sys.argv``). A tool calls this before it imports numpy, whose BLAS,
    OpenMP and numba thread pools read the environment once; :func:`cap_threads` then caps
    onnxruntime. Without the option the environment is left as it is."""
    import sys

    args = sys.argv[1:] if argv is None else list(argv)
    n = None
    for i, a in enumerate(args):
        if a == "--threads" and i + 1 < len(args):
            n = args[i + 1]
        elif a.startswith("--threads="):
            n = a.partition("=")[2]
    if n is not None and n.isdigit() and int(n) > 0:
        os.environ.update(thread_env(int(n)))


def cap_threads(n: int) -> None:
    """Cap this process at ``n`` threads: the environment (effective only before numpy and numba
    are imported; the driver also sets it for its children) and every onnxruntime session made
    afterwards (``intra_op_num_threads = n``, ``inter_op_num_threads = 1``)."""
    os.environ.update(thread_env(n))
    import onnxruntime as ort

    base = ort.InferenceSession
    if getattr(base, "_bench_capped", False):
        return

    class Capped(base):
        _bench_capped = True

        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None,
                     **kwargs):
            so = sess_options if sess_options is not None else ort.SessionOptions()
            so.intra_op_num_threads = int(n)
            so.inter_op_num_threads = 1
            super().__init__(path_or_bytes, so, providers, provider_options, **kwargs)

    ort.InferenceSession = Capped


def providers(name: str | None) -> list[str]:
    """onnxruntime provider list for ``auto``, ``dml``, ``cuda`` or ``cpu`` (``auto``: DirectML,
    else CUDA, where onnxruntime lists it, else the processor)."""
    name = (name or "auto").lower()
    cpu = ["CPUExecutionProvider"]
    if name == "cpu":
        return cpu
    if name == "dml":
        return ["DmlExecutionProvider", *cpu]
    if name == "cuda":
        return ["CUDAExecutionProvider", *cpu]
    if name != "auto":
        raise ValueError(f"unknown providers {name!r}: auto, dml, cuda or cpu")
    import onnxruntime as ort

    avail = ort.get_available_providers()
    for p in ("DmlExecutionProvider", "CUDAExecutionProvider"):
        if p in avail:
            return [p, *cpu]
    return cpu


def parse_models(specs: list[str] | None) -> dict[str, str]:
    """``["GW5=path.onnx", ...]`` -> ``{"GW5": "path.onnx"}`` (order kept)."""
    out: dict[str, str] = {}
    for s in specs or []:
        label, sep, path = s.partition("=")
        if not sep or not label or not path:
            raise ValueError(f"--model wants LABEL=PATH, got {s!r}")
        out[label] = path
    return out


def replace_file(src: str | Path, dst: str | Path, tries: int = 6) -> None:
    """``os.replace(src, dst)``, tried again for about 3 s while another program holds one of
    the two files open (Windows refuses the rename then: an antivirus scan, a sync or indexing
    service, a viewer)."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.1 * 2 ** i)


def write_atomic(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` through ``path.tmp`` and a rename (no half files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replace_file(tmp, path)


def _plain(o):
    """numpy scalars and arrays as the Python numbers and lists json writes."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def write_json(path: str | Path, obj) -> None:
    write_atomic(path, json.dumps(obj, indent=1, default=_plain))


def demote_headings(text: str, levels: int = 1) -> str:
    """Markdown ``text`` with every heading ``levels`` deeper (``#`` -> ``##``, at most
    ``######``), for a report placed under another's heading; fenced code is left as it is."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and (m := re.match(r"(#{1,6}) ", line)):
            line = "#" * min(6, len(m.group(1)) + levels) + line[len(m.group(1)):]
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def find_thermo_raws(folder: str | Path) -> dict[str, Path]:
    """The four Thermo runs under ``folder`` (searched recursively, case-insensitively; a leading
    ``<hex>-`` upload prefix is ignored), by key; missing ones are left out."""
    want = {v.lower(): k for k, v in THERMO_RUNS.items()}
    found: dict[str, Path] = {}
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file() or p.suffix.lower() != ".raw":
            continue
        name = re.sub(r"^[0-9a-f]{6,}-", "", p.name.lower())
        key = want.get(name)
        if key and key not in found:
            found[key] = p
    return found
