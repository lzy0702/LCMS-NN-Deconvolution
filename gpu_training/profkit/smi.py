"""nvidia-smi: where it is, which of its query fields this driver knows, the GPU's description,
and a sampler that reads utilisation, clocks, power, temperature, memory, performance state
and clock-event reasons at a fixed interval while the GPU works.

The sampler asks nvidia-smi once per sample (``--query-gpu ... --format=csv``) from a thread,
instead of leaving nvidia-smi's own loop running: that loop's output can stay in its buffer
when the process is stopped, and every sample here is timed by this process."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import percentile, stats

#: the GPU's description in the environment record (each kept only where this driver knows it)
INFO_FIELDS = ("name", "driver_version", "vbios_version", "memory.total", "pcie.link.gen.max",
               "pcie.link.gen.current", "pcie.link.width.max", "pcie.link.width.current",
               "power.limit", "power.default_limit", "power.max_limit", "clocks.max.sm",
               "clocks.max.mem", "driver_model.current", "compute_mode", "display_active",
               "persistence_mode", "temperature.gpu", "pstate")
#: what the sampler reads; a tuple names alternatives, newest name first (driver 535 renamed
#: the throttle reasons to clock-event reasons)
SAMPLE_FIELDS = ("utilization.gpu", "utilization.memory", "clocks.sm", "clocks.mem",
                 "power.draw", "temperature.gpu", "memory.used", "pstate",
                 ("clocks_event_reasons.active", "clocks_throttle_reasons.active"))
#: the bits of the clock-event (throttle) reasons
REASON_BITS = ((0x1, "gpu_idle"), (0x2, "applications_clocks_setting"), (0x4, "sw_power_cap"),
               (0x8, "hw_slowdown"), (0x10, "sync_boost"), (0x20, "sw_thermal_slowdown"),
               (0x40, "hw_thermal_slowdown"), (0x80, "hw_power_brake_slowdown"),
               (0x100, "display_clock_setting"))


def find_smi() -> str | None:
    """nvidia-smi on the PATH, or in Windows' System32 where the driver puts it."""
    smi = shutil.which("nvidia-smi")
    if smi is None and sys.platform == "win32":
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(cand) if cand.exists() else None
    return smi


def run_smi(smi: str, args: list[str], timeout: float = 30.0) -> tuple[int, str]:
    """nvidia-smi's exit code and output (both streams), or (-1, the reason) if it did not run."""
    try:
        proc = subprocess.run([smi, *args], capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"
    text = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    return proc.returncode, text


def known_fields(smi: str, fields) -> dict:
    """``{asked: field this driver answers}`` for each of ``fields`` it knows (each asked alone:
    one unknown field fails the whole query); a tuple of alternatives keeps the first known."""
    out = {}
    for item in fields:
        names = item if isinstance(item, tuple) else (item,)
        for name in names:
            rc, text = run_smi(smi, [f"--query-gpu={name}", "--format=csv,noheader"], 15.0)
            if rc == 0 and text.strip() and "not a valid field" not in text.lower():
                out[names[0]] = name
                break
    return out


def parse_value(text: str):
    """A field's value: a number where it is one, None for N/A, the text otherwise."""
    t = text.strip()
    if not t or t.strip("[]").upper() in ("N/A", "NOT SUPPORTED", "UNKNOWN ERROR"):
        return None
    try:
        return int(t, 16) if t.lower().startswith("0x") else float(t)
    except ValueError:
        return t


def query(smi: str, fields: dict) -> list[dict]:
    """One reading of ``fields`` (``{name: field}`` from :func:`known_fields`) for every GPU."""
    if not fields:
        return []
    rc, text = run_smi(smi, [f"--query-gpu={','.join(fields.values())}",
                             "--format=csv,noheader,nounits"])
    if rc != 0:
        raise RuntimeError(f"nvidia-smi exit code {rc}: {text.strip()[:300]}")
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append({k: parse_value(v) for k, v in zip(fields, parts)})
    return rows


def describe(smi: str | None = None) -> dict:
    """nvidia-smi's description of the GPUs (INFO_FIELDS where known), or why there is none."""
    smi = smi or find_smi()
    if smi is None:
        return {"available": False, "why": "nvidia-smi was not found"}
    fields = known_fields(smi, INFO_FIELDS)
    try:
        gpus = query(smi, fields)
    except RuntimeError as exc:
        return {"available": False, "path_found": True, "why": str(exc)}
    return {"available": True, "gpus": gpus, "unknown_fields": [f for f in INFO_FIELDS
                                                                if f not in fields]}


def decode_reasons(value) -> list[str]:
    """The names of the clock-event (throttle) reason bits set in ``value`` (an int or hex)."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = int(value, 16)
        except ValueError:
            return [value]
    v = int(value)
    names = [name for bit, name in REASON_BITS if v & bit]
    rest = v & ~sum(bit for bit, _n in REASON_BITS)
    return names + ([f"other_{rest:#x}"] if rest else [])


class Sampler:
    """Reads SAMPLE_FIELDS (those this driver knows) every ``interval`` seconds from a thread
    until stopped; each row has ``t`` (seconds since the start) and the fields of GPU 0."""

    def __init__(self, smi: str | None = None, interval: float = 0.5):
        self.smi = smi or find_smi()
        self.interval = float(interval)
        self.fields = known_fields(self.smi, SAMPLE_FIELDS) if self.smi else {}
        self.rows: list[dict] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0

    def start(self) -> Sampler:
        if not self.smi or not self.fields:
            return self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - self._t0
            try:
                gpus = query(self.smi, self.fields)
                if gpus:
                    self.rows.append({"t": round(t, 3), **gpus[0]})
            except RuntimeError as exc:
                if len(self.errors) < 20:
                    self.errors.append(str(exc)[:200])
            nxt += self.interval
            self._stop.wait(max(0.0, nxt - time.perf_counter()))

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        return self.rows

    def write_csv(self, path: Path) -> None:
        cols = ["t", *self.fields]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(cols) + "\n")
            for r in self.rows:
                fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")


def summarise(rows: list[dict], fields=None) -> dict:
    """Means and percentiles of each numeric field over ``rows``, how often each performance
    state and each clock-event reason was seen, and the sampling itself."""
    out: dict = {"samples": len(rows)}
    if not rows:
        return out
    ts = [r["t"] for r in rows if r.get("t") is not None]
    if len(ts) > 1:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        out["seconds"] = round(ts[-1] - ts[0], 2)
        out["interval_median_s"] = round(percentile(gaps, 0.5), 3)
    names = list(fields or [k for k in rows[0] if k != "t"])
    for name in names:
        vals = [r.get(name) for r in rows]
        if name.startswith("clocks_event_reasons") or name.startswith("clocks_throttle"):
            counts: dict[str, int] = {}
            for v in vals:
                for reason in decode_reasons(v) or ["none"]:
                    counts[reason] = counts.get(reason, 0) + 1
            out["clock_event_reasons"] = {k: round(c / len(rows), 3) for k, c in counts.items()}
        elif name == "pstate":
            counts = {}
            for v in vals:
                counts[str(v)] = counts.get(str(v), 0) + 1
            out["pstate"] = {k: round(c / len(rows), 3) for k, c in counts.items()}
        else:
            nums = [float(v) for v in vals if isinstance(v, (int, float))]
            if nums:
                out[name] = stats(nums, 1)
    return out
