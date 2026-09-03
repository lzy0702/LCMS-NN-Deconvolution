"""Integration-event vocabulary and time-resolved parameter schedule (Agilent CDS style)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Events that change a numeric integration parameter from their time onwards.
VALUE_EVENTS = {
    "slope_sensitivity": "slope_sensitivity",
    "peak_width": "peak_width",
    "area_reject": "area_reject",
    "height_reject": "height_reject",
    "area_pct_reject": "area_pct_reject",
    "max_area": "max_area",
    "max_height": "max_height",
    "tail_skim_height_ratio": "tail_skim_height_ratio",
    "front_skim_height_ratio": "front_skim_height_ratio",
    "skim_valley_ratio": "skim_valley_ratio",
}

#: Events that switch a mode on or off (value 1/0 or "on"/"off").
SWITCH_EVENTS = {
    "integration": "integration_on",
    "integration_off": "integration_on",
    "integration_on": "integration_on",
    "baseline_all_valleys": "baseline_all_valleys",
    "baseline_at_valleys": "baseline_all_valleys",
    "tangent_skim": "tangent_skim",
    "front_tangent_skim": "front_tangent_skim",
    "rear_tangent_skim": "tangent_skim",
    "negative_peak": "detect_negative_peaks",
    "negative_peaks": "detect_negative_peaks",
    "fixed_peak_width": "fixed_peak_width",
    "shoulders": "shoulders_mode",
}

#: Events that act at a single point in time.
POINT_EVENTS = {
    "baseline_now",
    "baseline_hold",
    "baseline_hold_off",
    "baseline_next_valley",
    "baseline_back",
    "split_peak",
    "solvent_peak",
    "peak_sum_slice",
    "peak_sum_slice_off",
    "set_baseline_start",
    "set_baseline_end",
}

ALL_EVENTS = sorted(set(VALUE_EVENTS) | set(SWITCH_EVENTS) | POINT_EVENTS)


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("on", "true", "yes", "1")
    return bool(value)


@dataclass
class EventSchedule:
    """Time-resolved integration parameters plus point events."""

    times: np.ndarray
    params: dict[str, np.ndarray]  # parameter name -> value per time point
    integration_on: np.ndarray
    baseline_now: list[float] = field(default_factory=list)
    baseline_hold: list[tuple[float, float]] = field(default_factory=list)
    baseline_next_valley: list[float] = field(default_factory=list)
    baseline_back: list[float] = field(default_factory=list)
    split_peak: list[float] = field(default_factory=list)
    solvent_peak: list[float] = field(default_factory=list)
    peak_sum_slice: list[tuple[float, float]] = field(default_factory=list)
    forced_baseline: list[tuple[float, str]] = field(default_factory=list)

    def value(self, name: str, t: float) -> float:
        arr = self.params.get(name)
        if arr is None:
            return 0.0
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return float(arr[max(0, min(i, arr.size - 1))])

    def is_on(self, t: float) -> bool:
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return bool(self.integration_on[max(0, min(i, self.integration_on.size - 1))])


def build_schedule(times: np.ndarray, settings) -> EventSchedule:
    """Expand initial settings plus timed events into per-time-point parameter arrays."""
    n = times.size
    names = ["slope_sensitivity", "peak_width", "area_reject", "height_reject",
             "area_pct_reject", "max_area", "max_height", "tail_skim_height_ratio",
             "front_skim_height_ratio", "skim_valley_ratio"]
    params = {k: np.full(n, float(getattr(settings, k, 0.0))) for k in names}
    switches = {
        "baseline_all_valleys": np.full(n, bool(settings.baseline_all_valleys)),
        "tangent_skim": np.zeros(n, dtype=bool),
        "front_tangent_skim": np.zeros(n, dtype=bool),
        "detect_negative_peaks": np.full(n, bool(settings.detect_negative_peaks)),
        "fixed_peak_width": np.full(n, bool(settings.fixed_peak_width)),
    }
    integration_on = np.ones(n, dtype=bool)
    sched = EventSchedule(times=times, params=params, integration_on=integration_on)

    hold_start: float | None = None
    slice_start: float | None = None
    for ev in sorted(settings.timed_events, key=lambda e: e.time):
        name = ev.event.strip().lower().replace(" ", "_")
        idx = int(np.searchsorted(times, ev.time))
        idx = max(0, min(idx, n))
        if name in VALUE_EVENTS:
            key = VALUE_EVENTS[name]
            try:
                params[key][idx:] = float(ev.value)
            except (TypeError, ValueError):
                pass
            continue
        if name in ("integration_off",) or (name == "integration" and not _as_bool(ev.value)):
            integration_on[idx:] = False
            continue
        if name in ("integration_on",) or (name == "integration" and _as_bool(ev.value)):
            integration_on[idx:] = True
            continue
        if name in SWITCH_EVENTS:
            key = SWITCH_EVENTS[name]
            if key in switches:
                switches[key][idx:] = _as_bool(ev.value)
            continue
        if name == "baseline_now":
            sched.baseline_now.append(float(ev.time))
        elif name == "baseline_hold":
            if _as_bool(ev.value):
                hold_start = float(ev.time)
            elif hold_start is not None:
                sched.baseline_hold.append((hold_start, float(ev.time)))
                hold_start = None
        elif name == "baseline_hold_off":
            if hold_start is not None:
                sched.baseline_hold.append((hold_start, float(ev.time)))
                hold_start = None
        elif name == "baseline_next_valley":
            sched.baseline_next_valley.append(float(ev.time))
        elif name == "baseline_back":
            sched.baseline_back.append(float(ev.time))
        elif name == "split_peak":
            sched.split_peak.append(float(ev.time))
        elif name == "solvent_peak":
            sched.solvent_peak.append(float(ev.time))
        elif name == "peak_sum_slice":
            if _as_bool(ev.value):
                slice_start = float(ev.time)
            elif slice_start is not None:
                sched.peak_sum_slice.append((slice_start, float(ev.time)))
                slice_start = None
        elif name == "peak_sum_slice_off":
            if slice_start is not None:
                sched.peak_sum_slice.append((slice_start, float(ev.time)))
                slice_start = None
        elif name in ("set_baseline_start", "set_baseline_end"):
            sched.forced_baseline.append((float(ev.time), name))
    if hold_start is not None:
        sched.baseline_hold.append((hold_start, float(times[-1])))
    if slice_start is not None:
        sched.peak_sum_slice.append((slice_start, float(times[-1])))
    sched.switches = switches  # type: ignore[attr-defined]
    return sched
