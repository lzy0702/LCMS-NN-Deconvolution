"""Link per-frame components across retention time into deconvolved species (EICs)."""

from __future__ import annotations

import numpy as np

from ..core.model import FrameResult, Species


def link_species(
    frame_results: list[FrameResult],
    mass_tolerance_ppm: float = 30.0,
    mass_tolerance_da: float = 0.5,
    min_frames: int = 2,
    max_gap_frames: int = 2,
    noise_peak_ratio: float = 3.0,
    region_id: int = -1,
    start_id: int = 0,
    edge_frames: int = 2,
    edge_fraction: float = 0.1,
    flat_test: bool = True,
) -> list[Species]:
    """Group components with consistent mass across consecutive frames."""
    if not frame_results:
        return []
    frame_results = sorted(frame_results, key=lambda f: f.rt)
    times = np.array([f.rt for f in frame_results])

    tracks: list[dict] = []
    for fi, fr in enumerate(frame_results):
        for comp in fr.components:
            best = None
            best_err = np.inf
            for tr in tracks:
                if fi - tr["last_frame"] > max_gap_frames + 1:
                    continue
                tol = max(tr["mass"] * mass_tolerance_ppm * 1e-6, mass_tolerance_da)
                err = abs(comp.mass - tr["mass"])
                if err <= tol and err < best_err:
                    best, best_err = tr, err
            if best is None:
                tracks.append({
                    "mass": comp.mass, "weight": comp.intensity, "last_frame": fi,
                    "frames": {fi: comp}, "polarity": fr.polarity,
                    "cls": comp.compound_class, "score": comp.score,
                    "spread": comp.mass_spread_ppm,
                })
            else:
                w0, w1 = best["weight"], comp.intensity
                if w0 + w1 > 0:
                    # an exact weighted mean: flooring the denominator (it was max(w0 + w1,
                    # 1e-12)) shrank the mass toward zero whenever the intensities were tiny,
                    # and the shrunken track then spawned species at meaningless masses
                    best["mass"] = (best["mass"] * w0 + comp.mass * w1) / (w0 + w1)
                best["weight"] = w0 + w1
                best["last_frame"] = fi
                prev = best["frames"].get(fi)
                if prev is None:
                    best["frames"][fi] = comp
                else:
                    prev.intensity += comp.intensity
                    for a, v in comp.adducts.items():
                        prev.adducts[a] = prev.adducts.get(a, 0.0) + v
                best["score"] = max(best["score"], comp.score)
                best["spread"] = min(best["spread"] or np.inf, comp.mass_spread_ppm or np.inf)

    species: list[Species] = []
    sid = start_id
    strongest = max((sum(c.intensity for c in tr["frames"].values()) for tr in tracks), default=0.0)
    n_frames = times.size
    for tr in tracks:
        if len(tr["frames"]) < min_frames:
            continue
        total = sum(c.intensity for c in tr["frames"].values())
        apex_frame = max(tr["frames"], key=lambda f: tr["frames"][f].intensity)
        edge = max(edge_frames, int(round(edge_fraction * n_frames)))
        if (n_frames > 6 * min_frames and total < strongest
                and (apex_frame < edge or apex_frame >= n_frames - edge)):
            # A region is built around elution bands, so a species whose trace peaks on the
            # region's edge is not eluting here: it is a template soaking up a neighbour's tail
            # or residue. The region's strongest species is exempt (it defines the region).
            continue
        inten = np.zeros(times.size)
        adducts: dict[str, np.ndarray] = {}
        charges: dict[int, float] = {}
        for fi, comp in tr["frames"].items():
            inten[fi] = comp.intensity
            for a, v in comp.adducts.items():
                adducts.setdefault(a, np.zeros(times.size))[fi] = v
            for z, v in comp.charges.items():
                charges[z] = charges.get(z, 0.0) + v
        if flat_test and inten.size > 6 * min_frames and _is_flat(inten, noise_peak_ratio):
            # flat profile across the whole window: chemical background, not a chromatographic peak
            continue
        spread = tr["spread"] if np.isfinite(tr["spread"]) else 0.0
        flags = []
        if len([z for z, v in charges.items() if v > 0]) == 1 and min(charges) > 1:
            flags.append("one charge state: charge read from isotope spacing")
        species.append(Species(
            id=sid, mass=tr["mass"], mass_type="average", polarity=tr["polarity"],
            compound_class=tr["cls"], time=times.copy(), intensity=inten,
            adduct_intensity=adducts, charges=charges, score=tr["score"],
            mass_spread_ppm=float(spread), region_id=region_id, flags=flags,
        ))
        sid += 1
    species.sort(key=lambda s: -s.total_intensity)
    return species


def merge_species_across_polarity(species: list[Species], tolerance_ppm: float = 50.0) -> list[Species]:
    """Merge species detected in both polarities of a polarity-switching run."""
    out: list[Species] = []
    for s in sorted(species, key=lambda x: -x.total_intensity):
        hit = None
        for k in out:
            if k.polarity != s.polarity and abs(k.mass - s.mass) / k.mass * 1e6 < tolerance_ppm:
                hit = k
                break
        if hit is None:
            out.append(s)
        else:
            hit.flags.append(f"also seen in {'positive' if s.polarity > 0 else 'negative'} mode")
    return out


def _is_flat(trace: np.ndarray, ratio: float, edge_frames: int = 2) -> bool:
    """True when a trace stays within ``ratio`` of its maximum at *both* ends of its window.

    Regions are built around elution bands, so a real peak's trace falls well below its maximum
    at one end of the window at least, while chemical background stays up at both. Comparing
    the maximum with the median instead calls a broad real peak that fills its region flat
    (its median is a third of its maximum), and that rule dropped the main species of real
    oligonucleotide runs.
    """
    peak = float(trace.max()) if trace.size else 0.0
    if peak <= 0:
        return True
    k = max(1, min(edge_frames, trace.size // 4))
    low_end = min(float(trace[:k].mean()), float(trace[-k:].mean()))
    return low_end * ratio >= peak


def merge_adjacent_species(species: list[Species], tolerance_ppm: float = 30.0,
                           tolerance_da: float = 0.5, max_gap_min: float = 0.2) -> list[Species]:
    """Merge one species that was linked separately in neighbouring regions.

    A broad elution band can be split into several regions; the same species is then found in
    each and would be reported several times at nearly the same retention time. Species of the
    same polarity and mass whose traces touch (within ``max_gap_min``) are merged into the
    strongest, concatenating their traces.
    """
    out: list[Species] = []
    for s in sorted(species, key=lambda x: -x.total_intensity):
        hit = None
        for k in out:
            if k.polarity != s.polarity or not s.time.size or not k.time.size:
                continue
            if abs(k.mass - s.mass) > max(k.mass * tolerance_ppm * 1e-6, tolerance_da):
                continue
            gap = max(float(s.time.min()) - float(k.time.max()), float(k.time.min()) - float(s.time.max()))
            if gap <= max_gap_min:
                hit = k
                break
        if hit is None:
            out.append(s)
            continue
        times = np.union1d(hit.time, s.time)

        def on(t_src, y_src, t_dst=times):
            y = np.zeros(t_dst.size)
            idx = np.searchsorted(t_dst, t_src)
            np.add.at(y, idx, y_src)
            return y

        merged_adducts = {}
        for name in set(hit.adduct_intensity) | set(s.adduct_intensity):
            y = np.zeros(times.size)
            if name in hit.adduct_intensity:
                y += on(hit.time, hit.adduct_intensity[name])
            if name in s.adduct_intensity:
                y += on(s.time, s.adduct_intensity[name])
            merged_adducts[name] = y
        hit.intensity = on(hit.time, hit.intensity) + on(s.time, s.intensity)
        hit.adduct_intensity = merged_adducts
        hit.time = times
        for z, v in s.charges.items():
            hit.charges[z] = hit.charges.get(z, 0.0) + v
    return out
