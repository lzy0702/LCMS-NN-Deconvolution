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
                best["mass"] = (best["mass"] * w0 + comp.mass * w1) / max(w0 + w1, 1e-12)
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
    for tr in tracks:
        if len(tr["frames"]) < min_frames:
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
        nz = inten[inten > 0]
        if nz.size and np.median(nz) > 0 and inten.max() / np.median(nz) < noise_peak_ratio and inten.size > 6 * min_frames:
            # flat profile across the whole window: chemical background, not a chromatographic peak
            continue
        spread = tr["spread"] if np.isfinite(tr["spread"]) else 0.0
        species.append(Species(
            id=sid, mass=tr["mass"], mass_type="average", polarity=tr["polarity"],
            compound_class=tr["cls"], time=times.copy(), intensity=inten,
            adduct_intensity=adducts, charges=charges, score=tr["score"],
            mass_spread_ppm=float(spread), region_id=region_id,
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
