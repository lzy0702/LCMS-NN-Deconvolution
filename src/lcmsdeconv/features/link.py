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
    shared_edges: tuple[bool, bool] = (False, False),
) -> list[Species]:
    """Group components with consistent mass across consecutive frames.

    A track whose apex falls within ``edge_fraction`` of the frames (at least ``edge_frames``)
    of either end of the window is dropped, unless it is the window's strongest: a region is
    built around elution bands, so such a track is a template soaking up a neighbour's tail.
    ``shared_edges`` marks the ends where the window abuts another region's (a boundary drawn at
    a valley between two bands): a species eluting near that valley peaks there, and the other
    region quantifies the rest of it, so the test is not made on that side.
    """
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
        at_start = apex_frame < edge and not shared_edges[0]
        at_end = apex_frame >= n_frames - edge and not shared_edges[1]
        if n_frames > 6 * min_frames and total < strongest and (at_start or at_end):
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


def _on(times: np.ndarray, t_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    """A trace sampled at ``t_src`` placed on ``times``, which hold every one of those times."""
    y = np.zeros(times.size)
    np.add.at(y, np.searchsorted(times, t_src), y_src)
    return y


def _coelute(a: Species, b: Species, min_r: float, max_apex_min: float) -> bool:
    """Do two species' traces peak together and rise and fall together?"""
    if abs(a.rt_apex - b.rt_apex) > max_apex_min:
        return False
    times = np.union1d(a.time, b.time)
    ya, yb = _on(times, a.time, a.intensity), _on(times, b.time, b.intensity)
    carried = (ya > 0) | (yb > 0)
    if np.count_nonzero(carried) < 4:
        return True  # too few scans to show a profile: the apexes decide
    ya, yb = ya[carried], yb[carried]
    if ya.std() <= 0 or yb.std() <= 0:
        return False
    return float(np.corrcoef(ya, yb)[0, 1]) >= min_r


def fold_adduct_forms(species: list[Species], libraries, tolerance_da: float = 1.0,
                      tolerance_ppm: float = 60.0, min_r: float = 0.8,
                      max_apex_min: float = 0.1) -> list[Species]:
    """Count every species' adduct forms with it, wherever the decoder reported them.

    Within a region the decoder attaches a species' adduct forms to it as adduct columns when
    the region shows those adducts next to its strongest species
    (:func:`lcmsdeconv.deconv.pipeline.evidenced_adducts`); where it does not, the same forms
    come out as species of their own. Amounts are then not comparable: in an IP-HILIC run an
    impurity counted its sodium and potassium forms while the main species, whose forms had
    become ten separate species worth 90 % of it, did not, and the impurity came out at twice
    its share.

    A species is an adduct form of another when its mass is the other's plus an adduct state
    of the method's library (average-mass shifts, within ``tolerance_da`` or
    ``tolerance_ppm``), the other is at least as strong, and the two elute together: apexes
    within ``max_apex_min`` and traces correlating at ``min_r`` or better. Its trace is added
    to the parent's under the combined adduct labels. Where several species qualify, the
    strongest is the parent, as a weaker species does not keep a stronger one's adduct. A mass
    difference that a known modification of the species' class explains as well (oxidation,
    +15.99 Da, lies 1.04 Da from an ammonium adduct; a PEG unit 0.07 Da from two sodium ions)
    is not folded: the species may be an impurity, and a report too many is the lesser error.
    ``libraries`` is an :class:`~lcmsdeconv.chem.adducts.AdductLibrary` or a dict of them by
    polarity.
    """
    from ..chem.adducts import AdductState
    from ..chem.modifications import annotate_delta

    cache: dict[int, tuple] = {}

    def states(pol: int):
        if pol not in cache:
            lib = libraries.get(pol) if isinstance(libraries, dict) else libraries
            st = [x for x in lib.states() if x.counts] if lib is not None else []
            cache[pol] = (st, np.array([x.avg_mass for x in st]), lib.names() if lib else [])
        return cache[pol]

    out: list[Species] = []
    folded: dict[int, list[str]] = {}
    for s in sorted(species, key=lambda x: -x.total_intensity):
        st, shifts, order = states(s.polarity)
        parent = state = None
        for k in out if st else ():
            d = s.mass - k.mass
            if k.polarity != s.polarity or d <= 0:
                continue
            j = int(np.argmin(np.abs(shifts - d)))
            tol = max(tolerance_da, k.mass * tolerance_ppm * 1e-6)
            if abs(shifts[j] - d) > tol:
                continue
            if annotate_delta(d, s.compound_class or k.compound_class, tolerance=tol):
                continue
            if _coelute(k, s, min_r, max_apex_min):
                parent, state = k, st[j]
                break
        if parent is None:
            out.append(s)
            continue
        times = np.union1d(parent.time, s.time)
        forms = {lab: _on(times, parent.time, y) for lab, y in parent.adduct_intensity.items()}
        if not forms:
            forms = {"base": _on(times, parent.time, parent.intensity)}
        own = s.adduct_intensity or {"base": s.intensity}
        for lab, y in own.items():
            key = state.plus(AdductState.from_label(lab, order), order).label
            forms[key] = forms.get(key, np.zeros(times.size)) + _on(times, s.time, y)
        parent.intensity = _on(times, parent.time, parent.intensity) + _on(times, s.time, s.intensity)
        parent.adduct_intensity = forms
        parent.time = times
        for z, v in s.charges.items():
            parent.charges[z] = parent.charges.get(z, 0.0) + v
        folded.setdefault(id(parent), []).append(f"{state.label} ({s.mass:.2f} Da)")
    for k in out:
        if id(k) in folded:
            k.flags.append("adduct forms reported apart counted with it: "
                           + ", ".join(folded[id(k)]))
    return out


def merge_adjacent_species(species: list[Species], tolerance_ppm: float = 30.0,
                           tolerance_da: float = 0.5, max_gap_min: float = 0.2) -> list[Species]:
    """Merge one species that was linked separately in neighbouring regions.

    A broad elution band can be split into several regions; the same species is then found in
    each and would be reported several times at nearly the same retention time. Species of the
    same polarity and mass whose traces touch (within ``max_gap_min``) are merged into the
    strongest, concatenating their traces. A scan both regions hold (the one on their shared
    boundary) is counted once, with the larger of its two readings: the other region's bank
    may hold only a weak copy of the species there, and adding the two counted it twice.
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
        y_hit, y_s = _on(times, hit.time, hit.intensity), _on(times, s.time, s.intensity)
        # per scan, one piece's reading: the only one, or on a shared scan the larger
        take = y_s > y_hit
        merged_adducts = {}
        for name in set(hit.adduct_intensity) | set(s.adduct_intensity):
            a = (_on(times, hit.time, hit.adduct_intensity[name])
                 if name in hit.adduct_intensity else np.zeros(times.size))
            b = (_on(times, s.time, s.adduct_intensity[name])
                 if name in s.adduct_intensity else np.zeros(times.size))
            merged_adducts[name] = np.where(take, b, a)
        hit.intensity = np.where(take, y_s, y_hit)
        hit.adduct_intensity = merged_adducts
        hit.time = times
        for z, v in s.charges.items():
            hit.charges[z] = hit.charges.get(z, 0.0) + v
    return out
