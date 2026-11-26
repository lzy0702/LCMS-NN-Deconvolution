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


def merge_species_across_polarity(species: list[Species], tolerance_ppm: float = 50.0,
                                  max_apex_min: float = 0.2) -> list[Species]:
    """Merge species detected in both polarities of a polarity-switching run, keeping the reading
    with the larger base form (the amount reported). The two readings must peak together
    (apexes within ``max_apex_min``): isomers of one mass that elute apart stay two species,
    whichever polarity shows each."""
    out: list[Species] = []
    for s in sorted(species, key=lambda x: -x.base_intensity):
        hit = None
        for k in out:
            if (k.polarity != s.polarity and abs(k.mass - s.mass) / k.mass * 1e6 < tolerance_ppm
                    and abs(k.rt_apex - s.rt_apex) <= max_apex_min):
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
    """Do two species' traces (all forms) peak together and rise and fall together?"""
    times = np.union1d(a.time, b.time)
    ya, yb = _on(times, a.time, a.intensity), _on(times, b.time, b.intensity)
    if abs(times[int(np.argmax(ya))] - times[int(np.argmax(yb))]) > max_apex_min:
        return False
    carried = (ya > 0) | (yb > 0)
    if np.count_nonzero(carried) < 4:
        return True  # too few scans to show a profile: the apexes decide
    ya, yb = ya[carried], yb[carried]
    if ya.std() <= 0 or yb.std() <= 0:
        return False
    return float(np.corrcoef(ya, yb)[0, 1]) >= min_r


def fold_adduct_forms(species: list[Species], libraries, tolerance_da: float = 0.6,
                      tolerance_ppm: float = 40.0, min_r: float = 0.8,
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
    within ``max_apex_min`` and traces correlating at ``min_r`` or better (strength: the base
    form's amount). Its trace is added to the parent's under the combined adduct labels. Where
    several species qualify, the strongest is the parent, as a weaker species does not keep a
    stronger one's adduct. A mass
    difference that a known modification of the species' class explains as well (peptide
    oxidation, +15.99 Da, lies 1.04 Da from an ammonium adduct; an oligonucleotide's n+1 can lie
    within a dalton of amine and HFIP clusters; average-mass deltas, multiples only of the
    modifications one molecule carries at several sites) is not folded: the species may be an
    impurity, and a report too many is the lesser error. Modifications listed for any class only
    (a PEG unit, 0.07 Da from two sodium ions) do not count.
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
    # strength is the base form's: a form reported apart can carry adduct columns of its own
    # that took up a neighbour's ladder, and outweigh its parent on all forms
    for s in sorted(species, key=lambda x: -x.base_intensity):
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
            cls = s.compound_class or k.compound_class
            if any(cls in m.classes for m in annotate_delta(d, cls, tolerance=tol,
                                                            mass_type="average",
                                                            multiple_tolerance=1.0)):
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
            k.flags.append(FOLD_FLAG + ", ".join(folded[id(k)]))
    return out


def _one_band(times: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, valley: float,
              comparable: float = 0.1, max_fill: int = 3) -> bool:
    """Do two pieces' traces (on one time axis) join into one elution band? A piece below
    ``comparable`` of the other's height is a fragment of it (a tail, a stray scan) and joins.

    A short run of scans where neither piece has a reading - a gap the linking bridged (two
    scans at most), the three frames whose fit gave the species nothing that end a track - is
    no evidence of a valley: the band is read across it from the readings either side. A run
    longer than ``max_fill`` scans is a valley: two isomers seen in a scan or two each, a
    minute apart, are two bands."""
    ia, ib = int(np.argmax(y_a)), int(np.argmax(y_b))
    ha, hb = float(y_a[ia]), float(y_b[ib])
    lo, hi = sorted((ia, ib))
    if hi - lo < 2 or min(ha, hb) < comparable * max(ha, hb):
        return True
    joined = np.maximum(y_a, y_b)[lo:hi + 1]
    have = joined > 0
    if 2 <= int(have.sum()) < joined.size:
        t = np.asarray(times, dtype=float)[lo:hi + 1]
        filled = np.interp(t, t[have], joined[have])
        short = np.zeros(joined.size, dtype=bool)
        i = 0
        while i < joined.size:
            if have[i]:
                i += 1
                continue
            j = i
            while j < joined.size and not have[j]:
                j += 1
            short[i:j] = j - i <= max_fill
            i = j
        joined = np.where(short, filled, joined)
    return float(joined.min()) >= valley * min(ha, hb)


def merge_adjacent_species(species: list[Species], tolerance_ppm: float = 30.0,
                           tolerance_da: float = 0.5, max_gap_min: float = 0.2,
                           valley: float = 0.5) -> list[Species]:
    """Merge one species that was linked separately in neighbouring regions.

    A broad elution band can be split into several regions; the same species is then found in
    each and would be reported several times at nearly the same retention time. Species of the
    same polarity and mass whose traces touch (within ``max_gap_min``) are merged into the
    strongest, concatenating their traces. A scan both regions hold (the one on their shared
    boundary) is counted once, with the larger of its two readings: the other region's bank
    may hold only a weak copy of the species there, and adding the two counted it twice.
    Pieces of comparable height whose joined trace falls between their apexes below ``valley``
    of the lower apex are two elution bands of one mass - isomers either side of a boundary -
    and stay apart.

    Pieces are taken strongest first, so a piece that touches only another weak piece of its
    band, not yet joined to the strong one, is compared with the strong piece's trace before
    that grew: the merge is repeated until nothing more joins (tails of 2653.32 and 2998.45 Da
    of the IP-RPLC conjugate run over three regions each left their last piece apart).
    """
    out = list(species)
    while True:
        merged = _merge_pass(out, tolerance_ppm, tolerance_da, max_gap_min, valley)
        if len(merged) == len(out):
            return merged
        out = merged


def _merge_pass(species: list[Species], tolerance_ppm: float, tolerance_da: float,
                max_gap_min: float, valley: float) -> list[Species]:
    """One pass of :func:`merge_adjacent_species`, strongest piece first."""
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
                t = np.union1d(k.time, s.time)
                if _one_band(t, _on(t, k.time, k.intensity), _on(t, s.time, s.intensity), valley):
                    hit = k
                    break
        if hit is None:
            out.append(s)
            continue
        _absorb(hit, s)
    return out


def _absorb(hit: Species, s: Species) -> None:
    """Join piece ``s`` of one elution band to ``hit``, which keeps its mass: on a scan both
    hold (the boundary of their regions), the larger reading."""
    times = np.union1d(hit.time, s.time)
    y_hit, y_s = _on(times, hit.time, hit.intensity), _on(times, s.time, s.intensity)
    # per scan, one piece's reading: the only one, or on a shared scan the larger
    take = y_s > y_hit
    merged_adducts = {}
    # in the order first met (hit's, then the piece's), so outputs list them the same every run
    for name in dict.fromkeys([*hit.adduct_intensity, *s.adduct_intensity]):
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
    _merge_flags(hit, s)


def _reading(sp: Species, t: float, tol: float) -> float:
    """``sp``'s trace at the scan at time ``t`` (0 where it has no reading within ``tol``)."""
    if not sp.time.size:
        return 0.0
    i = int(np.argmin(np.abs(sp.time - t)))
    return float(sp.intensity[i]) if abs(float(sp.time[i]) - t) <= tol else 0.0


def merge_boundary_pieces(species: list[Species], boundaries, same_species, steps,
                          tolerance_ppm: float = 30.0, tolerance_da: float = 0.5,
                          max_da: float = 1.5, min_presence: float = 0.3,
                          valley: float = 0.5) -> list[Species]:
    """Join to its species a piece of its band that a region boundary cut off and that the
    piece's region read up to a dalton off.

    A boundary drawn at the valley between two bands leaves a band's front or tail in the
    neighbouring region, whose summed spectrum mixes it with what elutes there and can put its
    mass beyond :func:`merge_adjacent_species`' tolerance (n-1's tail under IP-HILIC, 4833.14 Da
    against 4833.86 Da): it is reported as a species of its own, peaking on the boundary. Such a
    piece - its trace on one side of a shared boundary, its apex on it - is joined to a stronger
    species within ``max_da`` that holds that boundary scan (at least ``min_presence`` of the
    piece's reading there), runs on past it and peaks on the other side, without a valley
    between them (:func:`_one_band`), and only where the two are one molecule by their
    isotopes: ``same_species(piece, band, species, band)`` reads the piece's scans next to the
    boundary and the species' apex scans, and returns the average mass the isotopes put the
    molecule at (or True) where it is one, a false value where not; the joined species keeps the
    stronger piece's mass unless the piece's lies nearer that reading. A deaminated form, 0.98 Da
    from its parent, lies one tooth of the isotope comb away from it, and so do the minor species
    whose leading edges a region reads as a blend half a dalton or a dalton from a neighbour
    (6777.31 and 6792.99 Da of the IP-RPLC conjugate run, lighter than PO and FLP by more than a
    spacing, peaking at 19.0 min): those stay.

    ``boundaries`` holds (polarity, time) of every boundary two regions share, ``steps`` the
    scan spacing (min) by polarity.
    """
    removed: set[int] = set()
    for s in sorted(species, key=lambda x: x.base_intensity):
        if id(s) in removed or not s.time.size:
            continue
        step = float(steps.get(s.polarity, 0.0))
        if step <= 0:
            continue
        half = 0.5 * step
        apex = s.rt_apex
        tb = next((b for pol, b in boundaries if pol == s.polarity and abs(apex - b) <= half), None)
        if tb is None:
            continue
        if abs(float(s.time.max()) - tb) <= half:
            side = -1  # the piece ends on the boundary: the front of a later band
        elif abs(float(s.time.min()) - tb) <= half:
            side = 1  # it starts there: the tail of an earlier band
        else:
            continue
        tol = max(s.mass * tolerance_ppm * 1e-6, tolerance_da)
        here = _reading(s, tb, half)
        cands = []
        for k in species:
            if (k is s or id(k) in removed or k.polarity != s.polarity or not k.time.size
                    or k.base_intensity <= s.base_intensity):
                continue
            dm = abs(k.mass - s.mass)
            if dm <= tol or dm > max_da:
                continue
            if side < 0 and not (float(k.time.max()) > tb + half and k.rt_apex > tb + half):
                continue
            if side > 0 and not (float(k.time.min()) < tb - half and k.rt_apex < tb - half):
                continue
            if _reading(k, tb, half) < max(min_presence * here, 1e-300):
                continue
            t = np.union1d(k.time, s.time)
            if not _one_band(t, _on(t, k.time, k.intensity), _on(t, s.time, s.intensity), valley):
                continue
            cands.append((dm, k))
        # three scans of each: the piece's next to the boundary, the species' about its apex
        band_s = (tb - 2.5 * step, tb + half) if side < 0 else (tb - half, tb + 2.5 * step)
        for _dm, k in sorted(cands, key=lambda c: c[0]):
            tooth = same_species(s, band_s, k, (k.rt_apex - 1.5 * step, k.rt_apex + 1.5 * step))
            if tooth:
                # the joined species keeps the mass its isotopes support: the stronger piece's
                # unless the piece's lies nearer the tooth (a region's summed spectrum can read
                # the band's main piece most of a dalton off, n-1 under IP-HILIC at +0.88 Da)
                own = isinstance(tooth, float) and abs(s.mass - tooth) < abs(k.mass - tooth)
                _absorb(k, s)
                if own:
                    k.mass = s.mass
                removed.add(id(s))
                break
    return [x for x in species if id(x) not in removed]


LATER_FLAG = ("the same molecule found later in the run (a tail past its band, the column wash), "
              "not reported apart: ")


def set_aside_later_elutions(species: list[Species], same_species, steps,
                             tolerance_ppm: float = 30.0, tolerance_da: float = 0.5,
                             min_ratio: float = 20.0, min_gap_min: float = 0.2,
                             with_main_min: float = 0.1, min_rise: float = 2.0) -> list[Species]:
    """Name the main species' later elutions in its flags instead of reporting them as species.

    The run's main species loads the column far more than anything else, and part of it comes
    off after its band: a tail that runs on across region boundaries, and a peak when the
    column is washed (NC of the IP-RPLC NC run: a tail at 0.07-0.3 % of its height from 19 to
    25 min, then a peak at 1.3 % of its height at 25.4 min). Each region that sees it reports it
    at the main species' mass, as an impurity it is not. A species within the linking tolerance
    of the main species' mass (the strongest of its polarity), at most ``1 / min_ratio`` of it,
    whose trace starts more than ``min_gap_min`` after the main species' ends (closer pieces are
    :func:`merge_adjacent_species`' business) and that is the same molecule by its isotopes
    (``same_species``, read at both apexes) is taken out of the list, its mass, time and share
    named in the main species' flags; the main species' amount stays its own band's. The same
    event carries the main species' other forms off the column with it (NC's +201.4 Da form at
    25.4 min): a species peaking within ``with_main_min`` of one of the main species' later
    elutions is set aside the same way under an earlier, stronger species of its own mass. Only
    a later elution that rises and falls in its own trace (its highest reading ``min_rise``
    times its first and last) is such an event; a slice of the tail has its highest reading
    wherever a region boundary cut it (NC's at 20.01 min, three scans after its region's start,
    1.01 times its first), and nothing else leaves the column with it there. The flags name
    every species set aside.

    An isomer of the main species that elutes later, at most ``1 / min_ratio`` of it, is set
    aside too: its isotopes cannot tell it from the main species (an isoAsp or
    disulfide-scrambled variant of a protein, a branched polymer beside a linear one). That is
    why a method turns this on (``linking.set_aside_later_elutions``, off by default) only where
    the main species' tail and its re-elution in the column wash are the concern: the
    oligonucleotide methods.
    """
    parent: dict[int, Species] = {}

    def later_of(s: Species, p: Species, step: float) -> bool:
        if (p is s or id(p) in parent or p.polarity != s.polarity or not p.time.size
                or p.base_intensity < min_ratio * s.base_intensity):
            return False
        if abs(p.mass - s.mass) > max(p.mass * tolerance_ppm * 1e-6, tolerance_da):
            return False
        if float(s.time.min()) - float(p.time.max()) <= min_gap_min:
            return False
        return bool(same_species(s, (s.rt_apex - 1.5 * step, s.rt_apex + 1.5 * step),
                                 p, (p.rt_apex - 1.5 * step, p.rt_apex + 1.5 * step)))

    def event(s: Species) -> bool:
        y = np.asarray(s.intensity, dtype=float)
        return y.size > 2 and float(y.max()) >= min_rise * max(float(y[0]), float(y[-1]))

    for pol in sorted({s.polarity for s in species}):
        mine = sorted((s for s in species if s.polarity == pol and s.time.size),
                      key=lambda x: -x.base_intensity)
        step = float(steps.get(pol, 0.0))
        if len(mine) < 2 or step <= 0:
            continue
        main = mine[0]
        for s in mine[1:]:
            if later_of(s, main, step):
                parent[id(s)] = main
        times = [s.rt_apex for s in mine if parent.get(id(s)) is main and event(s)]
        for s in mine[1:]:
            if id(s) in parent or not any(abs(s.rt_apex - t) <= with_main_min for t in times):
                continue
            p = next((p for p in mine if later_of(s, p, step)), None)
            if p is not None:
                parent[id(s)] = p
    later: dict[int, list[Species]] = {}
    for s in species:
        if id(s) in parent:
            later.setdefault(id(parent[id(s)]), []).append(s)
    for p in species:
        if id(p) in later:
            p.flags.append(LATER_FLAG + ", ".join(
                f"{s.mass:.2f} Da at {s.rt_apex:.2f} min "
                f"({100 * s.base_intensity / max(p.base_intensity, 1e-300):.2f} %)"
                for s in sorted(later[id(p)], key=lambda x: x.rt_apex)))
    return [s for s in species if id(s) not in parent]


FOLD_FLAG = "adduct forms reported apart counted with it: "


def _merge_flags(hit: Species, s: Species) -> None:
    """The flags of two merged pieces: the adduct forms folded into either, each form once in
    one line; a one-charge-state flag only while the merged species still has one charge
    state; the stronger piece's other flags."""
    keep = [f for f in hit.flags if not f.startswith(FOLD_FLAG)]
    if sum(1 for v in hit.charges.values() if v > 0) > 1:
        keep = [f for f in keep if not f.startswith("one charge state")]
    forms: dict[str, str] = {}
    for f in hit.flags + s.flags:
        if f.startswith(FOLD_FLAG):
            for item in f[len(FOLD_FLAG):].split(", "):
                forms.setdefault(item.split(" (")[0], item)
    if forms:
        keep.append(FOLD_FLAG + ", ".join(forms.values()))
    hit.flags = keep
