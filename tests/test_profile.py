"""Vendor-thresholded profile scans: sample widths, interpolation, region sums, grid views."""

import numpy as np

from lcmsdeconv.core.model import Run, Spectrum
from lcmsdeconv.core.profile import gap_mask, profile_on_axis, sample_widths
from lcmsdeconv.nn.grid import LogMzGrid, spectrum_to_grid


def _segments(starts, n=7, spacing=0.002, height=1000.0):
    """Short profile segments of ``n`` points (a Gaussian peak each), gaps between them."""
    mz, y = [], []
    for s in starts:
        x = s + spacing * np.arange(n)
        mz.append(x)
        y.append(height * np.exp(-0.5 * ((x - x.mean()) / (1.5 * spacing)) ** 2))
    return np.concatenate(mz), np.concatenate(y)


def test_widths_follow_the_midpoint_rule_on_a_continuous_axis():
    u = np.log(1000.0 + 0.002 * np.arange(200))
    w = sample_widths(u)
    g = np.diff(u)
    # interior samples stand for half the distance to each neighbour
    assert np.allclose(w[1:-1], 0.5 * (g[:-1] + g[1:]))
    # the two ends stand for their own spacing, not half of it
    assert np.isclose(w[0], g[0]) and np.isclose(w[-1], g[-1])


def test_a_segment_edge_does_not_stand_for_the_gap_beside_it():
    mz, _ = _segments([1000.0, 1003.0, 1010.0])
    u = np.log(mz)
    w = sample_widths(u)
    own = np.median(np.diff(u[:7]))
    # every sample, the edges included, stands for about its own spacing
    assert np.all(w < 1.5 * own) and np.all(w > 0.5 * own)
    gaps = gap_mask(mz)
    assert gaps.sum() == 2 and gaps[6] and gaps[13]


def test_an_isolated_sample_takes_the_local_spacing():
    mz, _ = _segments([1000.0, 1010.0])
    mz = np.sort(np.concatenate([mz, [1005.0]]))
    w = sample_widths(np.log(mz))
    own = np.median(np.diff(np.log(mz[:7])))
    assert 0.5 * own < w[7] < 1.5 * own


def test_resampling_a_thresholded_scan_keeps_its_area():
    mz, y = _segments([1000.0, 1004.0, 1009.0])
    g = LogMzGrid(990.0, 1020.0, 2e-5, 1)
    out = g.resample_profile(mz, y)
    u = np.log(mz - g.carrier)
    # the grid holds the scan's own area (intensity x width), not the gaps' worth
    area_scan = float(np.sum(y * sample_widths(u)))
    assert np.isclose(out.sum() * g.step, area_scan, rtol=1e-6)
    # nothing is written between the segments
    between = (g.mz > 1000.02) & (g.mz < 1003.99)
    assert out[between].max() == 0.0


def test_profile_on_axis_is_zero_inside_gaps_and_outside_the_scan():
    mz, y = _segments([1000.0, 1004.0])
    axis = np.array([999.0, 1000.006, 1002.0, 1004.006, 1006.0])
    v = profile_on_axis(axis, mz, y)
    assert v[0] == 0.0 and v[2] == 0.0 and v[4] == 0.0
    assert np.isclose(v[1], np.interp(1000.006, mz, y))
    assert np.isclose(v[3], np.interp(1004.006, mz, y))


def test_region_sum_neither_ramps_across_gaps_nor_drops_points():
    # scan A has a peak at 1002 where scan B has a gap; B has one at 1006 where A has a gap
    a_mz, a_y = _segments([1000.0, 1002.0, 1010.0])
    b_mz, b_y = _segments([1000.0, 1006.0, 1010.0])
    frames = [Spectrum(a_mz, a_y.astype(np.float32), 1.0, 1, is_profile=True),
              Spectrum(b_mz, b_y.astype(np.float32), 1.01, 1, is_profile=True)]
    run = Run(frames)
    s = run.sum_frames(frames)
    # every point either scan measured is on the summed axis
    assert np.all(np.isin(a_mz, s.mz)) and np.all(np.isin(b_mz, s.mz))
    # A's peak is A's alone (B holds nothing in its gap), B's peak is B's alone
    for x, y0 in zip(a_mz[7:14], a_y[7:14]):
        assert np.isclose(s.intensity[np.searchsorted(s.mz, x)], y0, rtol=1e-5)
    for x, y0 in zip(b_mz[7:14], b_y[7:14]):
        assert np.isclose(s.intensity[np.searchsorted(s.mz, x)], y0, rtol=1e-5)
    assert len(s.meta["frames"]) == 2


def test_grid_view_of_a_region_is_the_sum_of_its_scans_grid_views():
    rng = np.random.default_rng(3)
    frames = []
    for k in range(5):
        starts = np.sort(rng.uniform(1000.0, 1050.0, 12))
        mz, y = _segments(starts, height=float(rng.uniform(500, 2000)))
        frames.append(Spectrum(mz, y.astype(np.float32), 1.0 + 0.01 * k, 1, is_profile=True))
    run = Run(frames)
    s = run.sum_frames(frames)
    grid, observed, _sigma = spectrum_to_grid(s.mz, s.intensity, 1, frames=s.meta["frames"])
    ref = np.zeros(grid.size)
    for f in frames:
        ref += grid.resample_profile(f.mz, f.intensity)
    assert np.allclose(observed, ref)
    # same total area as the scans themselves
    areas = sum(float(np.sum(f.intensity * sample_widths(grid.mz_to_u(f.mz)))) for f in frames)
    assert np.isclose(observed.sum() * grid.step, areas, rtol=1e-5)


def test_frames_on_one_axis_sum_point_by_point():
    ax = 1000.0 + 0.002 * np.arange(300)
    frames = [Spectrum(ax.copy(), np.full(300, float(k + 1), np.float32), 1.0 + 0.01 * k, 1,
                       is_profile=True) for k in range(3)]
    s = Run(frames).sum_frames(frames)
    assert s.mz.size == 300 and np.allclose(s.intensity, 6.0)
    grid, observed, _ = spectrum_to_grid(s.mz, s.intensity, 1, frames=s.meta["frames"])
    assert np.allclose(observed, grid.resample_profile(s.mz, s.intensity))


def test_isolated_first_and_last_points_stand_for_their_own_spacing():
    # one point, a long gap, peaks, a long gap, one point
    mz = np.concatenate([[1000.0], 1005.0 + 0.004 * np.arange(7), 1010.0 + 0.004 * np.arange(7),
                         [1020.0]])
    u = np.log(mz)
    own = np.median(np.diff(u[1:8]))
    w = sample_widths(u)
    assert np.all(w < 1.5 * own)
    gaps = gap_mask(mz)
    assert gaps[0] and gaps[-1]


def test_short_segments_among_gaps_keep_their_own_spacing():
    d = 4e-6
    idx = np.sort(np.concatenate([np.arange(0, 20000, 12), np.arange(1, 20000, 12)]))
    u = np.log(1000.0) + d * idx  # two-point segments: gaps are half the spacings
    assert np.allclose(np.median(sample_widths(u)), d, rtol=0.05)
    assert gap_mask(np.exp(u)).sum() == np.count_nonzero(np.diff(idx) > 1)


def test_scans_on_shifted_axes_keep_every_point_in_their_sum():
    """Scans calibrated one by one sit a fraction of a ppm apart: their sum keeps every point
    each measured, and the scans' own spacing is read from the scans, not from the union."""
    from lcmsdeconv.deconv.coelution import FrameStack

    rng = np.random.default_rng(5)
    base = 1000.0 + 0.004 * np.arange(5000)
    frames = []
    for k in range(20):
        ax = base * (1.0 + rng.uniform(-1.0, 1.0) * 1e-6)
        keep = (np.arange(5000) % 50) < 7 + k % 3  # 7- to 9-point segments, scan by scan
        keep[2500] = True  # and one isolated point
        y = np.where(keep, 100.0, 0.0)
        frames.append(Spectrum(ax[keep], y[keep].astype(np.float32), 1.0 + 0.01 * k, 1,
                               is_profile=True))
    s = Run(frames).sum_frames(frames)
    assert all(np.all(np.isin(f.mz, s.mz)) for f in frames)
    near = np.abs(s.mz - base[2500]) < 0.002
    assert near.any() and s.intensity[near].max() > 0.9 * 100.0
    # the union is much denser than a scan; the stack still knows the scans' spacing
    assert np.median(np.diff(s.mz)) < 0.2 * 0.004
    assert abs(FrameStack(frames).spacing(1000.0, 1010.0) - 0.004) < 2e-4


def test_unresolved_envelopes_survive_scans_on_shifted_axes():
    """A protein whose isotopes are not resolved, in scans whose axes differ by +-1 ppm: the
    continuity test reads the summed raw view at one scan's point density and keeps it."""
    from lcmsdeconv.chem.adducts import AdductLibrary, AdductState
    from lcmsdeconv.chem.classes import class_isotope_pattern
    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.deconv.oracle import OraclePredictor
    from lcmsdeconv.deconv.pipeline import DeconvParams, deconvolve_spectrum
    from lcmsdeconv.synth.charge import charge_distribution
    from lcmsdeconv.synth.render import ComponentInstance, Scene, build_sticks, render_profile
    from lcmsdeconv.synth.spec import Compound

    mass = 17017.34
    inst = InstrumentModel("tof", 30000)
    rng = np.random.default_rng(21)
    pat = class_isotope_pattern(mass, "peptide")
    comp = Compound(mass, "peptide", pat.shifted(mass - pat.average_mass), name="x")
    cd = charge_distribution(mass, "peptide", "denatured", 1, rng)
    ci = ComponentInstance(comp, 1e6, cd, {z: {AdductState(): 1.0} for z in cd})
    lib = AdductLibrary.from_mode("rplc", 1, max_per_type=2, max_total=2)
    sticks = build_sticks(Scene([ci], 1, lib))
    base = inst.profile_axis(500, 4000)
    frames = []
    for k in range(8):
        ax = base * (1.0 + rng.uniform(-1.0, 1.0) * 1e-6)
        y = render_profile(sticks, ax, inst) * np.exp(-0.5 * ((k - 3.5) / 2.0) ** 2)
        y = y + np.abs(rng.normal(0, 3.0, ax.size))
        keep = y > 9.0
        frames.append(Spectrum(ax[keep], y[keep].astype(np.float32), 1.0 + 0.01 * k, 1,
                               is_profile=True))
    summed = Run(frames).sum_frames(frames)
    params = DeconvParams(compound_class="peptide", adduct_mode="rplc", adduct_include=("Na",),
                          adduct_exclude=("K", "NH4"), adduct_max_total=2)
    fr = deconvolve_spectrum(summed, OraclePredictor(sticks, LogMzGrid(polarity=1)), params, inst)
    top = max(fr.components, key=lambda c: c.intensity)
    assert abs(top.mass - mass) < 1.0 and len(top.charges) > 10
