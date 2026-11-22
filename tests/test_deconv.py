import numpy as np

from lcmsdeconv.chem.adducts import AdductLibrary, AdductState
from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Spectrum
from lcmsdeconv.deconv.oracle import OraclePredictor
from lcmsdeconv.deconv.pipeline import DeconvParams, deconvolve_spectrum
from lcmsdeconv.deconv.templates import build_template, template_mz
from lcmsdeconv.nn.grid import LogMzGrid
from lcmsdeconv.synth.charge import charge_distribution
from lcmsdeconv.synth.render import ComponentInstance, Scene, build_sticks, render_profile
from lcmsdeconv.synth.spec import Compound


def _compound(mass: float, cls: str = "peptide", name: str = "x") -> Compound:
    pat = class_isotope_pattern(mass, cls)
    return Compound(mass, cls, pat.shifted(mass - pat.average_mass), name=name)


def _spectrum(components, instrument, polarity=1, noise=3.0, seed=0, mz_range=(500, 4000)):
    rng = np.random.default_rng(seed)
    lib = AdductLibrary.from_mode("rplc", polarity, max_per_type=2, max_total=2)
    sticks = build_sticks(Scene(components, polarity, lib))
    ax = instrument.profile_axis(*mz_range)
    prof = render_profile(sticks, ax, instrument) + np.abs(rng.normal(0, noise, ax.size))
    return Spectrum(ax, prof, rt=1.0, polarity=polarity, is_profile=True), sticks


def test_template_is_area_normalized_and_positioned():
    g = LogMzGrid(polarity=1)
    inst = InstrumentModel("tof", 30000)
    t = build_template(17000.0, 12, g, inst, "peptide")
    assert abs(t.values.sum() - 1.0) < 1e-9
    apex_mz = g.bin_to_mz(t.apex_bin)
    # apex sits at the most abundant isotopologue, below the average-mass position
    assert apex_mz < template_mz(17000.0, 12, 1)
    assert abs(apex_mz - template_mz(17000.0, 12, 1)) < 0.2


def test_deconvolution_recovers_mass_and_adducts():
    mass = 17017.34
    inst = InstrumentModel("tof", 30000)
    rng = np.random.default_rng(21)
    comp = _compound(mass)
    cd = charge_distribution(mass, "peptide", "denatured", 1, rng)
    na = AdductState((("Na", 1),))
    ci = ComponentInstance(comp, 1e6, cd, {z: {AdductState(): 0.9, na: 0.1} for z in cd})
    spec, sticks = _spectrum([ci], inst)
    g = LogMzGrid(polarity=1)
    fr = deconvolve_spectrum(spec, OraclePredictor(sticks, g),
                             DeconvParams(compound_class="peptide", adduct_mode="rplc",
                                          adduct_include=("Na",), adduct_exclude=("K", "NH4"),
                                          adduct_max_total=2), inst)
    assert fr.components
    top = max(fr.components, key=lambda c: c.intensity)
    assert abs(top.mass - mass) / mass * 1e6 < 50  # within 50 ppm
    assert top.mass_spread_ppm < 20
    frac = top.adduct_fractions()
    assert 0.03 < frac.get("+Na", 0.0) < 0.25  # true 10 %
    assert frac.get("base", 0.0) > 0.7
    assert fr.residual_fraction < 0.25


def test_negative_mode_oligo():
    mass = 6500.0
    inst = InstrumentModel("tof", 25000)
    rng = np.random.default_rng(5)
    comp = _compound(mass, "dna", "oligo")
    cd = charge_distribution(mass, "dna", "denatured", -1, rng)
    ci = ComponentInstance(comp, 5e5, cd, {z: {AdductState(): 1.0} for z in cd})
    spec, sticks = _spectrum([ci], inst, polarity=-1, mz_range=(400, 3000), seed=2)
    g = LogMzGrid(polarity=-1)
    fr = deconvolve_spectrum(spec, OraclePredictor(sticks, g),
                             DeconvParams(compound_class="dna", adduct_mode="iprp",
                                          adduct_max_total=1), inst)
    assert fr.components
    top = max(fr.components, key=lambda c: c.intensity)
    assert abs(top.mass - mass) / mass * 1e6 < 100


def test_no_components_in_empty_spectrum():
    inst = InstrumentModel("tof", 30000)
    ax = inst.profile_axis(500, 1000)
    spec = Spectrum(ax, np.zeros(ax.size), rt=0.0, polarity=1, is_profile=True)
    g = LogMzGrid(polarity=1)
    from lcmsdeconv.synth.spec import Sticks

    fr = deconvolve_spectrum(spec, OraclePredictor(Sticks.empty(), g), DeconvParams(), inst)
    assert fr.components == []


def _comp(mass, intensity, cols):
    from lcmsdeconv.core.model import Component

    total = sum(cols.values())
    return Component(mass, intensity, charges={z: intensity * v / total for z, v in cols.items()},
                     adducts={"base": intensity}, compound_class="ps_rna",
                     columns={(z, ""): intensity * v / total for z, v in cols.items()})


MP = 1.00727646688
T12 = np.arange(12)
PEAK = np.exp(-0.5 * ((T12 - 5.0) / 1.5) ** 2)  # the monomer's elution over a region's scans
LATE = np.exp(-0.5 * ((T12 - 9.5) / 1.0) ** 2)  # a species eluting apart from it


def _scan_stack(ions, n=12):
    """The scans of a region: narrow peaks ``ions`` = [(mass, z, height, profile over the
    scans)] at their m/z, on one axis."""
    from lcmsdeconv.deconv.coelution import FrameStack

    x = np.arange(780.0, 1060.0, 0.01)
    out = []
    for k in range(n):
        y = np.zeros_like(x)
        for mass, z, height, profile in ions:
            y[np.abs(x - (mass - z * MP) / z) < 0.02] += height * profile[k]
        out.append(Spectrum(x, y, rt=4.0 + 0.04 * k, polarity=-1, is_profile=True))
    return FrameStack(out)


def test_multimers_go_by_their_fitted_amount():
    from lcmsdeconv.deconv.refine import drop_multimers

    # the IP-RPLC NC run's 1580 Da shortmer and its [2M-3H]3- dimer, as fitted, and a trimer;
    # the multimers' own states (z = 3 of the dimer, z = 5 of the trimer) co-elute with z = 2
    mono = _comp(1580.18, 8.35e6, {1: 0.014, 2: 0.977, 3: 0.009})
    dimer = _comp(3160.68, 6.3e4, {2: 0.1, 3: 0.61, 4: 0.1, 5: 0.1, 6: 0.04, 7: 0.04})
    trimer = _comp(4740.9, 2e4, {4: 0.5, 5: 0.5})
    other = _comp(3171.8, 6.3e4, {3: 0.6, 4: 0.4})
    frames = _scan_stack([(1580.18, 2, 1e6, PEAK), (3160.68, 3, 1e4, PEAK),
                          (4740.9, 5, 5e3, PEAK)])
    kept = drop_multimers([dimer, other, mono, trimer], polarity=-1, frames=frames)
    assert {round(c.mass, 2) for c in kept} == {1580.18, 3171.8}
    # a multimer with most of its monomer's amount is a species of its own
    assert len(drop_multimers([mono, _comp(3160.68, 0.7 * 8.35e6, {3: 1.0})], polarity=-1,
                              frames=frames)) == 2


def test_a_multimer_must_coelute_with_its_monomer():
    """The monomer's strongest state and the multimer's strongest state of its own (odd for a
    dimer) are compared over the scans: a species at 2M that elutes on its own is kept."""
    from lcmsdeconv.deconv.refine import drop_multimers

    mono = _comp(1580.18, 8.35e6, {2: 1.0})
    dimer = _comp(3160.68, 6.3e4, {3: 0.7, 4: 0.3})

    def kept(profile, multimer=dimer):
        frames = _scan_stack([(1580.18, 2, 1e6, PEAK), (3160.68, 3, 1e4, profile)])
        return {round(c.mass, 2) for c in drop_multimers([mono, multimer], polarity=-1,
                                                          frames=frames)}

    assert kept(PEAK ** 2) == {1580.18}
    assert kept(LATE) == {1580.18, 3160.68}
    # the monomer's ions read at twice their charge, with little on odd states (the 1234 Da
    # shortmer's 2469 Da reading of the IP-HILIC NC run): a harmonic, whatever that little does
    harmonic = _comp(3160.68, 6.3e4, {2: 0.45, 3: 0.05, 4: 0.5})
    assert kept(LATE, harmonic) == {1580.18}
    # a genuine species at 2M has about half its fit on odd states: one whose charge states lean
    # to even ones is still judged by its elution
    even_leaning = _comp(3160.68, 6.3e4, {3: 0.4, 4: 0.6})
    assert kept(LATE, even_leaning) == {1580.18, 3160.68}
    assert kept(PEAK ** 2, even_leaning) == {1580.18}


def test_a_multimer_goes_only_where_its_coelution_is_judged():
    """Without the scans, over fewer than four of them (regions may be two scans long), or where
    the monomer's trace carries signal in fewer than four, the co-elution test cannot judge, and
    a species at 2M is kept: it may elute on its own. The monomer's trace decides that, not the
    multimer's: a weak dimer above the vendor threshold only in the monomer's three apex scans
    is judged, and goes. A harmonic (the monomer's ions read at twice their charge) goes without
    the test, also with the test turned off (``coelution_r`` 0), which keeps every other
    multimer."""
    from lcmsdeconv.deconv.refine import drop_multimers

    mono = _comp(1580.18, 8.35e6, {2: 1.0})
    dimer = _comp(3160.68, 6.3e4, {3: 0.7, 4: 0.3})
    harmonic = _comp(3160.68, 6.3e4, {2: 0.45, 3: 0.05, 4: 0.5})
    band = [0.5, 1.0, 0.5]
    spike = np.zeros(12)
    spike[4:7] = band  # the monomer in three scans of twelve: no profile
    two = _scan_stack([(1580.18, 2, 1e6, [1.0, 0.6]), (3160.68, 3, 1e4, [0.2, 1.0])], n=2)
    three = _scan_stack([(1580.18, 2, 1e6, band), (3160.68, 3, 1e4, band)], n=3)
    few = _scan_stack([(1580.18, 2, 1e6, spike), (3160.68, 3, 1e4, spike)])
    for frames in (None, two, three, few):
        for multimer, out in ((dimer, {1580.18, 3160.68}), (harmonic, {1580.18})):
            kept = drop_multimers([mono, multimer], polarity=-1, frames=frames)
            assert {round(c.mass, 2) for c in kept} == out
    # judged over four scans, the co-eluting dimer goes
    four = _scan_stack([(1580.18, 2, 1e6, PEAK[3:7]), (3160.68, 3, 1e4, PEAK[3:7])], n=4)
    assert {round(c.mass, 2) for c in drop_multimers([mono, dimer], polarity=-1,
                                                     frames=four)} == {1580.18}
    # the monomer over twelve scans, the dimer above the threshold in its three apex scans only
    apex = np.where(PEAK >= np.sort(PEAK)[-3], PEAK, 0.0)
    weak = _scan_stack([(1580.18, 2, 1e6, PEAK), (3160.68, 3, 1e4, apex)])
    assert {round(c.mass, 2) for c in drop_multimers([mono, dimer], polarity=-1,
                                                     frames=weak)} == {1580.18}
    for multimer, out in ((dimer, {1580.18, 3160.68}), (harmonic, {1580.18})):
        kept = drop_multimers([mono, multimer], polarity=-1, frames=weak, coelution_r=0.0)
        assert {round(c.mass, 2) for c in kept} == out


class _LadderEcho:
    """The comb predictor, with the second choice the real run's charge ladder made on the
    1580 Da shortmer's doubly charged isotopologues: z = 4, at p = 0.4-0.49, as 2M's ladder of
    charges 2, 3, 4 and 6 holds the shortmer's states 1, 2 and 3 and the dimer's own z = 3."""

    def __init__(self, inner, mz_lo, mz_hi):
        self.inner, self.lo, self.hi = inner, mz_lo, mz_hi

    def predict_grid(self, grid, intensity, noise_sigma):
        p = self.inner.predict_grid(grid, intensity, noise_sigma)
        mz = grid.bin_to_mz(np.arange(intensity.size))
        sel = (p.top1_z == 2) & (mz > self.lo) & (mz < self.hi)
        p.top2_z[sel] = 4
        p.top2_p[sel] = 0.49
        return p


def test_a_non_covalent_dimer_is_not_reported():
    """A shortmer's dimer made in the source ([2M-3H]3-, ...) is fitted, its odd charge states
    being its own. Its candidate scored 0.74 of the shortmer's on the IP-RPLC NC run, the
    shortmer's isotopologues read at twice their charge voting for teeth two daltons apart,
    past the candidate-stage harmonic test; the fitted amounts decide instead, and the dimer's
    own states co-elute with the shortmer over the region's scans."""
    from lcmsdeconv.core.model import Run
    from lcmsdeconv.deconv.classical import make_predictor

    orbi = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)
    mass = 1580.37
    pat = class_isotope_pattern(mass, "ps_rna")
    m1, a1 = pat.masses - pat.average_mass + mass, pat.abundances
    m2, a2 = np.add.outer(m1, m1).ravel(), np.multiply.outer(a1, a1).ravel()
    x = np.arange(400.0, 2483.0, 0.0005)
    y = np.abs(np.random.default_rng(0).normal(0.0, 300.0, x.size))
    # heights of the real run's summed spectrum
    for masses, ab, heights in ((m1, a1, {1: 6.3e4, 2: 5.3e6, 3: 5.6e4}),
                                (m2, a2, {3: 1.2e4, 4: 3e3, 5: 1.0e4, 6: 3e3})):
        for z, h in heights.items():
            mz = (masses - z * 1.00727646688) / z
            sig = orbi.fwhm(float(mz.mean())) / 2.3548
            for c, v in zip(mz, ab / ab.max()):
                if v > 1e-4:
                    lo, hi = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
                    y[lo:hi] += h * v * np.exp(-0.5 * ((x[lo:hi] - c) / sig) ** 2)
    keep = y > 1500.0
    # the region's scans, which sum to that spectrum: the dimer elutes with the shortmer, made of
    # its molecules in the source
    elution = np.exp(-0.5 * ((np.arange(6) - 2.5) / 1.2) ** 2)
    scans = [Spectrum(x[keep], y[keep] * (h / elution.sum()), rt=4.04 + 0.04 * k, polarity=-1,
                      is_profile=True) for k, h in enumerate(elution)]
    spec = Run(scans).sum_frames(scans)
    pred = _LadderEcho(make_predictor(None, z_max=40, instrument=orbi), 785.0, 793.0)
    params = DeconvParams(compound_class="ps_rna", mass_range=(1000.0, 60000.0),
                          charge_range=(2, 40), adduct_mode="iprp", isotope_charges=True,
                          max_components=200)
    trace = []
    fr = deconvolve_spectrum(spec, pred, params, orbi, trace=trace)
    masses = [c.mass for c in fr.components]
    assert any(abs(m - mass) < 0.3 for m in masses)
    assert not any(abs(m - 2 * mass) < 2.0 for m in masses)
    # the dimer passed the candidate-stage harmonic test and was fitted: the fit decided
    harm = next(r for r in trace if r.get("stage") == "harmonics" and r["pass"] == 0)
    assert any(abs(m - 2 * mass) < 2.0 for m in harm["masses"])
    assert any(r.get("outcome") == "multimer" and abs(r["candidate"] - 2 * mass) < 2.0
               for r in trace)
