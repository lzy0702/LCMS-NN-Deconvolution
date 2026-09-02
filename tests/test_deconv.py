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
