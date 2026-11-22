"""The co-elution of a species' charge states over the scans of a region."""

import numpy as np

from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Spectrum
from lcmsdeconv.deconv.coelution import FrameStack, coeluting, correlation
from lcmsdeconv.deconv.decode import Candidate, refine_candidate_masses
from lcmsdeconv.deconv.refine import observed_charges

MP = 1.00727646688
MASS = 5162.25
N_SCANS = 15
INST = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)


def _envelope_peaks(mass, z, height):
    pat = class_isotope_pattern(mass, "dna")
    avg = float((pat.abundances * pat.masses).sum() / pat.abundances.sum())
    mz = (pat.masses - avg + mass - z * MP) / z
    return mz, height * pat.abundances / pat.abundances.max()


def _scans(envelopes, lo=1000.0, hi=2000.0, step=0.002):
    """Scans on one axis; ``envelopes`` = [(mass, z, height, time profile over the scans)]."""
    x = np.arange(lo, hi, step)
    scans = []
    for k in range(N_SCANS):
        y = np.zeros_like(x)
        for mass, z, height, profile in envelopes:
            centres, heights = _envelope_peaks(mass, z, height * profile[k])
            fwhm = centres.mean() / float(INST.resolving_power(centres.mean()))
            sig = fwhm / 2.3548
            for c, h in zip(centres, heights):
                a, b = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
                y[a:b] += h * np.exp(-0.5 * ((x[a:b] - c) / sig) ** 2)
        scans.append(Spectrum(x, y, 10.0 + 0.02 * k, -1, is_profile=True))
    return scans


t = np.arange(N_SCANS)
ELUTES = np.exp(-0.5 * ((t - 7.0) / 2.0) ** 2)  # this species' elution profile
OTHER = np.clip((t - 2.0) / 12.0, 0.0, None)  # another ion's, rising through the region


def _summed(scans):
    return scans[0].mz, np.sum([s.intensity for s in scans], axis=0)


def _gate(scans, frames, charges=(3, 4)):
    x, y = _summed(scans)
    return refine_candidate_masses([Candidate(MASS, 1.0, set(charges))], x, y, -1,
                                   noise_sigma=1.0, compound_class="dna", resolution=50000.0,
                                   instrument=INST, profile=True, frames=frames)


def test_traces_sum_each_scans_window():
    scans = _scans([(MASS, 4, 1000.0, ELUTES)])
    stack = FrameStack(scans)
    centre = (MASS - 4 * MP) / 4
    tr = stack.traces([centre], [1.0])[0]
    expected = [float(s.intensity[np.searchsorted(s.mz, centre - 1.0):
                                 np.searchsorted(s.mz, centre + 1.0)].sum()) for s in scans]
    assert np.allclose(tr, expected)
    assert correlation(tr, ELUTES) > 0.99


def test_coeluting_keeps_shared_profiles_and_abstains_without_one():
    noise = np.random.default_rng(1).normal(0, 0.02, N_SCANS)
    traces = np.vstack([ELUTES, 0.3 * ELUTES + noise, OTHER])
    assert list(coeluting(traces, 0)) == [True, True, False]
    # a reference that carries signal in too few scans shows no profile: keep everything
    spike = np.zeros(N_SCANS)
    spike[7] = 1.0
    assert coeluting(np.vstack([spike, OTHER]), 0).all()


def test_gate_keeps_charge_states_that_elute_together():
    scans = _scans([(MASS, 3, 800.0, ELUTES), (MASS, 4, 1000.0, ELUTES)])
    out = _gate(scans, FrameStack(scans))
    assert len(out) == 1 and abs(out[0].mass - MASS) < 0.3 and out[0].charges >= {3, 4}


def test_gate_rejects_a_ladder_of_ions_that_elute_apart():
    # the z = 4 envelope is there with the right mass, but it rises while z = 3 elutes and falls
    scans = _scans([(MASS, 3, 800.0, ELUTES), (MASS, 4, 1000.0, OTHER)])
    assert len(_gate(scans, None)) == 1  # the summed spectrum alone cannot tell
    assert _gate(scans, FrameStack(scans)) == []


def test_gate_judges_a_state_on_its_own_isotopologues():
    """An ion of another elution between two isotopologues in a charge state's window: the
    window's trace no longer follows the species, its isotopologues still do (the 1.9 kDa
    shortmer of the IP-HILIC NC run lost its z = 3 this way)."""
    early = np.exp(-0.5 * ((t - 1.0) / 2.0) ** 2)
    mz4, _h = _envelope_peaks(MASS, 4, 1.0)
    k = int(np.argmin(np.abs(mz4 - (MASS - 4 * MP) / 4)))
    foreign = mz4[k] + 0.5 * 1.0033548 / 4  # half a spacing from the isotopologues
    sig = foreign / float(INST.resolving_power(foreign)) / 2.3548
    scans = _scans([(MASS, 3, 800.0, ELUTES), (MASS, 4, 1000.0, ELUTES)])
    for i, s in enumerate(scans):
        s.intensity = s.intensity + 2000.0 * early[i] * np.exp(-0.5 * ((s.mz - foreign) / sig) ** 2)
    x, y = _summed(scans)

    def gate(own_comb):
        return refine_candidate_masses([Candidate(MASS, 1.0, {3, 4})], x, y, -1, noise_sigma=1.0,
                                       compound_class="dna", resolution=50000.0, instrument=INST,
                                       profile=True, frames=FrameStack(scans), own_comb=own_comb)

    # z = 4, whose window holds the foreign ion, is the reference: z = 3 fails against it
    assert gate(own_comb=False) == []
    out = gate(own_comb=True)
    assert len(out) == 1 and out[0].gate_charges == {3, 4} and abs(out[0].mass - MASS) < 0.5


def _beside_a_foreign_ion(profile_z3):
    """z = 3 (eluting as ``profile_z3``) and z = 4 of the species, and an ion of another
    elution 0.8 peak widths from z = 3's strongest isotopologue: the gate with or without the
    test on the states' own isotopologues."""
    early = np.exp(-0.5 * ((t - 1.0) / 2.0) ** 2)
    mz3, h3 = _envelope_peaks(MASS, 3, 1.0)
    k = int(np.argmax(h3))
    fwhm = mz3[k] / float(INST.resolving_power(mz3[k]))
    foreign, sig = mz3[k] + 0.8 * fwhm, fwhm / 2.3548
    scans = _scans([(MASS, 3, 800.0, profile_z3), (MASS, 4, 2000.0, ELUTES)])
    for i, s in enumerate(scans):
        s.intensity = s.intensity + 1000.0 * early[i] * np.exp(-0.5 * ((s.mz - foreign) / sig) ** 2)
    x, y = _summed(scans)

    def gate(own_comb):
        return refine_candidate_masses([Candidate(MASS, 1.0, {3, 4})], x, y, -1, noise_sigma=1.0,
                                       compound_class="dna", resolution=50000.0, instrument=INST,
                                       profile=True, frames=FrameStack(scans), own_comb=own_comb)

    return gate


def test_gate_rescues_a_state_whose_isotopologue_has_a_foreign_neighbour():
    """The 1.9 kDa shortmer's geometry: an ion of another elution 0.8 peak widths from the
    strongest isotopologue of the weaker state, which is not the reference. Its window fails
    against the reference's; its isotopologues, read on the reference's comb, co-elute."""
    gate = _beside_a_foreign_ion(ELUTES)
    assert gate(own_comb=False) == []  # z = 3's window fails against z = 4's
    out = gate(own_comb=True)
    assert len(out) == 1 and out[0].gate_charges == {3, 4} and abs(out[0].mass - MASS) < 0.5


def test_a_state_whose_isotopologues_coelute_in_part_is_not_rescued(monkeypatch):
    """The rescue asks more of a state's own isotopologues than the window test asks of its
    window: ``COMB_RESCUE_R`` (0.9), not 0.5. Half of this z = 3 envelope elutes with z = 4 and
    half four scans later, and its isotopologues correlate with z = 4 at 0.65. Of the states
    that fail on their windows on the four real runs a third reach 0.5 on their isotopologues,
    most of them adduct forms of the main species beside other forms' peaks."""
    from lcmsdeconv.deconv import coelution

    assert coelution.COMB_RESCUE_R == 0.9  # this case separates it from 0.5, not from 0.7
    later = np.exp(-0.5 * ((t - 11.0) / 2.0) ** 2)
    gate = _beside_a_foreign_ion(0.5 * ELUTES + 0.5 * later)
    assert gate(own_comb=True) == []
    # at the window test's bar it would be rescued
    monkeypatch.setattr(coelution, "COMB_RESCUE_R", 0.5)
    out = gate(own_comb=True)
    assert len(out) == 1 and out[0].gate_charges == {3, 4}


def test_rescue_needs_the_comb_in_the_reference_window():
    """A non-covalent dimer's z = 3 beside its monomer's z = 1, which sits on every other
    position of the dimer's z = 2 comb: a mean shift for that z = 2 lands between the monomer's
    isotopologues, and its window holds only a stray bump. The comb the rescue reads past that
    window is the monomer's; it must not pair the monomer's z = 1 with the dimer's z = 3 (the
    3160 Da dimer of the IP-HILIC NC run's 1580 Da shortmer)."""
    md = 3200.0
    pat = class_isotope_pattern(md, "dna")
    avg = float((pat.abundances * pat.masses).sum() / pat.abundances.sum())
    mono = float(pat.masses.min()) - avg + md
    rel = pat.abundances / pat.abundances.max()
    bump = np.zeros(N_SCANS)
    bump[[2, 5, 9, 12, 13]] = [0.2, 1.0, 0.1, 0.15, 0.1]  # no elution profile of its own
    x = np.arange(900.0, 1800.0, 0.002)

    def add(y, centre, height):
        sig = centre / float(INST.resolving_power(centre)) / 2.3548
        a, b = np.searchsorted(x, [centre - 5 * sig, centre + 5 * sig])
        y[a:b] += height * np.exp(-0.5 * ((x[a:b] - centre) / sig) ** 2)

    scans = []
    for i in range(N_SCANS):
        y = np.zeros_like(x)
        for j, h in enumerate((1.0, 1.0, 0.3)):  # the monomer at z = 1
            add(y, mono / 2 - MP + j * 1.0033548, 3000.0 * h * ELUTES[i])
        add(y, mono / 2 - MP + 0.5016774, 40.0 * bump[i])  # in the z = 2 window
        for m, a in zip((pat.masses - avg + md - 3 * MP) / 3, rel):  # the dimer at z = 3
            if a > 1e-3:
                add(y, m, 1500.0 * a * ELUTES[i])
        scans.append(Spectrum(x, y, 10.0 + 0.02 * i, -1, is_profile=True))
    xs, ys = _summed(scans)

    def gate(own_comb):
        return refine_candidate_masses([Candidate(md, 1.0, {2, 3})], xs, ys, -1, noise_sigma=1.0,
                                       compound_class="dna", resolution=50000.0, instrument=INST,
                                       profile=True, frames=FrameStack(scans), own_comb=own_comb)

    assert gate(own_comb=False) == []  # the z = 2 window's bump does not follow z = 3
    assert gate(own_comb=True) == []


def test_adduct_evidence_reads_the_trial_windows_only(monkeypatch):
    """``evidenced_adducts`` gates each adduct trial as detection gates a candidate - with the
    scans, the co-elution bar and the method's z = 1 rule - but on its charge windows alone
    (``own_comb=False``): the adduct evidence stays as it was before charge states were also
    judged on their isotopologues."""
    from lcmsdeconv.chem.adducts import AdductLibrary
    from lcmsdeconv.deconv import pipeline

    calls = []

    def spy(cands, *args, **kw):
        calls.append(kw)
        return []

    monkeypatch.setattr(pipeline, "refine_candidate_masses", spy)
    scans = _scans([(MASS, 3, 800.0, ELUTES), (MASS, 4, 1000.0, ELUTES)])
    x, y = _summed(scans)
    frames = FrameStack(scans)
    lib = AdductLibrary.from_mode("iprp", -1)
    params = pipeline.DeconvParams(singly_charged_species=False, coelution_r=0.6)
    names = pipeline.evidenced_adducts([Candidate(MASS, 1.0, {3, 4})], lib, x, y, -1, 1.0, params,
                                       "dna", INST, True, frames=frames)
    assert names == set() and len(calls) == len(lib.adducts)
    for kw in calls:
        assert kw["own_comb"] is False and kw["singly_charged_ok"] is False
        assert kw["frames"] is frames and kw["coelution_r"] == 0.6


def test_charge_states_grow_only_while_they_coelute():
    # the species shows z = 3, 4, 5; at z = 6's position sits an ion with its own profile
    scans = _scans([(MASS, 3, 800.0, ELUTES), (MASS, 4, 1000.0, ELUTES),
                    (MASS, 5, 600.0, ELUTES), (MASS, 6, 700.0, OTHER)], lo=800.0)
    x, y = _summed(scans)
    kw = dict(polarity=-1, compound_class="dna", resolution=50000.0, mz_range=(800.0, 2000.0))
    assert observed_charges(MASS, {3, 4}, x, y, 3.0, **kw) == {3, 4, 5, 6}
    assert observed_charges(MASS, {3, 4}, x, y, 3.0, frames=FrameStack(scans), **kw) == {3, 4, 5}


def test_fit_weights_follow_the_peak_not_the_bin():
    """A peak's flank bins carry the peak's variance, so a template a bin off is not pulled down."""
    from scipy.special import erf

    from lcmsdeconv.deconv.nnls_solve import fit_weights, weighted_nnls

    n = 400
    x = np.arange(n, dtype=float)
    centres = np.arange(20.3, 380.0, 10.0)  # the template's peaks sit 0.8 bin to the right
    sig = 0.4
    obs = np.zeros(n)
    tpl = np.zeros(n)
    edges = np.arange(n + 1) - 0.5
    for c in centres:
        # observed: Gaussian peaks averaged over each bin; template: sampled at bin centres
        cdf = 0.5 * (1 + erf((edges - c) / (sig * np.sqrt(2))))
        obs += 1e7 * np.diff(cdf)
        tpl += np.exp(-0.5 * ((x - (c + 0.8)) / sig) ** 2)
    tpl /= tpl.sum()
    A = tpl[:, None]
    # the template's shape is off; plain least squares sets the scale that shape allows
    plain = weighted_nnls(A, obs)[0]
    own = weighted_nnls(A, obs, 1.0 / np.sqrt(obs + 10.0 ** 2))[0]
    local = weighted_nnls(A, obs, fit_weights(obs, 10.0, grid_step=2e-5, resolution=60000.0))[0]
    assert own < 0.5 * plain  # weighting each bin by its own value pulls the scale down
    assert 0.9 * plain < local < 1.1 * plain


def test_a_weaker_species_does_not_keep_a_stronger_ones_adduct():
    from lcmsdeconv.chem.adducts import AdductLibrary
    from lcmsdeconv.core.model import Component
    from lcmsdeconv.deconv.refine import drop_coincident_adducts

    lib = AdductLibrary.from_mode("iprp", -1)
    nc = Component(5162.28, 100.0, charges={3: 50.0, 4: 50.0}, adducts={"base": 100.0},
                   columns={(3, ""): 50.0, (4, ""): 50.0})
    # PO-NC + K (5184.19 Da) sits on NC + Na (5184.26 Da); its +Na (5168.21 Da) is its own
    po = Component(5146.23, 30.0, charges={3: 15.0, 4: 15.0},
                   adducts={"base": 10.0, "+K": 15.0, "+Na": 5.0},
                   columns={(3, ""): 5.0, (4, ""): 5.0, (3, "+K"): 8.0, (4, "+K"): 7.0,
                            (4, "+Na"): 5.0})
    # a component made only of NC's adduct ladder disappears
    ghost = Component(5146.23 + 0.02, 5.0, charges={4: 5.0}, adducts={"+K": 5.0},
                      columns={(4, "+K"): 5.0})
    out = drop_coincident_adducts([ghost, po, nc], lib)
    assert [c.mass for c in out] == [5162.28, 5146.23]
    po_out = out[1]
    assert set(po_out.adducts) == {"base", "+Na"} and po_out.intensity == 15.0
    assert (3, "+K") not in po_out.columns and po_out.charges == {3: 5.0, 4: 10.0}


def test_templates_align_with_resolved_isotopologues():
    """The class's isotope pattern a bin off the data is slid onto it; unresolved data are left."""
    from lcmsdeconv.deconv.refine import align_template
    from lcmsdeconv.deconv.templates import build_template
    from lcmsdeconv.nn.grid import LogMzGrid

    grid = LogMzGrid(1000.0, 2000.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    true_shift = 1.0 * grid.step  # where this molecule's isotopologues actually are
    for z in (3, 4):
        t = build_template(MASS * np.exp(true_shift), z, grid, INST, "dna")
        spectrum[t.bins] += 1e6 * t.values
    cand = Candidate(MASS, 1.0, {3, 4})
    assert abs(align_template(cand, spectrum, grid, INST, "dna") - true_shift) <= 0.25 * grid.step
    low = InstrumentModel("tof", 5000.0)  # isotopes of z = 3-4 not resolved: no shift
    assert align_template(cand, spectrum, grid, low, "dna") == 0.0
