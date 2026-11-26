"""Harmonic readings (refine.drop_harmonic_readings) and the adduct dedupe's hold, on synthetic
time-of-flight spectra of phosphorothioate oligonucleotides: resolving power 14 000 at m/z 800,
where the isotopologues of a synthetic 11.5 kDa species at charges 8-16 are one to two peak
widths apart and the comb also reads its ions at higher charges."""

import numpy as np

from lcmsdeconv.chem.adducts import AdductLibrary
from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.formula import Formula
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Component, Run, Spectrum
from lcmsdeconv.deconv import pipeline as P
from lcmsdeconv.deconv.classical import make_predictor
from lcmsdeconv.deconv.decode import Candidate, dedupe_adduct_candidates
from lcmsdeconv.deconv.refine import drop_harmonic_readings

MP = 1.00727646688
TOF = InstrumentModel("tof", 14000.0, mz_ref=800.0, mz_exponent=0.8)
#: tripropylamine N-oxide, the oxidised ion-pairing amine: an adduct ladder on every species
TPAO = "C9H21NO"
LIB = AdductLibrary.from_mode("iprp", -1, include=(TPAO,), exclude=("TEA", "DIPEA", "HFIP", "K"),
                              max_per_type=2, max_total=3)
SHIFT = {s.label: s.mass for s in LIB.states()}
#: a time-of-flight axis: equal steps in sqrt(m/z), 0.02 m/z at m/z 730
AXIS = np.arange(np.sqrt(500.0), np.sqrt(2000.0), 0.000371) ** 2
#: a synthetic 11.5 kDa species
M = 11500.0
#: its charge states
ZH = {8: 0.25, 9: 0.45, 10: 0.7, 11: 0.9, 12: 1.0, 13: 0.95, 14: 0.75, 15: 0.5, 16: 0.25}
#: its C9H21NO ladder: one and two C9H21NO, as fractions of its base form
LADDER = (0.12, 0.04)


def _render(mass, heights, add=0.0, x=AXIS):
    """Profile of a ps_rna species of average mass ``mass`` (its ``add`` form) at {z: height}."""
    pat = class_isotope_pattern(mass, "ps_rna")
    m = pat.masses - pat.average_mass + mass + add
    a = pat.abundances / pat.abundances.max()
    y = np.zeros_like(x)
    for z, h in heights.items():
        for c, v in zip((m - z * MP) / z, a):
            if v < 1e-3:
                continue
            s = float(TOF.fwhm(c)) / 2.3548
            lo, hi = np.searchsorted(x, [c - 5 * s, c + 5 * s])
            y[lo:hi] += h * v * np.exp(-0.5 * ((x[lo:hi] - c) / s) ** 2)
    return y


def _comp(mass, cols, scale=1.0):
    """A component as the fit leaves it: its (charge, adduct label) coefficients."""
    cols = {(z, lab): v * scale for (z, lab), v in cols.items()}
    charges: dict = {}
    adducts: dict = {}
    for (z, lab), v in cols.items():
        charges[z] = charges.get(z, 0.0) + v
        adducts[lab or "base"] = adducts.get(lab or "base", 0.0) + v
    return Component(mass, sum(cols.values()), charges=charges, adducts=adducts,
                     compound_class="ps_rna", columns=cols)


def _species_spectrum():
    """The species with its C9H21NO form (12 % of it), and a fitted component of it."""
    tp = "+" + TPAO
    y = _render(M, {z: 1e5 * h for z, h in ZH.items()})
    y += _render(M, {z: 1e5 * LADDER[0] * h for z, h in ZH.items()}, SHIFT[tp])
    mono = _comp(M, {**{(z, ""): h for z, h in ZH.items()},
                     **{(z, tp): LADDER[0] * h for z, h in ZH.items()}}, 1e6)
    return y, mono


def test_readings_of_a_stronger_species_go_and_species_of_their_own_stay():
    y, mono = _species_spectrum()
    tp = "+" + TPAO
    # three times the species less two sodium exchanges: its +2Na form lies on the species' ions
    # at three times their charge (no exact multiple of anything), with weak flank columns
    na2 = SHIFT["+2Na"]
    harm = _comp(3 * M - na2, {**{(3 * z, "+2Na"): 0.2 for z in (10, 11, 12, 13)},
                               (36, ""): 0.04, (39, ""): 0.03, (33, tp): 0.02}, 2e5)
    # 39/14 of the species: its z = 14 read as z = 39, with a weak column of its own
    rational = _comp(39 / 14 * (M - 14 * MP) + 39 * MP, {(39, ""): 0.5, (31, ""): 0.1}, 1e5)
    # a genuine 25.7 kDa species: unresolved envelopes of its own at z 15-30
    gz = {z: float(np.exp(-0.5 * ((z - 22) / 4.0) ** 2)) for z in range(15, 31)}
    y = y + _render(25700.0, {z: 3e3 * h for z, h in gz.items()})
    real = _comp(25700.0, {(z, ""): h for z, h in gz.items()}, 6e4)
    # a non-covalent dimer: its odd states are its own, its even ones the species' ions
    dz = {z: float(np.exp(-0.5 * ((z - 24) / 3.0) ** 2)) for z in range(18, 31)}
    y = y + _render(2 * M, {z: 2e3 * h for z, h in dz.items() if z % 2})
    dimer = _comp(2 * M, {(z, ""): h for z, h in dz.items()}, 5e4)
    # lighter: half the species, its even states read at half their charge, and a genuine
    # 5.5 kDa shortmer with resolved envelopes of its own
    half = _comp(M / 2, {(z // 2, ""): h for z, h in ZH.items() if z % 2 == 0}, 1e5)
    sz = {z: float(np.exp(-0.5 * ((z - 6) / 1.5) ** 2)) for z in range(4, 10)}
    y = y + _render(5500.0, {z: 4e3 * h for z, h in sz.items()})
    short = _comp(5500.0, {(z, ""): h for z, h in sz.items()}, 1e5)
    trace: list = []
    kept = drop_harmonic_readings([mono, harm, rational, real, dimer, half, short], AXIS, y, -1,
                                  TOF, LIB, trace=trace)
    assert {round(c.mass, 1) for c in kept} == {M, 25700.0, 2 * M, 5500.0}
    dropped = sorted(r["candidate"] for r in trace if r["outcome"] == "harmonic_reading")
    assert np.allclose(dropped, sorted([harm.mass, rational.mass, half.mass]))
    assert all(r["mass"] == M for r in trace)
    # where nothing pairs them (a stronger component at under 1.25 times their mass, or no
    # stronger one), nothing is judged
    assert drop_harmonic_readings([harm, rational], AXIS, y, -1, TOF, LIB) == [harm, rational]
    assert drop_harmonic_readings([mono, harm], None, None, -1, TOF, LIB) == [mono, harm]


def test_a_reading_at_three_times_an_impurity_does_not_remove_it(monkeypatch):
    """A later pass drops charge harmonics of the species found so far (decode.drop_harmonics_of),
    not of the harmonic readings among them: a reading at three times a co-eluting impurity's
    mass, fitted in the first pass on the impurity's ions at three times their charge, would
    otherwise remove the impurity as its third."""
    imp = M + 57.0  # 5 % of the species
    y = _render(M, {z: 1e5 * h for z, h in ZH.items()})
    y = y + _render(imp, {z: 5e3 * h for z, h in ZH.items()})
    y = y + np.abs(np.random.default_rng(0).normal(0.0, 20.0, y.size))
    spec = Spectrum(AXIS, y, rt=10.0, polarity=-1, is_profile=True)
    mono = _comp(M, {(z, ""): h for z, h in ZH.items()}, 1e6)
    reading = _comp(3 * imp, {(3 * z, ""): h for z, h in ZH.items() if 8 <= z <= 11}, 2e5)
    params = P.DeconvParams(compound_class="ps_rna", class_candidates=("ps_rna",),
                            mass_range=(1000.0, 60000.0), charge_range=(2, 40),
                            adduct_mode="iprp", max_components=200)

    def second_pass():
        seen: list = []

        def refine_frame(candidates, grid, observed, *a, **k):
            # the first pass "fits" the species and the reading; the second ends the passes
            seen.append([c.mass for c in candidates])
            return ([mono, reading] if len(seen) == 1 else []), 0.5, observed.astype(float)

        monkeypatch.setattr(P, "refine_frame", refine_frame)
        P.deconvolve_spectrum(spec, make_predictor(None, z_max=40), params, TOF)
        return seen[1] if len(seen) > 1 else []  # no second fit: nothing was left to fit

    assert any(abs(m - imp) < 1.0 for m in second_pass())
    # without the reading test the impurity is the reading's third
    monkeypatch.setattr(P, "drop_harmonic_readings", lambda components, *a, **k: components)
    assert not any(abs(m - imp) < 1.0 for m in second_pass())


def _tpao_avg() -> float:
    return float(Formula(TPAO).avg_mass)


def test_the_strongest_candidates_keep_their_mass_against_a_much_weaker_partner():
    """An impurity one C9H21NO (within the dedupe's 0.5 Da) below the main species is a species
    of its own, not the main species' base form: under any realistic adduct load the base form
    is the strongest member of its ladder."""
    deltas = {TPAO: _tpao_avg()}
    below = M - _tpao_avg() + 0.4

    def run(partner_score, hold):
        strong = Candidate(M, 1.0, {11, 12, 13}, gate_charges={11, 12, 13}, gate_failed=set())
        weak = Candidate(below, partner_score, {11, 12}, gate_charges={11, 12})
        return dedupe_adduct_candidates([strong, weak], deltas, hold=hold)

    kept = run(0.05, hold=2)
    assert [round(c.mass, 2) for c in kept] == [M, round(below, 2)]
    assert kept[0].gate_charges == {11, 12, 13} and kept[0].charges == {11, 12, 13}
    # a partner with half its score or more is taken for its base form, as are much weaker
    # partners of candidates beyond the held ones
    for partner_score, hold in ((0.6, 2), (0.05, 0)):
        kept = run(partner_score, hold)
        assert len(kept) == 1 and abs(kept[0].mass - below) < 1e-9
        assert kept[0].gate_charges is None and kept[0].charges == {11, 12, 13}
    # only the strongest candidates are held: the third keeps moving onto a weak partner
    cands = [Candidate(M, 1.0, {11, 12, 13}), Candidate(5000.0, 0.8, {5, 6}),
             Candidate(6000.0, 0.5, {6, 7}), Candidate(6000.0 - _tpao_avg(), 0.01, {6, 7})]
    kept = dedupe_adduct_candidates(cands, deltas, hold=2)
    assert sorted(round(c.mass, 2) for c in kept) == [5000.0, round(6000.0 - _tpao_avg(), 2), M]


def _adduct_ladder_scans(partner_offset=0.4):
    """Eight scans of the species with its C9H21NO ladder (12 and 4 %) and a 3 % impurity
    ``partner_offset`` Da above one C9H21NO below it, as a vendor writes them."""
    x = AXIS
    tp = SHIFT["+" + TPAO]
    y = _render(M, {z: 1e5 * h for z, h in ZH.items()}, x=x)
    for n, share in enumerate(LADDER, 1):
        y += _render(M, {z: 1e5 * share * h for z, h in ZH.items()}, n * tp, x=x)
    y += _render(M - _tpao_avg() + partner_offset, {z: 3e3 * h for z, h in ZH.items()}, x=x)
    rng = np.random.default_rng(1)
    scans = []
    for k, e in enumerate(np.exp(-0.5 * ((np.arange(8) - 3.5) / 1.5) ** 2)):
        s = y * e + np.abs(rng.normal(0.0, 30.0, x.size))
        keep = s > 90.0
        scans.append(Spectrum(x[keep], s[keep], rt=10.0 + 0.02 * k, polarity=-1, is_profile=True))
    return Run(scans).sum_frames(scans)


def test_an_impurity_one_adduct_below_does_not_cost_the_region_its_adduct_fold(monkeypatch):
    """The dedupe used to move the main species' candidate down onto a 3 % impurity whose mass
    lies 0.4 Da off one C9H21NO below it; the adduct evidence then tried the impurity's C9H21NO
    form, 0.4 Da off the main species, found none, and the region fitted no C9H21NO columns: the
    main species' adduct ladder (a seventh of its signal) was reported as two impurities."""
    spec = _adduct_ladder_scans()
    params = P.DeconvParams(compound_class="ps_rna", class_candidates=("ps_rna",),
                            mass_range=(1000.0, 60000.0), charge_range=(2, 40),
                            adduct_mode="iprp", adduct_include=(TPAO,), adduct_max_total=3,
                            adduct_max_per_type=2, max_components=200)

    def deconvolve():
        fr = P.deconvolve_spectrum(spec, make_predictor(None, z_max=40), params, TOF)
        return fr, max(fr.components, key=lambda c: c.intensity)

    fr, main = deconvolve()
    assert TPAO in fr.meta["adducts"]
    assert abs(main.mass - M) < 0.3
    assert 0.07 < main.adduct_fractions().get("+" + TPAO, 0.0) < 0.15  # 10 % of its forms
    assert not any(abs(c.mass - M - n * _tpao_avg()) < 1.0 for c in fr.components for n in (1, 2))
    # without the hold the fold is lost
    monkeypatch.setattr(P, "ADDUCT_EVIDENCE_SPECIES", 0)
    fr, main = deconvolve()
    assert TPAO not in fr.meta["adducts"]
    assert any(abs(c.mass - M - _tpao_avg()) < 1.0 for c in fr.components)


#: nusinersen, a published phosphorothioate 18-mer (7.1 kDa); 2 and 3 times its composition
NUSINERSEN = "C234H340N61O128P17S17"


def _nominal_pattern(formula):
    """Average masses and abundances of a formula's isotopologues, one per nominal mass."""
    from lcmsdeconv.chem.isotopes import isotope_pattern

    p = isotope_pattern(formula, threshold=1e-5)
    k = np.rint((p.masses - p.masses.min()) / 1.0033548).astype(int)
    a = np.bincount(k, weights=p.abundances)
    w = np.bincount(k, weights=p.abundances * p.masses)
    keep = a > 1e-4 * a.max()
    return w[keep] / a[keep], a[keep] / a[keep].sum()


def _oligo_family_scans(heavy: bool):
    """Twelve scans of the 7.1 kDa oligonucleotide (exact isotopes, so the class template is a
    little off its isotope comb, as on real data) at charges 4-12 with a C9H21NO ladder (0-4)
    and sodium forms, a PS-to-PO and a +16 Da impurity (5 and 3 %), on a time-of-flight axis
    with vendor thresholding; with ``heavy`` a co-eluting genuine 21 kDa species at charges
    15-30 (base and +C9H21NO) as well."""
    tp = Formula(TPAO).mono_mass
    na = (Formula("Na") - Formula("H")).mono_mass
    zh = {4: 0.15, 5: 0.35, 6: 0.60, 7: 0.85, 8: 1.0, 9: 0.90, 10: 0.65, 11: 0.40, 12: 0.20}
    forms = [(0.0, 0.75), (tp, 0.10), (2 * tp, 0.03), (3 * tp, 0.008), (4 * tp, 0.002),
             (na, 0.04), (na + tp, 0.03), (na + 2 * tp, 0.01), (2 * na, 0.007)]
    ions = []
    for formula, scale, fs in ((NUSINERSEN, 1.0, forms),
                               ("C234H340N61O129P17S16", 0.05, forms[:4]),
                               ("C234H340N61O129P17S17", 0.03, forms[:4])):
        m, a = _nominal_pattern(formula)
        ions += [(m + add, a, z, scale * h * f) for z, h in zh.items() for add, f in fs]
    if heavy:
        m, a = _nominal_pattern("C690H1010N180O380P51S51")
        hz = {z: 0.3 * float(np.exp(-0.5 * ((z - 22) / 4.0) ** 2)) for z in range(15, 31)}
        ions += [(m + add, a, z, h * f) for z, h in hz.items() for add, f in ((0.0, 0.8),
                                                                               (tp, 0.2))]
    x = np.arange(np.sqrt(500.0), np.sqrt(2200.0), 0.000371) ** 2
    y = np.zeros_like(x)
    for masses, ab, z, h in ions:
        for c, v in zip((masses - z * MP) / z, ab):
            s = float(TOF.fwhm(c)) / 2.3548
            lo, hi = np.searchsorted(x, [c - 5 * s, c + 5 * s])
            y[lo:hi] += 2.0e5 * h * v * np.exp(-0.5 * ((x[lo:hi] - c) / s) ** 2)
    rng = np.random.default_rng(3)
    scans = []
    for k, e in enumerate(np.exp(-0.5 * ((np.arange(12) - 5.0) / 2.0) ** 2)):
        s = y * e + np.abs(rng.normal(0.0, 20.0, x.size))
        keep = s > 60.0
        scans.append(Spectrum(x[keep], s[keep], rt=10.0 + 0.0167 * k, polarity=-1,
                              is_profile=True))
    return Run(scans).sum_frames(scans)


def test_an_oligonucleotide_family_on_a_time_of_flight_leaves_no_readings():
    """The comb reads the 7.1 kDa family's ions at higher charges as heavier species (the rule
    off, they are reported); they go, and a genuine co-eluting 21 kDa species stays."""
    params = P.DeconvParams(compound_class="ps_rna", class_candidates=("ps_rna",),
                            mass_range=(1000.0, 60000.0), charge_range=(2, 40),
                            adduct_mode="iprp", adduct_include=(TPAO,), adduct_max_total=3,
                            adduct_max_per_type=2, max_components=200)
    for heavy in (False, True):
        trace: list = []
        fr = P.deconvolve_spectrum(_oligo_family_scans(heavy), make_predictor(None, z_max=40),
                                   params, TOF, trace=trace)
        main = max(fr.components, key=lambda c: c.intensity)
        assert abs(main.mass - 7127.2) < 0.5
        above = [c for c in fr.components if c.mass > 1.25 * main.mass]
        if heavy:
            genuine = [c for c in above if abs(c.mass - 21121.4) < 2.0]
            assert len(genuine) == 1 and genuine[0].intensity > 0.3 * main.intensity
            # nothing else of any size (a 0.1 % sliver at z 36-40 is left)
            total = sum(c.intensity for c in fr.components)
            assert sum(c.intensity for c in above if c not in genuine) < 0.005 * total
        else:
            assert above == []
        assert sum(r.get("outcome") == "harmonic_reading" for r in trace) >= 4
