
import pytest

from lcmsdeconv.chem.adducts import AdductLibrary, mass_from_mz, mz_from_mass
from lcmsdeconv.chem.classes import class_isotope_pattern, class_names, get_class
from lcmsdeconv.chem.formula import Formula
from lcmsdeconv.chem.isotopes import isotope_pattern
from lcmsdeconv.chem.modifications import annotate_delta


def test_formula_parse_and_mass():
    f = Formula("C6H12O6")
    assert abs(f.mono_mass - 180.0633881) < 1e-4
    assert abs(f.avg_mass - 180.156) < 1e-2
    assert f["C"] == 6 and f["H"] == 12 and f["O"] == 6


def test_formula_arithmetic():
    water = Formula("H2O")
    assert (Formula("C6H12O6") - water)["O"] == 5
    assert (water * 3)["H"] == 6
    assert (-water)["O"] == -1
    d = Formula("Na") - Formula("H")
    assert abs(d.mono_mass - 21.98194) < 1e-3


def test_isotope_matches_pyteomics():
    pyteomics = pytest.importorskip("pyteomics.mass")
    for formula in ["C6H12O6", "C50H80N15O18", "C100H150N30O40S2"]:
        p = isotope_pattern(Formula(formula))
        assert abs(p.mono_mass - pyteomics.calculate_mass(formula=formula)) < 2e-3
        assert abs(p.abundances.sum() - 1.0) < 1e-9


def test_isotope_apex_shifts_with_mass():
    small = isotope_pattern(Formula("C50H80N15O18"))
    big = class_isotope_pattern(50000, "peptide")
    assert small.most_abundant_index <= 2
    assert big.most_abundant_index > 10  # apex far from monoisotopic for large mass


def test_averagine_mass_scaling():
    for m in (5000, 20000, 148000):
        f = get_class("peptide").average_formula(m, average=True)
        assert abs(f.avg_mass - m) / m < 5e-4


def test_class_pattern_speed_and_size():
    p = class_isotope_pattern(148000, "peptide")
    assert 100 < len(p) < 400
    assert abs(p.average_mass - 147857) / 147857 < 1e-3


def test_mz_mass_roundtrip():
    for pol in (1, -1):
        for z in (1, 5, 42):
            mz = mz_from_mass(15000.0, z, pol)
            assert abs(mass_from_mz(mz, z, pol) - 15000.0) < 1e-6


def test_adduct_library_modes():
    lib = AdductLibrary.from_mode("iprp", -1)
    assert "TEA" in lib.names() and "HFIP" in lib.names()
    states = lib.states()
    assert states[0].label == ""  # base state first
    assert any(s.label == "+Na" for s in states)
    # sodium delta
    d = lib.deltas()
    assert abs(d["Na"] - 21.98194) < 1e-3


def test_modification_annotation():
    hits = annotate_delta(15.9949, "peptide")
    assert any("oxidation" in h.name for h in hits)
    hits2 = annotate_delta(-15.977, "ps_dna")
    assert any("PS" in h.name for h in hits2)


def test_all_classes_constructible():
    for name in class_names():
        f = get_class(name).average_formula(3000)
        assert f.avg_mass > 0
