"""Scoring rules of the real-run validation (scripts/validate_oligo_runs.py) on made-up runs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lcmsdeconv.core.model import Run, Spectrum

MP = 1.00727646688


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_oligo_runs.py"
    spec = importlib.util.spec_from_file_location("validate_oligo_runs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(peaks, times=np.arange(0.0, 30.0, 0.05), background=True, polarity=-1):
    """Frames (negative mode unless ``polarity`` says otherwise): a flat background every
    0.37 Th (unless ``background`` is off, as in vendor-thresholded data), plus ``peaks``, each
    (mass, charge, apex time, height, elution width) with a Gaussian elution profile."""
    bg = np.arange(400.0, 2500.0, 0.37) if background else np.array([400.0])
    spectra = []
    for t in times:
        mz = list(bg)
        inten = list((20.0 + 5.0 * np.sin(bg * 7.0 + t)) if background else [0.0])
        for mass, z, apex, height, width in peaks:
            mz.append((mass + polarity * z * MP) / z)
            inten.append(height * np.exp(-0.5 * ((t - apex) / width) ** 2))
        order = np.argsort(mz)
        spectra.append(Spectrum(np.array(mz)[order], np.array(inten)[order], float(t),
                                polarity=polarity))
    return Run(spectra)


def test_eic_check_confirms_a_weak_species_next_to_a_strong_signal_elsewhere():
    vor = _script()
    mass = 3208.66
    run = _run([(mass, 2, 14.2, 1e3, 0.1), (mass, 3, 14.2, 6e2, 0.1),
                # the injection peak passes through the z = 2 m/z window at 1.5 min
                (mass, 2, 1.5, 1e6, 0.1)])
    assert vor.joint_eic_peak(run, mass, 14.2)
    assert not vor.joint_eic_peak(run, mass, 16.0)  # nothing elutes there
    assert not vor.joint_eic_peak(run, 3333.3, 14.2)  # background only


def test_eic_check_confirms_an_isomer_eluting_next_to_a_stronger_one():
    vor = _script()
    mass = 4842.0  # two deletion isomers, 0.36 min apart, the later one stronger
    run = _run([(mass, 3, 17.46, 1e4, 0.08), (mass, 4, 17.46, 8e3, 0.08),
                (mass, 3, 17.82, 1.3e4, 0.08), (mass, 4, 17.82, 1e4, 0.08)])
    assert vor.joint_eic_peak(run, mass, 17.46)
    assert vor.joint_eic_peak(run, mass, 17.82)
    assert not vor.joint_eic_peak(run, mass, 18.4)  # the tail of the later one


def test_eic_check_needs_two_charge_states_eluting_together():
    vor = _script()
    mass = 4503.7
    run = _run([(mass, 3, 10.0, 1e4, 0.1), (mass, 4, 10.6, 1e4, 0.1)])
    assert not vor.joint_eic_peak(run, mass, 10.0)
    run = _run([(mass, 3, 10.0, 1e4, 0.1), (mass, 4, 10.0, 5e3, 0.1)])
    assert vor.joint_eic_peak(run, mass, 10.0)


def test_charge_harmonics_of_a_found_species_are_labelled_before_the_eic_check():
    vor = _script()
    nc = SimpleNamespace(mass=5162.27, rt_apex=18.21)
    half = SimpleNamespace(mass=5162.27 / 2, rt_apex=18.21)
    other = SimpleNamespace(mass=3333.3, rt_apex=18.21)
    asked = []

    def confirm(m, t):
        asked.append(m)
        return False
    sc = vor.score("IPRPLC_NC", [nc, half, other], confirm)
    why = {round(s.mass, 2): w for s, w in sc["others"]}
    assert why[round(half.mass, 2)].startswith("harmonic of 5162")
    assert why[3333.3] == "unconfirmed"
    assert asked == [3333.3]


def test_eic_check_wants_a_peak_at_the_reported_time_not_the_flank_of_one():
    vor = _script()
    mass = 4842.0
    run = _run([(mass, 3, 17.82, 1e4, 0.08), (mass, 4, 17.82, 8e3, 0.08)])
    # within 0.25 min of 17.45 the traces only rise toward the peak at 17.82
    assert not vor.joint_eic_peak(run, mass, 17.45)


def test_eic_check_wants_a_peak_several_scans_long():
    vor = _script()
    mass = 3500.0
    # zero-filled traces with one coinciding spike in two charge states
    spikes = _run([(mass, 3, 10.0, 1e4, 0.001), (mass, 4, 10.0, 1e4, 0.001)], background=False)
    assert not vor.joint_eic_peak(spikes, mass, 10.0)
    peak = _run([(mass, 3, 10.0, 1e4, 0.08), (mass, 4, 10.0, 1e4, 0.08)], background=False)
    assert vor.joint_eic_peak(peak, mass, 10.0)


def test_eic_check_wants_the_peak_to_stand_out():
    vor = _script()
    mass = 3500.0
    # a hump a minute wide: highest at 10.0, but not three times its surroundings' median
    run = _run([(mass, 3, 10.0, 1e4, 1.0), (mass, 4, 10.0, 1e4, 1.0)])
    assert not vor.joint_eic_peak(run, mass, 10.0)


def test_eic_check_wants_the_charge_states_to_co_vary():
    vor = _script()
    mass = 3500.0
    # both peak at 10.0, but the z = 4 window also carries a stronger, later species
    run = _run([(mass, 3, 10.0, 1e4, 0.05), (mass, 4, 10.0, 2e3, 0.05), (mass, 4, 10.45, 2e4, 0.05)])
    assert not vor.joint_eic_peak(run, mass, 10.0)


def test_eic_check_reads_the_run_polarity():
    vor = _script()
    mass = 3500.0
    peaks = [(mass, 3, 10.0, 1e4, 0.08), (mass, 4, 10.0, 1e4, 0.08)]
    assert vor.joint_eic_peak(_run(peaks, polarity=1), mass, 10.0)
    negative = _run(peaks)
    assert vor.joint_eic_peak(negative, mass, 10.0)
    for s in negative.spectra:  # the same ions labelled positive: no one mass explains both
        s.polarity = 1
    assert not vor.joint_eic_peak(negative, mass, 10.0)
