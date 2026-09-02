import numpy as np
import pytest

from lcmsdeconv.chrom.integrate import integrate_chromatogram, manual_integrate
from lcmsdeconv.core.method import IntegrationEvent, IntegrationSettings
from lcmsdeconv.core.model import Chromatogram


def _gauss(t, center, sigma, height):
    return height * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _area(sigma, height):
    return height * sigma * np.sqrt(2 * np.pi) * 60.0  # signal*s


@pytest.fixture
def three_peaks():
    t = np.linspace(0, 10, 2000)
    y = _gauss(t, 2, 0.05, 1000) + _gauss(t, 5, 0.05, 500) + _gauss(t, 5.18, 0.05, 200) + 10
    return Chromatogram(t, y, "test", "tic")


def test_isolated_peak_area_is_exact(three_peaks):
    table = integrate_chromatogram(three_peaks, IntegrationSettings(peak_width=0.12))
    assert len(table) == 3
    p = table.peaks[0]
    assert abs(p.area - _area(0.05, 1000)) / _area(0.05, 1000) < 0.01
    assert p.code == "BB"


def test_valley_cluster_conserves_total_area(three_peaks):
    table = integrate_chromatogram(three_peaks, IntegrationSettings(peak_width=0.12))
    cluster = table.peaks[1].area + table.peaks[2].area
    expected = _area(0.05, 500) + _area(0.05, 200)
    assert abs(cluster - expected) / expected < 0.02
    assert table.peaks[1].code == "BV"
    assert table.peaks[2].code == "VB"


def test_area_percent_sums_to_100(three_peaks):
    table = integrate_chromatogram(three_peaks, IntegrationSettings(peak_width=0.12))
    assert abs(sum(p.area_pct for p in table.peaks) - 100.0) < 1e-6


def test_area_reject_removes_small_peak(three_peaks):
    small = _area(0.05, 200)
    table = integrate_chromatogram(three_peaks, IntegrationSettings(peak_width=0.12,
                                                                   area_reject=small * 1.5))
    assert all(p.area > small * 1.5 for p in table.peaks)
    assert len(table) < 3


def test_height_reject(three_peaks):
    table = integrate_chromatogram(three_peaks, IntegrationSettings(peak_width=0.12,
                                                                    height_reject=300))
    # the 1000 and 500 high peaks survive; the 200 high rider is rejected
    assert len(table) == 2
    assert min(p.height for p in table.peaks) > 300


def test_integration_off_event(three_peaks):
    st = IntegrationSettings(peak_width=0.12,
                             timed_events=[IntegrationEvent(4.0, "integration_off", None)])
    table = integrate_chromatogram(three_peaks, st)
    assert len(table) == 1
    assert abs(table.peaks[0].rt - 2.0) < 0.05


def test_split_peak_event(three_peaks):
    st = IntegrationSettings(peak_width=0.12,
                             timed_events=[IntegrationEvent(2.0, "split_peak", None)])
    table = integrate_chromatogram(three_peaks, st)
    split = [p for p in table.peaks if "split" in p.flags]
    assert len(split) == 2


def test_peak_sum_slice_event(three_peaks):
    st = IntegrationSettings(peak_width=0.12, timed_events=[
        IntegrationEvent(4.5, "peak_sum_slice", "on"),
        IntegrationEvent(5.6, "peak_sum_slice", "off"),
    ])
    table = integrate_chromatogram(three_peaks, st)
    slices = [p for p in table.peaks if "peak sum slice" in p.flags]
    assert len(slices) == 1
    expected = _area(0.05, 500) + _area(0.05, 200)
    assert abs(slices[0].area - expected) / expected < 0.1


def test_manual_integration(three_peaks):
    p = manual_integrate(three_peaks, 1.7, 2.3)
    assert abs(p.area - _area(0.05, 1000)) / _area(0.05, 1000) < 0.02
    assert p.code == "MM"


def test_tailing_and_width_of_symmetric_peak(three_peaks):
    table = integrate_chromatogram(three_peaks, IntegrationSettings(peak_width=0.12))
    p = table.peaks[0]
    assert 0.9 < p.symmetry < 1.1
    assert abs(p.width_half - 2.3548 * 0.05) / (2.3548 * 0.05) < 0.1
    assert 0.85 < p.tailing < 1.15
