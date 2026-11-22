"""Pieces of one band over several regions: merged until nothing more joins."""

import numpy as np

from lcmsdeconv.core.model import Species
from lcmsdeconv.features.link import merge_adjacent_species

STEP = 0.04


def _species(sid, mass, t, y, charges=None):
    y = np.asarray(y, dtype=float)
    return Species(sid, mass, "average", -1, "ps_rna", np.asarray(t, dtype=float), y,
                   {"base": y.copy()}, charges or {4: float(y.sum())})


def test_merging_repeats_until_a_chain_of_pieces_is_joined():
    """Pieces are joined strongest first: a tail piece that touches only a weaker piece of the
    band, not yet joined, stayed apart (2998.30 Da of the IP-RPLC conjugate run)."""
    def band(t):  # a peak with a slow tail
        peak = 100 * np.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
        return np.where(t <= 1.0, peak, 100 * np.exp(-0.5 * 2.0 ** 2) * np.exp(-(t - 1.0) / 1.5))

    ta, tb, tc = np.linspace(0.0, 1.0, 26), np.linspace(1.0, 1.24, 7), np.linspace(1.24, 2.4, 30)
    a = _species(0, 2998.45, ta, band(ta))
    b = _species(1, 2998.32, tb, band(tb))
    c = _species(2, 2998.30, tc, band(tc))
    assert c.total_intensity > b.total_intensity  # taken before b, while a's trace ends at 1.0
    out = merge_adjacent_species([a, b, c])
    assert len(out) == 1 and out[0].mass == 2998.45
    assert np.isclose(out[0].time.max(), 2.4)
