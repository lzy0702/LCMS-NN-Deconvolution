"""Thermo .raw reading (needs the optional 'opentfraw' package and a raw file)."""

import os
from pathlib import Path

import pytest

opentfraw = pytest.importorskip("opentfraw")

RAW = os.environ.get("LCMSDECONV_TEST_RAW")


@pytest.mark.skipif(not RAW or not Path(RAW).exists(), reason="set LCMSDECONV_TEST_RAW to a .raw file")
def test_read_thermo_raw_profile_scans():
    from lcmsdeconv.io.thermo import read_thermo_raw

    run = read_thermo_raw(RAW)
    assert run.spectra
    s = run.spectra[len(run.spectra) // 2]
    assert s.is_profile in (True, False)
    assert s.mz.size == s.intensity.size and s.mz.size > 0
    assert (s.mz[1:] >= s.mz[:-1]).all()
    assert run.polarities[0] in (-1, 1)
