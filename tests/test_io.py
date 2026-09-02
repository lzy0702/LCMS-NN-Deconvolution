import numpy as np

from lcmsdeconv.core.model import Chromatogram, Run, Spectrum
from lcmsdeconv.io.mzml import read_mzml
from lcmsdeconv.io.mzml_writer import write_mzml
from lcmsdeconv.io.text import read_text_spectrum, write_text_spectrum


def _demo_run():
    spectra = []
    rng = np.random.default_rng(0)
    for i in range(5):
        mz = np.linspace(500, 2000, 400)
        it = (rng.random(400) * 10 + 100 * np.exp(-0.5 * ((mz - 1000) / 5) ** 2)).astype(float)
        spectra.append(Spectrum(mz, it, rt=0.1 * i, polarity=1 if i % 2 == 0 else -1, index=i, is_profile=True))
    uv = Chromatogram(np.linspace(0, 0.4, 20), np.abs(np.sin(np.linspace(0, 3, 20))) * 100, "UV1", "uv", "AU")
    return Run(spectra, {"UV1": uv}, name="demo")


def test_mzml_roundtrip(tmp_path):
    run = _demo_run()
    p = write_mzml(run, tmp_path / "demo.mzML")
    back = read_mzml(p)
    assert len(back.spectra) == 5
    assert set(back.polarities) == {1, -1}
    s0 = back.spectra[0]
    assert s0.mz.size == 400
    assert abs(s0.rt - 0.0) < 1e-6
    # chromatogram preserved
    assert any(c.kind == "uv" for c in back.chromatograms.values())


def test_text_spectrum_roundtrip(tmp_path):
    mz = np.linspace(100, 200, 50)
    it = np.abs(np.sin(mz))
    s = Spectrum(mz, it, rt=1.0)
    p = write_text_spectrum(s, tmp_path / "s.txt")
    back = read_text_spectrum(p, rt=1.0)
    assert back.mz.size == 50
    assert np.allclose(back.mz, mz)


def test_run_helpers():
    run = _demo_run()
    tic = run.tic(polarity=1)
    assert tic.time.size == 3
    eic = run.eic(995, 1005, polarity=1)
    assert eic.intensity.max() > 0
    summed = run.sum_frames(run.frames(polarity=1))
    assert summed.mz.size == 400
