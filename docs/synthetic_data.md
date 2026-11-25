# Synthetic data

The model is trained entirely on synthesized data. No real measurements are used at any point in
training; public data is used only to sanity-check the finished pipeline.

The generator has two tiers that share one rendering path, so a training label can never
disagree with the features it accompanies.

## What is simulated

**Compounds** (`synth/compounds.py`)

| Class | Range | Notes |
| --- | --- | --- |
| Peptides and proteins | 0.6–150 kDa | explicit sequences below 6 kDa from UniProt residue frequencies, averagine above; monoclonal antibodies sampled as a separate template |
| DNA, RNA, phosphorothioate DNA/RNA | 5–120 residues | phosphorothioate replaces one backbone oxygen with sulfur |
| Glycans | DP 2–60 | hexose-rich average residue |
| PEG, PPG, PLGA | DP 3–900 | repeat-unit compositions |
| Small molecules | 100–1500 Da | singly charged with occasional 2+ |

**Impurities** (`synth/impurities.py`) are class-specific mass deltas at log-uniform abundances
between 0.01 % and 10 % of the parent: oxidation, deamidation, pyro-Glu, C-terminal lysine loss,
glycoforms and afucosylation for proteins; n−1 and n+1 deletions, depurination, cyanoethyl
adducts and phosphorothioate-to-phosphodiester conversion for oligonucleotides; end-group
variants for polymers; sugar additions and sulfation for glycans. A fifth of frames also carry
an unrelated co-eluting compound.

**Charge states** (`synth/charge.py`) follow `z_apex = a · M^b` with class- and mode-dependent
coefficients — wide distributions for denatured protein, narrow ones near `0.078 · M^0.5` for
native, and `0.00664 · M^0.8` for oligonucleotides under ion pairing — with skewed and
occasionally bimodal shapes. A
product-related impurity inherits its parent's envelope, with the apex moved by the mass ratio
and a few percent of jitter: an oxidised or clipped form charges like its parent, and drawing
its envelope independently gave impurities charge states their parent never showed.

**Adducts** (`synth/adduct_sampler.py`) get a per-run load for each type in the active library:
the mean number of that adduct an ion carries at the envelope's apex charge, up to
`adduct_max_per_ion` (0.3 by default; 30 % of runs have none of a given type). The count on an
ion is binomial in its charge with that mean at the apex, so higher charge states carry more
adducts, as they do in practice, while the un-adducted form stays visible. Each adduct state is
rendered as its own partial spectrum, so the truth table records exact adduct fractions. The
older per-charge propensity (`adduct_max_lambda`) is kept for reproducing old benchmarks and as
a stress test: at 0.15 per charge a z = 40 protein ion carries about six adducts and its
un-adducted form is a few percent of its signal.

**Instrument** (`chem/instrument.py`, `synth/noise.py`): time-of-flight with constant resolving
power between 8 000 and 60 000, Orbitrap with resolving power falling as the square root of m/z,
or FT-ICR; Gaussian to Gaussian-Lorentzian line shapes; calibration offsets of up to ten parts
per million; Poisson shot noise with a variable gain; electronic noise; sparse singly-charged
chemical noise; and detector saturation as either hard clipping or dead-time compression.

**Chromatography** (`synth/chromatography.py`): exponentially modified Gaussian peaks (computed
with the scaled complementary error function; the textbook form underflowed to zero when the
tail was short against the width and silently removed about 2 % of peaks from the data), impurities
offset slightly in retention time, a UV trace with class-dependent response and a realistic
detector delay, polarity switching, and ionisation saturation applied to the frame's total ion
current so the TIC flat-tops while the UV trace stays linear — exactly the situation the software
is asked to detect.

## The two tiers

**Frame tier** (`synth/frames.py`) produces one training sample: a 32 768-bin crop of the
logarithmic grid with its four feature channels, soft charge-share labels (top three classes per
bin), and an apex heatmap. Crops are centred on a component 70 % of the time and placed at random
otherwise, so the network also learns what empty spectrum looks like.

By default (`render_path = "profile"`) a frame follows the path inference takes: the scene, with
up to 40 singly charged background ions, is rendered as profile scans on the instrument's own m/z
axis; one to eight scans are drawn with shot noise at a random gain, electronic noise and a
vendor-style threshold (every point above it is written with two neighbours either side, as
segments: the real Thermo scans hold no segment shorter than five points, and most hold five or
six; never noise points one by one), and summed (inference deconvolves the summed spectrum of a
chromatographic region); the sum is resampled onto a grid restricted to the measured range, and
the noise level is estimated from it by `estimate_noise_sigma`, exactly as `deconvolve_spectrum`
does. Labels are the shares `y · c_z / max(y, Σ c_z)` of the observed intensity `y`, with each
charge's noise-free contribution `c_z` rendered and resampled the same way. Generation takes
a few tenths of a second per frame on one core (see *How fast frames are made*).
`render_path = "grid"` renders straight onto the grid with the generator's own noise level, as
earlier versions did, which is faster but describes a different input from the one inference
sees.

**Run tier** (`synth/chromatography.py`) produces a whole LC-MS run as mzML plus a `truth.json`
listing every peak, its members, their masses and relative abundances. This is what
`lcmsdeconv synth` writes and what `lcmsdeconv evaluate` scores against.

## How fast frames are made

When a network trains on a GPU, the data workers that make its frames usually set the pace. The
steps that dominate a frame are compiled (`synth/fast.py`, with numba) and give exactly - bit for
bit - what the code they stand for gives, so with numba installed a frame is the same with them
or without them and a paused run resumes on the same spectra. Without numba they fall back to
that code, and the line-shape renderer, which numba compiles with fast-math
(`render._splat_profile`), runs as plain Python and rounds differently: the labels then differ
from those made with numba, as they did before these steps were compiled. The largest gains are
in whole-spectrum frames (recipes v3 and v4, 131 072-bin windows):

- **labels** (`fast.build_labels`): the reference sorts every bin's shares of all classes, one
  sort over a `[classes, bins]` matrix; here the three largest shares are kept as they are
  computed. In most bins (background) all classes but one or two have a share of exactly zero,
  and where numpy's sort puts such tied zeros depends on the sort it runs on the processor
  (AVX-512, AVX2 or neither): numpy's own sort orders one bin of each pattern, and bins whose
  nonzero shares tie are sorted by numpy itself, so the labels are numpy's on every processor;
- **label rows**: each charge is rendered only on the part of the m/z axis its peaks reach
  (`render.profile_reach`) and resampled from there, its zero samples skipped, into one matrix;
- **scans** (`fast.frame_scans`): each scan is one compiled kernel that draws its shot noise and
  electronic noise with numpy's own algorithms (`synth/rng.py`: the PCG64 bit generator, the
  ziggurat normal and the Poisson sampler, operation for operation, on the generator's state),
  saturates, thresholds and adds the scan to the sum (a normal draw by numpy costs about three
  times a compiled one, most of it in calling overhead);
- **comb features**: at the nonzero bins only in a window that is mostly zeros (thresholded
  Orbitrap spectra), tile by tile in the others;
- **isotope limit** (`nn.features.isotope_limit`, which inference shares): the highest resolved
  charge from an estimate, settled at each bin by the reader's own test at that charge and the
  next, instead of the test at every charge.

numpy does not promise that a distribution's algorithm stays the same between its versions, and
C runtimes compute `exp` and `log` differently, so the compiled draws are checked where they run:
the first frame a process makes runs `rng.self_check()` (a tenth to a quarter of a second), which
compares the compiled `exp`, `log`, `log1p` and `sqrt` with the C runtime's on a million inputs,
then the draws and whole scans with numpy's, draw for draw. If anything differs, numba is
missing, or `LCMSDECONV_COMPILED_DRAWS=0` is set, numpy draws the scans as before (the same
frames, slower), and `lcmsdeconv.synth.rng.status()` tells which and why. The isotope limit's
shortcut is checked against the loop over the charges once per process in the same way.
Measured on one core of the same busy machine (thread CPU time of paired, interleaved runs; two
samples of 120 and 90 items, every frame the same as before), the compiled draws and the isotope
limit's shortcut take another 13-14 % off a v4 whole-spectrum item, with its kinds weighted as
training draws them (1.15-1.17 times faster): an Orbitrap item takes 0.67-0.72 of its time, a
time-of-flight item 0.91-0.92 (a shorter axis, and Poisson draws that cost about what numpy's
do: one call of the C runtime's `exp` each).

`tests/test_synth_speed.py` compares each with the code it replaces (the labels with ties of
every kind, under each of numpy's sorts), whole frames of every recipe with the generator as it
was (`tests/_synth_reference.py`), and full-size v3 and v4 frames with the digests of the frames
commit 2a5fcb5 made. Measured item by item against that commit on one core of a busy 4-core
machine (medians of paired, interleaved timings), a v4 whole-spectrum item takes 191 ms instead
of 322 ms (paired ratio 1.58, 95 % interval 1.35-1.78) and a v3 item is 1.68 times faster; items
of the default mixture (32 768-bin crops) are 1.04-1.08 times faster. A process making v4 items
peaks about 0.2 GB lower (850-870 MB resident instead of 1 070-1 080 MB, PyTorch loaded, as in
a data worker).

## Why labels cannot drift from features

Every ion is rendered once, as a set of isotopologue sticks, by `synth/render.py`. The same
sticks produce the per-charge contributions used to build labels and the observed spectrum the
features are computed from, through the same rendering and resampling. A bin's label is literally the fraction of that bin's rendered
intensity belonging to each charge, so a change to the peak shape, the resolution model or the
noise automatically appears in both.

## Generating data

```bash
lcmsdeconv synth --preset protein --out runs/demo --peaks 3 --minutes 6 --saturate
lcmsdeconv synth --preset oligo --out runs/oligo --peaks 2      # negative mode, ion pairing
lcmsdeconv synth --preset mixture --out runs/mix --switch-polarity
```

Profile frames are thresholded the way vendor software writes them, so a run is tens rather than
hundreds of megabytes: every point above the threshold is kept with three neighbours either side,
so no segment is shorter than seven points (training frames keep two, which gives the five-point
minimum of the real Thermo scans; whole runs keep three because the benchmarks were measured on
them).
