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
about 0.35 s per frame. `render_path = "grid"` renders straight onto the grid with the
generator's own noise level, as earlier versions did, which is faster but describes a different
input from the one inference sees.

**Run tier** (`synth/chromatography.py`) produces a whole LC-MS run as mzML plus a `truth.json`
listing every peak, its members, their masses and relative abundances. This is what
`lcmsdeconv synth` writes and what `lcmsdeconv evaluate` scores against.

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
