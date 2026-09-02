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
native, length-scaled negative charging for oligonucleotides — with skewed and occasionally
bimodal shapes.

**Adducts** (`synth/adduct_sampler.py`) get a per-run propensity for each type in the active
library; the count on an ion is binomial in its charge, so higher charge states carry more
adducts, as they do in practice. Each adduct state is rendered as its own partial spectrum, so
the truth table records exact adduct fractions.

**Instrument** (`chem/instrument.py`, `synth/noise.py`): time-of-flight with constant resolving
power between 8 000 and 60 000, Orbitrap with resolving power falling as the square root of m/z,
or FT-ICR; Gaussian to Gaussian-Lorentzian line shapes; calibration offsets of up to ten parts
per million; Poisson shot noise with a variable gain; electronic noise; sparse singly-charged
chemical noise; and detector saturation as either hard clipping or dead-time compression.

**Chromatography** (`synth/chromatography.py`): exponentially modified Gaussian peaks, impurities
offset slightly in retention time, a UV trace with class-dependent response and a realistic
detector delay, polarity switching, and ionisation saturation applied to the frame's total ion
current so the TIC flat-tops while the UV trace stays linear — exactly the situation the software
is asked to detect.

## The two tiers

**Frame tier** (`synth/frames.py`) produces one training sample: a 32 768-bin crop of the
logarithmic grid with its four feature channels, soft charge-share labels (top three classes per
bin), and an apex heatmap. Crops are centred on a component 70 % of the time and placed at random
otherwise, so the network also learns what empty spectrum looks like. Generation takes about
60 ms, so one or two loader workers keep a CPU training run fed.

**Run tier** (`synth/chromatography.py`) produces a whole LC-MS run as mzML plus a `truth.json`
listing every peak, its members, their masses and relative abundances. This is what
`lcmsdeconv synth` writes and what `lcmsdeconv evaluate` scores against.

## Why labels cannot drift from features

Every ion is rendered once, as a set of isotopologue sticks, by `synth/render.py`. The same
render produces the per-charge contributions used to build labels and the observed spectrum the
features are computed from. A bin's label is literally the fraction of that bin's rendered
intensity belonging to each charge, so a change to the peak shape, the resolution model or the
noise automatically appears in both.

## Generating data

```bash
lcmsdeconv synth --preset protein --out runs/demo --peaks 3 --minutes 6 --saturate
lcmsdeconv synth --preset oligo --out runs/oligo --peaks 2      # negative mode, ion pairing
lcmsdeconv synth --preset mixture --out runs/mix --switch-polarity
```

Profile frames are thresholded the way vendor software writes them, so a run is tens rather than
hundreds of megabytes.
