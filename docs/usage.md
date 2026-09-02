# Usage

## Command line

```bash
lcmsdeconv methods                       # list the bundled method templates
lcmsdeconv process run.mzML --method rplc_pos_protein --out results/
lcmsdeconv report results/ --open
lcmsdeconv gui                           # desktop application
```

`process` writes `results.json`, `species.csv`, one `peaks_<signal>.csv` per integrated signal,
and `report.html`. Pass `--uv-csv trace.csv` when the UV trace is exported separately rather than
stored in the mzML.

## Methods

A method is a YAML file describing the instrument, the deconvolution settings, the adduct
library, the integration events for each signal, and the quantitation setup. Start from a
bundled template:

```bash
python -c "from lcmsdeconv.core.method import Method; Method.load('iprp_neg_oligo').to_yaml('my_method.yaml')"
lcmsdeconv process run.mzML -m my_method.yaml -o results/
```

| Template | For |
| --- | --- |
| `rplc_pos_protein` | reversed-phase, positive, intact protein or peptide |
| `iprp_neg_oligo` | ion-pairing reversed-phase (triethylamine/hexafluoroisopropanol), negative, oligonucleotides |
| `sec_native` | size exclusion, native electrospray, complexes and aggregates |
| `hilic_glycan` | HILIC, positive, released glycans |
| `polymer_pos` | synthetic polymers |
| `small_molecule_switching` | small molecules with polarity switching |
| `orbitrap_topdown` | Orbitrap or FT-ICR, isotopically resolved |

### Settings worth knowing

```yaml
instrument:
  kind: tof                # tof | orbitrap | fticr
  resolution: 30000        # FWHM resolving power at mz_ref
deconvolution:
  mass_range: [2000.0, 200000.0]
  charge_range: [3, 80]
  compound_class: peptide  # or auto, dna, rna, ps_dna, ps_rna, glycan, peg, small_molecule
  min_relative_abundance: 0.0001   # 0.01 % detection floor
  min_charge_support: 2    # charge states a species must show; set 1 for small molecules
  adducts:
    mode: rplc             # rplc | rplc_tfa | iprp | hilic | ip-hilic | sec | native | polymer
    include: []            # extra adducts, e.g. ["TEA"] or a formula like "C7H17N"
    exclude: []
    max_total: 3           # adducts per ion
quant:
  purity_signal: uv        # uv | tic
  impurity_floor_pct: 0.01
saturation:
  esi_ratio_drop: 0.25     # apex-versus-flank response drop that raises a warning
```

### Compound class

Set `compound_class` to the class you are running: it selects the average composition used to
build isotope envelopes. `auto` compares how well each candidate class explains the strongest
envelope, but at time-of-flight resolving power the envelopes of a peptide, an oligonucleotide
and a polymer of the same mass differ by less than the peak width, so the choice is unreliable:
in a three-way test it picked the right class once. Components carry a flag when the class was
chosen automatically. Every bundled method names its class explicitly, and so should yours.

The class affects envelope shape and therefore the fit quality and the impurity annotations; it
has only a small effect on the reported mass, which came out within 65 parts per million even
when the wrong class was chosen.

### Adducts

The adduct library decides which mass differences are merged into a base species rather than
reported as separate compounds. It is chromatography-mode specific: sodium, potassium and
ammonium under normal reversed-phase; triethylamine, di-isopropylethylamine and
hexafluoroisopropanol under ion pairing; trifluoroacetate, formate or acetate where those are
in the mobile phase.

Every reported species carries its adduct fractions, so the contribution of each adduct type is
visible rather than folded silently into the total.

One caveat is worth stating plainly. Ammonium (+17.027) and oxidation (+15.995) differ by about
1 Da; at modest resolving power on a large molecule these are not separable, and an oxidation
impurity can be reported as an ammonium adduct. Where a reported adduct is within 1.5 Da of a
known modification of that compound class, the species is flagged. Excluding adducts that the
mobile phase cannot produce removes the ambiguity.

## Reading the output

`species.csv` and the report's species table give, per species: mass, difference from the main
species, retention time, area, percentage of the deconvolved ion signal, charge range,
mass spread across charge states (a quality measure — a real species agrees to a few parts per
million), adduct fractions, and the modification that matches the mass difference.

`peaks_<signal>.csv` is a chromatographic peak table with retention time, boundaries, area,
height, width at half height, area percentage, tailing factor and the baseline code.

## Saturation warnings

Two independent checks run on every processed file:

- **Detector saturation** — flat-topped MS frames clipped at a common ceiling. Intensities and
  isotope ratios in those frames are unreliable.
- **Ionisation saturation** — the MS response falls at a peak's apex relative to its own flanks
  when compared with the UV trace. The column is delivering more analyte than the source can
  ionise, so the MS area under-reports the amount while the UV area does not.

Both name the affected peaks in the report. Quantify flagged peaks from UV, or dilute and
re-inject.

## Python

```python
from lcmsdeconv.core.method import Method
from lcmsdeconv.io.mzml import read_mzml
from lcmsdeconv.process import process_run

run = read_mzml("run.mzML")
result = process_run(run, Method.load("rplc_pos_protein"))

for s in result.species[:5]:
    print(f"{s.mass:10.2f} Da  RT {s.rt_apex:.2f}  {s.adduct_fractions()}")
```
