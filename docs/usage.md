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
| `orbitrap_iprp_neg_oligo` | Orbitrap, ion-pairing reversed phase or HILIC, negative, oligonucleotides and conjugates (800-60 000 Da, z 1-40, singly charged species off) |
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
  singly_charged_species: true   # z = 1 alone makes a species; false: z = 1 makes none alone,
                                 # and the gate counts it only where its envelope is singly charged
  suppress_multimers: true       # drop a weaker co-eluting species at n x M (false: native, SEC)
  coelution_r: 0.5         # charge states of a region must co-elute (Pearson r); 0 = no test
                           # (multimers then go only as harmonic readings)
  adducts:
    mode: rplc             # rplc | rplc_tfa | iprp | hilic | ip-hilic | sec | native | polymer
    include: []            # extra adducts, e.g. ["TEA"] or a formula like "C7H17N"
    exclude: []
    max_total: 3           # adducts per ion
regions:
  min_snr: 10.0            # a TIC peak this many times the trace's noise makes a region
  min_rel_height: 0.002    # ... and at least this share of the tallest TIC peak's height
linking:
  set_aside_later_elutions: false  # name the main species' later elutions in its flags
quant:
  purity_signal: uv        # uv | tic
  impurity_floor_pct: 0.01
saturation:
  esi_ratio_drop: 0.25     # apex-versus-flank response drop that raises a warning
```

The oligonucleotide methods (`iprp_neg_oligo` and `orbitrap_iprp_neg_oligo`) change several of
these. Their regions need a TIC peak only three times the trace's noise and 0.1 % of the tallest
peak's height (`min_snr: 3.0`, `min_rel_height: 0.001`): a 2 % impurity eluting on its own makes a
TIC peak about four times the noise, and at 0.2 % the height floor deleted the region holding a
0.04 % shortmer, whose TIC peak (mostly a co-eluting 3198.6 Da shortmer's) stood 3.47 times above
the noise. `orbitrap_iprp_neg_oligo` reads masses from 800 Da and charges from 1, with
`singly_charged_species: false`: its species of 0.8-2.5 kDa show z = 1 beside z = 2. A z = 1 ion
alone makes no species there, as singly charged background, or an envelope of any charge read at
z = 1, would make one; the gate counts z = 1 as one of a species' charge states only where its
envelope has no peaks between its isotopologues, while the candidate pick counts every z = 1 vote
beside another charge state and the fit still covers a voted z = 1 the gate rejected
(`architecture.md`, section 3). Both methods set `set_aside_later_elutions: true`: a species at the
main species' mass, at most a twentieth of it, starting more than 0.2 min after the main species'
band and the same molecule by its resolved isotopes - its tail past its band and its re-elution when
the column is washed - is named in the main species' flags instead of being reported, and so are the
forms that leave the column with such a wash peak. The setting is off by default because an isomer
eluting later at that share (an isoAsp or disulfide-scrambled protein variant, a branched polymer
beside a linear one) would be set aside the same way.

With `suppress_multimers` a weaker species at n times a stronger one's mass is dropped as a harmonic
reading of the stronger one's ions where its own charge states carry under a quarter of its fit, and
otherwise only where its elution follows the stronger one's over four scans or more (with
`coelution_r: 0` only as a harmonic reading); methods for native and size-exclusion work, where
oligomers are the point, turn it off. Method files written before a setting existed load with its
default.

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

`adducts.include` adds library adducts by name, or any adduct by formula ("C7H17N"). A formula
must add mass: an exchange is written with a negative count ("MgH-2" for magnesium in place of two
protons; "Mg-H2" reads as Mg^-1 H^2 and is refused). A formula with the mass of an adduct already
listed is that adduct ("NH3" where NH4 is listed), since two columns of one mass trade intensity.

Every reported species carries its adduct fractions, so the contribution of each adduct type is
visible rather than folded silently into the total.

One caveat is worth stating plainly. Ammonium (+17.027) and oxidation (+15.995) differ by about
1 Da; at modest resolving power on a large molecule these are not separable, and an oxidation
impurity can be reported as an ammonium adduct. Where a reported adduct is within 1.5 Da of a
known modification of that compound class, the species is flagged. Excluding adducts that the
mobile phase cannot produce removes the ambiguity.

## Reading the output

`species.csv` and the report's species table give, per species: mass, difference from the main
species, retention time, area of its base form (`area`) and of every adduct form resolved with it
(`area_all_forms`), percentage of the deconvolved base-form signal ("% (base forms)"), charge range,
mass spread across charge states (a quality measure — a real species agrees to a few parts per
million), adduct fractions, and the modification that matches the mass difference. Adduct forms
that were reported apart and belong to a co-eluting species are counted with it and listed in its
flags. Percentages compare base forms because which adduct forms are resolved depends on the
chromatographic region (`architecture.md`, section 5); where the adduct load differs between
species - heavy ion pairing with acetate, the main species at its apex - they understate
impurities by the ratio of the loads, which the adduct fractions of the two species show.

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
