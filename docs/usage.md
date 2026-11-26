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

Runs can be mzML or mzMLb (mzML in HDF5, `msconvert --mzMLb`). A Waters run's scan IDs carry
its function numbers (`function=N process=P scan=S`); one function is read, the one with the
most MS1 scans (on a tie the lowest number, the low-energy function of MSE data), so the
lockspray reference scans interleaved with it stay out, and `--waters-function N` reads another.
The scans of a function not read whose scan window is narrow (at most 100 m/z: the lockspray) are
kept apart for the lock-mass correction (below).
Chromatograms stored in the file are read with their type: UV traces (electromagnetic radiation
or absorption chromatograms) serve as the UV signal, while pressure, flow-rate and temperature
traces of the LC are kept apart and never taken for a detector trace.

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
| `rda_iprp_neg_oligo` | Waters RDa (BioAccord) time-of-flight, resolving power 15 000, lockspray correction on Leu-enkephalin [M-H]-, ion-pairing reversed phase with tripropylamine/HFIP (TPAO adducts; for TEA/HFIP remove TPAO from `deconvolution.adducts.include`), negative, oligonucleotides and conjugates |
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
acquisition:
  lock_mass_mz: null       # the lockspray reference, e.g. 554.2615 (Waters runs; see below)
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
    include: []            # extra adducts, e.g. ["TEA"], ["TPAO", "TPA"] or a formula "C7H17N"
    exclude: []
    max_total: 3           # adducts per ion
regions:
  min_snr: 10.0            # a TIC peak this many times the trace's noise makes a region
  min_rel_height: 0.002    # ... and at least this share of the tallest TIC peak's height
  rt_window: null          # analyse only [start, end] minutes, e.g. [2, 30]; null: the whole run
linking:
  set_aside_later_elutions: false  # name the main species' later elutions in its flags
quant:
  purity_signal: uv        # uv | tic
  impurity_floor_pct: 0.01
saturation:
  esi_ratio_drop: 0.25     # apex-versus-flank response drop that raises a warning
```

The oligonucleotide methods (`iprp_neg_oligo`, `orbitrap_iprp_neg_oligo` and `rda_iprp_neg_oligo`)
change several of these. Their regions need a TIC peak only three times the trace's noise and 0.1 %
of the tallest peak's height (`min_snr: 3.0`, `min_rel_height: 0.001`): a 2 % impurity eluting on
its own makes a TIC peak about four times the noise, and at 0.2 % the height floor deleted the
region holding a 0.04 % shortmer, whose TIC peak (mostly a co-eluting 3198.6 Da shortmer's) stood
3.47 times above the noise. `orbitrap_iprp_neg_oligo` reads masses from 800 Da and charges from 1,
with `singly_charged_species: false`: its species of 0.8-2.5 kDa show z = 1 beside z = 2. A z = 1
ion alone makes no species there, as singly charged background, or an envelope of any charge read at
z = 1, would make one; the gate counts z = 1 as one of a species' charge states only where its
envelope has no peaks between its isotopologues, while the candidate pick counts every z = 1 vote
beside another charge state and the fit still covers a voted z = 1 the gate rejected
(`architecture.md`, section 3). All three set `set_aside_later_elutions: true`: a species at the
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

### Analysis window

`regions.rt_window: [start, end]` (minutes) restricts the analysis to that part of the run, and is
the way to leave out an injection front and a column wash, which would otherwise make regions,
species and TIC and UV peaks of their own and count in purity and content:

```yaml
regions:
  rt_window: [2, 30]       # from 2 to 30 min; [2, null]: from 2 min to the end
```

Regions are found and deconvolved only within the window (a region's bounds are cut to it), the TIC
and UV traces are integrated only within it, and the area-% purity, the MS purity, the impurity
table and the saturation checks see only its peaks and species. The regions' relative floors
(`min_rel_height` of the tallest TIC peak, and the shares of the total area and ion current) are
taken within the window, so a window can add regions that a tall injection front suppressed; the
noise their `min_snr` multiplies is read on the whole run's trace, since a window crowded with peaks
reads a higher noise on its own and would lose weak bands near the floor. The window applies to
every trace on its own time axis, the UV trace included. `null` (the default) analyses the whole
run; a window covering the whole run gives the same results. `results.json` records the window
(`rt_window`), and the report names it and shades what lies outside it on the whole traces it plots.
Whether a species' trace falls away, as chemical background's does not, is judged with the run's
scans either side of its region, beyond the window too. Re-integrating a trace in the desktop
application keeps to the window the result was processed with, and the area-% purity follows the
purity signal's new peak table. A window that holds no MS scans raises a warning.

### Lock-mass correction (Waters)

A Waters run sprays a reference compound apart (the lockspray) and records a few scans of it in a
function of their own, at the start and the end of the run on the RDa. With
`acquisition.lock_mass_mz` set (`rda_iprp_neg_oligo` sets Leu-enkephalin [M-H]-, 554.2615; its
[M+H]+ is 556.2766 for positive mode), the reference peak is located in each lockspray scan within
0.5 m/z of the lock mass, at the centroid of its part above its half-height points (a scan that does
not show it is skipped). A reading more than 100 ppm from the lock mass is not the reference but
another ion - a background ion in the window while the lockspray is off, a stray point of a
thresholded scan - and is skipped too, named in a warning; an instrument in calibration reads the
reference within a few tens of ppm. Readings at one time are taken at their median, with a warning
where they spread more than 5 ppm. The ratio of the reference's observed to its true m/z is
interpolated over time (held constant before the first reading and after the last) to one time, the
TIC-weighted mean retention time of the run's MS scans (of those within the analysis window where
`regions.rt_window` is set, so that an injection front or a column wash outside it does not pull
the time away from the species analysed). Every MS scan's m/z is divided by that one
ratio: a one-point lock-mass correction, with one ratio for the whole run rather than one per scan.
The scans of a region must share one m/z axis to be summed into the spectrum that is deconvolved
(they are added point by point); scans scaled each by its own ratio no longer share one, and their
sum spreads every peak over the scans' several axes. On the RDa runs that cost more than the drift
it removed: scaled scan by scan, the comb left more of a run's signal in species re-read at other
charges than with no correction at all, and one ratio left as little as none (`architecture.md`,
section 5). The drift over the run stays in the masses: a scan keeps the drift between its own time
and the time the ratio is read at, up to the whole spread of the readings (`drift_ppm`) at the end
of the run farthest from that time; where the lockspray reads +30 ppm at the start and +20 ppm at
the end and the ion current is centred in the run, the scans at its ends keep about 5 ppm.

`process` applies the correction once, before anything else (the run read is left as it was; a run
corrected already is not corrected again), and prints the scaling applied, how far the readings
spread over the run and how many lockspray scans were read ("lock mass 554.2615: m/z scaled by X ppm
(readings spread Y ppm over the run; n of m lockspray scans read)"); the report's header says the
same. `results.json` records it (`lock_mass`: each scan's reading, the scans read and skipped,
`tic_weighted_rt`, the time the ratio is read at, `applied_ppm`, the scaling applied to every scan,
negative where the instrument read high, and `drift_ppm`, the spread of the readings' ppm over the
run). With `lock_mass_mz` set but no lockspray scans in the run (or none showing the reference
within 100 ppm), nothing is corrected and a warning says so. Species masses can still differ from
theory by a few ppm where the instrument's error depends on m/z or drifts over the run.

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

Ion pairing with tripropylamine (with HFIP) is not among the defaults: name its adducts,
`include: [TPAO]` (as `rda_iprp_neg_oligo` does). Tripropylamine N-oxide (TPAO, C9H21NO,
159.16 Da monoisotopic, 159.27 Da on average masses) can decorate a species with a ladder of
several. TPA is the amine itself (C9H21N, 143.27 Da on average masses), whose adducts are
usually weak: add it (`include: [TPAO, TPA]`) where its forms are seen, since a listed adduct
also lets the fit read a species that lies within a dalton or two of a neighbour's adduct form
as that form, which a time-of-flight's resolving power cannot tell apart. Named, they label
adduct columns "+TPAO" and "+TPA"; given as formulas ("C9H21NO"), the formula labels them.

Every reported species carries its adduct fractions, so the contribution of each adduct type is
visible rather than folded silently into the total. A region's decoder fits an adduct as columns
of a species only where it sees that adduct next to the region's strongest species, and up to
`max_per_type` of a kind, so some adduct forms come out as species of their own. A species one
to four adducts of one kind of the method's library (the mode's adducts and `include`) above a
stronger species of its polarity - within the tolerance of a mass difference below - that
co-elutes with it (apexes within 0.05 min, or in one region each apex inside the other's band at
half height) is that species' adduct form, unless a modification of its class explains the
difference as well: it is annotated "adduct form of main (... Da): +2TPAO", has no row in the
impurity table and no share of the base-form signal, and its amount counts in its parent's
all-forms area and adduct fractions. The parent's flags name its forms.

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
million), adduct fractions, and the modification that matches the mass difference. The masses are
average masses, so the difference is compared with the modifications' average-mass deltas (PS to
PO is -16.07 Da on average masses, -15.98 Da monoisotopic), within 0.15 Da or 15 ppm of the main
species' mass, whichever is larger (0.05 Da or 12 ppm for monoisotopic masses): the difference of
two species of one run carries little of its calibration error. A species whose charge states
disagree is read less exactly, and the tolerance grows to the two species' combined mass spread,
up to 40 ppm (a PS-to-PO impurity whose envelope overlaps its parent's at time-of-flight
resolving power can read a few tenths of a dalton off). Where several modifications
match, each is named, the closest first ("n+1 (A, PS) or n+1 (G)": a phosphorothioate A and a
phosphodiester G residue differ by 0.07 Da). Twice or three times a modification is read only for
those one molecule carries at several sites (oxidation and dioxidation, deamidation, nucleobase
deamination, PS to PO, PEG units, phosphorylation, sulfation, acetylation, formylation,
carbamylation, the glycan units, C-terminal lysine loss, disulfide reduction), within half the
tolerance; cyanoethyl, protecting groups, end groups, depurination and n-1 / n+1 residues are
read once. Adduct forms that were reported apart and belong to a co-eluting species are counted
with it and listed in its flags (above). Percentages compare base forms because which adduct
forms are resolved depends on the chromatographic region (`architecture.md`, section 5); where
the adduct load differs between species - heavy ion pairing with acetate, the main species at its
apex - they understate impurities by the ratio of the loads, which the adduct fractions of the
two species show.

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
