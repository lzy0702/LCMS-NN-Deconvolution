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
stored in the mzML. A CSV file is named after its signal, in Unicode's composed form (NFC), with
any character but letters, digits, `-`, `_` and `.` made `_`, and a name of over 100 bytes cut
short and ended with 8 hex digits of its hash; where two signals' names come out the same that way
(also apart from case), the later ones get `_2`, `_3`, ... added, so no table overwrites another.
The report escapes run, method and trace names, so characters such as `&`, `<`, quotes and `$`
show as written in its text and are drawn as written in its plots (where a trace or species name
holds characters the plot fonts lack, Chinese say, the plots fall back to a CJK font if one is
installed: Microsoft YaHei or SimHei on Windows). Names that hold a project number can be shown
as another text everywhere, `--alias 123456=保密项目1` (see "Name aliases" below). With
`--provenance` it also writes `provenance.json`, `provenance.html` and an evidence figure per
species: where each mass was read from (see "Provenance" below).

Runs can be mzML or mzMLb (mzML in HDF5, `msconvert --mzMLb`). They are read, and `lcmsdeconv
synth` writes mzML, with the PSI-MS and Units of Measurement Ontology vocabularies saved in the
package (`src/lcmsdeconv/data/cv/`), which pyteomics (5 and later) would otherwise fetch from
purl.obolibrary.org for every file it reads, and psims for the first file a process writes. A
term the saved PSI-MS lacks is read with a warning, as pyteomics reads any term it has no type
for, instead of failing the file. A Waters run's scan IDs carry
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
sum spreads every peak over the scans' several axes. On the RDa runs of Confidential sample 1
(`training.md`) that cost more than the drift it removed: scaled scan by scan, the comb left more of
a run's signal in species re-read at other charges than with no correction at all, and one ratio
left as little as none (`architecture.md`, section 5). The drift over the run stays in the masses: a
scan keeps the drift between its own time and the time the ratio is read at, up to the whole spread
of the readings (`drift_ppm`) at the end of the run farthest from that time; where the lockspray
reads +30 ppm at the start and +20 ppm at the end and the ion current is centred in the run, the
scans at its ends keep about 5 ppm.

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

## Provenance

Every reported mass can be traced to the retention-time range, the scans and the m/z signals it
was read from:

```bash
lcmsdeconv process run.mzML -m rplc_pos_protein -o results/ --provenance
```

```python
result = process_run(run, method, provenance=True)       # result.provenance
write_outputs(result, "results/")                         # lcmsdeconv.io.results
```

Beside the usual outputs this writes `provenance.json`, `provenance.html` (an index of the
species: mass, RT range, regions, charge states, pieces, each linked to its figure) and one
evidence figure per species under `evidence/` (`species-<id>-<mass>.png`, the 300 largest
species by base-form amount; `provenance.json` says which were drawn).
`write_outputs(..., provenance=False)` leaves them out, `True` requires them; by default they are
written where the result holds provenance. Without `--provenance` (in Python, without
`provenance=True`) nothing is collected or written for it; with it, every other output is the
same, byte for byte, apart from the processing time in `results.json` (`timings.total_s`) and in
`report.html`.

`provenance.json` names the `results.json` it was written with by its SHA-256 (`results.sha256`,
shown in `provenance.html` too). A folder written to again without `--provenance` keeps the
provenance files of the earlier run, which then describe other results: check that the hash is
that of the `results.json` beside them (`sha256sum results.json`, `Get-FileHash results.json` in
PowerShell). Writing the provenance again removes the evidence figures it does not draw.

**What is recorded, and where it comes from.** A run is split into regions around the peaks of
its total ion current; each region's scans are summed, and the summed spectrum is deconvolved on
the ln(m/z) grid: the charge estimator assigns charges bin by bin, the votes give candidate
masses, and each candidate's isotope-envelope templates (one per charge state and adduct form)
are fitted by weighted non-negative least squares, giving *components*. After the fit, what a
stronger component explains is dropped: multimers of it, its ions read at other charges,
envelopes its columns already hold, adduct columns that coincide with its adduct ladder. Each
component is then quantified in every scan of the region (the trace), and a species is a
component's trace linked across the scans. Species found in pieces are then joined: an adduct
form reported apart is folded into its parent, readings of one mass whose traces touch as one
elution band are joined (a neighbouring region's piece of the band, or a reading of the same
region that linking kept apart at a gap in its trace), a piece a region boundary cut off is
joined where its isotopes say it is the same molecule, and the main species' later elutions
(and, in polarity-switching runs, the other polarity's reading) are named in its flags without
being counted. `provenance.json` follows that chain backwards:

- per region: its retention-time span, its scans (number and IDs), whether its start or end is a
  boundary it shares with a neighbour, and its deconvolution (number of components and fit
  passes, compound class, the adducts it evidenced, noise level, the share of its signal the fit
  left, the grid, and what was dropped after the fit: each component or adduct column with its
  step, its fitted columns and the stronger components it was attributed to);
- per species: id, mass, mass type, polarity, apex and charges (as `results.json` gives them),
  its retention range, the region(s) it came from (each with the species' pieces from it and
  whether any of them is counted), its *pieces* (each as linked in its region: region, the id it
  was linked with, its mass, amount and charges then, and whether it is counted in the species'
  amount), the *merge steps* that joined them (in the order taken, with what each joined), which
  piece the reported mass is from and how that mass was obtained, and its trace per scan (scan
  ID, RT, intensity, and each adduct form's part);
- per component of a piece (the component of the region-summed spectrum it was read from): its
  id and the fit pass that found it (`pass`; the two name it), its mass, the shift of its
  isotope templates (`template_shift`, ln mass; the templates sit at `template_mass` = mass ×
  exp(shift)), whether its own isotope comb placed them (`comb_placed`; the mass is then their
  position where that lies within half an isotope spacing of the mass measured, which stands
  otherwise: `template_mass` shows where they sit), its score, the spread of its charge states'
  masses, its fitted intensity, and, for a component the region's fit merged from parts, where
  each part's templates sit (`fitted_parts`, below);
- per charge state z of a component: the theoretical m/z of its envelope (centroid, and the
  monoisotopic peak's where it holds at least 1 % of the tallest isotopologue), whether the
  instrument resolves its isotopes, the envelope's m/z window (`window_mz`, the span of the grid
  bins holding 99.5 % of the area of its base-form template as fitted, a merged component's
  parts' added up; the fit of the summed spectrum read the template's whole support,
  `support_mz`), what the region's scans measured in that window, the
  fitted intensity of its base-form template and of all its forms, its share of the component,
  and on the grid, summed over the window's bins: the signal, the fitted templates of every
  reported component and of this one, the signal they leave unexplained and how far they
  overshoot it, what the fit itself left there, and the templates of what was dropped after the
  fit;
- per adduct-form column (z, adduct state) with a fitted intensity: the same window record;
- the charge estimator: `comb` (with whether it reads isotope spacing), or the network's file
  name and SHA-256 (with its backend and view); and per charge state, the estimator's reading
  on the envelope's bins above the vote threshold, weighted by their signal: the probability of
  the assigned charge (`p_assigned`), the share of signal on bins whose first choice it is
  (`top1_share`) and the strongest other charge (`alternative`). The decoder keeps each bin's
  two most likely charges only, so a charge that is neither counts 0: a low `p_assigned` says the
  estimator did not read that charge there (a charge state the envelope search added beyond the
  voted ones, say). For the comb the values are its evidence shares, not probabilities;
- run-level: the region list, the lock-mass correction and the analysis window (as
  `results.json` gives them), the instrument model, the SHA-256 of `results.json`, and what each
  merge step and each drop step means.

**Units.** Masses in Da (average masses unless `mass_type` says mono), times in minutes. Every
m/z is read on the scans as they were analysed: where the method corrects the lock mass, after
the correction (`units.mz` then says so), and the m/z the file holds is m/z / (1 +
`lock_mass.applied_ppm` × 1e-6). `observed` adds up the region's scans' own points in the
window, all scans together: the instrument's counts (`sum`), the number of those points
(`points`) and of the scans with signal there (`scans`), and their intensity-weighted mean m/z
(`centroid_mz`, and the neutral mass it implies, `centroid_mass`); `max` is the tallest point of
their sum, the
region-summed spectrum (which, for profile scans each on an m/z axis of its own, reads every scan
on the union of their points). `observed` and `centroid_mass` hold every signal in the window, a
neighbour's or an adduct form's envelope as well as the species': `grid.component_model` against
`grid.model` gives the species' own share of it. The fit reads the summed spectrum resampled on
the ln(m/z) grid (intensity integrated over each bin and divided by its width); `fitted`, `grid`
and a component's `intensity` are in those units (a template's area on the grid), which differ
from the points' by the points' spacing. Compare them with each other, not with `observed`.

In `grid`, `model` adds up the fitted templates of every component the region reports
(`component_model`: this component's); `unexplained` sums, bin by bin, the signal above them and
`overshoot` what they put above the signal (`unexplained` − `overshoot` = `observed` − `model`).
`fit_residual` is what the fit itself left in the window: its residual once every pass had
subtracted its components, clipped at zero after each as the fit clips it, before the steps that
drop components after the fit. Where a dropped component explained the signal, it stays in
`unexplained` but not in `fit_residual`, and `dropped_model` holds the dropped templates; a
region's `deconvolution.residual_fraction` is the fit's residual over the grid bins holding
signal (above ten noise levels), as a share of their signal. Every template is read where the
fit placed and subtracted it, so `model` and `dropped_model` add up to what the fit subtracted.

**Components merged by the fit.** Within a fit pass, components found at almost one mass
(duplicate candidates, within the larger of 150 ppm and 0.4 Da) are merged into the strongest,
and a weaker component one, two or three adduct masses above a stronger one (within 0.5 Da) into
it as its adduct form; the merged component reports the parts' columns added up. The fit had subtracted each part's templates where the
part's own fit placed them, which can lie a dalton or more apart: `fitted_parts` lists each
part, its template mass and its columns (`adduct`: the merged component's column it is counted
in; `template_adduct`: the adduct state its own template carries; `fitted`), and the windows, the
`grid` sums and the figures read each part's templates where they sit. It is null for a
component fitted as one, whose templates all sit at `template_mass`. A component dropped after
the fit (`dropped[]`) says the same of itself.

**Naming a component.** A fit pass numbers its components on from the number the earlier
passes kept, after the merges, so an id can recur in a region: the fit pass and the id name a
component (`pass` and `id` of its record, of `dropped[]` and of `attributed_to[]`), and
`mass_from.how` gives both ("component 17 (fit pass 1) of region 19's summed spectrum").

**Schema** (`schema` "lcmsdeconv.provenance", `version` 1):

| Key | Contents |
| --- | --- |
| `schema`, `version`, `lcmsdeconv` | what the file is, the version of its layout, the version that wrote it |
| `run`, `method`, `alias_pairs` | as `results.json` |
| `results` | `file` ("results.json"), `sha256`: the `results.json` it was written with |
| `units`, `merge_steps`, `drop_steps` | what the numbers are in; what each merge step and each drop step name means |
| `estimator` | `{"kind": "comb", "z_max", "isotope_charges"}` or `{"kind": "network", "backend", "file", "sha256", "view", "isotope_channels", "z_max"}` |
| `instrument`, `rt_window`, `lock_mass` | the method's instrument model; the analysis window and the lock-mass correction as `results.json` gives them |
| `regions[]` | `id`, `polarity`, `start`, `end`, `scans` {`n`, `ids`, `rt`: first and last}, `shared_boundaries` {`start`, `end`}, `species` (ids of the reported species with a counted piece from it), `deconvolution` {`engine`, `components`, `passes`, `compound_class`, `adducts`, `noise_sigma`, `residual_fraction`, `dropped[]`, `grid`} |
| `dropped[]` | `step` (`multimer`, `harmonic_reading`, `shared_evidence`, `coincident_adducts`), `id`, `pass`, `mass`, `template_mass`, `component_dropped` (false: adduct columns taken from a component the region reports), `attributed_to` [{`id`, `pass`, `mass`}] (the stronger components, the largest share first), `columns` [{`z`, `adduct`, `fitted`}], `fitted_parts` (as a component's, for the columns dropped) |
| `figures` | `directory`, `max`, `drawn`, `species`, `rule` |
| `timings` | `collect_s` (while processing), `figures_s` (drawing the figures and making the records, before the two files are written) |
| `species[]` | in `results.json`'s order: `id`, `name`, `mass`, `mass_type`, `polarity`, `compound_class`, `rt_apex`, `rt_range` (its trace's first and last scan), `rt_signal` (first and last scan above zero), `amount` {`base`, `all_forms`, `max`}, `adduct_fractions`, `charges` [{`z`, `share`, `value`}], `adduct_form_of`, `adduct_form`, `flags`, `annotations`, `regions` [{`id`, `polarity`, `start`, `end`, `scans` (their number), `first_scan`, `last_scan`, `shared_boundaries`, `pieces` (the species' pieces from it), `counted`}], `mass_from` {`piece`, `how`}, `pieces[]`, `merges[]`, `eic` {`per_scan`, `scan_ids`, `rt`, `intensity`, `forms`}, `figure` (its path, or null) |
| `pieces[]` | the species' own first: `region`, `species_id`, `mass`, `rt_apex`, `rt_range`, `scans`, `amount` {`base`, `all_forms`}, `charges` [{`z`, `value`}], `counted`, `joined_by` (null for its own), `evidence` |
| `evidence` | `kind` `"grid_components"`: `mass_rule`, `components[]` {`id`, `mass`, `mass_type`, `template_shift`, `template_mass`, `comb_placed`, `score`, `mass_spread_ppm`, `compound_class`, `intensity`, `pass`, `flags`, `fitted_parts`, `charge_states[]`, `adduct_columns[]`} |
| `fitted_parts` | null for a component fitted as one; for one the fit merged from parts, each part: `template_mass`, `columns` [{`z`, `adduct`, `template_adduct`, `fitted`}] |
| `charge_states[]` | `z`, `mz_centroid`, `mz_mono`, `isotopes_resolved`, `fitted`, `fitted_all_forms`, `share`, `window_mz` [low, high], `support_mz` [low, high], `observed` {`points`, `sum`, `max`, `scans`, `centroid_mz`}, `centroid_mass`, `grid` {`bins` [first, last], `observed`, `model`, `component_model`, `unexplained`, `overshoot`, `fit_residual`, `dropped_model`}, `charge_map` {`pass`, `bins`, `p_assigned`, `top1_share`, `alternative` {`z`, `p`}} |
| `adduct_columns[]` | `z`, `adduct`, `fitted`, `mz_centroid`, `window_mz`, `support_mz`, `observed`, `centroid_mass`, `grid` |
| `merges[]` | `step` (`adduct_fold`, `adjacent_pieces`, `boundary_piece`, `set_aside`, `other_polarity`), `kept_piece` (the kept species' own piece), `joined_pieces` (the pieces the step brought in: the joined species' own first, then those it had joined before, each also under the step that joined it), `joined` {`species_id`, `mass`, `rt_apex`, `amount`} (as it was when joined), `counted`; `adduct_state` (a fold), `boundary_rt` and `mass_from_piece` (a boundary piece) |

A reader should check `version`, and `evidence.kind`: a decoder that reports whole regions rather
than grid components (a maximum-entropy engine, say) describes its species with
`"region_report"` and fields of its own, and names itself in the region's
`deconvolution.engine`. The pieces' amounts are as they were linked: a joined species counts a
scan two pieces share once (the larger reading), and pieces that are not `counted` add nothing,
so they need not add up to the species' amount; their charges do: a species' charges
(`results.json`'s) are those of its counted pieces added up. A piece joined as a reading of the
same region can be a reading of the very component another piece holds, whose record its
`evidence` then repeats. A method that quantifies the summed spectra only
(`deconvolution.quantify_frames: false`) reads each species once per region, at the mean time of
the region's scans: its trace's points have no scan ID (`eic.per_scan` false).

**Reading a figure.** The title names the run, the species (id, name, mass) and its mass type,
polarity, apex, charge range and number of pieces. The top panel is the summed spectrum of the
region the reported mass comes from (grey, as the fit reads it, on the grid; in a polarity-switching
run, whose region ids restart in each polarity, the id carries its + or -), over the m/z span of the
species' charge states: each charge state's window is shaded, and numbered with its z along the top
where the number has room. The species' fitted templates (all its forms) are blue, each where the
fit placed it (for a component the fit merged from parts, the legend says how many); those of its
other pieces counted from the same region (an adduct form folded in, a reading joined; the legend
numbers them) purple, their windows shaded and numbered in purple; the other components' orange. A
window where grey stands well above blue, purple and orange holds signal no fitted component
explains. The middle row shows the strongest charge states close up (up to four), titled, as shares
of the window's signal on the grid, with what the fit left there (fit residual), the signal the
reported templates leave unexplained and how far they overshoot it, and what the templates fitted
and then dropped hold there (from 1 %; they are drawn red, dotted); the orange dashes are every
reported component's templates added up. The bottom panel is the species' trace over the scans
(black, a dot per scan; blue, the base form where adduct forms were resolved), its regions shaded
with their bounds dotted, its apex red, and, for a species joined from pieces, each piece's trace as
it was linked, labelled with the id it was linked as, its region, mass, apex and how it was joined
(dash-dot: not counted); of a species of more than seven pieces the first six are so labelled and
the others drawn grey under one entry (`provenance.json` lists them all). The figure's title, the
only text it draws that holds names (the run's and the species'), is also kept as text in the PNG
file, where the leak check of name aliases reads it.

**Cost.** Collecting the provenance (per region, the templates of every fitted column added up once
and the windows measured, each window over all the region's scans' points at once; each region's
grid view and fitted models are then kept, in single precision, until the species are final, for
their figures) took from 0.06 s to about 1 s per run on the runs measured: 0.3 s on the Thermo runs
of the validation, 1.0 s on the synthetic oligonucleotide preset (`lcmsdeconv synth --preset oligo`:
3,003 charge states and adduct columns, rich in adduct forms and in components its fit merged from
parts); drawing the figures takes about 0.2 s each, so 300 of them about a minute. On the IP-RPLC NC
run of the validation (Thermo, 692 scans, 31 regions, comb estimator) `process` took 46 s with
`--provenance` against 27 s without (101 species: 0.3 s collecting, 20 s drawing), and its peak
memory rose from 324 to 351 MB: a region's scans, summed spectrum, the fit's residual and the
estimator's output are kept only while the region is processed. `provenance.json` holds every charge
state and adduct column of every component and every species' trace: about 2-2.7 kB per charge state
or adduct column recorded (0.6 MB for the 234 of a synthetic run of 7 species, 2.5 MB for the 1,000
of the NC run's 101, 7.4 MB for the 3,003 of the oligonucleotide preset); a figure is about 90-190
kB.

**Name aliases.** The run's, its traces' and the method's names in the provenance files come
aliased with the run and the method (`--alias`, `process_run(..., aliases=...)`), and so do the
scan IDs. The network's file name, taken from the command line or the method, is aliased by
`process`, and by `process_run(..., aliases=...)`; for a run and a method aliased already,
processed without `aliases=`, pass the pairs to `write_outputs(..., aliases=...)` (or
`write_provenance`, or call `result.provenance.alias_texts(aliases)`), or the file name stays as
given. `process`'s check of its outputs covers `provenance.json` and `provenance.html` by their
text and the figures by their names and the title each holds as text (what they draw comes from
the aliased names). A figure an earlier run drew shows that run's names: the check finds the
OLD in its title, among what else `--out` holds.

## Saturation warnings

Two independent checks run on every processed file:

- **Detector saturation** — flat-topped MS frames clipped at a common ceiling. Intensities and
  isotope ratios in those frames are unreliable.
- **Ionisation saturation** — the MS response falls at a peak's apex relative to its own flanks
  when compared with the UV trace. The column is delivering more analyte than the source can
  ionise, so the MS area under-reports the amount while the UV area does not.

Both name the affected peaks in the report. Quantify flagged peaks from UV, or dilute and
re-inject.

## Name aliases

Run files, traces and methods are often named after a project number that must not leave the lab.
A pair OLD=NEW makes lcmsdeconv show NEW wherever such a name holds OLD, in everything it writes
and prints, while the files keep their names on disk:

```bash
lcmsdeconv process LC-123456-S.mzMLb -m rplc_pos_protein -o results/ --alias 123456=保密项目1
lcmsdeconv process run.mzML -o results/ --alias-file aliases.txt
LCMSDECONV_ALIAS_FILE=~/aliases.txt lcmsdeconv gui
```

**Giving pairs.** `--alias OLD=NEW` and `--alias-file FILE` (both repeatable) are taken by
`process`, `gui`, `report`, `evaluate` and `synth`, and the environment variable
`LCMSDECONV_ALIAS_FILE` names a file every one of them reads, as does the desktop application
however it is started. Its value may be in double quotes, as Windows' "Copy as path" gives a path,
and may start with `~` or hold environment variables (`%USERPROFILE%` on Windows, which sets no
`HOME`; `$HOME` elsewhere); give an absolute path, as a relative one is read from whatever folder
lcmsdeconv starts in (an error then shows the folder it was looked for in). A variable named
`LCMSDECONV_ALIAS_FILE` with a space after the name, as cmd's `set LCMSDECONV_ALIAS_FILE = ...`
makes, stops the command with an error. The pairs
of `LCMSDECONV_ALIAS_FILE` come first, then those of each `--alias-file` in turn, then each
`--alias`; a later pair for one OLD replaces an earlier one. No pair is built into lcmsdeconv, and
no output records an OLD or a pair: `results.json` records only how many pairs were in effect
(`alias_pairs`, absent without any). To see that pairs are in effect, `process` prints `name
aliases: 2 pairs` first (unless `--quiet`), and the desktop application shows the same in its
status bar; neither shows anything without pairs. An alias file holding no pair (only comments,
say) aliases nothing, and a warning says so (in a dialog in the desktop application). Keep the
alias file itself where the outputs do not go. `train`, `export-onnx` and `methods` handle no run,
trace or method names: they take no pairs, ignore `LCMSDECONV_ALIAS_FILE` and print their paths as
given.

**The file.** UTF-8 (a byte order mark is accepted, and one between pairs or comments within a
line starts a pair there, as where two files were joined with `cat` or cmd's `copy a.txt+b.txt`,
and so are end-of-file marks, Ctrl-Z, at its end, as `copy` adds), LF, CRLF or CR line ends, one
pair per line:

```text
# project numbers: OLD=NEW
123456=保密项目1
PRJ-7 = Project seven
```

Each line is split at its first `=` (so NEW may hold `=`, OLD cannot), and the whitespace around OLD
and NEW is stripped. Notepad puts no line end after the last line: press Enter at its end before
you join the file to another or add a line to it otherwise, or the next pair lands on that line,
in the NEW of its pair, which then shows the next OLD wherever the first OLD is. A NEW holding `=`
is therefore warned of (`WARNING: alias file aliases.txt, line 3: NEW holds '=': where the line
holds two pairs ...`), and `process` prints the number of pairs in effect, which should be that of
the pairs in your files. Blank lines and lines whose first non-blank character is `#` are skipped
(an OLD starting with `#` is given with `--alias`); there are no comments at the end of a line. A
file that is missing or cannot be read, is not UTF-8 (a file saved as UTF-16 or in an ANSI code
page included: the error says how to save it as UTF-8; for lines added to a UTF-8 file in another
encoding, it says to type the pairs into a new file), or has a line without `=` (the error names a
full-width `＝` typed with an input method), with an empty OLD or NEW, with an invisible or control
character in OLD (a zero-width space or direction mark pasted with the number, a tab, NUL), with
U+FFFD in OLD or NEW (which an editor puts for a character it could not read), or with a
full-width or compatibility character in OLD (`１２３４５６` typed with an input method in
full-width mode, a no-break or ideographic space pasted from a document, a Roman numeral, a radical
pasted from a PDF file for the Chinese character it looks like) stops the command with an error
naming the file and the line, never quoting the line, which holds the text being hidden; the
desktop application shows the error in a dialog and does not start, as it may run without a
console. Such an OLD would not match the plain characters names are written in (where the names
hold such a character themselves, give an OLD without it). A NEW may hold format characters
(Persian's zero-width non-joiner, an emoji's zero-width joiner, a soft hyphen), but no control
character, line break or byte order mark. The errors read `Error: alias file aliases.txt, line 3: no '='
(expected OLD=NEW)`, `Error: alias file aliases.txt, line 2: OLD holds U+200B, an invisible or
control character` or `Error: alias file aliases.txt, line 1: OLD holds U+FF11, a full-width or
compatibility form of a character, which does not match the plain one: type it half-width (the
input method in English or half-width mode)`. The `--alias` pairs are checked the same way
(`Error: --alias #1: ...`). The file's path in such an error is shown with the pairs read before
it and those of `--alias`.

**Matching.** Case-sensitive, with the pairs and the names compared in Unicode's composed form
(NFC). At each position the longest OLD wins, in a single pass, so a NEW is never matched again. An
OLD that begins with a digit does not match where a digit comes right before it, and one that ends
with a digit does not match where a digit comes right after it: 123456 matches in `LC-123456-S`,
`LC123456`, `123456_2` and `123456.2`, but not in `1234567` or `0123456`.

**Where they apply.** To every name, path and text lcmsdeconv takes from the run, the method or the
command line and then writes or prints: the run's name and source path; the traces' names (the
UV trace of `--uv-csv` too), units and metadata; the run's metadata strings (Waters function
information, source files, instrument text, the lock-mass record); the method's name and
description, `report.title`, `instrument.kind`, `quant.calibration.amount_unit` and `weighting`,
each integration setting's `area_unit`, and the names and text values of its timed events; the
input and output paths in the console messages; the warnings and errors printed (Python warnings,
log records, the traceback of an unexpected error). So the report's text, titles, tables and
plots, `results.json`'s keys and values, the CSV files' names, headers and cells, and the desktop
application's status bar, legends, tables (the integration events' too), dialogs and the reports
it exports all show NEW. The aliases are applied at the source, to the run as it is read and the
method as it is loaded, never to formatted output: an area of 1234560 or a mass of 123456.7 stays
as it is. Text that only exists as text, a warning or an error quoting a file's path, is aliased
as text; a number in such a message that equals an OLD in full (a count of 123456 scans) is
replaced with it. Where aliasing makes two names equal (two traces named `UV 123456` and
`UV 654321` with both numbers aliased to one NEW, or a trace named `UV 保密项目1` already), no data
is lost: the later ones get ` (2)`, ` (3)`, ... (the peak-table CSV files' names are made unique
besides, as always). The method's references to the run's traces (`acquisition.uv_channel`, a
`quant.purity_signal` naming a trace) are resolved against the run as read, so they name the same
trace as without the pairs (`UV 保密项目1 (2)`, say), and one naming no trace still names none:
the numbers are those of a run processed without pairs.

**What is not renamed.** The files and directories you name (the run, `--out`, `--uv-csv`, the
method file) keep their names on disk; lcmsdeconv only shows them aliased. The method's settings
(keywords, adduct names and formulas, numbers, the model's path) are left as they are; so is a
setting that takes keywords (`instrument.kind`, `weighting`, a timed event's name or switch) where
its value is one (`tof`, `1/x`, `baseline_now`, `on`) or would become one, and a timed event's value
that is a number. Outputs written without the pairs are not rewritten: `lcmsdeconv report DIR`
shows the report's path aliased and warns where the report still holds an OLD, and processing again
with the pairs writes it aliased. The operating system's own file dialogs in the desktop
application show your folders as they are.

**The leak check.** After `process` has written its outputs with pairs set, it reads every file it
wrote, its name relative to `--out` and its text (`results.json`'s strings decoded, the report's
text with its entities decoded and its embedded images left out), for each OLD. An occurrence that
is part of a number, with a digit, or a `.` or `,` and a digit, right before or after it
(`0.123456`, `123456.7`), is not counted. For each file holding any, it prints one warning on
stderr naming the file, each OLD in the name shown as `***`, and the count, never the text found
(`WARNING: report.html still holds 2 occurrences of an alias's OLD text`). It then checks what else
`--out` holds, at any depth (an earlier run's outputs, there or in a folder in it), which it does
not rewrite or delete: the name of every file and folder, and the text of those named as outputs
are (`results.json`, `species.csv`, `report.html`, `peaks_*.csv`, `provenance.json`,
`provenance.html`): `WARNING: run1/peaks_UV_***.csv, which this run did not write, holds 2
occurrences of an alias's OLD text` (at most 10,000 entries are looked at, and no text above 50 MB
is read). An evidence figure (`evidence/species-*.png`, written with `--provenance`) is checked by
its name and by the title it holds as text in the PNG file (the run's and the species' names, as
drawn; its pixels are not read), among the files written and the others alike: a figure an
earlier run drew without the pairs shows that run's names in its title, and is flagged. Give
`--out` a fresh folder, or delete such files (an earlier run's `evidence/` with its provenance
files), before handing it on. The check never fails the command. A NEW that holds its OLD outside
a number trips it.

**Chinese names in the plots.** A NEW in Chinese (`保密项目1`) is drawn in the report's plots
through the same CJK font fallback as any trace or species name in Chinese: Microsoft YaHei or
SimHei on Windows, PingFang SC or Hiragino Sans GB on macOS, Noto Sans CJK SC, Source Han Sans SC
or WenQuanYi Zen Hei on Linux, the first of them installed; for a run processed with pairs, Noto
Sans CJK's TC, HK, JP and KR faces, each of which holds simplified Chinese, are tried after those
(a run without pairs is drawn as before). A Linux machine with none of these draws each Chinese
character in the plots as an empty box (`UV □□□□1`), and matplotlib warns
`Glyph 20445 (\N{CJK UNIFIED IDEOGRAPH-4FDD}) missing from font(s) DejaVu Sans`; the report's text
is unaffected (the browser draws it with its own fonts). Install a CJK font (`sudo apt install
fonts-noto-cjk` or `fonts-wqy-zenhei` on Debian or Ubuntu, `sudo dnf install
google-noto-sans-cjk-fonts` on Fedora), delete matplotlib's font cache (`rm -rf
~/.cache/matplotlib`) so that it finds the new font, and process the run again. (matplotlib before
3.11 reads only the first face of a font collection, Noto Sans CJK JP in Debian's and Ubuntu's
`fonts-noto-cjk`; that face draws simplified Chinese too.)

**On Windows.** The commands below are for cmd or Windows PowerShell 5.1 (the PowerShell of
Windows 10 and 11), as noted; type or paste each one as a single line.

1. Make the alias file as UTF-8, at an absolute path outside the folders you hand on or
   synchronise. In PowerShell (`Set-Content` does not make a missing folder, hence `New-Item`):

   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\lcmsdeconv" | Out-Null
   Set-Content -Encoding UTF8 -Path "$env:USERPROFILE\lcmsdeconv\aliases.txt" -Value '123456=保密项目1'
   ```

   To add a pair later:

   ```powershell
   Add-Content -Encoding UTF8 -Path "$env:USERPROFILE\lcmsdeconv\aliases.txt" -Value '654321=保密项目2'
   ```

   Both end the line they write with a line end. In Notepad, use File > Save as (文件 > 另存为),
   make the folder with New folder (新建文件夹) where it is not there yet, and pick Encoding UTF-8
   (编码 UTF-8); Notepad puts no line end after the last line, so press Enter at its end before
   you add a pair in any other way. With file name extensions shown in
   Explorer, check that the file is not `aliases.txt.txt`. PowerShell's `>` and `Out-File`,
   `Set-Content` and `Add-Content` without `-Encoding UTF8`, and cmd's `echo ... >>` write UTF-16
   or code page 936: such a file is refused as a rule (one whose text also happens to be valid
   UTF-8 is read with its Chinese garbled: check the names `process` prints), and so is a UTF-8
   file a line was added to that way (type its pairs into a new file).
2. Set the variable for your account, which needs no administrator: in cmd,
   `setx LCMSDECONV_ALIAS_FILE "%USERPROFILE%\lcmsdeconv\aliases.txt"`; in PowerShell,
   `setx LCMSDECONV_ALIAS_FILE "$env:USERPROFILE\lcmsdeconv\aliases.txt"`. Or search the Start menu
   for 环境变量 (environment), open Edit environment variables for your account (编辑帐户的环境变量;
   `rundll32 sysdm.cpl,EditEnvironmentVariables` opens it too), and under User variables (用户变量)
   click New (新建): name `LCMSDECONV_ALIAS_FILE`, value the file's full path, without quotes. Not
   through Advanced system settings > Environment Variables (高级系统设置 > 环境变量): on a
   standard account it asks for an administrator's password and then edits that administrator's
   variables, not yours.
3. For every user of the computer, as an administrator (in a console opened with Run as
   administrator, 以管理员身份运行): make the folder (`mkdir C:\ProgramData\lcmsdeconv` in cmd,
   `New-Item -ItemType Directory -Force C:\ProgramData\lcmsdeconv` in PowerShell), save the file
   there as in step 1 (users can read it there), and set the variable for the computer with
   `setx /M LCMSDECONV_ALIAS_FILE "C:\ProgramData\lcmsdeconv\aliases.txt"` or under System
   Properties > Advanced > Environment Variables > System variables (系统属性 > 高级 > 环境变量 >
   系统变量). Give that path itself: `%USERPROFILE%` in an administrator's console is the
   administrator's folder, which other users cannot read. A user variable of the same name (set
   earlier without `/M`) takes precedence for that user: delete it, in Edit environment variables
   for your account.
4. Programs started afterwards from the Start menu, the taskbar or Explorer see the variable, and
   so does a console opened from there. A console or terminal opened from a program that was
   already running (an editor, an AI assistant, a terminal window opened earlier) does not:
   restart that program, or sign out and in. To use the variable at once in one console, set it
   there too, for that console and the programs started from it: in cmd,
   `set "LCMSDECONV_ALIAS_FILE=%USERPROFILE%\lcmsdeconv\aliases.txt"`; in PowerShell,
   `$env:LCMSDECONV_ALIAS_FILE = "$env:USERPROFILE\lcmsdeconv\aliases.txt"`. Check it with
   `echo %LCMSDECONV_ALIAS_FILE%` (cmd) or `$env:LCMSDECONV_ALIAS_FILE` (PowerShell). The desktop
   application reads the file when it starts: restart it after editing the file.
5. Check that the pairs are in effect: `process` prints `name aliases: 1 pair`, the number of
   pairs in your files, and paths showing NEW; `results.json` holds `alias_pairs`; the desktop
   application's status bar shows the same count.
6. In a batch file or a Windows PowerShell 5.1 script (`.ps1`), give a NEW in Chinese in the alias
   file, made in Notepad or with the commands of step 1 typed or pasted in a console, rather than
   in the script (`--alias`, `Set-Content -Value`): cmd and Windows PowerShell 5.1 read a script
   saved without a byte order mark in code page 936, which garbles it. Alternatively, save the
   `.ps1` as UTF-8 with BOM (Notepad's 带有 BOM 的 UTF-8).

**Python.** `lcmsdeconv.aliases.Aliases` holds the pairs (without any it changes nothing), and
`lcmsdeconv.io.results.write_outputs` writes what `process` writes:

```python
from lcmsdeconv.aliases import Aliases
from lcmsdeconv.core.method import Method
from lcmsdeconv.io.mzml import read_mzml, read_uv_csv
from lcmsdeconv.io.results import write_outputs
from lcmsdeconv.process import process_run

aliases = Aliases.from_env() | Aliases.parse(["123456=保密项目1"])  # or .from_file(alias_path)
with aliases.console():  # warnings, log records and tracebacks printed aliased
    run = read_mzml(path)
    uv = read_uv_csv(uv_path)  # where the UV trace comes as a CSV file (--uv-csv)
    run.chromatograms[uv.name] = uv
    result = process_run(run, Method.load("rplc_pos_protein"), aliases=aliases)
    written = write_outputs(result, "results")  # results.json, species.csv, peaks_*.csv, report
print(aliases.apply("results/results.json"))
# the files written, then what else the folder holds (best a fresh one), as process checks them
for name, n in aliases.leaks(written, "results") + aliases.leaks_in("results", exclude=written):
    print(f"WARNING: {name} holds {n} occurrences of an alias's OLD text")
```

`Aliases.collect(files, pairs)` puts pairs together as the command line does (the file
`LCMSDECONV_ALIAS_FILE` names, then `files`, then `pairs`), `merged` or `|` combine two sets (the
later wins), `apply(text)` aliases a text, `leaks(files, folder)` and `leaks_in(folder, exclude)`
check files and what else a folder holds at any depth, and `alias_run(run)` and
`alias_method(method, run)` return aliased copies (the objects passed in are not changed), which
is what `process_run(..., aliases=...)` processes; it records the number of pairs in
`result.alias_pairs`. Give `alias_method` the run as read, before `alias_run`: it resolves the
method's trace references against it (without it, they are aliased as text, which names another
trace where aliasing made two trace names equal). Pass a run and a method aliased already to
`process_run` without `aliases=`: aliasing twice can differ from aliasing once. The network's file
name, which `process_run(..., provenance=True)` records, comes with neither: for such a run pass
the pairs to `write_outputs(..., aliases=aliases)` too, which applies them to the provenance's
texts only (or call `result.provenance.alias_texts(aliases)`); `process` does so itself.

Within `aliases.console()`, what Python prints as text is aliased. A warning is passed on, aliased,
to the warnings hook in effect, so `warnings.catch_warnings`, `pytest.warns` and
`logging.captureWarnings` still get it. A log record has its message, exception and stack text
and its other text fields (file, module, function and thread names, `extra` fields) aliased as it
is made and as a handler takes it; a record holding no OLD is left as it is. The traceback of an
exception no one catches, in any thread, an exception Python ignores (one raised in `__del__`, say)
and the message of `sys.exit("...")` are aliased too. An exception leaving the block is printed
aliased where nothing catches it, even after the block, and so is a group of such exceptions:
`sys.excepthook` and `threading.excepthook` keep a hook for this until it has printed every such
exception, for good where one is caught. Blocks open at once, nested or in other threads (one for
each worker of a pool, say), share the hooks, which apply the pairs of all of them and are put
back when the last block closes. Join a thread started within a block before the block ends: one
that fails after it is printed as Python prints it.

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
