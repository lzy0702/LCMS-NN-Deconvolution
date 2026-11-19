# Probes behind `docs/iwsp_transformer.md`

Prototypes and measurements made while designing the hand-written (iWSP) transformer. None of
them changes the package: each change is patched into the running process only, and real runs
are scored with `scripts/validate_oligo_runs.py`'s truth and scoring. They are evidence for the
design, not the harness it describes (that is stage 0 of its plan).

Every probe that reads single scans resamples them with `scan_grid.resample_scan`, which caps each
profile point's width at its smaller neighbour spacing (at most three median spacings). The
package's `resample_profile` lets the edge point of a vendor-thresholded segment stand for half
the gap beside it (`tile_statistics.py gaps`); the probes that run the pipeline itself
(`real_run_probes.py validate` and `trace`, the predictor probes) measure the package as it runs,
unless `--resample` says otherwise.

`RAW` below stands for the four Thermo runs of `docs/benchmarks.md`
(`IPRPLC_GalNAc_conjugated_mix.raw IPRPLC_NC_mix.raw IPHILIC_GalNAc_conjugated_mix.raw
IPHILIC_NC_mix.raw`); `GALNAC`, `NC`, `HGALNAC` and `HNC` for each of them in that order. Each
four-run validation takes about 30 s.

| Command | Reproduces (design section) |
| --- | --- |
| `python scripts/iwsp_probes/comb_transformer_v0.py synthetic --tof 64 --orbitrap 64` | version 0 equals `CombPredictor` in every field on 128 synthetic spectra at two charge ceilings, and its time against the comb's (4.3) |
| `python scripts/iwsp_probes/comb_transformer_v0.py real RAW` | the same on every predictor call of the four real runs, identical species tables; active bins per call; time per call (4.1, 4.3) |
| `python scripts/iwsp_probes/comb_transformer_v0.py leakage NC` | softmax leakage changes 823 top charges without the floor neuron at beta = 400, none with it (4.2) |
| `python scripts/iwsp_probes/physics_checks.py idioms` (`pointer`, `interp`, `isobaric`, `consensus`) | the comparator and select idioms; the pointer score in float64 and float32 and its leakage bounds; the interpolation head with relative and absolute positions; FLP against FLP + 0.984 Da; mean-shift steps (4.2, 4.5) |
| `python scripts/iwsp_probes/pointer_attention.py check GALNAC` | the ladder part as softmax pointer heads (beta 200) with a sink, against the comb |
| `python scripts/iwsp_probes/pointer_attention.py inrange` | the in-range ladder mean per bin on the 96-spectrum benchmark (3) |
| `python scripts/iwsp_probes/real_run_probes.py validate RAW` | the reference: 13/20, 5/29, 39 reported, 7 unconfirmed (3) |
| `... validate --resample capped RAW`, `... validate --resample scans RAW` | the pipeline with its resampling patched in-process (not the fix: the envelope gate is largely disabled; section 2): 13-14/20, 9-10/29, 154-158 reported, 52-55 unconfirmed, and the changed amounts (2) |
| `... validate --rule oracle RAW` (and `--single-charge`) | the charge oracle: 13/20 and 6/29 (16/20 and 8/29); the 1 908 Da fragment's reported time (3, 7) |
| `... validate --single-charge RAW` | the comb accepting single charge states: 15/20, 7/29, 74, 19, and the reports it adds (3, 4.6, 7) |
| `... validate --rule inrange RAW` | the in-range ladder mean on the real runs (3) |
| `... validate --rule adjacent RAW` (and `--single-charge`) | the adjacent-isotopologue condition in the decoder's isotope check (3) |
| `... trace RAW`, `... trace --rule oracle --single-charge RAW` | where the missed species are lost; the nine envelope-gate rejections; each run's TIC regions (3) |
| `... snr RAW` | the missed species' signal-to-noise ratios, resampled scan by scan and in the package's own view of the region (3) |
| `python scripts/iwsp_probes/isotope_chains.py RAW` | the time-resolved isotope-chain test on known species, unconfirmed reports and random positions (3, 4.5) |
| `python scripts/iwsp_probes/isotope_filter.py real --variant c2b RAW` (`--variant c2`, `--variant comb`) | the isotope matched filter C2b (and C2: absolute half-spacing inhibition, margin s/4) on the real runs, with the truth species each misses per run (3, 4.4) |
| `python scripts/iwsp_probes/isotope_filter.py bins` | C2 and C2b per bin on the 96-spectrum benchmark (3, 4.4) |
| `python scripts/iwsp_probes/coelution_gate.py --theta 0.5 RAW` (0.3, 0.7) | the untuned co-elution gate on the real runs (3, 4.4) |
| `python scripts/iwsp_probes/tile_statistics.py scans GALNAC --at 19.25 16.1 27.0 12.0`, `... scans NC --at 18.21 27.0 10.0`, `... scans HGALNAC --at 16.4`, `... scans HNC --at 13.41` | scan rate and signal bins per scan and per 24-scan tile (2) |
| `... scans GALNAC --scans 48 --at 19.25 14.5 27.0`, `... evidence GALNAC --scans 48 --at 19.25 14.5 27.0` | tokens per 48-scan tile, their one- or two-scan share and intensity, version 0's and the comb's time on a tile (4.1, 4.7, 5.4) |
| `... widths GALNAC --species 6793.98@19.3 5162.25@20.2`, `... widths NC --species 5162.25@18.21`, `... widths HGALNAC --species 6793.98@16.40`, `... widths HNC --species 5162.25@13.41` | elution widths and charge-state shares of the main peaks (2) |
| `... widths GALNAC --species 6448.71@18.71 7139.25@20.34 6777.92@19.21`, `... widths NC --species 4832.98@17.54 5146.21@18.09 4503.71@16.55`, `... widths HGALNAC --species 6448.71@15.94 7139.25@16.98 6777.92@16.98 5162.25@13.72`, `... widths HNC --species 4832.98@13.03 5146.21@14.11 4503.71@12.60` | impurity widths (2) |
| `... profiles GALNAC` | trace correlations of FLP's charge states and of the other bins in its tile (2, 4.4) |
| `... evidence GALNAC --at 19.25 16.1 27.0 12.0` | one- and two-scan events and the origin of the comb's ladder evidence (2) |
| `... envelopes GALNAC --at 19.21 18.71 20.3` | known species' envelope bins: version 0, its charge confusion, C2b, C2 and ladder variants (3, 4.5) |
| `... traces NC --species 5162.25@18.21 --ref-z 4 --scans 41` | NC's charge-state traces and their log-SNR (2, 4.4) |
| `... pairs NC --scans 32 --at 18.21 27.0 10.0` | trace correlations of random peak pairs and of chance ladder pairs (2) |
| `... wash GALNAC --scans 96 --at 27.0`, `... wash NC --scans 96 --at 27.0` | the column wash: breadth, z = 1 partners, apex spread, co-variation (2) |
| `... reelution NC --species 5162.25@18.21`, `... reelution GALNAC --species 6793.98@19.21` | later elutions of NC and FLP (2) |
| `... gaps GALNAC --at 19.29 27.0`, `... gaps NC --at 18.21`, `... gaps HGALNAC --at 15.91`, `... gaps HNC --at 12.5` | points that stand for a gap under `resample_profile`, in scans and in the pipeline's region sums; the region sum's noise level three ways; the ramps `Run.sum_frames` interpolates across gaps (2) |
| `... forms RUN` (each run) | the co-eluting +201.4 Da, Na, K and -16.06 Da forms of the paper species (2) |
| `... events RUN` (each run) | elution-event tokens of a whole run and their one- or two-scan share (4.1) |

The per-bin benchmarks draw their spectra as `scripts/eval_charge_bins.py` does (seed
20 000 000); `comb_transformer_v0.py synthetic` draws its own (seed 7).
