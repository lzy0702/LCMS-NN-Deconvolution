# Probes behind `docs/iwsp_transformer.md`

Prototypes and measurements made while designing the hand-written (iWSP) transformer. None of
them changes the package: each change is patched into the running process only, and real runs
are scored with `scripts/validate_oligo_runs.py`'s truth and scoring. They are evidence for the
design, not the harness it describes (that is stage 0 of its plan).

`RAW` below stands for the four Thermo runs of `docs/benchmarks.md`
(`IPRPLC_GalNAc_conjugated_mix.raw IPRPLC_NC_mix.raw IPHILIC_GalNAc_conjugated_mix.raw
IPHILIC_NC_mix.raw`); `GALNAC` for the first. Each four-run validation takes about 30 s.

| Command | Reproduces (design section) |
| --- | --- |
| `python scripts/iwsp_probes/comb_transformer_v0.py synthetic --tof 64 --orbitrap 64` | version 0 equals `CombPredictor` in every field on 128 synthetic spectra at two charge ceilings (4.3) |
| `python scripts/iwsp_probes/comb_transformer_v0.py real RAW` | the same on every predictor call of the four real runs; identical species tables (4.3) |
| `python scripts/iwsp_probes/comb_transformer_v0.py leakage IPRPLC_NC_mix.raw` | softmax leakage changes 823 top charges without the floor neuron at beta = 400, none with it (4.2) |
| `python scripts/iwsp_probes/pointer_attention.py check GALNAC` | the ladder part as softmax pointer heads with a sink, against the comb (4.2) |
| `python scripts/iwsp_probes/pointer_attention.py inrange` | the in-range ladder mean per bin on the 96-spectrum benchmark (3) |
| `python scripts/iwsp_probes/real_run_probes.py validate --rule comb RAW` | the reference: 13/20, 5/29, 39 reported, 7 unconfirmed (3) |
| `... validate --rule oracle RAW` (and `--single-charge`) | the charge oracle: 13/20 and 6/29 (16/20 and 8/29) (3) |
| `... validate --rule inrange RAW` | the in-range ladder mean on the real runs (3) |
| `... validate --rule adjacent RAW` (and `--single-charge`) | the adjacent-isotopologue condition in the decoder's isotope check (3) |
| `... trace RAW`, `... trace --rule oracle --single-charge RAW` | where the missed species are lost (3) |
| `... snr RAW` | the missed species' signal-to-noise ratios (3) |
| `python scripts/iwsp_probes/isotope_chains.py RAW` | the time-resolved isotope-chain test on known species, unconfirmed reports and random positions (3, 4.5) |
| `python scripts/iwsp_probes/isotope_filter.py real --variant c2b RAW` (and `--variant c2`) | the isotope matched filter C2b (and C2, with a decision margin) on the real runs (3, 4.4) |
| `python scripts/iwsp_probes/isotope_filter.py bins` | C2 and C2b per bin on the 96-spectrum benchmark (3) |
| `python scripts/iwsp_probes/coelution_gate.py --theta 0.5 RAW` (0.3, 0.7) | the untuned co-elution gate on the real runs (3, 4.4) |
| `python scripts/iwsp_probes/tile_statistics.py scans GALNAC --at 19.25 16.1 27.0 12.0` | scan rate and signal bins per scan and per tile (2) |
| `... widths GALNAC --species 6793.98@19.3 5162.25@17.0` | elution widths (2) |
| `... profiles GALNAC` | trace statistics of FLP's tile and two others (2) |
| `... evidence GALNAC --at 19.25 16.1 27.0 12.0` | spikes and the origin of the comb's ladder evidence (2) |
| `... envelopes GALNAC --at 19.21 18.71 20.3` | four ladder variants on known species' envelope bins (3) |
| `... traces IPRPLC_NC_mix.raw --species 5162.25@18.21 --ref-z 4 --scans 41` | NC's charge-state traces (4.4) |
| `... scans GALNAC --scans 48 --at 19.25 14.5 27.0`, `... evidence GALNAC --scans 48 --at 19.25 14.5 27.0` | tokens per 48-scan tile and their one- or two-scan share (4.1) |
| `... forms RUN` (each run) | the co-eluting +201.4 Da, Na, K and -16.06 Da forms of the known species (2) |
| `... events RUN` | elution-event tokens of a whole run and their spike share (4.1) |
| `... pairs IPRPLC_NC_mix.raw --scans 32 --at 18.21 27.0 10.0` | trace correlations of random peak pairs and of chance ladder pairs (2) |

The per-bin benchmarks draw their spectra as `scripts/eval_charge_bins.py` does (seed
20 000 000); `comb_transformer_v0.py synthetic` draws its own (seed 7).
