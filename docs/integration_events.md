# Integration events

Integration follows the Agilent ChemStation / OpenLab CDS model: a set of *initial* parameters
that apply to the whole signal, plus a *timed event table* that changes them, or forces a
particular baseline treatment, from a given retention time onwards.

Each signal (UV, TIC, BPC, and every deconvolved ion chromatogram) has its own event table in
the method, keyed by `tic`, `uv` or `deic`.

## Initial parameters

| Parameter | Meaning |
| --- | --- |
| `slope_sensitivity` | Signal change per minute at which a peak is considered to start. `0` derives it from the noise of the first derivative. Lower values detect smaller peaks and more noise. |
| `peak_width` | Expected width at half height, in minutes. Sets the smoothing and the minimum number of points a peak must span. Too small detects noise; too large merges peaks. |
| `area_reject` | Peaks with less area are not reported. |
| `height_reject` | Peaks lower than this are not reported. |
| `area_pct_reject` | Peaks below this percentage of the total area are not reported. |
| `max_area`, `max_height` | Upper limits; peaks above them are dropped (useful to exclude a solvent front). |
| `shoulders` | `off`, `drop` or `tangent`: how unresolved shoulders on a peak flank are treated. |
| `baseline_all_valleys` | When on, the baseline is reset at every valley (valley-to-valley) instead of running under the whole cluster. |
| `tail_skim_height_ratio` | Parent-to-rider height ratio above which a rider on the tail is tangent-skimmed. `0` disables skimming. |
| `front_skim_height_ratio` | The same for a rider on the leading edge. |
| `skim_valley_ratio` | The valley between parent and rider must exceed this percentage of the rider height before skimming applies. |
| `detect_negative_peaks` | Also integrate peaks below the baseline. |
| `fixed_peak_width` | Keep the peak-width parameter fixed instead of tracking it. |
| `area_unit` | `signal*s` (default) or `signal*min`. |

## Timed events

Add these to `timed_events` as `{time, event, value}`. Value-changing events take a number;
switches take `on`/`off`; point events take no value.

| Event | Effect |
| --- | --- |
| `integration_off` / `integration_on` | Stop and resume peak detection. Everything between is ignored. |
| `slope_sensitivity`, `peak_width`, `area_reject`, `height_reject`, `area_pct_reject`, `max_area`, `max_height` | Change that parameter from this time onwards. |
| `baseline_now` | Force a baseline point at this time. |
| `baseline_hold` (`on`/`off`) | Hold the baseline at its level at the start time until switched off, instead of following the signal. |
| `baseline_at_valleys` / `baseline_all_valleys` | Switch valley-to-valley baselines on or off. |
| `baseline_next_valley` | Place the next baseline point at the following valley. |
| `baseline_back` | Move the baseline point back to the preceding valley. |
| `tangent_skim`, `rear_tangent_skim`, `front_tangent_skim` | Enable skimming of riders on the tail or the front of a parent peak. |
| `split_peak` | Drop a vertical line at this time, splitting one peak into two. |
| `solvent_peak` | Mark the peak containing this time as a solvent peak (code `F`). |
| `peak_sum_slice` (`on`/`off`) | Report everything between the two times as a single summed peak, ignoring peak detection inside it. |
| `negative_peak` | Enable or disable negative-peak detection. |
| `fixed_peak_width` | Fix or release the peak-width tracking. |
| `shoulders` | Change the shoulder mode. |

## Baseline codes

Two letters: the first describes the peak start, the second the end.

| Code | Meaning |
| --- | --- |
| `B` | Baseline. The signal returned to baseline on that side. |
| `V` | Valley. The peak is separated from its neighbour by a drop line. |
| `P` | Penetrated. The signal dipped below the constructed baseline. |
| `T` | Tangent-skimmed on that side. |
| `M` | Manually integrated. |
| `F` | Forced or solvent peak. |
| `S` | Shoulder. |

`BB` is a fully baseline-resolved peak; `BV` and `VB` are the first and last peaks of a cluster;
`VV` is a peak with a neighbour on each side.

## How valley clusters are integrated

Peaks that are not baseline-resolved form a cluster. One baseline is drawn from the start of the
first peak to the end of the last, and the peaks inside are separated by vertical drop lines at
the valleys. This is Agilent's default and it conserves the cluster's total area: on two
overlapping Gaussians, each individual area is within a few percent while the sum is exact.
Setting `baseline_all_valleys` switches to valley-to-valley baselines, which assigns less area
to the trailing peaks.

## Manual integration

`lcmsdeconv.chrom.integrate.manual_integrate(chrom, start, end, baseline_start, baseline_end)`
integrates a user-drawn peak and returns a peak with code `MM`. The desktop application exposes
this by dragging peak boundaries.
