# Offline benchmark: the charge networks and the pipeline on your computer

## 快速开始（中文）

1. **解压**：把 `lcmsdeconv_bench_<日期>.zip` 解压到一个短路径，例如 `C:\lcms`，得到
   `C:\lcms\lcmsdeconv\bench`。电脑不能上网时，把 wheels 包
   （`lcmsdeconv_bench_wheels_cp312-win_amd64_<日期>.zip`）解压到同一个位置，它会填好
   `bench\wheels\cp312-win_amd64`。不要放在 OneDrive 同步的文件夹里（桌面、文档）。
2. **Python 3.12**：没有的话用 `winget install -e --id Python.Python.3.12` 安装，或从
   python.org 下载安装程序（勾选 "Add python.exe to PATH"）。离线时必须是 3.12（wheels 包只适用于
   3.12），请事先准备好安装程序；能上网时 3.13、3.11 也可以。setup 提示缺少 Visual C++ 运行库时，
   安装 `vc_redist.x64.exe`（https://aka.ms/vs/17/release/vc_redist.x64.exe，离线时请事先下载）。
3. **安装（一次）**：双击 `bench\setup_bench.bat`。它建立 `.venv-bench` 环境、安装依赖、
   用 DirectML（GPU）和 CPU 对每个网络各做一次短推理并比较结果，写出 `bench\machine.json`。
   可以重复运行。
4. **Thermo 原始数据**：把四个 `.raw` 文件（`IPRPLC_NC_mix.raw`、
   `IPRPLC_GalNAc_conjugated_mix.raw`、`IPHILIC_NC_mix.raw`、`IPHILIC_GalNAc_conjugated_mix.raw`）
   放进 `data\thermo_raw`（子文件夹也可以），或运行时用 `--thermo-dir 文件夹` 指定。
5. **运行**：先 `bench\run_bench.bat --smoke`（几分钟的端到端检查），再 `bench\run_bench.bat`
   （完整测试）。中断（Ctrl-C、关机）后再次运行即可继续，已完成的任务不会重做。运行期间请接通
   电源、不要最小化窗口（程序会阻止 Windows 睡眠）。
6. **上传**：结束时窗口里显示的 `bench\results\<id>_upload.zip`。本机的总结在
   `bench\results\<id>\summary.md`。

保密样品1（全硫代 DNA，C252H328N72O138P24S24，碱基组成 T13 C4 G6）的测试只在包内有
`bench\private\confidential_sample_1\` 时运行；它的结果只写在本机的 `bench\results` 和上传包里。

## What is measured

Everything runs on this package's own code (`src\lcmsdeconv`), with the comb (no network) as
the baseline and every network found (below) beside it:

- **thermo**: the pipeline on each of the four Thermo runs (method `orbitrap_iprp_neg_oligo`),
  once per predictor, scored against the paper's and the SI's species: species found, species
  reported, reports that nothing confirms, and the time taken.
- **bars**: the networks against the go/no-go bars, beside the comb: charges read on the real
  envelopes of the Thermo runs, lone envelopes at 4512.8, 1884.5, 3206.4 and 6011.5 Da,
  synthetic envelopes (four frame and charge-model settings), the reports only a network makes,
  and each network's validation scores from its training console log when one is next to it.
  The table is `bars\bars.md` in the results.
- **rda** (only when the package holds `bench\private\confidential_sample_1\`): the Waters RDa
  runs of Confidential sample 1 (fully phosphorothioated DNA, C252H328N72O138P24S24, base
  composition T13 C4 G6), once per run, predictor and method variant, then the comparison
  tables.
- **time and memory**: every job's seconds and peak memory (`timing.json`), the computer
  (`machine.json`: processor, cores, RAM, GPU, the onnxruntime providers that work, DirectML
  against the processor on a short inference).

## Requirements

- Windows 10 or 11, 64-bit (Linux works too: `python3 bench/setup_bench.py`, then
  `.venv-bench/bin/python bench/run_bench.py`).
- Python 3.12. Without internet it has to be 3.12: the wheels package holds the dependencies for
  Python 3.12 only. With internet, 3.13 also works (the same checked versions, from PyPI), and so
  does 3.11 (newer versions of some dependencies than the ones checked). Setup prefers 3.12, and
  an environment whose install failed for want of internet is made again with 3.12 once it is
  installed.
- The Microsoft Visual C++ 2015-2022 runtime (x64), which onnxruntime and numba need. Most
  computers have it (games and drivers install it); setup says when it is missing. Without
  internet, bring `vc_redist.x64.exe` along (https://aka.ms/vs/17/release/vc_redist.x64.exe).
- A GPU that DirectML supports (any DirectX 12 GPU, such as the RTX 4090; keep its driver up to
  date). CUDA is not needed. Without a working GPU the networks run on the processor.
- About 2 GB of free disk space. No internet is needed when the wheels package is unzipped.

The environment `.venv-bench` is separate from the training package's `.venv`
(onnxruntime-directml and onnxruntime cannot share an environment), so both packages can live
in the same folder.

## How to run

1. `bench\setup_bench.bat` (once; safe to run again). It finds Python 3.12, creates
   `.venv-bench` in the package's top folder, installs the dependencies at the versions the
   package was checked with (from `bench\wheels\<tag>` without internet, else from PyPI), checks
   that `lcmsdeconv` imports from `src` and that numba and onnxruntime load, lists onnxruntime's
   providers, runs every network of `bench\models` on DirectML and on the processor, compares
   the two and writes `bench\machine.json`, which names the provider the benchmark uses (DirectML
   when it reads every network as the processor does). Its output is also in
   `bench\setup_bench.log`. Options: `--recreate` (a fresh environment), `--source pypi` or
   `--source wheels`, `--python PATH`.
2. `bench\run_bench.bat --smoke`: every suite shrunk (one Thermo run, two runs of Confidential
   sample 1, a few frames, one lone mass), to see everything work end to end. Its results go
   to `bench\results\<id>_smoke`.
3. `bench\run_bench.bat`: the whole benchmark. The jobs run in parallel within the budgets
   (by default all logical cores but two, 4 threads per job, and 70 % of the RAM counted by
   each job's peak estimate); the window shows each job as it starts and ends. A job that fails
   does not stop the others: its log is in the results, and the next run retries it. While it
   runs, Windows is kept from going to sleep (the screen may still turn off) and the jobs are
   kept at full speed; still, keep the laptop plugged in and the window open rather than
   minimized, and the lid open. If the computer sleeps after all, the window says so and
   `timing.json` marks the jobs whose times include the pause.

Run `run_bench.bat` again after Ctrl-C, a crash or a reboot: jobs whose output exists are kept,
the rest run (the tables are rebuilt when anything under them ran again, and an output made with
other networks or Thermo runs than the ones found now is redone). `--dry-run` shows which jobs
would run. The results folder `<id>` is named after the package's build, date and time (with
`_smoke` for a smoke run, and `_<provider>`, such as `_cpu`, when `--providers` names another
provider than the one setup chose), so a new package starts afresh.

Options (all optional; `run_bench.bat --help` lists them):

| option | meaning |
| --- | --- |
| `--smoke` | a few minutes' end-to-end check |
| `--suites thermo,bars,rda` | run only these suites |
| `--thermo-dir DIR` | where the Thermo `.raw` files are (default `data\thermo_raw`) |
| `--rda-dir DIR` | where the RDa `.mzMLb` runs are (default `bench\private\confidential_sample_1\data`) |
| `--model LABEL=PATH.onnx` | evaluate this network too (its `.json` next to it); repeatable |
| `--no-default-models` | only the `--model` networks |
| `--providers auto/dml/cpu` | where the networks run (default: setup's choice); a GPU provider setup found not working is refused (`cuda` needs onnxruntime's CUDA build, which setup does not install) |
| `--threads N` | threads of all jobs together (default: logical cores - 2) |
| `--threads-per-job N` | threads of each job (default 4) |
| `--mem-gb G` | memory of all jobs together, by their peak estimates (default 0.7 x RAM) |
| `--results-id NAME` | the results folder under `bench\results` |
| `--dry-run` | show the settings and the jobs, run nothing |

A relative path in an option is taken from the folder you run the command in; the window shows
the full paths used. Settings can also go into `bench\config.local.json` (the keys of
`bench\config.json`; a relative path there is taken from the package's top folder).

## Where the Thermo .raw files go

Put the four files in `data\thermo_raw` in the package's top folder (next to `bench` and `src`),
or anywhere and give the folder with `--thermo-dir`. Subfolders are searched, the names'
case does not matter and an upload prefix such as `4935a102-` before the name is ignored:

- `IPRPLC_NC_mix.raw`
- `IPRPLC_GalNAc_conjugated_mix.raw`
- `IPHILIC_NC_mix.raw`
- `IPHILIC_GalNAc_conjugated_mix.raw`

Runs that are missing are left out (the summary says so); without any, the thermo suite and the
real-envelope parts of bars are skipped.

## How long it takes

On the development machine: 4 cores, the processor only (no GPU), each job on 2 threads, one
job at a time. The full run was not timed there: its time is the sum of its jobs as measured
there (the smoke run's jobs, and the same work at full size in the evaluations this benchmark
ports).

| run | there |
| --- | --- |
| `setup_bench`, first time | half a minute to install from the wheels, then the check |
| `run_bench --smoke`, one network | 7.5-8 minutes (21 jobs) |
| `run_bench --smoke`, two networks | 12 minutes (26 jobs) |
| `run_bench`, two networks | about 80 minutes (89 jobs, estimated; two thirds of it the runs of Confidential sample 1) |
| each further network | about 35 minutes more (estimated) |

A 16-24 core laptop runs several jobs at once and DirectML runs the networks, so expect a
fraction of these times. What each job took on your computer is in `timing.json` and in the job
table at the end of `summary.md`.

## What to upload

`bench\results\<id>_upload.zip`, named in the window at the end. It holds the results folder
(the summary, the tables, each job's record and log, `timing.json`, `config.json`) with
`machine.json` and the setup log, and no raw data (`.raw` and `.mzMLb` files are left out).
Results of Confidential sample 1 are in it as well: send it only the way you send us that
sample's data.

## Adding a network

Copy the network's `.onnx` file and its `.json` file (same name, written next to it by the
export) into `bench\models`, then run `run_bench.bat` again: the new network's jobs run, the bars
jobs that read every network at once (real, lone and synthetic envelopes, network-only reports)
run again with all of them, and the tables are rebuilt with it. A network replaced by another
file of the same name is evaluated again as well. The label is the file name without `ladder_`
(`ladder_v6_comb.onnx` becomes `v6_comb`). Networks the training package wrote to `runs\` in
the same folder are found as well (not its smoke networks, nor one marked `.unchecked`), and
`--model LABEL=PATH.onnx` adds one from anywhere. A training console log
(`<name>.console.log`) next to the network adds its validation scores to the bars table. The
same file found twice is evaluated once.

## When something goes wrong

- **Setup cannot install without internet**: the wheels package is for Python 3.12 only.
  Install Python 3.12 (the installer brought along) and run `setup_bench.bat` again: it makes
  the environment again with 3.12 and installs from `bench\wheels` (unzip the wheels package
  first if `bench\wheels\cp312-win_amd64` is missing).
- **The Visual C++ runtime is missing** (setup says so, or the check fails with
  `DLL load failed while importing onnxruntime_pybind11_state`): install the Microsoft Visual
  C++ 2015-2022 Redistributable (x64) with `winget install -e --id Microsoft.VCRedist.2015+.x64`
  or with `vc_redist.x64.exe` from https://aka.ms/vs/17/release/vc_redist.x64.exe (download it
  on another computer without internet), then run `setup_bench.bat` again.
- **DirectML does not work** (setup says so; the networks then run on the processor): update
  the graphics driver and run `setup_bench.bat` again. `--providers cpu` forces the processor;
  `--providers dml` is refused until setup finds DirectML working.
- **Out of memory**: lower `--mem-gb` or `--threads`; the peak memory of every job is in
  `timing.json`, the estimates the budget uses in `summary.md`.
- **A job failed**: its log is named in the window and in `summary.md`; run `run_bench.bat`
  again to retry the failed jobs only.
- **"Another run_bench is working on ..."**: a second window runs the same results folder; wait
  for it, or delete `bench\results\<id>\.lock` if none runs.
