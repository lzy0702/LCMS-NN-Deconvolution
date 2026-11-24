"""What the training machine offers: cores, memory, and how many data workers fit a budget."""

from __future__ import annotations

import os

#: memory of one data worker generating spectra (Python, PyTorch, numba and a spectrum in work):
#: measured at 0.8 GB on Linux and 0.9 GB on Windows (private working set); 1.2 stays on the safe
#: side until gpu_training/probe.py has measured it on the computer at hand
WORKER_GB = 1.2
#: memory of the training process itself: PyTorch with a CUDA context, the model, the validation
#: spectra kept in memory and the batches in flight
MAIN_GB = 3.5
#: kept free below a memory budget, so that the budget is never reached in normal running
MARGIN_GB = 1.5
#: the RAM budget of gpu_training's setup and train_all when none is given
#: (default_ram_budget_gb): half the computer's memory, the other half left to the system and
#: other programs; at least BUDGET_MIN_GB, the fixed default of earlier packages, which a
#: computer of 20 to 32 GB keeps; at most BUDGET_MAX_GB, which holds the training process and 40
#: workers of 1 GB, more than the processor's cap (cpu_workers) of any desktop processor; and
#: never more than the memory less SYSTEM_GB, which binds on a computer of less than 20 GB. There
#: it departs from earlier packages on purpose: 12 GB on a computer of 16 GB, where their 16 GB
#: left the system nothing and the computer would page before the budget was reached
BUDGET_MIN_GB, BUDGET_MAX_GB, SYSTEM_GB = 16.0, 48.0, 4.0


def physical_ram_gb() -> float | None:
    """Physical memory of this computer in GB as psutil reads it (training installs psutil), or
    None without psutil."""
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:  # not installed, or not readable on this platform
        return None


def default_ram_budget_gb(total_gb: float | None = None) -> float:
    """The RAM budget of the training process and its data workers together when gpu_training's
    setup or train_all is given none, in whole GB: half of ``total_gb`` (the computer's physical
    memory, :func:`physical_ram_gb`, if not given), within BUDGET_MIN_GB and BUDGET_MAX_GB and
    at most ``total_gb`` less SYSTEM_GB; BUDGET_MIN_GB where the memory cannot be read. 4 GB on a
    computer of 8 GB, 12 on 16, 16 on 20 to 32, 32 on 64 and 48 on 96 or more: on 96 GB room for
    every worker the processor is given (on a processor of 24 cores, 20 of 0.9 GB each with a
    training process of 2.4 GB)."""
    total = physical_ram_gb() if total_gb is None else total_gb
    if not total or total <= 0:
        return BUDGET_MIN_GB
    budget = min(BUDGET_MAX_GB, max(BUDGET_MIN_GB, total / 2), total - SYSTEM_GB)
    return float(max(1, int(budget + 0.5)))


def half_ram_gb(total_gb: float | None = None) -> float:
    """Half of ``total_gb`` (the computer's physical memory, :func:`physical_ram_gb`, if not
    given), 8 GB where the memory cannot be read: the budget :func:`auto_workers` takes when given
    none, which is the default of scripts/train_charge_net.py without --max-ram-gb, as it always
    was."""
    total = physical_ram_gb() if total_gb is None else total_gb
    return 0.5 * (total if total and total > 0 else 16.0)


def cpu_cores() -> int:
    return os.cpu_count() or 2


def cpu_workers(cores: int | None = None) -> int:
    """The processor's cap on data workers: all cores but two of eight or fewer, four of more,
    left to the training process, its pinned-memory thread and the system."""
    cores = cpu_cores() if cores is None else cores
    return max(1, cores - 2 if cores <= 8 else cores - 4)


def ram_workers(budget_gb: float, worker_gb: float = WORKER_GB, main_gb: float = MAIN_GB,
                margin_gb: float = MARGIN_GB) -> int:
    """Data workers of ``worker_gb`` each that ``budget_gb`` holds beside the training process
    (``main_gb``) and the margin; 0 or less where not even one does."""
    return int((budget_gb - main_gb - margin_gb) // max(worker_gb, 0.1))


def auto_workers(ram_budget_gb: float | None = None, worker_gb: float = WORKER_GB,
                 main_gb: float = MAIN_GB, margin_gb: float = MARGIN_GB) -> int:
    """Data workers that fit the processor and the memory: the processor's cap
    (:func:`cpu_workers`) and, within ``ram_budget_gb``, as many workers as the memory left by
    the training process and a margin holds; at least one. Without a budget half the computer's
    memory (:func:`half_ram_gb`), as scripts/train_charge_net.py has always taken without
    --max-ram-gb; gpu_training's setup and train_all pass theirs (:func:`default_ram_budget_gb`
    unless told another)."""
    budget = ram_budget_gb if ram_budget_gb else half_ram_gb()
    return max(1, min(cpu_workers(), ram_workers(budget, worker_gb, main_gb, margin_gb)))


def process_ram_gb(proc) -> float:
    """RAM a process holds for itself, as the task manager's memory column shows it: its unique
    set (on Windows the private working set). Memory merely reserved - Windows charges every
    process that loads the CUDA libraries of PyTorch several GB of commit it never touches - and
    libraries shared between processes are not counted. ``proc`` is a psutil.Process."""
    try:
        return proc.memory_full_info().uss / 2**30
    except Exception:  # access denied, or a platform without it: the resident set
        return proc.memory_info().rss / 2**30


def single_threaded_workers() -> None:
    """One thread per data worker: the workers are processes, and numerical libraries that
    each start a thread per core would make them fight over the processor. Spawned workers
    inherit this environment; set before they start."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")
