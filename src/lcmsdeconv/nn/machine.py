"""What the training machine offers: cores, memory, and how many data workers fit a budget."""

from __future__ import annotations

import os
import sys

#: memory of one data worker generating spectra (Python, PyTorch, numba and a spectrum in work),
#: measured at 0.8 GB on Linux; Windows loads more of PyTorch into every process
WORKER_GB = 1.2
#: memory of the training process itself: PyTorch with a CUDA context, the model, the validation
#: spectra kept in memory and the batches in flight
MAIN_GB = 3.5
#: kept free below a memory budget, so that the budget is never reached in normal running
MARGIN_GB = 1.5


def total_ram_gb() -> float:
    """Physical memory of this computer in GB (16 if it cannot be read)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 2**30
        else:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        pass
    return 16.0


def cpu_cores() -> int:
    return os.cpu_count() or 2


def auto_workers(ram_budget_gb: float | None = None, worker_gb: float = WORKER_GB,
                 main_gb: float = MAIN_GB, margin_gb: float = MARGIN_GB) -> int:
    """Data workers that fit the processor and the memory: all cores but a few (two of eight or
    fewer, four of more) and, within ``ram_budget_gb`` (half the computer's memory without
    one), as many workers as the memory left by the training process and a margin holds."""
    cores = cpu_cores()
    by_cpu = cores - 2 if cores <= 8 else cores - 4
    budget = ram_budget_gb if ram_budget_gb else 0.5 * total_ram_gb()
    by_ram = int((budget - main_gb - margin_gb) // max(worker_gb, 0.1))
    return max(1, min(by_cpu, by_ram))


def single_threaded_workers() -> None:
    """One thread per data worker: the workers are processes, and numerical libraries that
    each start a thread per core would make them fight over the processor. Spawned workers
    inherit this environment; set before they start."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")
