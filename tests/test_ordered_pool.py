"""The order-preserving loader (lcmsdeconv.synth.pool), workers started the Windows way.

Its batches are those of torch's DataLoader, on recipe-v4 frames, over epochs and resumes, by
both paths (shared slots and torch's shared memory); a failed item is raised at its turn, a
dead worker named with its item; Ctrl+C reaches the workers from their birth and they ignore
it; a stream abandoned while its batches are being assembled loses and duplicates no slot; an
item held too long is made again elsewhere; training through it ends with the weights of a run
without workers, through a pause and with fewer workers after it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CTX = "spawn"  # the workers start as on Windows


# ------------------------------------------------------- helpers (imported by spawned workers)
class Tiny:
    """Cheap deterministic items shaped like training items: (x [C, L] float32, z [L, k] int64,
    w [L, k] float32, h [L] float32). ``fail_at``: an index that raises (``how="raise"``) or
    ends its worker (``how="exit"``); ``sleep``: seconds an item; ``stall``: seconds more an
    item takes on worker ``stall_worker`` (a worker on a core another program occupies)."""

    def __init__(self, n=10**9, C=2, L=64, k=3, fail_at=None, how="raise", sleep=0.0,
                 stall_worker=None, stall=0.0):
        self.n, self.C, self.L, self.k = n, C, L, k
        self.fail_at, self.how, self.sleep = fail_at, how, sleep
        self.stall_worker, self.stall = stall_worker, stall

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i == self.fail_at:
            if self.how == "exit":
                os._exit(7)
            raise ValueError(f"item {i} is broken on purpose")
        if self.sleep:
            time.sleep(self.sleep)
        if self.stall_worker is not None:
            from lcmsdeconv.synth.pool import current_worker

            if current_worker().get("id") == self.stall_worker:
                time.sleep(self.stall)
        g = torch.Generator().manual_seed(int(i))
        return (torch.rand((self.C, self.L), generator=g),
                torch.randint(0, 50, (self.L, self.k), generator=g),
                torch.rand((self.L, self.k), generator=g), torch.rand((self.L,), generator=g))


SPEC = [((2, 64), torch.float32), ((64, 3), torch.int64), ((64, 3), torch.float32),
        ((64,), torch.float32)]


def same(a, b, where=""):
    """Equal batches: the same types, structure, dtypes, shapes and values."""
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, (where, a.dtype, b.dtype, a.shape)
        assert torch.equal(a, b), where
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (where, len(a), len(b))
        for k, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{k}]")
    else:
        assert a == b, where


def expected(ds, indices, batch_size):
    from torch.utils.data import default_collate

    return [default_collate([ds[i] for i in indices[k:k + batch_size]])
            for k in range(0, len(indices) // batch_size * batch_size, batch_size)]


def _children():
    """This process's children but multiprocessing's resource tracker and torch's shared-memory
    manager (which outlive any one pool)."""
    import psutil

    return [p for p in psutil.Process().children(recursive=True)
            if not any(w in " ".join(p.cmdline()) for w in ("resource_tracker",
                                                             "torch_shm_manager"))]


def _gone(pids, seconds=10.0) -> bool:
    import psutil

    deadline = time.time() + seconds
    while time.time() < deadline:
        if not [p for p in pids if psutil.pid_exists(p)
                and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture(autouse=True)
def _alarm():
    # a stalled stream fails the test instead of hanging it
    signal.alarm(900)
    yield
    signal.alarm(0)


# ------------------------------------------------------------------- recipe-v4 frames
def _v4_mixture(window: int = 4096):
    """The v4 training mixture as train() sets it for an isotope-channel LadderNet, on crops of
    ``window`` bins (cheap items)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_charge_net",
                                                  ROOT / "scripts" / "train_charge_net.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    mix = script.mixture_v4(window)
    for _w, sc in mix:
        sc.evidence_channels = 0
        sc.singly_charged = "keep"
        sc.iso_channels = True
        sc.measured_channel = False
        sc.crop_size = window
    return mix


@pytest.fixture(scope="module")
def v4_mixture():
    from lcmsdeconv.nn.train import _warm_up

    mix = _v4_mixture()
    _warm_up(mix)  # numba compiled and cached once, not by every worker
    return mix


def test_batches_equal_the_dataloaders_on_recipe_v4_frames(v4_mixture, monkeypatch):
    """Every batch of the ordered loader, through the shared slots and through torch's shared
    memory, equals the DataLoader's that make_dataloaders builds, over two epochs of new
    spectra and a resume inside the second; the validation spectra and the comb's items too."""
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import CombBarFrames, EpochSampler, make_dataloaders
    from lcmsdeconv.synth.pool import OrderedBatches, ordered_batches

    monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", CTX)
    plan = ((0, 0), (1, 0), (1, 8))
    ref_tl, ref_vl = make_dataloaders(v4_mixture, 18, 5, batch_size=4, num_workers=2,
                                      base_seed=3, loader="dataloader")
    ref = {}
    for epoch, start in plan:
        ref_tl.sampler.epoch, ref_tl.sampler.start = epoch, start
        ref[(epoch, start)] = (len(ref_tl), list(ref_tl))
    ref_tl._iterator._shutdown_workers()
    x = ref[(0, 0)][1][0]
    assert [t.dtype for t in x] == [torch.float32, torch.int64, torch.float32, torch.float32]
    assert x[0].shape == (4, 7, 4096)  # 4 base channels and 3 isotope channels
    tl, vl = make_dataloaders(v4_mixture, 18, 5, batch_size=4, num_workers=3, base_seed=3,
                              loader="ordered", window=7)
    try:
        assert tl.pool.ring is not None and tl.num_workers == 3
        generic = OrderedBatches(tl.dataset, 4, EpochSampler(18), tl.pool, window=5,
                                 use_ring=False, name="generic")
        for epoch, start in plan:
            n, want = ref[(epoch, start)]
            for name, loader in (("slots", tl), ("generic", generic)):
                loader.sampler.epoch, loader.sampler.start = epoch, start
                got = list(loader)
                assert len(loader) == n == len(got) == (18 - start) // 4
                same(got, want, f"{name} epoch {epoch} start {start}")
        assert len(vl.dataset) == len(ref_vl.dataset) == 5
        same(vl.dataset.items, ref_vl.dataset.items, "validation")
        same(list(vl), list(ref_vl), "validation batches")
        comb = CombBarFrames(vl.dataset.frames)
        got = list(ordered_batches(tl.pool, "comb", comb, len(comb), 4))
        want = list(DataLoader(comb, batch_size=4, num_workers=0))
        same(got, want, "comb")  # tensors, and the strings as tuples
        assert isinstance(got[0][3], tuple) and isinstance(got[0][3][0], str)
        assert sum(tl.pool.items_done) >= 2 * 40 + 5 + 5
    finally:
        tl.close()
    assert not _children()


def test_an_item_that_fails_is_raised_in_order_with_its_index():
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import DataWorkerError, OrderedBatches, WorkerPool

    ds = Tiny(40, fail_at=5)
    with WorkerPool(2, context=CTX) as pool:
        tl = OrderedBatches(ds, 4, EpochSampler(40), pool, window=8)
        it = iter(tl)
        same(next(it), expected(Tiny(), list(range(4)), 4)[0], "batch 0")
        with pytest.raises(DataWorkerError, match="broken on purpose") as err:
            next(it)
        assert err.value.index == 5 and isinstance(err.value, RuntimeError)
        assert all(p.is_alive() for p in pool._procs)  # the worker itself goes on
        tl.sampler.epoch = 1  # the next epoch is whole
        same(list(tl), expected(ds, list(range(40, 80)), 4), "epoch 1")


@pytest.mark.parametrize("use_ring", [False, True])
def test_a_worker_that_dies_is_reported_with_its_item(use_ring):
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import DataWorkerError, OrderedBatches, SlotRing, WorkerPool

    ds = Tiny(400, fail_at=6, how="exit", sleep=0.05)
    with WorkerPool(3, context=CTX, ring=SlotRing(12, SPEC) if use_ring else None) as pool:
        pids = pool.pids
        tl = OrderedBatches(ds, 4, EpochSampler(400), pool, window=8)
        t0 = time.time()
        got = 0
        with pytest.raises(DataWorkerError, match="exit code 7 while generating item 6") as err:
            for _b in tl:
                got += 1
        assert err.value.index == 6 and got <= 1 and time.time() - t0 < 20
    assert _gone(pids)


def test_a_worker_that_died_while_idle_fails_the_stream_at_once():
    """Nothing sent to it yet when it died: the stream fails, it does not wait for ever."""
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import DataWorkerError, OrderedBatches, WorkerPool

    with WorkerPool(2, context=CTX) as pool:
        tl = OrderedBatches(Tiny(10**6), 4, EpochSampler(400), pool, window=8)
        same(next(iter(tl)), expected(Tiny(), list(range(4)), 4)[0], "batch 0")
        time.sleep(0.5)
        os.kill(pool.pids[1], signal.SIGKILL)
        t0 = time.time()
        with pytest.raises(DataWorkerError, match="data worker 1"):
            tl.sampler.epoch = 1
            for _b in tl:
                pass
        assert time.time() - t0 < 20
    # one whose pipe breaks as an item is sent to it: the same, from the sending side
    with WorkerPool(2, context=CTX) as pool:
        tl = OrderedBatches(Tiny(10**6), 4, EpochSampler(400), pool, window=8)

        class Broken:
            def send(self, msg):
                raise BrokenPipeError(32, "Broken pipe")

        real, pool._task_conns[0] = pool._task_conns[0], Broken()
        try:
            with pytest.raises(DataWorkerError, match="data worker 0 .* cannot be given work"):
                list(tl)
        finally:
            pool._task_conns[0] = real


def test_the_consumer_fails_when_the_receiver_thread_does():
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import DataWorkerError, OrderedBatches, WorkerPool

    with WorkerPool(2, context=CTX) as pool:
        def broken(*_a, **_k):
            raise MemoryError("on purpose")

        pool._on_result = broken
        tl = OrderedBatches(Tiny(10**6), 4, EpochSampler(40), pool, window=8)
        t0 = time.time()
        with pytest.raises(DataWorkerError, match="receiver thread"):
            list(tl)
        assert time.time() - t0 < 20


@pytest.mark.parametrize("use_ring", [False, True])
def test_abandoned_epochs_leave_nothing_behind(use_ring):
    """An epoch left part way (a pause, a time budget) and the next one started: the next
    one's batches are exactly its own, no slot of the ring is lost, and no batcher thread is
    left waiting on a queue nobody reads."""
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import OrderedBatches, SlotRing, WorkerPool, ring_slots

    ds = Tiny(10**6, sleep=0.01)
    ring = SlotRing(ring_slots(8, 4, 2), SPEC) if use_ring else None
    with WorkerPool(2, context=CTX, ring=ring) as pool:
        tl = OrderedBatches(ds, 4, EpochSampler(24), pool, window=8)
        for epoch in range(6):
            tl.sampler.epoch, tl.sampler.start = epoch, 0
            it = iter(tl)
            next(it)
            next(it)
            del it  # abandoned with items in flight and batches queued
        tl.sampler.epoch, tl.sampler.start = 7, 4
        same(list(tl), expected(ds, list(range(7 * 24 + 4, 8 * 24)), 4), "epoch 7")
        deadline = time.time() + 5
        while time.time() < deadline and any(t.name.startswith("lcmsdeconv-batcher")
                                             for t in threading.enumerate()):
            time.sleep(0.1)
        assert not [t.name for t in threading.enumerate()
                    if t.name.startswith("lcmsdeconv-batcher")]
        if use_ring:
            assert sorted(pool._free_slots) == list(range(ring.n))


def test_a_new_stream_while_the_last_ones_batch_is_copied_out_loses_no_slot():
    """The regression of the slot-ring race: an epoch started while the previous epoch's
    batcher holds slots it is copying out. Once it frees them, no slot is free twice and every
    batch of the new epoch is right (before: 3 slots doubled, 8 of 31 batches wrong)."""
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import OrderedBatches, SlotRing, WorkerPool

    ds = Tiny(10**6)
    ring = SlotRing(8, SPEC)
    with WorkerPool(2, context=CTX, ring=ring) as pool:
        tl = OrderedBatches(ds, 4, EpochSampler(200), pool, window=8, use_ring=True)
        orig = ring.batch
        calls = {"n": 0}
        entered = threading.Event()

        def slow_batch(slots, out=None):
            calls["n"] += 1
            if calls["n"] == 2:  # the old stream's second batch: its slots in hand
                entered.set()
                time.sleep(1.5)
            return orig(slots, out)

        ring.batch = slow_batch
        it1 = iter(tl)
        next(it1)
        assert entered.wait(60)
        it2 = iter(tl)
        first = next(it2)  # the new stream
        time.sleep(2.0)  # the old batcher wakes, copies, frees its 4 slots
        with pool._cond:
            free = list(pool._free_slots)
            in_use = [s for s in range(ring.n) if pool._slot_owner[s] is not None]
        assert len(free) == len(set(free)) and sorted(free + in_use) == list(range(ring.n))
        got = [first] + [b for _, b in zip(range(30), it2)]
        same(got, expected(ds, list(range(4 * len(got))), 4), "new stream")
        del it1, it2


@pytest.mark.parametrize("use_ring", [True, False])
def test_an_item_held_too_long_is_made_again_elsewhere(use_ring):
    """Worker 0 stalls 6 s on every item (bound to a core another program occupies): its items
    are handed to the free workers as well, the first result wins, the batches stay exact, and
    the slots of the copies that lost come back."""
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import OrderedBatches, SlotRing, WorkerPool, ring_slots

    ds = Tiny(10**6, sleep=0.01, stall_worker=0, stall=6.0)
    ring = SlotRing(ring_slots(12, 4, 3), SPEC) if use_ring else None
    with WorkerPool(3, context=CTX, ring=ring) as pool:
        pool.speculate_min_seconds = 0.5
        tl = OrderedBatches(ds, 4, EpochSampler(48), pool, window=12)
        t0 = time.time()
        same(list(tl), expected(ds, list(range(48)), 4), "epoch 0")
        assert pool.duplicates >= 1 and time.time() - t0 < 30
        if use_ring:
            deadline = time.time() + 15
            while time.time() < deadline and len(pool._free_slots) < ring.n:
                time.sleep(0.2)
            assert sorted(pool._free_slots) == list(range(ring.n))


def test_close_stops_busy_workers_and_leaves_no_process():
    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import OrderedBatches, WorkerPool

    pool = WorkerPool(3, context=CTX)
    tl = OrderedBatches(Tiny(10**6, sleep=0.5), 4, EpochSampler(10**6), pool, window=12)
    it = iter(tl)
    next(it)
    t0 = time.time()
    tl.close()
    assert time.time() - t0 < 5.0
    assert not _children()
    tl.close()  # idempotent


FILE_SYSTEM_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from torch import multiprocessing as tmp
tmp.set_sharing_strategy("file_system")
from lcmsdeconv.synth.dataset import EpochSampler
from lcmsdeconv.synth.pool import OrderedBatches, SlotRing, WorkerPool, ring_slots
from tests.test_ordered_pool import SPEC, Tiny, expected, same
ds = Tiny(10**6)
with WorkerPool(2, context="spawn", ring=SlotRing(ring_slots(8, 4, 2), SPEC)) as pool:
    for use_ring in (True, False):
        tl = OrderedBatches(ds, 4, EpochSampler(24), pool, window=8, use_ring=use_ring,
                            name=f"t{use_ring}")
        tl.sampler.epoch = 2
        same(list(tl), expected(ds, list(range(48, 72)), 4), f"ring {use_ring}")
print("equal", tmp.get_sharing_strategy(), flush=True)
"""


def test_the_slots_under_the_file_system_sharing_windows_uses():
    """Shared memory by file name, as on Windows (in a process of its own: the strategy is
    process-wide, and torch's shared-memory manager it starts outlives the test)."""
    out = subprocess.run([sys.executable, "-c", FILE_SYSTEM_CHILD, str(ROOT)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.split() == ["equal", "file_system"], \
        out.stdout + out.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
def test_pinned_batches_equal_the_dataloaders():  # pragma: no cover - CUDA machines only
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import EpochSampler
    from lcmsdeconv.synth.pool import OrderedBatches, SlotRing, WorkerPool, ring_slots

    ds = Tiny(10**6)
    want = list(DataLoader(ds, batch_size=4, sampler=EpochSampler(24), drop_last=True,
                           pin_memory=True))
    with WorkerPool(2, context=CTX, ring=SlotRing(ring_slots(8, 4, 2), SPEC)) as pool:
        for use_ring in (True, False):
            tl = OrderedBatches(ds, 4, EpochSampler(24), pool, window=8, pin_memory=True,
                                use_ring=use_ring, name=f"p{use_ring}")
            got = list(tl)
            same(got, want, f"pinned, ring {use_ring}")
            assert all(t.is_pinned() for b in got for t in b)
            assert len({b[0].data_ptr() for b in got}) == len(got)  # a fresh buffer a batch


def test_many_workers_are_waited_on_in_turns_where_python_cannot_at_once(monkeypatch):
    """Windows before Python 3.12 waits on 63 handles at most: the pipes and process handles
    of 65 workers are waited on in turns of MAX_WAIT_HANDLES; elsewhere at once."""
    import multiprocessing.connection as mpc
    from types import SimpleNamespace

    from lcmsdeconv.synth import pool as P

    calls, ready = [], {125}

    def fake_wait(objs, timeout):
        calls.append((len(objs), timeout))
        return [o for o in objs if o in ready]

    monkeypatch.setattr(mpc, "wait", fake_wait)
    monkeypatch.setattr(P, "sys", SimpleNamespace(platform="win32", version_info=(3, 11, 9)))
    assert P._wait(list(range(130)), 0.3) == [125]
    assert [n for n, _t in calls[:3]] == [60, 60, 10] and calls[0][1] == pytest.approx(0.1)
    calls.clear()
    monkeypatch.setattr(P, "sys", SimpleNamespace(platform="win32", version_info=(3, 13, 1)))
    assert P._wait(list(range(130)), 0.3) == [125] and calls == [(130, 0.3)]


def test_windows_workers_inherit_ignoring_ctrl_c_only_while_they_start(monkeypatch):
    """SetConsoleCtrlHandler(NULL, TRUE) before a worker starts (children inherit it), then
    FALSE: the training process handles Ctrl+C again (a fake kernel32 records the calls)."""
    from types import SimpleNamespace

    from lcmsdeconv.nn import cpus
    from lcmsdeconv.synth import pool as P

    calls = []

    class Kernel32:
        def SetConsoleCtrlHandler(self, handler, add):
            calls.append((handler, add))
            return 1

    monkeypatch.setattr(cpus, "win_kernel32", lambda: Kernel32())
    monkeypatch.setattr(P, "sys", SimpleNamespace(platform="win32", version_info=(3, 13, 1)))
    with P.children_ignore_ctrl_c():
        assert calls == [(None, True)]
    assert calls == [(None, True), (None, False)]


# ----------------------------------------------------------------------- Ctrl+C, as a console
CHILD = r"""
import json, os, signal, sys, time
sys.path.insert(0, sys.argv[2])
import torch
from torch.utils.data import default_collate
from lcmsdeconv.synth.dataset import EpochSampler
from lcmsdeconv.synth.pool import OrderedBatches, SlotRing, WorkerPool
from tests.test_ordered_pool import SPEC, Tiny

mode = sys.argv[1]
pause = {"requested": 0}

def on_interrupt(signum, frame):  # train.py's: the first pauses, the second aborts
    pause["requested"] += 1
    if pause["requested"] > 1 and mode != "startup":
        raise KeyboardInterrupt

signal.signal(signal.SIGINT, on_interrupt)
if mode == "startup":
    print("starting", flush=True)  # Ctrl+C now, while the workers start
# with shared slots a spawned worker imports PyTorch before it runs any of the pool's code
pool = WorkerPool(2, context="spawn", ring=SlotRing(16, SPEC) if mode == "startup" else None)
n = 40 if mode in ("once", "startup") else 400
tl = OrderedBatches(Tiny(10**6, sleep=0.05), 4, EpochSampler(n), pool, window=8)
try:
    got, alive_after = [], None
    for k, b in enumerate(tl):
        if k == 0:
            print("ready", *pool.pids, flush=True)
            if mode == "kill":  # the training process killed outright (Task Manager)
                os.kill(os.getpid(), signal.SIGKILL)
        got.append(b)
        time.sleep(0.05)  # the training step
        if pause["requested"] and alive_after is None:
            alive_after = [p.is_alive() for p in pool._procs]
        if mode == "twice" and pause["requested"]:
            time.sleep(0.5)  # a long step: the second Ctrl+C lands here
    ds = Tiny()
    want = [default_collate([ds[i] for i in range(k, k + 4)]) for k in range(0, n, 4)]
    equal = len(got) == len(want) and all(
        all(torch.equal(x, y) for x, y in zip(a, b)) for a, b in zip(got, want))
    print(json.dumps({"pause_requested": pause["requested"], "batches": len(got),
                      "workers_alive_after_signal": alive_after,
                      "expected_batches": len(want), "batches_equal": equal}), flush=True)
finally:
    tl.close()
"""


def _run_child(mode):
    """The child's exit code, its workers' pids, and the rest of its output and its errors
    (read whole: lines read ahead with the first ones are not lost)."""
    import tempfile

    with tempfile.TemporaryFile("w+") as errors:
        proc = subprocess.Popen([sys.executable, "-c", CHILD, mode, str(ROOT)],
                                stdout=subprocess.PIPE, stderr=errors, text=True,
                                start_new_session=True, env=os.environ.copy())
        try:
            if mode == "startup":
                assert proc.stdout.readline().startswith("starting")
                for _ in range(6):  # Ctrl+C reaches every process of the console
                    os.killpg(proc.pid, signal.SIGINT)
                    time.sleep(0.4)
            line = proc.stdout.readline()  # "ready <pids>"
            pids = [int(p) for p in line.split()[1:]] if line.startswith("ready") else []
            if pids and mode in ("once", "twice"):
                os.killpg(proc.pid, signal.SIGINT)
            if pids and mode == "twice":
                time.sleep(1.0)
                os.killpg(proc.pid, signal.SIGINT)
            out = proc.stdout.read()
            proc.wait(timeout=300)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        errors.seek(0)
        err = errors.read()
    assert pids, line + out + err
    return proc.returncode, pids, out, err


def test_ctrl_c_pauses_after_the_step_with_the_workers_still_there():
    """Ctrl+C reaches the workers too (as in Windows' console): they ignore it, the batches
    keep coming in order, and the training stops when it chooses."""
    rc, pids, out, err = _run_child("once")
    assert rc == 0, out + err
    res = json.loads(out.strip().splitlines()[-1])
    assert res["pause_requested"] and res["workers_alive_after_signal"] == [True, True]
    assert res["batches_equal"] and res["batches"] == res["expected_batches"]
    assert _gone(pids)


def test_ctrl_c_while_the_workers_start_does_not_reach_them():
    """Spawned workers import PyTorch for seconds before they run any of the pool's code:
    they ignore Ctrl+C from their birth all the same."""
    rc, pids, out, err = _run_child("startup")
    assert rc == 0, out + err
    res = json.loads(out.strip().splitlines()[-1])
    assert res["pause_requested"] >= 1 and res["batches_equal"]
    assert res["batches"] == res["expected_batches"]


def test_a_second_ctrl_c_aborts_and_the_workers_go_with_it():
    rc, pids, out, err = _run_child("twice")
    # an unhandled KeyboardInterrupt ends Python by SIGINT itself (-2), as Ctrl+C would
    assert rc in (1, -signal.SIGINT) and "KeyboardInterrupt" in err, out + err
    assert _gone(pids)


def test_the_workers_end_when_the_training_process_is_killed():
    """No Ctrl+C, no clean-up: the training process killed outright. Its workers read the end
    of their task pipes and end within seconds instead of lingering."""
    rc, pids, out, err = _run_child("kill")
    assert rc == -signal.SIGKILL, out + err
    assert _gone(pids, 20.0)


# ------------------------------------------------------------------------------- training
def _synth():
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig

    return SynthConfig(classes=[ClassConfig("peptide", (8000.0, 20000.0))], mode="rplc",
                       polarity=1)


def _cfg(out, **kw):
    base = dict(arch="ladder", model_size="small", epochs=2, train_len=6, val_len=2,
                batch_size=2, num_workers=0, threads=1, out=str(out), log_every=100,
                fresh_each_epoch=True, device="cpu")
    base.update(kw)
    return base


def _same_weights(a, b):
    for (k, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(x, y), k


def test_training_through_the_ordered_loader_is_exact_through_a_pause_with_fewer_workers(
        tmp_path, monkeypatch, capsys):
    """Paused after a step with three workers bound one to a core, resumed with one worker
    (train_all's step-down after a memory pause): the weights equal those of a run without
    workers, bit for bit. Each epoch records what the workers made, where; this process gets
    its processors back."""
    from lcmsdeconv.nn.train import TrainConfig, train

    torch.manual_seed(0)
    plain, _h = train(TrainConfig(**_cfg(tmp_path / "plain.pt")), synth_config=_synth())
    monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", CTX)
    before = os.sched_getaffinity(0)
    out = tmp_path / "ordered.pt"
    torch.manual_seed(0)
    capsys.readouterr()
    train(TrainConfig(max_seconds=1e-6, loader="ordered", cpu_binding="core",
                      worker_priority="normal", worker_qos="high", **_cfg(out, num_workers=3)),
          synth_config=_synth())
    text = capsys.readouterr().out
    assert "data loading: ordered" in text and "data workers bound as planned" in text
    assert os.sched_getaffinity(0) == before  # given back
    state = torch.load(out.with_suffix(".resume.pt"), weights_only=False)
    assert (state["epoch"], state["step"], state["stop_reason"]) == (0, 1, "time")
    resumed, history = train(TrainConfig(resume=True, loader="ordered",
                                         **_cfg(out, num_workers=1)), synth_config=_synth())
    assert [h["epoch"] for h in history] == [0, 1]
    _same_weights(plain, resumed)
    w = history[-1]["pace"]["workers"]
    assert (w["loader"], w["binding"]) == ("ordered", "none")
    assert sum(d["items"] for d in w["by_kind"].values()) >= 6 and w["made_twice"] == 0
    assert not _children()


def test_the_dataloader_stays_the_default_and_the_environment_forces_it(tmp_path, monkeypatch,
                                                                         capsys):
    """LCMSDECONV_LOADER=dataloader takes torch's DataLoader whatever the run asks; with it,
    binding each worker to a core becomes sharing the workers' cores. The same weights, and
    the per-class record of the workers there too."""
    from lcmsdeconv.nn.train import TrainConfig, train

    torch.manual_seed(0)
    plain, _h = train(TrainConfig(**_cfg(tmp_path / "plain.pt", epochs=1)),
                      synth_config=_synth())
    monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", CTX)
    monkeypatch.setenv("LCMSDECONV_LOADER", "dataloader")
    torch.manual_seed(0)
    capsys.readouterr()
    dl, history = train(TrainConfig(loader="ordered", cpu_binding="core",
                                    **_cfg(tmp_path / "dl.pt", epochs=1, num_workers=2)),
                        synth_config=_synth())
    text = capsys.readouterr().out
    assert "torch's DataLoader" in text and "share the cores" in text
    _same_weights(plain, dl)
    w = history[-1]["pace"]["workers"]
    assert (w["loader"], w["binding"]) == ("dataloader", "pool")
    assert sum(d["items"] for d in w["by_kind"].values()) == 6
    assert "spectra made: 6 (" in text and "/s a busy worker, on processors " in text


def test_one_pool_serves_validation_comb_and_training(tmp_path, monkeypatch):
    """The comb is scored by the training's own workers: one pool of num_workers processes
    for the whole run (never a second one beside it), and none left once training ends."""
    import lcmsdeconv.nn.train as tr

    seen = []
    original = tr.comb_baseline

    def counting(loader, num_workers=None, **kw):
        seen.append((len(_children()), kw.get("pool") is not None))
        return original(loader, num_workers=num_workers, **kw)

    monkeypatch.setattr(tr, "comb_baseline", counting)
    monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", CTX)
    tr.train(tr.TrainConfig(loader="ordered", **_cfg(tmp_path / "net.pt", num_workers=2,
                                                      epochs=1)),
             synth_config=_synth())
    assert seen == [(2, True)]
    assert not _children()


def test_the_memory_watch_counts_the_shared_slots():
    pytest.importorskip("psutil")
    from lcmsdeconv.nn.train import _RamWatch

    assert _RamWatch(None).used_gb() > 0
    assert _RamWatch(None, extra_gb=0.5).used_gb() >= 0.5 + _RamWatch(None).used_gb() - 0.1


def test_the_loader_is_checked():
    from lcmsdeconv.synth.pool import choose_loader, default_window, ring_slots

    assert choose_loader(None) in ("dataloader", "ordered")
    with pytest.raises(ValueError):
        choose_loader("fast")
    assert default_window(22, 4) == 110 and default_window(1, 8) == 16
    assert ring_slots(110, 4, 22) == 110 + 4 + 5
