"""Order-preserving dynamic data loading for training: every batch exactly as
``torch.utils.data.DataLoader`` gives it, made by whichever worker is free.

torch's DataLoader hands whole batches to its workers strictly in turn and delivers them in
order, so on a processor with fast and slow cores (performance and efficient) the slowest core
sets the pace of all. A :class:`WorkerPool` keeps persistent data-worker processes that make
dataset items by index; the main process hands out one item at a time to whichever worker is
free (the fastest free one first, in the order the placement lists the workers), never more
than ``window`` items ahead of the consumer, and gives the items back in index order. Fast
workers make more items than slow ones, and every batch of :class:`OrderedBatches` holds
exactly the items, in the order, with the collation and dtypes, that the DataLoader would give
(``batch_size`` consecutive indices of the sampler, the last incomplete batch dropped; an
``EpochSampler``'s ``epoch`` and ``start`` read when an epoch starts, as the DataLoader reads
them): training is bit for bit the same with either loader, any number of workers, and through
pauses and resumes.

Items travel either through torch's shared-memory reductions (any picklable item, as the
DataLoader's batches do: the validation and comb items) or, for items of fixed shapes (the
training items), through a ring of preallocated shared slots (:class:`SlotRing`): the worker
copies its item into a slot and sends the slot's number, and the main process copies a batch's
slots out in one copy, into pinned memory for the GPU where asked. Each slot belongs to the
task it was handed out with; it is freed once, by that task's owner (the consumer once it has
copied it, or the pool when the item is no longer wanted), so that starting a new epoch while
the previous one's batches are still being assembled cannot hand a slot out twice.

Robustness: workers ignore Ctrl+C from their birth (the training pauses after its step and
stops them itself); a worker that dies is reported with the item it was making; an item that
fails is raised at its turn, with its index; an item held much longer than items usually take
(a worker stalled on a core another program occupies) is handed to a free worker as well, and
the first result wins; the training process killed outright, the workers end by themselves.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import queue
import signal
import statistics
import sys
import threading
import time
import traceback
from collections import deque

#: seconds between checks of the other side's health while waiting
POLL = 0.25
#: an item in work for SPECULATE_AFTER times the median item time (at least
#: SPECULATE_MIN_SECONDS, once SPECULATE_MIN_ITEMS items have been timed) is handed to a free
#: worker as well: items are pure functions of their index, so the first result wins
SPECULATE_AFTER, SPECULATE_MIN_SECONDS, SPECULATE_MIN_ITEMS = 4.0, 2.0, 8
#: WaitForMultipleObjects waits on 64 handles at most; Pythons before 3.12 do not split longer
#: lists on Windows
MAX_WAIT_HANDLES = 60
#: no slot of the ring free and nothing of the stream in work for this long: a slot was lost
SLOT_STALL_SECONDS = 120.0
#: the loaders make_dataloaders and train() choose between (LCMSDECONV_LOADER overrides)
LOADERS = ("dataloader", "ordered")

_WORKER: dict = {}  # in a worker process: {"id": ...}


def current_worker() -> dict:
    """In a data worker of a :class:`WorkerPool`: ``{"id": worker id}``; {} elsewhere."""
    return _WORKER


def choose_loader(loader: str | None = None) -> str:
    """The training loader to use: ``LCMSDECONV_LOADER`` where set (a way back to torch's
    DataLoader whatever setup chose), else ``loader``, else "dataloader"."""
    name = (os.environ.get("LCMSDECONV_LOADER") or loader or "dataloader").strip().lower()
    if name not in LOADERS:
        raise ValueError(f"loader must be one of {LOADERS}, not {name!r}")
    return name


class DataWorkerError(RuntimeError):
    """A data worker failed on an item (``index``) or died. A RuntimeError, as the DataLoader's
    worker errors are, so that the training loop's handling of errors stays the same."""

    def __init__(self, message: str, index=None, worker=None):
        super().__init__(message)
        self.index = index
        self.worker = worker


# ------------------------------------------------------------------ what the workers made
class ItemStats:
    """Items made by each data worker on each logical processor (the one an item ended on):
    how many, their seconds (wall clock), their seconds of processor time, and how many moved
    to another processor while being made. In shared memory with ``shared`` (filled by the
    DataLoader's worker processes); a :class:`WorkerPool` fills its own from the results."""

    FIELDS = ("items", "seconds", "cpu_seconds", "moved")

    def __init__(self, num_workers: int, num_cpus: int | None = None, shared: bool = False):
        import torch

        self.num_workers = max(1, int(num_workers))
        self.num_cpus = max(1, int(num_cpus or os.cpu_count() or 1))
        table = torch.zeros((self.num_workers, self.num_cpus + 1, len(self.FIELDS)),
                            dtype=torch.float64)
        self.table = table.share_memory_() if shared else table
        self._np = None

    def __getstate__(self):
        return {"num_workers": self.num_workers, "num_cpus": self.num_cpus, "table": self.table}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._np = None

    def add(self, worker: int, cpu: int | None, seconds: float, cpu_seconds: float,
            moved: bool = False) -> None:
        if self._np is None:
            self._np = self.table.numpy()  # a view: in shared memory where the table is
        c = cpu if cpu is not None and 0 <= cpu < self.num_cpus else self.num_cpus
        row = self._np[int(worker) % self.num_workers, c]
        row[0] += 1
        row[1] += seconds
        row[2] += cpu_seconds
        row[3] += bool(moved)

    def snapshot(self):
        """A copy of the table (numpy), to subtract from a later one."""
        return self.table.numpy().copy()

    def per_worker(self, since=None) -> list[dict]:
        d = self.snapshot() - (since if since is not None else 0)
        return [{"items": int(r[:, 0].sum()), "seconds": float(r[:, 1].sum()),
                 "cpu_seconds": float(r[:, 2].sum())} for r in d]

    def summary(self, since=None, kind_of_cpu: dict | None = None,
                seconds: float | None = None) -> dict:
        """What was made since the snapshot ``since``, by class of processor (``kind_of_cpu``:
        {processor: "P", "E", "LP" or "?"}): items, items a second over ``seconds`` (the time it
        took), items a second each worker made while busy, the share of their time that was
        processor time, items that moved processor while made, and the processors that made
        them; and the share of the time the workers were busy."""
        from ..nn.cpus import cpu_ranges

        d = self.snapshot() - (since if since is not None else 0)
        per_cpu = d.sum(axis=0)
        kind_of_cpu = kind_of_cpu or {}
        acc: dict = {}
        cpus: dict = {}
        for c in range(self.num_cpus + 1):
            if per_cpu[c, 0] <= 0:
                continue
            kind = kind_of_cpu.get(c, "?") if c < self.num_cpus else "?"
            acc[kind] = acc.get(kind, 0) + per_cpu[c]
            if c < self.num_cpus:
                cpus.setdefault(kind, []).append(c)
        out = {}
        for kind, (items, secs, cpu_secs, moved) in sorted(acc.items()):
            out[kind] = {"items": int(items),
                         "per_second": round(items / seconds, 2) if seconds else None,
                         "per_worker_per_second": round(items / secs, 3) if secs > 0 else None,
                         "on_processor": round(cpu_secs / secs, 3) if secs > 0 else None,
                         "moved": int(moved), "processors": cpu_ranges(cpus.get(kind, []))}
        busy = float(d[:, :, 1].sum())
        return {"by_kind": out, "items": int(d[:, :, 0].sum()),
                "workers_busy": round(busy / (self.num_workers * seconds), 3) if seconds
                else None}


# ---------------------------------------------------------------- shared slots for fixed items
class SlotRing:
    """``n`` slots for items that are tuples of tensors of fixed shapes and dtypes (``spec``:
    [(shape, dtype)]), in shared memory: created by the main process, mapped by every worker
    once (at its start)."""

    def __init__(self, n: int, spec):
        import torch

        self.n = int(n)
        self.spec = [(tuple(shape), dtype) for shape, dtype in spec]
        self.tensors = []
        for shape, dtype in self.spec:
            numel = self.n
            for s in shape:
                numel *= int(s)
            proto = torch.empty(0, dtype=dtype)
            try:  # straight into shared memory, as default_collate does in a worker
                storage = proto._typed_storage()._new_shared(numel)
                t = proto.new(storage).resize_(self.n, *shape)
            except Exception:  # pragma: no cover - a PyTorch without that private path
                t = torch.empty((self.n, *shape), dtype=dtype).share_memory_()
            self.tensors.append(t)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def misfit(self, item) -> str | None:
        """Why ``item`` does not fit a slot, or None."""
        import torch

        if not isinstance(item, (tuple, list)) or len(item) != len(self.spec):
            return f"an item of {len(self.spec)} tensors expected, got {type(item).__name__}"
        for k, (x, (s, d)) in enumerate(zip(item, self.spec)):
            if not isinstance(x, torch.Tensor) or tuple(x.shape) != s or x.dtype != d:
                got = (tuple(x.shape), x.dtype) if isinstance(x, torch.Tensor) else type(x)
                return f"element {k}: {got} does not match the slots' {(s, d)}"
        return None

    def write(self, slot: int, item) -> None:
        for ring, x in zip(self.tensors, item):
            ring[slot].copy_(x)

    def item(self, slot: int) -> tuple:
        """The item in ``slot`` (views into the ring: copy before the slot is freed)."""
        return tuple(ring[slot] for ring in self.tensors)

    def batch(self, slots, out=None) -> list:
        """The items in ``slots`` stacked, one copy (a list of tensors, as default_collate makes
        of tuple items), into ``out`` if given (fresh pinned tensors for the GPU)."""
        import torch

        idx = torch.as_tensor(list(slots), dtype=torch.long)
        if out is None:
            return [torch.index_select(ring, 0, idx) for ring in self.tensors]
        return [torch.index_select(ring, 0, idx, out=o) for ring, o in zip(self.tensors, out)]


# ---------------------------------------------------------------------------- the workers
def _ctrl_c_ignored(ignore: bool) -> bool:
    """Windows: make this process ignore Ctrl+C (``ignore``) or process it again; children
    started meanwhile inherit the setting (SetConsoleCtrlHandler(NULL, TRUE))."""
    try:
        from ..nn.cpus import win_kernel32

        return bool(win_kernel32().SetConsoleCtrlHandler(None, bool(ignore)))
    except Exception:
        return False


@contextlib.contextmanager
def children_ignore_ctrl_c():
    """Processes started inside this block ignore Ctrl+C from their birth, before they run any
    of our code (a spawned worker imports PyTorch for seconds first): on Windows the console's
    Ctrl+C is ignored meanwhile (an attribute children inherit), elsewhere SIGINT is blocked in
    this thread (children inherit the mask, through exec too; the worker sets SIGINT to ignored
    and unblocks it, which discards one that came meanwhile). The training process's own Ctrl+C
    handling is back once the block ends (a Ctrl+C pressed on Windows during the few
    milliseconds of a start is lost; elsewhere it is delivered then)."""
    if sys.platform == "win32":
        ignored = _ctrl_c_ignored(True)
        try:
            yield
        finally:
            if ignored:
                _ctrl_c_ignored(False)
        return
    old = None
    if hasattr(signal, "pthread_sigmask"):
        try:
            old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        except (OSError, ValueError):
            old = None
    try:
        yield
    finally:
        if old is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _parent_alive() -> bool:
    import multiprocessing as mp

    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def _worker_loop(wid, task_conn, result_conn, ring, init, env):
    # The main process decides when to stop: a Ctrl+C asks the training to pause, which saves
    # its state after the current step, and the workers must still be there to finish it.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    except (ValueError, OSError):  # pragma: no cover
        pass
    os.environ.update(env)
    import torch

    from ..nn.cpus import current_cpu

    torch.set_num_threads(1)
    _WORKER.clear()
    _WORKER.update(id=wid)
    init_error = None
    if init is not None:
        try:
            init(wid)  # a DataLoader worker_init_fn: priority, power throttling, core binding
        except Exception:  # reported with every item, as the DataLoader does
            init_error = traceback.format_exc()
    datasets: dict = {}
    unreadable = None  # the last dataset that could not be read here
    while True:
        try:
            if not task_conn.poll(1.0):
                if not _parent_alive():
                    return
                continue
            msg = task_conn.recv()
        except (EOFError, OSError):  # the main process is gone
            return
        except Exception:  # a dataset that cannot be unpickled here
            unreadable = traceback.format_exc()
            continue
        if msg is None:
            return
        if msg[0] == "dataset":
            if msg[2] is None:
                datasets.pop(msg[1], None)
            else:
                datasets[msg[1]] = msg[2]
            continue
        _kind, gen, task, name, idx, slot = msg
        t0, c0, cpu0 = time.perf_counter(), time.process_time(), current_cpu()
        try:
            if init_error:
                raise RuntimeError(f"the data worker's start failed:\n{init_error}")
            dataset = datasets.get(name)
            if dataset is None:
                raise RuntimeError(f"no dataset {name!r} in this worker" + (
                    f"; one could not be read:\n{unreadable}" if unreadable else ""))
            item = dataset[idx]
            if slot is not None and ring.misfit(item) is None:
                ring.write(slot, item)
                payload = ("slot", slot)
            else:  # whole, through shared memory (an item of another shape: generic batch)
                payload = ("item", item)
        except Exception:
            payload = ("error", traceback.format_exc())
        item = None
        cpu1 = current_cpu()
        took = (time.perf_counter() - t0, time.process_time() - c0, cpu1,
                cpu0 is not None and cpu1 is not None and cpu0 != cpu1)
        try:
            result_conn.send((wid, gen, task, payload) + took)
        except (BrokenPipeError, EOFError, OSError):
            return
        except Exception:  # an item that cannot be pickled
            try:
                result_conn.send((wid, gen, task, ("error", traceback.format_exc())) + took)
            except Exception:
                return
        payload = None


def _wait(objs, timeout: float):
    """multiprocessing.connection.wait, in turns of MAX_WAIT_HANDLES objects where this Python
    cannot wait on more at once (Windows before 3.12)."""
    from multiprocessing.connection import wait

    if sys.platform != "win32" or sys.version_info >= (3, 12) or len(objs) <= MAX_WAIT_HANDLES:
        return wait(objs, timeout)
    chunks = [objs[i:i + MAX_WAIT_HANDLES] for i in range(0, len(objs), MAX_WAIT_HANDLES)]
    for c in chunks:
        ready = wait(c, timeout / len(chunks))
        if ready:
            return ready + [x for d in chunks if d is not c for x in wait(d, 0)]
    return []


# ------------------------------------------------------------------------------ the main side
class _Stream:
    """One ordered pass over indices (an epoch, the validation items): its own generation."""

    __slots__ = ("gen", "name", "use_ring", "sent", "taken", "results", "duplicated",
                 "exhausted", "refill")

    def __init__(self, gen: int, name: str, use_ring: bool):
        self.gen, self.name, self.use_ring = gen, name, use_ring
        self.sent = self.taken = 0  # positions handed out, and given to the consumer
        self.results: dict = {}  # position -> (payload, task), the first to come
        self.duplicated: set = set()  # positions handed out a second time
        self.exhausted = False
        self.refill = None  # hands out more items, up to the window (with the pool's lock)


class _Task:
    __slots__ = ("id", "stream", "pos", "idx", "slot", "worker", "sent", "duplicate")

    def __init__(self, tid, stream, pos, idx, slot, duplicate=False):
        self.id, self.stream, self.pos, self.idx, self.slot = tid, stream, pos, idx, slot
        self.worker, self.sent, self.duplicate = None, None, duplicate


class WorkerPool:
    """``num_workers`` persistent worker processes making items of registered datasets.

    ``init(worker_id)`` runs first in every worker, as a DataLoader's ``worker_init_fn`` does
    (priority, power throttling, binding to the worker's cores: a functools.partial carrying the
    placement pickles for spawned workers); ``rank``: the order in which free workers get items
    (default: worker order, in which the placement lists the fastest cores first); ``labels``:
    how messages name each worker's cores. ``context``: the start method (None:
    LCMSDECONV_MP_CONTEXT, else the platform's; "spawn" as on Windows). ``ring``: a
    :class:`SlotRing` for fixed-shape items. ``speculate``: hand an item held far longer than
    items usually take to a free worker too (SPECULATE_AFTER). ``stats`` counts what every
    worker made where (:class:`ItemStats`), ``duplicates`` the items handed out twice."""

    def __init__(self, num_workers: int, init=None, context: str | None = None,
                 ring: SlotRing | None = None, rank=None, labels=None, env: dict | None = None,
                 speculate: bool = True):
        import torch.multiprocessing as tmp

        if int(num_workers) < 1:
            raise ValueError("a worker pool needs at least one worker")
        self.num_workers = int(num_workers)
        self.labels = list(labels) if labels is not None else [""] * self.num_workers
        self.ring = ring
        self.context = context or os.environ.get("LCMSDECONV_MP_CONTEXT") or None
        ctx = tmp.get_context(self.context)
        env = dict(env or {})
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS"):
            env.setdefault(var, "1")
        # one pipe each way per worker: a worker that dies cannot leave a shared queue's lock
        # taken, its pipe reads as closed, and the main process waits on the pipes and on the
        # processes' sentinels at once
        self._task_conns: list = []
        self._result_conns: list = []
        self._procs: list = []
        try:
            for w in range(self.num_workers):
                task_r, task_w = ctx.Pipe(duplex=False)
                res_r, res_w = ctx.Pipe(duplex=False)
                self._task_conns.append(task_w)
                self._result_conns.append(res_r)
                p = ctx.Process(target=_worker_loop, daemon=True,
                                name=f"lcmsdeconv-data-worker-{w}",
                                args=(w, task_r, res_w, ring, init, env))
                try:
                    with children_ignore_ctrl_c():
                        p.start()
                finally:
                    task_r.close()  # the worker's ends: closed here, so that either side's
                    res_w.close()  # death reads as end of file on the other
                self._procs.append(p)
        except BaseException:
            self._closed = True
            self._stop_processes(5.0)
            raise
        rank = [w for w in (rank or ()) if 0 <= w < self.num_workers]
        self._rank = rank + [w for w in range(self.num_workers) if w not in rank]
        self._cond = threading.Condition()
        self._idle = set(range(self.num_workers))
        self._dead: set = set()
        self._held: list = [None] * self.num_workers  # task id each worker is making
        self._pending: deque = deque()  # task ids waiting for a free worker, in order
        self._tasks: dict = {}  # task id -> _Task, handed out and not yet answered
        self._stream: _Stream | None = None
        self._gen = 0
        self._fatal: DataWorkerError | None = None
        self._closed = False
        self._task_ids = itertools.count()
        self._free_slots: deque = deque(range(ring.n)) if ring is not None else deque()
        self._slot_owner: list = [None] * (ring.n if ring is not None else 0)
        self.stats = ItemStats(self.num_workers)
        self._durations: deque = deque(maxlen=256)  # seconds of the latest items
        self.speculate = bool(speculate)
        self.speculate_after = SPECULATE_AFTER
        self.speculate_min_seconds = SPECULATE_MIN_SECONDS
        self.duplicates = 0
        self._rx = threading.Thread(target=self._receive, name="lcmsdeconv-pool-receiver",
                                    daemon=True)
        self._rx.start()

    @property
    def pids(self) -> list[int]:
        return [p.pid for p in self._procs]

    @property
    def items_done(self) -> list[int]:
        return [w["items"] for w in self.stats.per_worker()]

    @property
    def busy_seconds(self) -> list[float]:
        return [w["seconds"] for w in self.stats.per_worker()]

    def _name(self, w: int) -> str:
        label = self.labels[w] if w < len(self.labels) else ""
        pid = self._procs[w].pid if w < len(self._procs) else None
        return f"data worker {w} (pid {pid}" + (f", {label})" if label else ")")

    def _check(self) -> None:
        """Raise what makes this pool unusable (caller holds the lock)."""
        if self._fatal is not None:
            f = self._fatal
            raise DataWorkerError(str(f), index=f.index, worker=f.worker)
        if self._closed:
            raise DataWorkerError("the data workers were stopped")

    def register(self, name: str, dataset) -> None:
        """Send ``dataset`` (pickled for each worker) under ``name``; None forgets it."""
        with self._cond:
            self._check()
            for w, c in enumerate(self._task_conns):
                try:
                    c.send(("dataset", name, dataset))
                except Exception as exc:  # the worker died: its pipe is closed
                    self._lost(w, exc)
                    self._check()

    # -- receiving and dispatching (receiver thread)
    def _receive(self) -> None:
        try:
            self._receive_loop()
        except BaseException as exc:  # never silently: the consumer must not wait for it
            self._fail(DataWorkerError(f"the data workers' receiver thread failed: {exc!r}"))

    def _receive_loop(self) -> None:
        conns = {c: w for w, c in enumerate(self._result_conns)}
        sentinels = {p.sentinel: w for w, p in enumerate(self._procs)}
        while not self._closed:
            for r in _wait(list(conns) + list(sentinels), POLL):
                if r in conns:
                    w = conns[r]
                    try:
                        msg = r.recv()
                    except (EOFError, OSError):
                        del conns[r]  # the worker is gone; its sentinel says how
                        continue
                    except Exception as exc:  # a result that cannot be unpickled
                        self._fail(DataWorkerError(f"reading {self._name(w)}'s result failed: "
                                                   f"{exc!r}", worker=w))
                        continue
                    self._on_result(*msg)
                elif r in sentinels:
                    w = sentinels.pop(r)
                    self._drain(w, conns)
                    self._on_exit(w)
            for w, p in enumerate(self._procs):  # a backstop for the sentinels
                if p.sentinel in sentinels and p.exitcode is not None:
                    del sentinels[p.sentinel]
                    self._drain(w, conns)
                    self._on_exit(w)
            self._speculate()

    def _drain(self, w: int, conns: dict) -> None:
        """What worker ``w`` sent before it ended comes first."""
        conn = self._result_conns[w]
        try:
            while conn in conns and conn.poll(0):
                self._on_result(*conn.recv())
        except Exception:
            pass

    def _on_result(self, wid, gen, tid, payload, seconds=0.0, cpu_seconds=0.0, cpu=None,
                   moved=False) -> None:
        with self._cond:
            if self._held[wid] == tid:
                self._held[wid] = None
            if wid not in self._dead:
                self._idle.add(wid)
            self.stats.add(wid, cpu, seconds, cpu_seconds, moved)
            if payload[0] != "error":
                self._durations.append(seconds)
            task = self._tasks.pop(tid, None)
            if task is not None:
                s = task.stream
                if s is self._stream and task.pos >= s.taken and task.pos not in s.results:
                    s.results[task.pos] = (payload, task)  # the first result wins
                else:  # an item no longer wanted: a stream left, or the other copy came first
                    self._free_slot(task.slot, task.id)
            self._refill()
            self._cond.notify_all()

    def _on_exit(self, w: int) -> None:
        p = self._procs[w]
        p.join(1.0)
        with self._cond:
            self._dead.add(w)
            self._idle.discard(w)
            if self._closed:
                return
            task = self._tasks.get(self._held[w]) if self._held[w] is not None else None
            what = (f" while generating item {task.idx} of {task.stream.name!r}"
                    if task is not None else "")
            self._fail(DataWorkerError(f"{self._name(w)} exited unexpectedly with exit code "
                                       f"{p.exitcode}{what}",
                                       index=task.idx if task is not None else None, worker=w))

    def _lost(self, w: int, exc: BaseException) -> None:
        """Worker ``w`` cannot be sent work (it died while idle): the streams fail at once,
        before its sentinel says how (caller holds the lock)."""
        self._dead.add(w)
        self._idle.discard(w)
        if not self._closed:
            self._fail(DataWorkerError(f"{self._name(w)} cannot be given work, it has ended "
                                       f"({exc!r})", worker=w))

    def _fail(self, err: DataWorkerError) -> None:
        with self._cond:
            if self._fatal is None:
                self._fatal = err
            self._cond.notify_all()

    def _dispatch(self) -> None:
        # caller holds the lock: give pending items to free workers, fastest first
        while self._pending and self._idle and not self._closed and self._fatal is None:
            w = next(x for x in self._rank if x in self._idle)
            tid = self._pending[0]
            task = self._tasks.get(tid)
            if task is None:
                self._pending.popleft()
                continue
            try:
                self._task_conns[w].send(("item", task.stream.gen, tid, task.stream.name,
                                          task.idx, task.slot))
            except Exception as exc:
                self._lost(w, exc)
                return
            self._pending.popleft()
            self._idle.discard(w)
            self._held[w] = tid
            task.worker, task.sent = w, time.monotonic()

    def _refill(self) -> None:
        """Items freed slots or workers make room for, into work (caller holds the lock)."""
        if self._closed or self._fatal is not None:
            return
        s = self._stream
        if s is not None and s.refill is not None:
            s.refill()
        else:
            self._dispatch()

    def _speculate(self) -> None:
        """Hand an item held far longer than items usually take to a free worker as well (a
        worker bound to a core another program occupies cannot move): the stream waits for
        that item, in order, while the others are idle."""
        if not self.speculate:
            return
        with self._cond:
            s = self._stream
            if (s is None or not self._idle or self._closed or self._fatal is not None
                    or len(self._durations) < SPECULATE_MIN_ITEMS):
                return
            limit = max(self.speculate_min_seconds,
                        self.speculate_after * statistics.median(self._durations))
            now = time.monotonic()
            late = sorted((t for t in (self._tasks.get(h) for h in self._held if h is not None)
                           if t is not None and t.stream is s and not t.duplicate
                           and t.sent is not None and now - t.sent > limit
                           and t.pos >= s.taken and t.pos not in s.results
                           and t.pos not in s.duplicated), key=lambda t: t.pos)
            for t in late:
                if not self._idle:
                    break
                tid = next(self._task_ids)
                slot = None
                if t.slot is not None:
                    if not self._free_slots:
                        break
                    slot = self._take_slot(tid)
                self._tasks[tid] = _Task(tid, s, t.pos, t.idx, slot, duplicate=True)
                self._pending.appendleft(tid)
                s.duplicated.add(t.pos)
                self.duplicates += 1
                self._dispatch()

    # -- ring slots: each owned by the task it was handed out with, freed once by its owner
    def _take_slot(self, tid: int) -> int:
        slot = self._free_slots.popleft()
        self._slot_owner[slot] = tid
        return slot

    def _free_slot(self, slot, tid) -> None:
        if slot is not None and self._slot_owner[slot] == tid:
            self._slot_owner[slot] = None
            self._free_slots.append(slot)

    def release(self, tokens) -> None:
        """Free the slots of ``tokens`` ((slot, task) as :meth:`imap` gave them) once their
        items are copied. A token already freed, or of an abandoned stream whose slot was
        handed out since, is ignored."""
        with self._cond:
            for slot, tid in tokens:
                self._free_slot(slot, tid)
            self._refill()
            self._cond.notify_all()

    # -- ordered streams (consumer side)
    def _end_stream(self, s: _Stream | None) -> None:
        """Drop what stream ``s`` left behind, if it is the current one: its items not yet
        handed out (their slots freed now) and its results not taken (freed now); its items in
        work free their slots when they come (caller holds the lock)."""
        if s is None or s is not self._stream:
            return
        self._stream = None
        keep: deque = deque()
        for tid in self._pending:
            t = self._tasks.get(tid)
            if t is not None and t.stream is s:
                del self._tasks[tid]
                self._free_slot(t.slot, t.id)
            else:
                keep.append(tid)
        self._pending = keep
        for _payload, t in s.results.values():
            self._free_slot(t.slot, t.id)
        s.results.clear()
        self._cond.notify_all()

    def imap(self, name: str, indices, window: int, use_ring: bool = False, stop=None):
        """``dataset[i]`` for ``i`` in ``indices``, in that order, as (kind, data) pairs: kind
        "item" (data the item) or "slot" (data a (slot, task) token: the item is in that slot of
        the ring until the caller frees it with :meth:`release`). At most ``window`` items are
        in work or waiting. Starting a stream abandons the previous one (its items in work are
        dropped when they come); ``stop`` (a threading.Event) ends this one early. A failed
        item raises :class:`DataWorkerError` at its turn; a dead worker or a closed pool raises
        at once."""
        if use_ring and self.ring is None:
            raise ValueError("this pool has no slot ring")
        window = max(1, int(window))
        if use_ring:
            window = min(window, self.ring.n)
        src = iter(indices)
        with self._cond:
            self._check()
            self._end_stream(self._stream)
            self._gen += 1
            s = self._stream = _Stream(self._gen, name, use_ring)

        def fill():  # caller holds the lock: more items into work, up to the window
            if self._closed or self._fatal is not None:
                return
            while not s.exhausted and s.sent - s.taken < window and s is self._stream:
                if use_ring and not self._free_slots:
                    break
                try:
                    idx = next(src)
                except StopIteration:
                    s.exhausted = True
                    break
                tid = next(self._task_ids)
                slot = self._take_slot(tid) if use_ring else None
                self._tasks[tid] = _Task(tid, s, s.sent, idx, slot)
                self._pending.append(tid)
                s.sent += 1
            self._dispatch()

        s.refill = fill  # also run by the receiver thread when slots or workers come free
        try:
            while True:
                with self._cond:
                    fill()
                    if s.exhausted and s.taken >= s.sent:
                        return
                    stalled = None
                    while s.taken not in s.results:
                        if s.exhausted and s.taken >= s.sent:  # found out by a later fill
                            return
                        self._check()
                        if s is not self._stream:
                            raise DataWorkerError("this stream was abandoned for a newer one")
                        if stop is not None and stop.is_set():
                            return
                        if not self._rx.is_alive():
                            raise DataWorkerError("the data workers' receiver thread has ended")
                        if use_ring and not self._free_slots and s.sent == s.taken:
                            stalled = stalled or time.monotonic()
                            if time.monotonic() - stalled > SLOT_STALL_SECONDS:
                                raise DataWorkerError(
                                    f"no slot of the ring came back in {SLOT_STALL_SECONDS:g} s "
                                    f"({self.ring.n} slots)")
                        else:
                            stalled = None
                        self._cond.wait(POLL)
                        fill()
                    payload, task = s.results.pop(s.taken)
                    s.taken += 1
                    fill()
                if payload[0] == "error":
                    if task.slot is not None:
                        self.release([(task.slot, task.id)])
                    raise DataWorkerError(f"a data worker failed generating item {task.idx} of "
                                          f"{name!r}:\n{payload[1]}", index=task.idx)
                if payload[0] == "slot":
                    yield "slot", (task.slot, task.id)
                else:
                    if task.slot is not None:  # did not fit the ring: sent whole instead
                        self.release([(task.slot, task.id)])
                    yield payload
        finally:
            with self._cond:
                self._end_stream(s)

    def forget(self, name: str) -> None:
        """Drop the dataset registered as ``name`` from the workers."""
        with contextlib.suppress(DataWorkerError):
            self.register(name, None)

    # -- shutdown
    def _stop_processes(self, timeout: float) -> None:
        for c in self._task_conns:
            try:
                c.send(None)
            except Exception:
                pass
        deadline = time.monotonic() + timeout
        for p in self._procs:
            p.join(max(0.0, deadline - time.monotonic()))
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(2.0)
            if p.is_alive():  # pragma: no cover
                p.kill()
                p.join(1.0)
        for c in [*self._task_conns, *self._result_conns]:
            try:
                c.close()
            except Exception:
                pass

    def close(self, timeout: float = 10.0) -> None:
        """Stop the workers (each finishes the item it is on, for ``timeout`` seconds at most),
        then the receiver thread. Idempotent."""
        with self._cond:
            if self._closed:
                return
            self._closed = True  # nothing is dispatched any more: no other sender on the pipes
            self._pending.clear()
            self._cond.notify_all()
        if self._rx is not threading.current_thread():
            self._rx.join(2 * POLL + 1.0)
        self._stop_processes(timeout)
        with self._cond:
            self._tasks.clear()
            if self._stream is not None:
                self._stream.results.clear()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class OrderedBatches:
    """The training loader: batches of ``batch_size`` consecutive indices of ``sampler``, the
    last incomplete batch dropped, made by ``pool`` (item by item, the free workers taking the
    next items, ``window`` items at most in work or waiting), stacked out of the slot ring in
    one copy (into fresh pinned memory with ``pin_memory``) or collated by ``default_collate``,
    in a thread, ``prefetch_batches`` ahead, as the DataLoader's. ``len`` is the number of
    batches; ``sampler.epoch`` and ``sampler.start`` are read when iteration starts."""

    def __init__(self, dataset, batch_size: int, sampler, pool: WorkerPool,
                 window: int | None = None, pin_memory: bool = False, name: str = "train",
                 prefetch_batches: int = 2, use_ring: bool | None = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.sampler = sampler
        self.pool = pool
        self.window = int(window or default_window(pool.num_workers, batch_size))
        self.pin_memory = bool(pin_memory)
        self.name = name
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.use_ring = pool.ring is not None if use_ring is None else bool(use_ring)
        self.item_stats = pool.stats
        self._stop: threading.Event | None = None
        pool.register(name, dataset)

    @property
    def num_workers(self) -> int:
        return self.pool.num_workers

    @property
    def shared_gb(self) -> float:
        """Shared memory of the slot ring (counted in no process's private memory)."""
        return self.pool.ring.nbytes / 2**30 if self.pool.ring is not None else 0.0

    def __len__(self) -> int:
        return len(self.sampler) // self.batch_size

    def _collate(self, got: list):
        import torch
        from torch.utils.data import default_collate

        ring = self.pool.ring
        if got and all(kind == "slot" for kind, _d in got):
            slots = [d[0] for _k, d in got]
            out = None
            if self.pin_memory:  # fresh each batch: a non_blocking copy may still read the last
                out = [torch.empty((len(slots), *shape), dtype=dtype, pin_memory=True)
                       for shape, dtype in ring.spec]
            return ring.batch(slots, out=out)
        items = [ring.item(d[0]) if kind == "slot" else d for kind, d in got]
        if self.pin_memory and _stackable(items):  # one copy, into fresh pinned memory
            return [torch.stack([it[j] for it in items], 0,
                                out=torch.empty((len(items), *items[0][j].shape),
                                                dtype=items[0][j].dtype, pin_memory=True))
                    for j in range(len(items[0]))]
        batch = default_collate(items)
        if self.pin_memory:
            from torch.utils.data._utils.pin_memory import pin_memory as pin

            batch = pin(batch)
        return batch

    def _batches(self, stream):
        B = self.batch_size
        while True:
            got: list = []
            tokens: list = []
            try:
                for _ in range(B):
                    try:
                        kind, data = next(stream)
                    except StopIteration:
                        break
                    got.append((kind, data))
                    if kind == "slot":
                        tokens.append(data)
                if len(got) < B:  # the last incomplete batch is dropped
                    return
                batch = self._collate(got)
            finally:
                if tokens:
                    self.pool.release(tokens)
            yield batch

    def __iter__(self):
        indices = list(iter(self.sampler))
        indices = indices[:len(indices) // self.batch_size * self.batch_size]
        if self._stop is not None:
            self._stop.set()  # the previous iteration's batcher, if still at work
        stop = threading.Event()
        self._stop = stop
        out: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        done = object()
        stream = self.pool.imap(self.name, indices, self.window, use_ring=self.use_ring,
                                stop=stop)

        def put(x) -> bool:  # never blocks for good: an abandoned queue is left once stopped
            while not stop.is_set():
                try:
                    out.put(x, timeout=POLL)
                    return True
                except queue.Full:
                    continue
            return False

        def run():
            batches = self._batches(stream)
            try:
                for b in batches:
                    if not put(b):
                        return
                put(done)
            except BaseException as exc:  # handed to the consumer
                put(exc)
            finally:
                batches.close()
                stream.close()

        t = threading.Thread(target=run, name=f"lcmsdeconv-batcher-{self.name}", daemon=True)
        t.start()
        try:
            while True:
                try:
                    b = out.get(timeout=POLL)  # a timeout: Ctrl+C is handled while waiting
                except queue.Empty:
                    if not t.is_alive() and out.empty():
                        raise DataWorkerError("the batcher thread ended without a batch"
                                              ) from None
                    continue
                if b is done:
                    return
                if isinstance(b, BaseException):
                    raise b
                yield b
        finally:
            stop.set()

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        self.pool.close()


def _stackable(items: list) -> bool:
    """Whether default_collate would stack ``items`` element by element into a list of tensors:
    tuples of tensors, element k of the same shape and dtype in every item."""
    import torch

    first = items[0] if items else None
    if not isinstance(first, tuple) or not first:
        return False
    for it in items:
        if not isinstance(it, tuple) or len(it) != len(first):
            return False
        for x, y in zip(it, first):
            if not (isinstance(x, torch.Tensor) and type(x) is torch.Tensor
                    and x.shape == y.shape and x.dtype == y.dtype and not x.is_nested):
                return False
    return True


def default_window(num_workers: int, batch_size: int) -> int:
    """Items in work or waiting at most: five per worker (the simulation of 6-8 performance and
    12-16 efficient cores at the measured spread of item costs keeps them 99 % busy; two per
    worker gives 70-77 %), and at least two batches."""
    return max(5 * int(num_workers), 2 * int(batch_size))


def ring_slots(window: int, batch_size: int, num_workers: int) -> int:
    """Slots of a ring for ``window`` items in work or waiting: those, the batch being copied
    out, and a few for items handed out twice (SPECULATE_AFTER)."""
    return int(window) + int(batch_size) + max(2, int(num_workers) // 4)


def ordered_batches(pool: WorkerPool, name: str, dataset, n: int, batch_size: int,
                    window: int | None = None):
    """``default_collate`` batches of ``dataset[0 .. n-1]`` made by ``pool`` (the last one may
    be short, as a DataLoader without drop_last gives): the validation spectra and the comb's
    items, which made DataLoaders of their own before."""
    from torch.utils.data import default_collate

    pool.register(name, dataset)
    try:
        stream = pool.imap(name, range(n), window or default_window(pool.num_workers,
                                                                    batch_size))
        while True:
            got = [data for _k, data in itertools.islice(stream, batch_size)]
            if not got:
                return
            yield default_collate(got)
    finally:
        pool.forget(name)
