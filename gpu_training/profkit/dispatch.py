"""An order-preserving dynamic dispatcher (a prototype): training's data loading, with each
worker taking the next item the moment it is free instead of a fixed share of whole batches.

PyTorch's DataLoader gives batch k to worker k mod N and delivers the batches in order; each
worker keeps at most ``prefetch_factor`` batches in hand. On a processor with fast and slow
cores the fast workers finish their batches and then wait for the slow ones, so the slowest
cores set the pace of all. Here the item indices go into one queue, and every worker takes the
next one as soon as it has delivered the last: fast cores make more items. The consumer keeps a
bounded window of items issued but not yet consumed (``window``), and assembles batches of
``batch_size`` strictly in order, stacking the items as DataLoader's collate does (into pinned
memory where asked), so that training sees exactly the batches DataLoader would give it. An item
that failed is raised at its turn, as DataLoader raises a failed batch at its turn: the batches
before it are delivered first, however early its error arrived. Items travel from the workers
through shared memory, as DataLoader's do (the tensor reductions of torch.multiprocessing).
"""

from __future__ import annotations

import os
import queue
import time
import traceback

#: the worker this process is (set in a dispatcher's worker; None elsewhere)
_WORKER_ID: int | None = None
#: seconds between checks that the workers are alive while the consumer waits
POLL = 1.0


def current_worker() -> int | None:
    """The id of the dispatcher worker running this code, or None outside one."""
    return _WORKER_ID


def _worker_loop(wid: int, dataset, tasks, results, stop, init) -> None:
    """A worker: take ``(position, index)`` from ``tasks`` until None (or ``stop``), make the
    item, send ``(position, worker, start, end, item, error)``."""
    global _WORKER_ID
    import torch

    _WORKER_ID = wid
    torch.set_num_threads(1)
    if init is not None:
        init(wid)
    while True:
        task = tasks.get()
        if task is None or stop.is_set():
            break
        pos, idx = task
        t0 = time.perf_counter()
        try:
            item, err = dataset[idx], None
        except Exception:  # sent to the consumer, which raises it
            item, err = None, traceback.format_exc()
        results.put((pos, wid, t0, time.perf_counter(), item, err))


def collate(items: list, pin_memory: bool = False) -> list:
    """Items (tuples of tensors) stacked into a batch, element by element, as DataLoader's
    default collate does (a list of tensors); into page-locked memory with ``pin_memory``."""
    import torch

    out = []
    for j in range(len(items[0])):
        parts = [it[j] for it in items]
        if pin_memory:
            first = parts[0]
            buf = torch.empty((len(parts), *first.shape), dtype=first.dtype, pin_memory=True)
            out.append(torch.stack(parts, 0, out=buf))
        else:
            out.append(torch.stack(parts, 0))
    return out


class OrderedDispatcher:
    """Batches of ``dataset[indices[...]]`` in order, made by ``num_workers`` processes that each
    take the next index when free. ``window``: items issued but not yet consumed (default three
    per worker, and at least two batches); ``worker_init(worker_id)`` runs first in every worker
    (priority, affinity); ``mp_context``: the start method (default: the platform's, or
    LCMSDECONV_MP_CONTEXT, as training's loaders). ``log`` holds ``(position, worker, start,
    end)`` of every item made (received without an error) and ``wait_seconds`` the consumer's
    time spent waiting."""

    def __init__(self, dataset, indices, batch_size: int, num_workers: int, window=None,
                 worker_init=None, pin_memory: bool = False, mp_context: str | None = None,
                 timeout: float = 600.0, drop_last: bool = True):
        if num_workers < 1:
            raise ValueError("the dispatcher needs at least one worker")
        self.dataset = dataset
        self.indices = indices
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.window = int(window or max(3 * num_workers, 2 * batch_size))
        self.worker_init = worker_init
        self.pin_memory = bool(pin_memory)
        self.timeout = float(timeout)
        self.drop_last = drop_last
        self.method = mp_context or os.environ.get("LCMSDECONV_MP_CONTEXT") or None
        self.log: list[tuple] = []
        self.wait_seconds = 0.0
        self.assembly_seconds = 0.0
        self._workers: list = []

    def __len__(self) -> int:
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _start(self) -> None:
        import torch.multiprocessing as mp

        ctx = mp.get_context(self.method)
        self._tasks, self._results, self._stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        self._workers = [ctx.Process(target=_worker_loop, args=(w, self.dataset, self._tasks,
                                                                self._results, self._stop,
                                                                self.worker_init), daemon=True)
                         for w in range(self.num_workers)]
        for p in self._workers:
            p.start()

    def _receive(self):
        start = time.perf_counter()
        while True:
            try:
                return self._results.get(timeout=POLL)
            except queue.Empty:
                dead = [(w, p.exitcode) for w, p in enumerate(self._workers)
                        if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"dispatcher worker {dead[0][0]} exited (code "
                                       f"{dead[0][1]})") from None
                if time.perf_counter() - start > self.timeout:
                    raise RuntimeError(f"no item from the workers within {self.timeout:g} s"
                                       ) from None

    def __iter__(self):
        n_items = len(self.indices)
        n_batches = len(self)
        self._start()
        issued = consumed = 0
        pending: dict = {}

        def issue():
            nonlocal issued
            while issued < n_items and issued - consumed < self.window:
                self._tasks.put((issued, self.indices[issued]))
                issued += 1

        try:
            issue()
            for b in range(n_batches):
                items = []
                for pos in range(b * self.batch_size, min((b + 1) * self.batch_size, n_items)):
                    while pos not in pending:
                        t = time.perf_counter()
                        p, wid, t0, t1, item, err = self._receive()
                        self.wait_seconds += time.perf_counter() - t
                        pending[p] = (item, err, wid)  # an error waits for its turn, as an item
                        if err is None:
                            self.log.append((p, wid, t0, t1))
                    item, err, wid = pending.pop(pos)
                    if err is not None:
                        raise RuntimeError(f"dispatcher worker {wid} failed on item "
                                           f"{self.indices[pos]}:\n{err}")
                    items.append(item)
                    consumed = pos + 1
                    issue()
                t = time.perf_counter()
                batch = collate(items, self.pin_memory)
                self.assembly_seconds += time.perf_counter() - t
                yield batch
        finally:
            self.close()

    def close(self) -> None:
        """Stop the workers: each finishes the item it is making; items still queued are
        dropped."""
        if not self._workers:
            return
        self._stop.set()
        for _ in self._workers:
            try:
                self._tasks.put(None)
            except (OSError, ValueError):
                break
        deadline = time.perf_counter() + 15.0
        while any(p.is_alive() for p in self._workers) and time.perf_counter() < deadline:
            try:  # a worker blocked on a full pipe ends once it is read
                self._results.get(timeout=0.1)
            except (queue.Empty, OSError, EOFError):
                pass
        for p in self._workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        for q in (self._tasks, self._results):
            q.cancel_join_thread()
            q.close()
        self._workers = []

    def per_worker(self) -> dict:
        """Items each worker made and the seconds it spent making them, over the whole run."""
        out: dict = {}
        for _p, wid, t0, t1 in self.log:
            w = out.setdefault(wid, {"items": 0, "busy_seconds": 0.0})
            w["items"] += 1
            w["busy_seconds"] += t1 - t0
        return out
