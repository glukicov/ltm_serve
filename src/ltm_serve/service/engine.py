"""Serving engine: a context registry with a memory budget, and an exact dynamic batcher per context.

TabFM is an in-context model, so a "model" in the serving sense is (weights + a training table). The weights are
shared; each registered context owns its encoded cache (hundreds of MB to GB). That makes cache memory, not
weights, the scaling limit — so contexts live in an LRU bounded by bytes, and eviction is explicit.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from tabfm import TabFMClassifier

from ltm_serve.context_cache import CachedTabFMClassifier
from ltm_serve.model import LoadedModel, synchronize

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
QUEUE_WAIT = Histogram("ltm_queue_wait_seconds", "Wait before the request's batch started", buckets=LATENCY_BUCKETS)
INFERENCE = Histogram("ltm_inference_seconds", "Preprocessing + forward per batch", buckets=LATENCY_BUCKETS)
BATCH_ROWS = Histogram("ltm_batch_rows", "Query rows per batch", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
BATCH_REQUESTS = Histogram("ltm_batch_requests", "Requests merged per executed batch", buckets=(1, 2, 4, 8, 16, 32, 64))
CONTEXTS = Gauge("ltm_contexts", "Registered contexts")
CACHE_BYTES = Gauge("ltm_cache_bytes", "Bytes held by context caches")
EVICTIONS = Counter("ltm_context_evictions_total", "Contexts evicted to respect the cache budget")
INFLIGHT = Gauge("ltm_inflight_requests", "Prediction requests queued or executing (the autoscaling signal)")


@dataclass
class _Pending:
    rows: pd.DataFrame
    enqueued: float
    future: asyncio.Future[tuple[np.ndarray, dict[str, float]]]


@dataclass
class Context:
    context_id: str
    columns: list[str]
    classifier: CachedTabFMClassifier
    encode_seconds: float
    warmup_seconds: float = 0.0
    batcher: DynamicBatcher = field(init=False)

    @property
    def cache_bytes(self) -> int:
        return self.classifier.cache.nbytes() if self.classifier.cache is not None else 0

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.classifier.clf.classes_]


class DynamicBatcher:
    """Merge concurrent requests for one context into a single forward pass.

    The worker takes the first queued request, then keeps collecting until either `max_rows` rows are gathered or
    `max_delay_s` has passed since that first request. With `max_delay_s = 0` it never waits on purpose but still
    merges everything that queued up while the previous batch was running — the throughput win under load comes
    for free, the latency cost is opt-in. Exactness: query rows never attend to one another, so batching does not
    change any prediction.
    """

    def __init__(self, ctx: Context, executor: ThreadPoolExecutor, max_rows: int, max_delay_s: float) -> None:
        self.ctx = ctx
        self.executor = executor
        self.max_rows = max_rows
        self.max_delay_s = max_delay_s
        self.queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._carry: _Pending | None = None  # a request that did not fit in the previous batch

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        while not self.queue.empty():  # fail waiters instead of leaving them hanging
            pending = self.queue.get_nowait()
            if not pending.future.done():
                pending.future.set_exception(LookupError(f"context {self.ctx.context_id} was removed"))

    async def submit(self, rows: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
        loop = asyncio.get_running_loop()
        pending = _Pending(rows, time.perf_counter(), loop.create_future())
        INFLIGHT.inc()
        try:
            await self.queue.put(pending)
            return await pending.future
        finally:
            INFLIGHT.dec()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first, self._carry = (self._carry, None) if self._carry is not None else (await self.queue.get(), None)
            batch = [first]
            n_rows = len(first.rows)
            deadline = first.enqueued + self.max_delay_s
            while n_rows < self.max_rows:
                timeout = deadline - time.perf_counter()
                try:
                    if timeout <= 0:
                        item = self.queue.get_nowait()
                    else:
                        item = await asyncio.wait_for(self.queue.get(), timeout)
                except (asyncio.QueueEmpty, TimeoutError):
                    break
                if n_rows + len(item.rows) > self.max_rows:
                    # Keep batches within the warmed-up shape buckets: a compiled/CUDA-graph forward would otherwise
                    # recompile for a never-seen shape in the middle of serving.
                    self._carry = item
                    break
                batch.append(item)
                n_rows += len(item.rows)
            started = time.perf_counter()
            frame = pd.concat([p.rows for p in batch], ignore_index=True) if len(batch) > 1 else batch[0].rows
            try:
                proba = await loop.run_in_executor(self.executor, self._infer, frame)
            except Exception as exc:  # propagate to every waiter rather than killing the worker
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                continue
            inference_s = time.perf_counter() - started
            INFERENCE.observe(inference_s)
            BATCH_ROWS.observe(n_rows)
            BATCH_REQUESTS.observe(len(batch))
            offset = 0
            for p in batch:
                n = len(p.rows)
                QUEUE_WAIT.observe(started - p.enqueued)
                timing = {
                    "queue_ms": (started - p.enqueued) * 1e3,
                    "inference_ms": inference_s * 1e3,
                    "batch_rows": float(n_rows),
                    "batch_requests": float(len(batch)),
                }
                if not p.future.done():
                    p.future.set_result((proba[offset : offset + n], timing))
                offset += n

    def _infer(self, frame: pd.DataFrame) -> np.ndarray:
        proba = self.ctx.classifier.predict_proba(frame)
        synchronize(self.ctx.classifier.device)
        return proba


class Engine:
    """Owns the model, the single GPU/CPU executor, and the context registry."""

    def __init__(
        self,
        loaded: LoadedModel,
        cache_budget_bytes: int,
        max_batch_rows: int,
        max_delay_ms: float,
        compile_mode: str | None = None,
        bucket_rows: bool = False,
    ) -> None:
        self.loaded = loaded
        self.compile_mode = compile_mode
        self.bucket_rows = bucket_rows or compile_mode is not None
        self.cache_budget_bytes = cache_budget_bytes
        self.max_batch_rows = max_batch_rows
        self.max_delay_s = max_delay_ms / 1e3
        # One worker thread: the accelerator is a single queue anyway, and serialising access avoids
        # oversubscribing CPU threads or interleaving MPS/CUDA command buffers across contexts.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tabfm")
        self._contexts: OrderedDict[str, Context] = OrderedDict()

    def contexts(self) -> list[Context]:
        return list(self._contexts.values())

    def get(self, context_id: str) -> Context | None:
        ctx = self._contexts.get(context_id)
        if ctx is not None:
            self._contexts.move_to_end(context_id)  # LRU touch
        return ctx

    async def register(
        self, X: pd.DataFrame, y: np.ndarray, n_estimators: int, context_id: str | None = None
    ) -> Context:
        loop = asyncio.get_running_loop()
        clf = TabFMClassifier(self.loaded.model, n_estimators=n_estimators, random_state=0)

        def build() -> tuple[CachedTabFMClassifier, float]:
            # Fit, encode and warm up on the executor thread that will serve the context: CUDA-graph trees are
            # recorded per thread, so capturing them anywhere else would force a second capture under traffic.
            cached = CachedTabFMClassifier(clf, compile_mode=self.compile_mode, bucket_rows=self.bucket_rows).fit(X, y)
            warm_s = cached.warm_up(self.max_batch_rows) if self.compile_mode else 0.0
            return cached, warm_s

        cached, warm_s = await loop.run_in_executor(self.executor, build)
        ctx = Context(context_id or uuid.uuid4().hex[:12], list(X.columns), cached, cached.encode_seconds, warm_s)
        ctx.batcher = DynamicBatcher(ctx, self.executor, self.max_batch_rows, self.max_delay_s)
        ctx.batcher.start()
        # Registry mutations all happen on the event-loop thread, so no lock is needed around them.
        if (old := self._contexts.pop(ctx.context_id, None)) is not None:
            await old.batcher.stop()
        self._contexts[ctx.context_id] = ctx
        await self._evict_over_budget(keep=ctx.context_id)
        self._update_gauges()
        return ctx

    async def delete(self, context_id: str) -> bool:
        ctx = self._contexts.pop(context_id, None)
        if ctx is None:
            return False
        await ctx.batcher.stop()
        self._update_gauges()
        return True

    async def _evict_over_budget(self, keep: str) -> None:
        while sum(c.cache_bytes for c in self._contexts.values()) > self.cache_budget_bytes and len(self._contexts) > 1:
            victim_id = next(k for k in self._contexts if k != keep)
            victim = self._contexts.pop(victim_id)
            await victim.batcher.stop()
            EVICTIONS.inc()

    def _update_gauges(self) -> None:
        CONTEXTS.set(len(self._contexts))
        CACHE_BYTES.set(sum(c.cache_bytes for c in self._contexts.values()))
