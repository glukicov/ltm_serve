"""FastAPI service for online TabFM inference.

    uv run uvicorn ltm_serve.service.app:app --port 8080

Two API styles over the same engine:
* native JSON (`/v1/contexts...`): register a training table once, then predict against its id;
* Open Inference Protocol v2 (`/v2/models/{context_id}/infer`): the tensor API that Triton and KServe also speak,
  so one load generator can drive all three stacks with identical payloads.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel, Field

from ltm_serve.data import make_task
from ltm_serve.model import load_model
from ltm_serve.service.engine import LATENCY_BUCKETS, Context, Engine

REQUEST_LATENCY = Histogram(
    "ltm_request_seconds", "End-to-end server-side request time", ["endpoint"], buckets=LATENCY_BUCKETS
)


@dataclass(frozen=True)
class Settings:
    device: str = "auto"
    dtype: str = "bf16"
    checkpoint_dir: str | None = None
    torch_threads: int = 0
    cache_budget_mb: int = 4096
    max_batch_rows: int = 256
    max_delay_ms: float = 0.0
    # e.g. "demo:n_train=512,n_features=10,n_classes=2,n_estimators=4" -> a synthetic context registered at start-up
    demo_context: str | None = None
    compile_mode: str | None = None  # torch.compile mode for the query forward, e.g. "reduce-overhead"
    bucket_rows: bool = False  # implied by compile_mode

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            device=env.get("LTM_DEVICE", cls.device),
            dtype=env.get("LTM_DTYPE", cls.dtype),
            checkpoint_dir=env.get("LTM_CHECKPOINT_DIR"),
            torch_threads=int(env.get("LTM_TORCH_THREADS", cls.torch_threads)),
            cache_budget_mb=int(env.get("LTM_CACHE_BUDGET_MB", cls.cache_budget_mb)),
            max_batch_rows=int(env.get("LTM_MAX_BATCH_ROWS", cls.max_batch_rows)),
            max_delay_ms=float(env.get("LTM_MAX_DELAY_MS", cls.max_delay_ms)),
            demo_context=env.get("LTM_DEMO_CONTEXT"),
            compile_mode=env.get("LTM_COMPILE_MODE") or None,
            bucket_rows=env.get("LTM_BUCKET_ROWS", "0") == "1",
        )


class ContextIn(BaseModel):
    columns: list[str]
    rows: list[list[Any]] = Field(description="training rows, values in `columns` order")
    labels: list[Any]
    n_estimators: int = Field(default=4, ge=1, le=64)
    context_id: str | None = None


class SyntheticContextIn(BaseModel):
    n_train: int = Field(default=512, ge=8, le=50_000)
    n_features: int = Field(default=10, ge=1, le=500)
    n_classes: int = Field(default=2, ge=2, le=10)
    n_estimators: int = Field(default=4, ge=1, le=64)
    seed: int = 0
    context_id: str | None = None


class ContextOut(BaseModel):
    context_id: str
    n_train: int
    columns: list[str]
    classes: list[str]
    n_estimators: int
    encode_ms: float
    cache_mb: float


class PredictIn(BaseModel):
    rows: list[list[Any]]


class PredictOut(BaseModel):
    classes: list[str]
    labels: list[str]
    probabilities: list[list[float]]
    timing: dict[str, float]


class V2Tensor(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[Any]


class V2Request(BaseModel):
    id: str | None = None
    inputs: list[V2Tensor]


def _describe(ctx: Context) -> ContextOut:
    assert ctx.classifier.cache is not None
    return ContextOut(
        context_id=ctx.context_id,
        n_train=ctx.classifier.cache.n_train,
        columns=ctx.columns,
        classes=ctx.classes,
        n_estimators=ctx.classifier.clf.n_estimators,
        encode_ms=ctx.encode_seconds * 1e3,
        cache_mb=ctx.cache_bytes / 1e6,
    )


def _frame(rows: list[list[Any]], columns: list[str]) -> pd.DataFrame:
    """JSON rows -> DataFrame with real dtypes. A numeric column containing `null` would otherwise stay `object`, and
    TabFM's encoder would silently treat it as categorical, changing predictions."""
    return pd.DataFrame(rows, columns=columns).infer_objects()


def _parse_demo(spec: str) -> tuple[str, SyntheticContextIn]:
    name, _, params = spec.partition(":")
    kwargs = {k: int(v) for k, v in (kv.split("=") for kv in params.split(",") if kv)}
    return name, SyntheticContextIn(**kwargs, context_id=name)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    state: dict[str, Engine] = {}

    async def register_synthetic(body: SyntheticContextIn) -> Context:
        task = make_task(body.n_train, 1, body.n_features, body.n_classes, seed=body.seed)
        return await state["engine"].register(task.X_train, task.y_train, body.n_estimators, body.context_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        loaded = load_model(cfg.device, cfg.dtype, Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None)
        print(f"model loaded on {loaded.device} ({cfg.dtype}) in {loaded.load_seconds:.1f}s", flush=True)
        state["engine"] = Engine(
            loaded,
            cfg.cache_budget_mb * 1_000_000,
            cfg.max_batch_rows,
            cfg.max_delay_ms,
            compile_mode=cfg.compile_mode,
            bucket_rows=cfg.bucket_rows,
        )
        if cfg.demo_context:
            name, body = _parse_demo(cfg.demo_context)
            ctx = await register_synthetic(body)
            print(
                f"demo context {name!r} encoded in {ctx.encode_seconds:.1f}s, warm-up {ctx.warmup_seconds:.1f}s "
                f"(compile_mode={cfg.compile_mode})",
                flush=True,
            )
        yield
        state["engine"].executor.shutdown(wait=False)

    app = FastAPI(title="ltm-serve", lifespan=lifespan)

    def engine() -> Engine:
        if "engine" not in state:
            raise HTTPException(503, "model is loading")
        return state["engine"]

    def context(context_id: str) -> Context:
        ctx = engine().get(context_id)
        if ctx is None:
            raise HTTPException(404, f"unknown context {context_id!r}")
        return ctx

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        eng = engine()
        return {"status": "ready", "device": str(eng.loaded.device), "contexts": len(eng.contexts())}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/contexts", response_model=ContextOut)
    async def create_context(body: ContextIn) -> ContextOut:
        X = _frame(body.rows, body.columns)
        ctx = await engine().register(X, np.asarray(body.labels), body.n_estimators, body.context_id)
        return _describe(ctx)

    @app.post("/v1/contexts/synthetic", response_model=ContextOut)
    async def create_synthetic_context(body: SyntheticContextIn) -> ContextOut:
        return _describe(await register_synthetic(body))

    @app.get("/v1/contexts", response_model=list[ContextOut])
    async def list_contexts() -> list[ContextOut]:
        return [_describe(c) for c in engine().contexts()]

    @app.delete("/v1/contexts/{context_id}")
    async def delete_context(context_id: str) -> dict[str, bool]:
        if not await engine().delete(context_id):
            raise HTTPException(404, f"unknown context {context_id!r}")
        return {"deleted": True}

    @app.post("/v1/contexts/{context_id}/predict", response_model=PredictOut)
    async def predict(context_id: str, body: PredictIn) -> PredictOut:
        t0 = time.perf_counter()
        ctx = context(context_id)
        proba, timing = await ctx.batcher.submit(_frame(body.rows, ctx.columns))
        classes = ctx.classes
        REQUEST_LATENCY.labels("v1_predict").observe(time.perf_counter() - t0)
        return PredictOut(
            classes=classes,
            labels=[classes[i] for i in proba.argmax(axis=1)],
            probabilities=proba.tolist(),
            timing=timing | {"server_ms": (time.perf_counter() - t0) * 1e3},
        )

    @app.get("/v2/models/{context_id}/ready")
    async def v2_ready(context_id: str) -> dict[str, bool]:
        context(context_id)
        return {"ready": True}

    @app.post("/v2/models/{context_id}/infer")
    async def v2_infer(context_id: str, body: V2Request) -> dict[str, Any]:
        t0 = time.perf_counter()
        ctx = context(context_id)
        tensor = body.inputs[0]
        rows = np.asarray(tensor.data, dtype=np.float32).reshape(tensor.shape)
        proba, _ = await ctx.batcher.submit(pd.DataFrame(rows, columns=ctx.columns))
        REQUEST_LATENCY.labels("v2_infer").observe(time.perf_counter() - t0)
        return {
            "model_name": context_id,
            "id": body.id,
            "outputs": [
                {
                    "name": "probabilities",
                    "shape": list(proba.shape),
                    "datatype": "FP32",
                    "data": proba.astype(np.float32).ravel().tolist(),
                }
            ],
        }

    return app


app = create_app()
