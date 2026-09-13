"""Latency benchmark harness: one row of results per configuration, written to parquet.

Timing rules that make the numbers trustworthy:
* the model is loaded once per (device, dtype) and reused, so load time never leaks into latency;
* every configuration gets warm-up calls first (allocator growth, kernel selection, lazy init);
* the device is synchronised before reading the clock (GPU calls return before the work is done);
* we report percentiles over repeats, not a single run, and accuracy next to latency.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tabfm import TabFMClassifier

from ltm_serve.context_cache import CachedTabFMClassifier
from ltm_serve.data import Task, make_task
from ltm_serve.model import LoadedModel, load_model, synchronize

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass(frozen=True)
class Config:
    mode: str = "cached"  # "stock" = TabFMClassifier.predict_proba, "cached" = CachedTabFMClassifier
    device: str = "mps"
    dtype: str = "bf16"
    n_train: int = 256
    n_query: int = 1  # rows per request
    n_features: int = 10
    n_classes: int = 2
    n_estimators: int = 4
    member_batch_size: int | None = None  # ensemble members per forward; None = all at once
    repeats: int = 5
    warmup: int = 2
    eval_rows: int = 256  # extra held-out rows to measure accuracy (0 = skip)
    compile_mode: str | None = None  # torch.compile mode for the cached query forward
    bucket_rows: bool = False  # pad query rows to powers of two (stable shapes for compiled / CUDA graphs)
    class_sep: float = 1.0  # synthetic task difficulty: lower = harder, so accuracy can move with ensemble size


@dataclass
class Result:
    config: Config
    latencies_s: list[float]
    fit_s: float
    encode_s: float = 0.0
    cache_mb: float = 0.0
    accuracy: float = float("nan")
    extra: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        lat = np.asarray(self.latencies_s)
        return {
            **asdict(self.config),
            "p50_ms": float(np.percentile(lat, 50) * 1e3),
            "p95_ms": float(np.percentile(lat, 95) * 1e3),
            "p99_ms": float(np.percentile(lat, 99) * 1e3),
            "mean_ms": float(lat.mean() * 1e3),
            "min_ms": float(lat.min() * 1e3),
            "rows_per_s": float(self.config.n_query / np.median(lat)),
            "fit_s": self.fit_s,
            "encode_s": self.encode_s,
            "cache_mb": self.cache_mb,
            "accuracy": self.accuracy,
            **self.extra,
        }


class ModelPool:
    """Keeps one loaded model per (device, dtype) for the whole sweep."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], LoadedModel] = {}

    def get(self, device: str, dtype: str) -> LoadedModel:
        key = (device, dtype)
        if key not in self._models:
            self._models.clear()  # at most one 1.6B model in memory at a time
            if device == "mps":
                torch.mps.empty_cache()
            self._models[key] = load_model(device, dtype)
        return self._models[key]


def timed(fn: Callable[[], object], device: torch.device, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        out.append(time.perf_counter() - t0)
    return out


def run_config(cfg: Config, pool: ModelPool) -> Result:
    lm = pool.get(cfg.device, cfg.dtype)
    task = make_task(
        cfg.n_train, max(cfg.n_query, cfg.eval_rows), cfg.n_features, cfg.n_classes, class_sep=cfg.class_sep
    )
    query = task.X_test.iloc[: cfg.n_query]
    base = TabFMClassifier(lm.model, n_estimators=cfg.n_estimators, batch_size=cfg.member_batch_size, random_state=0)

    t0 = time.perf_counter()
    if cfg.mode == "stock":
        clf = base.fit(task.X_train, task.y_train)
        fit_s = time.perf_counter() - t0
        lat = timed(lambda: clf.predict_proba(query), lm.device, cfg.warmup, cfg.repeats)
        # Accuracy is only measured on the cached path: it is exact, and stock inference on eval rows is slow.
        return Result(cfg, lat, fit_s)

    cached = CachedTabFMClassifier(
        base, member_batch_size=cfg.member_batch_size, compile_mode=cfg.compile_mode, bucket_rows=cfg.bucket_rows
    ).fit(task.X_train, task.y_train)
    fit_s = time.perf_counter() - t0
    assert cached.cache is not None
    breakdown: list[dict[str, float]] = []

    def call() -> None:
        cached.predict_proba(query)
        breakdown.append(cached.last_timings)

    lat = timed(call, lm.device, cfg.warmup, cfg.repeats)
    measured = breakdown[cfg.warmup :]
    accuracy, log_loss = _quality(cached.predict_proba, task, cfg.eval_rows)
    return Result(
        cfg,
        lat,
        fit_s,
        encode_s=cached.encode_seconds,
        cache_mb=cached.cache.nbytes() / 1e6,
        accuracy=accuracy,
        extra={k: float(np.median([t[k] for t in measured])) for k in measured[0]} | {"log_loss": log_loss},
    )


def _quality(predict_proba: Callable[[pd.DataFrame], np.ndarray], task: Task, eval_rows: int) -> tuple[float, float]:
    """Accuracy and log loss on held-out rows. Log loss also sees calibration, which ensembles tend to improve."""
    if eval_rows == 0:
        return float("nan"), float("nan")
    X, y = task.X_test.iloc[:eval_rows], task.y_test[:eval_rows]
    proba = np.concatenate([predict_proba(X.iloc[s : s + 64]) for s in range(0, len(X), 64)])
    log_loss = float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))
    return float((proba.argmax(axis=1) == y).mean()), log_loss


@contextmanager
def stage_timer(model: Any, device: torch.device) -> Iterator[dict[str, float]]:
    """Accumulate synchronised wall time per top-level TabFM stage via forward hooks."""
    totals: dict[str, float] = {}
    starts: dict[str, float] = {}
    stages = ["cell_embedder", "col_embedder", "row_interactor", "col_embedder_2", "row_interactor_2", "icl_predictor"]
    handles = []
    for name in stages:
        module = getattr(model, name)

        def pre(_m: Any, _i: Any, name: str = name) -> None:
            synchronize(device)
            starts[name] = time.perf_counter()

        def post(_m: Any, _i: Any, _o: Any, name: str = name) -> None:
            synchronize(device)
            totals[name] = totals.get(name, 0.0) + time.perf_counter() - starts[name]

        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    try:
        yield totals
    finally:
        for h in handles:
            h.remove()


def stage_breakdown(cfg: Config, pool: ModelPool) -> dict[str, Any]:
    """Where does stock-inference time go, per stage, for one shape?"""
    lm = pool.get(cfg.device, cfg.dtype)
    task = make_task(cfg.n_train, cfg.n_query, cfg.n_features, cfg.n_classes)
    clf = TabFMClassifier(lm.model, n_estimators=cfg.n_estimators, random_state=0).fit(task.X_train, task.y_train)
    clf.predict_proba(task.X_test)  # warm-up
    with stage_timer(lm.model, lm.device) as totals:
        t0 = time.perf_counter()
        clf.predict_proba(task.X_test)
        total = time.perf_counter() - t0
    return {**asdict(cfg), "total_s": total, **{f"{k}_s": v for k, v in totals.items()}}


def bf16_noise_floor(cfg: Config, pool: ModelPool) -> dict[str, Any]:
    """How much do *stock* bf16 outputs move when only the batch composition changes?

    Mathematically, predicting rows [a, b] together or separately is identical (queries never attend to each
    other), so any difference is floating-point kernel noise. The cache should be judged against this floor.
    """
    lm = pool.get(cfg.device, cfg.dtype)
    task = make_task(cfg.n_train, cfg.n_query, cfg.n_features, cfg.n_classes)
    clf = TabFMClassifier(lm.model, n_estimators=cfg.n_estimators, random_state=0).fit(task.X_train, task.y_train)
    together = clf.predict_proba(task.X_test)
    half = cfg.n_query // 2
    split = np.concatenate([clf.predict_proba(task.X_test.iloc[:half]), clf.predict_proba(task.X_test.iloc[half:])])
    cached = CachedTabFMClassifier(TabFMClassifier(lm.model, n_estimators=cfg.n_estimators, random_state=0)).fit(
        task.X_train, task.y_train
    )
    via_cache = cached.predict_proba(task.X_test)
    return {
        **asdict(cfg),
        "stock_vs_stock_split_maxabs": float(np.abs(together - split).max()),
        "stock_vs_cached_maxabs": float(np.abs(together - via_cache).max()),
        "stock_vs_stock_split_agree": float((together.argmax(1) == split.argmax(1)).mean()),
        "stock_vs_cached_agree": float((together.argmax(1) == via_cache.argmax(1)).mean()),
    }


def grid(base: Config, **axes: list[Any]) -> list[Config]:
    """Cartesian product of axis values applied to `base`."""
    configs = [base]
    for name, values in axes.items():
        configs = [replace(c, **{name: v}) for c in configs for v in values]
    return configs


def run_sweep(name: str, configs: list[Config], runner: Callable[[Config, ModelPool], Any] | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.parquet"
    pool = ModelPool()
    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.perf_counter()
        res = (runner or run_config)(cfg, pool)
        row = res.row() if isinstance(res, Result) else res
        rows.append(row)
        pd.DataFrame(rows).to_parquet(out)  # checkpoint after every config: long sweeps survive interruption
        brief = {k: row[k] for k in ("p50_ms", "encode_s", "accuracy", "total_s") if k in row}
        print(f"[{name} {i}/{len(configs)}] {_describe(cfg)} -> {brief} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def _describe(cfg: Config) -> str:
    return (
        f"{cfg.mode} {cfg.device}/{cfg.dtype} train={cfg.n_train} query={cfg.n_query} feat={cfg.n_features} "
        f"cls={cfg.n_classes} est={cfg.n_estimators} mbs={cfg.member_batch_size}"
    )
