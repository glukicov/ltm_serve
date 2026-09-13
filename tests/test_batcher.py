"""The dynamic batcher must merge concurrent requests and hand every caller back exactly its own rows."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from ltm_serve.service.engine import Context, DynamicBatcher


@dataclass
class _FakeClassifier:
    """Row-wise 'model': probability = sigmoid(first column). Records the size of each forward it served."""

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_sizes: list[int] = field(default_factory=list)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.batch_sizes.append(len(frame))
        p = 1 / (1 + np.exp(-frame.iloc[:, 0].to_numpy()))
        return np.stack([1 - p, p], axis=1)


def _run(max_rows: int, max_delay_s: float, requests: list[pd.DataFrame]) -> tuple[list[np.ndarray], list[int]]:
    fake = _FakeClassifier()
    ctx = Context("t", ["a"], cast(Any, fake), 0.0)

    async def main() -> list[np.ndarray]:
        batcher = DynamicBatcher(ctx, ThreadPoolExecutor(max_workers=1), max_rows, max_delay_s)
        batcher.start()
        results = await asyncio.gather(*(batcher.submit(r) for r in requests))
        await batcher.stop()
        return [proba for proba, _ in results]

    return asyncio.run(main()), fake.batch_sizes


def test_each_caller_gets_its_own_rows() -> None:
    requests = [pd.DataFrame({"a": np.arange(n, dtype=float) - i}) for i, n in enumerate([1, 3, 2, 5])]
    results, _ = _run(max_rows=64, max_delay_s=0.05, requests=requests)
    for req, got in zip(requests, results, strict=True):
        expected = 1 / (1 + np.exp(-req["a"].to_numpy()))
        np.testing.assert_allclose(got[:, 1], expected)


def test_concurrent_requests_are_merged_up_to_max_rows() -> None:
    requests = [pd.DataFrame({"a": [float(i)]}) for i in range(10)]
    _, batch_sizes = _run(max_rows=4, max_delay_s=0.05, requests=requests)
    assert sum(batch_sizes) == 10
    assert max(batch_sizes) <= 4
    assert len(batch_sizes) < 10  # some merging happened


def test_batches_never_exceed_max_rows_with_multi_row_requests() -> None:
    """Compiled forwards are warmed up for row buckets <= max_rows; a larger batch would recompile under traffic."""
    requests = [pd.DataFrame({"a": [float(i)] * 3}) for i in range(10)]
    results, batch_sizes = _run(max_rows=4, max_delay_s=0.05, requests=requests)
    assert max(batch_sizes) <= 4
    assert sum(batch_sizes) == 30
    for req, got in zip(requests, results, strict=True):
        assert got.shape == (3, 2)
        np.testing.assert_allclose(got[:, 1], 1 / (1 + np.exp(-req["a"].to_numpy())))
