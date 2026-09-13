"""Fast tests: no model weights needed."""

from __future__ import annotations

import numpy as np

from ltm_serve.bench import Config, grid
from ltm_serve.context_cache import _chunks
from ltm_serve.data import make_task, penguins_task


def test_make_task_has_exact_shape() -> None:
    task = make_task(n_train=100, n_test=7, n_features=13, n_classes=5)
    assert task.X_train.shape == (100, 13)
    assert task.X_test.shape == (7, 13)
    assert task.n_classes == 5
    assert set(np.unique(task.y_test)) <= set(np.unique(task.y_train))


def test_penguins_split_is_deterministic() -> None:
    a, b = penguins_task(), penguins_task()
    assert a.X_test.equals(b.X_test)
    assert len(a.X_train) + len(a.X_test) == 333


def test_grid_is_cartesian() -> None:
    configs = grid(Config(), mode=["stock", "cached"], n_train=[1, 2, 3])
    assert len(configs) == 6
    assert {(c.mode, c.n_train) for c in configs} == {(m, n) for m in ("stock", "cached") for n in (1, 2, 3)}


def test_chunks_cover_range_without_overlap() -> None:
    slices = _chunks(10, 4)
    assert [(s.start, s.stop) for s in slices] == [(0, 4), (4, 8), (8, 10)]
    assert _chunks(5, None) == [slice(0, 5)]
