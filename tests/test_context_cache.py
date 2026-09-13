"""The cache must be *exact*: in fp32 it matches stock TabFM to kernel precision. Loads the real checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
from tabfm import TabFMClassifier

from ltm_serve.context_cache import CachedTabFMClassifier
from ltm_serve.data import make_task, penguins_task
from ltm_serve.model import LoadedModel, load_model


@pytest.fixture(scope="module")
def fp32_model() -> LoadedModel:
    return load_model("cpu", "fp32")


@pytest.mark.slow
@pytest.mark.parametrize("task_name", ["penguins", "synthetic_wide_multiclass"])
@pytest.mark.parametrize("member_batch_size", [None, 1])
def test_cached_matches_stock_fp32(fp32_model: LoadedModel, task_name: str, member_batch_size: int | None) -> None:
    # Penguins exercises categorical columns; the synthetic task exercises padding (d) and >2 classes.
    task = penguins_task() if task_name == "penguins" else make_task(96, 12, n_features=23, n_classes=4)
    query = task.X_test.iloc[:12]

    stock = TabFMClassifier(fp32_model.model, n_estimators=3, random_state=0).fit(task.X_train, task.y_train)
    cached = CachedTabFMClassifier(
        TabFMClassifier(fp32_model.model, n_estimators=3, random_state=0), member_batch_size=member_batch_size
    ).fit(task.X_train, task.y_train)

    expected = stock.predict_proba(query)
    actual = cached.predict_proba(query)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    assert (actual.argmax(1) == expected.argmax(1)).all()


@pytest.mark.slow
def test_query_rows_are_independent(fp32_model: LoadedModel) -> None:
    """Predicting rows together or one by one gives the same answer: this is what makes dynamic batching exact."""
    task = make_task(64, 6, n_features=5, n_classes=3)
    cached = CachedTabFMClassifier(TabFMClassifier(fp32_model.model, n_estimators=2, random_state=0)).fit(
        task.X_train, task.y_train
    )
    together = cached.predict_proba(task.X_test)
    one_by_one = np.concatenate([cached.predict_proba(task.X_test.iloc[[i]]) for i in range(len(task.X_test))])
    np.testing.assert_allclose(together, one_by_one, atol=1e-5)
