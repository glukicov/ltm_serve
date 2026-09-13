"""Datasets for benchmarking: synthetic tables with controllable shape, plus the real penguins table.

Latency depends on the *shape* of the in-context problem (training rows, features, classes, query rows),
not on what the numbers mean. `make_task` lets a sweep dial each axis independently while keeping the
problem learnable, so accuracy can be tracked next to latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

REPO_ROOT = Path(__file__).resolve().parents[2]
PENGUINS_CSV = REPO_ROOT / "data" / "penguins.csv"


@dataclass(frozen=True)
class Task:
    """An in-context classification problem: context rows (train) and query rows (test)."""

    name: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y_train))


def make_task(
    n_train: int, n_test: int, n_features: int, n_classes: int, seed: int = 0, class_sep: float = 1.0
) -> Task:
    """Synthetic classification task of an exact shape.

    `n_informative` scales with the feature count so wide tables are not trivially easy, and
    `n_clusters_per_class=1` keeps many-class problems feasible with few features.
    """
    n_informative = max(2, min(n_features, n_features // 2 + 1))
    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, min(n_features - n_informative, n_features // 4)),
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=seed,
    )
    columns = [f"f{i}" for i in range(n_features)]
    frame = pd.DataFrame(X.astype(np.float32), columns=columns)
    return Task(
        name=f"synthetic_tr{n_train}_te{n_test}_f{n_features}_c{n_classes}",
        X_train=frame.iloc[:n_train].reset_index(drop=True),
        y_train=y[:n_train],
        X_test=frame.iloc[n_train:].reset_index(drop=True),
        y_test=y[n_train:],
    )


PENGUIN_FEATURES = ["species", "island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]


def penguins_task(seed: int = 0) -> Task:
    """Predict penguin sex from a seeded 80/20 train/test split of the Palmer Penguins table."""
    df = pd.read_csv(PENGUINS_CSV)
    df = df[df["sex"].notna() & (df["sex"] != ".")]
    order = np.random.default_rng(seed).permutation(len(df))
    cut = int(0.8 * len(df))
    train, test = df.iloc[order[:cut]], df.iloc[order[cut:]]
    return Task(
        name="penguins",
        X_train=train[PENGUIN_FEATURES].reset_index(drop=True),
        y_train=train["sex"].to_numpy(),
        X_test=test[PENGUIN_FEATURES].reset_index(drop=True),
        y_test=test["sex"].to_numpy(),
    )
