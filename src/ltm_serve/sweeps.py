"""Named benchmark sweeps. Each isolates one axis of problem complexity or one optimisation knob.

Shapes are sized for an Apple M4 (the 1.6B-parameter model costs roughly 1 s per ensemble member per few hundred
rows on the Mac GPU). The same sweeps run unchanged on a CUDA device with `--device cuda`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ltm_serve.bench import Config, ModelPool, bf16_noise_floor, grid, stage_breakdown

Runner = Callable[[Config, ModelPool], Any] | None


def sweeps(device: str, dtype: str) -> dict[str, tuple[list[Config], Runner]]:
    base = Config(device=device, dtype=dtype, repeats=5, warmup=1, eval_rows=0)
    both = ["stock", "cached"]
    # A 24 GB datacenter GPU reaches contexts the Mac cannot in reasonable time.
    contexts = [64, 128, 256, 512, 1024, 2048] + ([4096, 8192] if device.startswith("cuda") else [])
    return {
        # Axis 1: context size. Stock cost should grow ~quadratically, cached per-request cost ~linearly.
        "context_scaling": (
            grid(
                replace(base, n_query=16, repeats=3, eval_rows=256),
                mode=both,
                n_train=contexts,
            ),
            None,
        ),
        # Axis 2: table width. Width enters the row transformer (quadratic in columns) and column stages.
        "feature_scaling": (grid(replace(base, n_query=16), mode=both, n_features=[4, 8, 16, 32, 64, 100]), None),
        # Axis 3: rows per request. Throughput vs latency trade-off that dynamic batching exploits.
        "query_scaling": (grid(replace(base, n_train=512), mode=both, n_query=[1, 4, 16, 64, 256]), None),
        # Axis 4: number of classes: should cost nothing (only the decoder width changes).
        "class_scaling": (grid(replace(base, n_train=512, eval_rows=256), n_classes=[2, 3, 5, 10]), None),
        # Knob: ensemble size, the main accuracy-for-latency dial.
        "ensemble": (
            grid(
                replace(base, n_train=512, n_features=20, n_classes=5, eval_rows=1024, class_sep=0.5),
                n_estimators=[1, 2, 4, 8, 16, 32],
            ),
            None,
        ),
        # Knob: how many ensemble members share one forward pass (kernel-launch overhead vs peak memory).
        "member_batching": (
            grid(replace(base, n_train=512, n_estimators=16), n_query=[1, 64], member_batch_size=[1, 2, 4, 8, 16]),
            None,
        ),
        # Knob: compilation. Row bucketing keeps shapes stable so compiled graphs (and CUDA graphs) are reused.
        "compile": (
            grid(
                replace(base, n_train=512, repeats=20, warmup=5),
                n_query=[1, 16],
                compile_mode=[None, "default", "reduce-overhead"],
                bucket_rows=[True],
            ),
            None,
        ),
        # Where a small request's time goes once the model is cached: host-side preprocessing vs the forward pass.
        "overhead": (
            grid(replace(base, n_train=512, repeats=20, warmup=3), n_query=[1, 8, 64], n_estimators=[1, 4, 16]),
            None,
        ),
        # Knob: device and precision.
        "precision": (
            grid(
                replace(base, n_train=128, n_query=16, repeats=3, eval_rows=128),
                mode=both,
                device=["cpu", "mps"],
                dtype=["fp32", "bf16"],
            ),
            None,
        ),
        # Where stock inference time goes, by model stage.
        "stages": (
            grid(replace(base, mode="stock", n_query=16), n_train=[128, 512, 2048], n_features=[10, 64]),
            stage_breakdown,
        ),
        # Is the cached-vs-stock bf16 difference bigger than stock's own batch-composition noise?
        "noise_floor": (
            [
                replace(base, n_train=256, n_query=128, dtype="bf16"),
                replace(base, n_train=256, n_query=128, dtype="fp32"),
            ],
            bf16_noise_floor,
        ),
    }
