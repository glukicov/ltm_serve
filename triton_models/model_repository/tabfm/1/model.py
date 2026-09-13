"""Triton Python-backend model: TabFM with an exact, pre-encoded context.

Triton owns everything around the model: HTTP/gRPC endpoints, the dynamic batcher (config.pbtxt), instance
placement, and Prometheus metrics on :8002. This file only turns a batch of requests into a batch of responses.

Design choice — "context as model": the training table is fixed per Triton model, encoded once in `initialize`.
Different tables become different models in the repository, loaded and unloaded through Triton's model-control
API (`POST /v2/repository/models/<name>/load`), so Triton's own lifecycle management doubles as the context
registry and each context gets its own batcher queue and metrics for free.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils
from tabfm import TabFMClassifier

from ltm_serve.context_cache import CachedTabFMClassifier
from ltm_serve.data import make_task
from ltm_serve.model import load_model


def _params(model_config: dict[str, Any]) -> dict[str, str]:
    return {k: v["string_value"] for k, v in model_config.get("parameters", {}).items()}


def _parse_context(spec: str) -> dict[str, int]:
    kind, _, params = spec.partition(":")
    if kind != "synthetic":
        raise ValueError(f"unsupported context spec {spec!r}")
    return {k: int(v) for k, v in (kv.split("=") for kv in params.split(",") if kv)}


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        config = json.loads(args["model_config"])
        # Environment overrides config.pbtxt so one model repository serves every deployment (docker, KServe, GPU).
        overrides = {"checkpoint_dir": "LTM_CHECKPOINT_DIR", "dtype": "LTM_DTYPE", "context": "LTM_CONTEXT"}
        params = _params(config) | {key: os.environ[env] for key, env in overrides.items() if env in os.environ}
        # KIND_GPU instances receive their device id; KIND_CPU instances run on CPU.
        device = f"cuda:{args['model_instance_device_id']}" if args["model_instance_kind"] == "GPU" else "cpu"
        checkpoint = params.get("checkpoint_dir")
        dtype = params.get("dtype", "fp32")
        loaded = load_model(device, dtype, Path(checkpoint) if checkpoint else None)
        ctx = _parse_context(params["context"])
        task = make_task(ctx["n_train"], 1, ctx["n_features"], ctx["n_classes"], seed=ctx.get("seed", 0))
        self.columns = list(task.X_train.columns)
        clf = TabFMClassifier(loaded.model, n_estimators=ctx.get("n_estimators", 4), random_state=0)
        # LTM_COMPILE_MODE=reduce-overhead: torch.compile + CUDA graphs, with query rows padded to power-of-two buckets
        # and every bucket up to max_batch_size captured here, before Triton reports the model READY.
        compile_mode = os.environ.get("LTM_COMPILE_MODE") or None
        self.classifier = CachedTabFMClassifier(clf, compile_mode=compile_mode, bucket_rows=compile_mode is not None)
        self.classifier.fit(task.X_train, task.y_train)
        warm_s = self.classifier.warm_up(int(config["max_batch_size"])) if compile_mode else 0.0
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(config, "PROBABILITIES")["data_type"]
        )
        pb_utils.Logger.log_info(
            f"tabfm on {device}/{dtype}: weights {loaded.load_seconds:.1f}s, "
            f"context encoded {self.classifier.encode_seconds:.1f}s, "
            f"warm-up {warm_s:.1f}s (compile_mode={compile_mode})"
        )

    def execute(self, requests: list[Any]) -> list[Any]:
        # The dynamic batcher hands us several requests at once; one forward over all their rows is exact because
        # query rows never attend to each other.
        arrays = [pb_utils.get_input_tensor_by_name(r, "ROWS").as_numpy() for r in requests]
        sizes = [a.shape[0] for a in arrays]
        t0 = time.perf_counter()
        proba = self.classifier.predict_proba(pd.DataFrame(np.concatenate(arrays), columns=self.columns))
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        responses = []
        offset = 0
        for n in sizes:
            out = pb_utils.Tensor("PROBABILITIES", proba[offset : offset + n].astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            offset += n
        pb_utils.Logger.log_verbose(f"batch of {len(requests)} requests / {sum(sizes)} rows in {elapsed_ms:.1f} ms")
        return responses

    def finalize(self) -> None:
        self.classifier = None  # type: ignore[assignment]
