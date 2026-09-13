"""Loading TabFM for serving: device selection, device sync for honest timing, and a compact checkpoint.

The published checkpoint is fp32 (6.6 GB for 1.64B parameters) and `tabfm_v1_0_0.load()` materialises it in
fp32 before casting to bf16, so peak memory at start-up is ~2x the serving footprint. `export_checkpoint`
writes the already-cast bf16 weights once; `load_model` then builds the module on the meta device and
assigns those tensors directly — half the download, half the peak RAM, and a much faster cold start.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm.src.pytorch.model import TabFM

DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued kernels finish; without it GPU timings measure only the kernel *launch*."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class LoadedModel:
    model: TabFM
    device: torch.device
    dtype: torch.dtype
    load_seconds: float


def model_config(model: TabFM) -> dict[str, Any]:
    """Recover the constructor arguments from a loaded model (the HF mixin keeps them on `_hub_mixin_config`)."""
    config = getattr(model, "_hub_mixin_config", None)
    if not config:
        raise ValueError("model has no hub config; load it through tabfm_v1_0_0.load() first")
    return dict(config)


def export_checkpoint(out_dir: Path, dtype: str = "bf16", model_type: str = "classification") -> Path:
    """Write the cast weights and constructor config to `out_dir` (one-off, at image build time)."""
    model = tabfm_v1_0_0.load(model_type=model_type, dtype=DTYPES[dtype], use_cache=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out_dir / WEIGHTS_FILE))
    config = model_config(model) | {"dtype": dtype, "model_type": model_type}
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    return out_dir


def load_model(device: str = "auto", dtype: str = "bf16", checkpoint_dir: Path | None = None) -> LoadedModel:
    """Load TabFM onto `device` in `dtype`, preferring a pre-cast checkpoint when one is given."""
    target = pick_device(device)
    torch_dtype = DTYPES[dtype]
    t0 = time.perf_counter()
    if checkpoint_dir is not None and (checkpoint_dir / WEIGHTS_FILE).exists():
        config = json.loads((checkpoint_dir / CONFIG_FILE).read_text())
        saved_dtype = config.pop("dtype")
        config.pop("model_type", None)
        with torch.device("meta"):
            model = TabFM(**config)
        state = load_file(str(checkpoint_dir / WEIGHTS_FILE), device=str(target))
        model.load_state_dict(state, assign=True)
        if saved_dtype != dtype:
            model = model.to(torch_dtype)
    else:
        model = tabfm_v1_0_0.load(dtype=torch_dtype, device=str(target), use_cache=False)
    model.eval()
    synchronize(target)
    return LoadedModel(model=model, device=target, dtype=torch_dtype, load_seconds=time.perf_counter() - t0)
