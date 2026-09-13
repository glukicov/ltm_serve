# ltm-serve — online inference for TabFM: benchmark, optimise, serve

A learning project that takes Google's [TabFM](https://github.com/google-research/tabfm), a 1.64B-parameter
in-context tabular foundation model, from "a `predict_proba` call in a notebook" to low-latency online serving on
Kubernetes. It covers measurement, model-level optimisation, and three serving stacks (FastAPI, Triton, KServe),
locally and on GPUs in GKE.

**Start with [`docs/WRITEUP.md`](docs/WRITEUP.md)**: what was built in each phase, the numbers, and why.
The laptop phase (CPU containers, kind + KServe) is detailed in [`docs/local_phase.md`](docs/local_phase.md).

## Layout

| Path | What it is |
|---|---|
| `src/ltm_serve/model.py` | Loading on cpu/mps/cuda, honest device sync, pre-cast bf16 checkpoint export |
| `src/ltm_serve/context_cache.py` | **Exact context (KV) cache**: encode training rows once, serve queries in O(query × context) |
| `src/ltm_serve/bench.py`, `sweeps.py` | Benchmark harness and named sweeps → `results/*.parquet` |
| `src/ltm_serve/plots.py` | Figures → `docs/figures/` |
| `src/ltm_serve/service/` | FastAPI service: context registry with a memory budget, exact dynamic batcher, Prometheus, OIP v2 |
| `src/ltm_serve/loadgen.py` | Open-loop (Poisson) load generator for native and v2 APIs |
| `triton_models/model_repository/` | Triton Python-backend model with dynamic batching |
| `docker/` | CPU (arm64) and CUDA (amd64) images for the service and Triton; `requirements.sh` regenerates pins from `uv.lock` |
| `k8s/kind/`, `k8s/kserve/` | Local kind cluster + KServe InferenceServices (custom container and Triton runtime) |
| `k8s/gke/` | GKE with an L4 pool that scales from zero, Cloud Build, GPU InferenceServices, KEDA, bench/load pods, `experiment-compiled.sh` |
| `tests/` | Harness and batcher tests; `-m slow` loads the real model: cache == stock TabFM, and the HTTP service end to end |

## Quick start (local, Apple silicon or CUDA)

```bash
uv sync --all-extras
uv run pytest -m "not slow"                      # fast tests
uv run pytest -m slow                            # cache == stock TabFM (downloads 6.6 GB of weights once)
uv run ltm-serve bench context_scaling --device mps
uv run ltm-serve plots

# online service with a synthetic demo context, then load it
LTM_DEVICE=mps LTM_DEMO_CONTEXT="demo:n_train=512,n_features=10,n_classes=2,n_estimators=4" \
  uv run uvicorn ltm_serve.service.app:app --port 8080
uv run python -m ltm_serve.loadgen --url http://localhost:8080 --model demo --rates 1,2,4 --duration 30
```

Kubernetes: `k8s/kind/up.sh` (local) and `k8s/gke/up.sh` / `k8s/gke/down.sh` (GPU; `down.sh` deletes everything
billable). Both use a dedicated kubeconfig (`~/.kube/ltm-serve`), so your current kubectl context is left alone.
The GKE scripts need your GCP project: `export PROJECT=<your-project-id>` (otherwise the active `gcloud` project is
used); the weights bucket defaults to `$PROJECT-ltm-serve`.

**Licence note:** this repository's code is Apache-2.0 ([`LICENSE`](LICENSE)), as is the tabfm code, but the
pretrained TabFM weights are non-commercial. That's fine for learning and benchmarking; don't use them in production.

**Data:** `data/penguins.csv` is the Palmer Penguins dataset (Gorman, Williams & Fraser, 2014, Palmer Station
Antarctica LTER; distributed via the [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) package, CC0).
