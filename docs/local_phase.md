# Local phase (Apple M4, CPU fp32): validation, checkpoint, Docker, kind + KServe

Machine: Apple M4 (10 cores), 24 GB RAM, macOS; Docker Desktop 29.7.2 (VM raised to 8 CPUs / 15.6 GiB / 120 GB disk).
All inference on CPU, fp32, synthetic context `n_train=512, n_features=10, n_classes=2, n_estimators=4`,
1 row per request. Logs: `results/logs/local_*.log`.

**Status:** all steps complete.

## 1. Cache exactness tests

`uv run pytest -m slow -q` → **5 passed**, 7 deselected, 41.4 s (42.1 s wall incl. uv). The cache matches stock
TabFM within `atol=1e-4` (penguins + synthetic wide multiclass, member batching on/off), and query rows are
independent (`atol=1e-5`).

## 2. fp32 checkpoint export and load time

`uv run ltm-serve export-checkpoint ~/.cache/ltm_serve/checkpoints/tabfm-fp32 --dtype fp32`: 18.7 s wall, peak memory
footprint 12.45 GB (max RSS 6.67 GB). Output: `model.safetensors` 6,557,888,408 bytes + `config.json` 377 bytes.

`load_model('cpu', 'fp32', ...)`, two runs each, separate processes:

| Source | Load time |
|---|---|
| Exported checkpoint (meta-device build + safetensors) | 1.34 s, 0.75 s |
| Stock `tabfm_v1_0_0.load()` from the local HF cache | 22.5 s, 27.5 s |

The checkpoint number is optimistic: safetensors mmaps the file, so pages are faulted in during the first forward
(the context encoding) instead of during "load". In the containers the load was 2.1 s (FastAPI) / 1.0 s (Triton),
and encoding the demo context took 35.4 s / 44.8 s.

## 3. Docker Desktop resources

Before: Cpus 5, MemoryMiB 8192, DiskSizeMiB 40960. After: Cpus 8, MemoryMiB 16384, DiskSizeMiB 122880.
`docker info`: `CPUs: 8`, `Total Memory: 15.6GiB`; container `/` is 118 G.

## 4. Images

| Image | Build | Size (unpacked / compressed) |
|---|---|---|
| `nvcr.io/nvidia/tritonserver:26.08-py3` (arm64) | pull 4 min 01 s | 26.6 GB / – |
| `ltm-serve:cpu` | 2 min 38 s cold (concurrent with the Triton pull); 30.7 s after the fix below | 1.85 GB / 394 MB (was 7.86 GB / 2.87 GB) |
| `ltm-triton:cpu` | 1 min 03 s (base already pulled) | 28.5 GB / 10.2 GB |

## 5. Plain Docker

FastAPI:

```bash
docker run -d --name ltm-fastapi-smoke -p 8080:8080 -v ~/.cache/ltm_serve/checkpoints/tabfm-fp32:/models:ro \
  -e LTM_CHECKPOINT_DIR=/models -e LTM_DEVICE=cpu -e LTM_DTYPE=fp32 -e LTM_TORCH_THREADS=6 \
  -e LTM_DEMO_CONTEXT=demo:n_train=512,n_features=10,n_classes=2,n_estimators=4 ltm-serve:cpu
```

`docker run` → `/v2/models/demo/ready`: **42.2 s** (weights 2.1 s, context encode 35.4 s). First request 1.33 s,
then 0.52-0.76 s per 1-row request (v1 and v2). Container memory 3.1 GiB (mmapped weights not counted).
`/metrics` exposes `ltm_inference_seconds`, `ltm_batch_rows`, `ltm_inflight_requests`, `ltm_cache_bytes`
(893 MB for this context) etc.

Triton:

```bash
docker run -d --name ltm-triton-smoke -p 8000:8080 -p 8002:8002 -v ~/.cache/ltm_serve/checkpoints/tabfm-fp32:/models:ro \
  -e LTM_CHECKPOINT_DIR=/models -e OMP_NUM_THREADS=6 ltm-triton:cpu tritonserver --model-repository=/model_repository --http-port=8080
```

`docker run` → `/v2/health/ready`: **51.2 s** (weights 1.0 s, context encode 44.8 s). First request 1.33 s, then
0.50-0.64 s. Probabilities agree with FastAPI to 7e-7. `nv_inference_count`, `nv_inference_exec_count`,
`nv_inference_*_duration_us` present on :8002.

Load (`uv run python -P -m ltm_serve.loadgen ... --protocol v2 --duration 40`, client on the host through Docker
Desktop port forwarding, results in `results/load/local_docker_cpu.jsonl`). FastAPI 0.5-4 req/s had 10 s warm-up;
8 and 16 were a second run without warm-up. Latency is from the scheduled send time; `achieved` divides by elapsed
time including the drain after the last send, so it reads slightly below the offered rate even when unsaturated.

| Offered req/s | FastAPI achieved | FastAPI p50 / p95 / p99 ms | Triton achieved | Triton p50 / p95 / p99 ms |
|---|---|---|---|---|
| 0.5 | 0.49 | 515 / 927 / 1015 | 0.49 | 502 / 861 / 918 |
| 1 | 0.86 | 508 / 898 / 923 | 0.87 | 488 / 887 / 905 |
| 2 | 1.83 | 758 / 1238 / 1546 | 1.85 | 669 / 1232 / 1383 |
| 4 | 3.83 | 1397 / 2082 / 2280 | 3.82 | 1342 / 2110 / 2375 |
| 8 | 7.35 | 2114 / 3518 / 4508 | 7.27 | 2968 / 5887 / 6769 |
| 16 | 13.42 | 7495 / 12045 / 13145 | 11.60 (49 dropped) | 11738 / 18978 / 19899 |

Both stacks keep up with the offered rate to 8 req/s only because the dynamic batchers merge queued requests
(FastAPI over the whole run: 1,244 requests in 215 batches; Triton: 1,192 inferences in 204 executions); latency
grows with the batch. 16 req/s is past saturation for both. No errors.

## Problems and fixes

| Problem | Symptom | Fix |
|---|---|---|
| CPU images carried CUDA 13 wheels | `ltm-serve:cpu` 7.86 GB, 3.3 GB of it `site-packages/nvidia`; `docker/requirements.txt` pins `nvidia-*`/`cuda-*` with `sys_platform == 'linux'` markers | Both Dockerfiles drop `^(nvidia-|cuda-)` lines when `TORCH_VARIANT=cpu`; CUDA builds use the file unchanged. Image 1.85 GB |
| `triton/` directory shadows the `triton` package | `python -c "...load_model(..., checkpoint_dir)"` run from the repo root: `AttributeError: module 'triton' has no attribute 'language'` (building the model on the meta device imports `torch._dynamo`, which imports `triton`; cwd is `sys.path[0]`) | Renamed the directory to `triton_models/`, so nothing in the repo root can shadow the real package. Before that, `python -P` or another working directory avoided it |
| `osascript -e 'quit app "Docker"'` did not stop Docker Desktop | backend still running after 2 min | `docker desktop stop` (took ~3 min in total), edit settings, `docker desktop start` |
| `kind create cluster` rewrites current-context | `~/.kube/ltm-serve` current-context switched from the GKE context to `kind-ltm` | Every script passes `--context` explicitly (`kind-ltm`, or the GKE context via `k8s/gke/kctl`), so the current-context is irrelevant to them; bare `kubectl` with that kubeconfig targets kind |
| Checkpoint file is mode 0600 | – | Not a problem: Docker Desktop's virtiofs let uid 10001 read it |

## 6. kind + KServe (Standard mode)

`k8s/kind/up.sh` → cluster `ltm` (kindest/node v1.37.0) with cert-manager v1.21.2, KServe v0.20.0, KEDA and
Prometheus, both images loaded into the node, namespace `ltm` and the hostPath-backed `tabfm-weights` PV/PVC: **508 s**
end to end (component installs done after ~1.5 min; the rest was Prometheus readiness and `kind load` of the 28.5 GB
Triton image).

| InferenceService | apply → Ready | What dominated |
|---|---:|---|
| `tabfm-fastapi` (custom container, `STORAGE_URI=pvc://tabfm-weights/tabfm-fp32`) | **66 s** | context encode 56.5 s (weights 2.3 s: PVC mounted, no copy; image already on the node) |
| `tabfm-triton` (ServingRuntime `ltm-triton-cpu` + `storageUri: pvc://...`) | **71 s** | context encode 56.0 s (weights 1.3 s) |

Both answer the same request identically (`[0.887156, 0.112844]` vs `[0.887157, 0.112843]`), and Prometheus scrapes
both pods through the `prometheus.io/*` annotations (`ltm_batch_rows_count`, `nv_inference_count` returned per pod).

Load, client on the host through `kubectl port-forward` (adds latency; the kind node also hosts the control plane,
KServe, KEDA and Prometheus on the same 8 CPUs), 1 row per request, `--procs 2`, 40 s per rate. Results in
`results/load/local_kind_cpu.jsonl`.

| Offered req/s | KServe FastAPI achieved | p50 / p95 / p99 ms | KServe Triton achieved | p50 / p95 / p99 ms |
|---|---|---|---|---|
| 0.5 | 0.58 | 636 / 1529 / 1582 | 0.58 | 584 / 1381 / 1760 |
| 1 | 1.11 | 671 / 1104 / 1195 | 1.13 | 598 / 1315 / 1512 |
| 2 | 1.94 | 1574 / 2778 / 3283 | 1.91 | 1200 / 2250 / 2456 |
| 4 | 3.81 | 2320 / 3656 / 3800 | 3.85 | 1679 / 2535 / 2700 |
| 8 | 7.31 | 4186 / 5955 / 6701 | 7.56 | 2455 / 3391 / 3830 |

On CPU inside kind, Triton's tail is clearly better at load (p99 3.8 s vs 6.7 s at 8 req/s): its HTTP front end is
C++, while FastAPI parses JSON on the same Python process that runs the model.

## Also validated afterwards

- `uv run pytest tests/test_service_e2e.py -m slow` → **4 passed** (66 s): over HTTP with the real fp32 model, a
  penguins context registered as JSON (string categoricals + missing values) predicts identically to stock TabFM;
  v2 infer, health, 404 and metrics paths behave.
- Service CPU image rebuilt after the requirements/gcc/rename changes: `docker build` OK in 1 min 35 s.
- The Docker VM disk reached 94 % after `kind load` of the Triton image; the local Triton CPU rebuild was stopped to
  avoid filling it (the only change was a `COPY` path, validated by the Cloud Build amd64 build).
