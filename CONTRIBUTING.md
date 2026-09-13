# Contributing

Issues and pull requests are welcome: a result that does not reproduce on your hardware, another serving stack or
optimisation, a fix to a script, or a clearer explanation in the write-up.

## Setup and checks

```bash
uv sync --all-extras                                  # project, serve + bench extras, dev tools (pinned in uv.lock)
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests                                 # strict
uv run pytest -m "not slow"                           # no weights needed
git ls-files -z '*.sh' | xargs -0 shellcheck && shellcheck k8s/gke/kctl
```

CI runs the same checks on every push and pull request (with CPU-only PyTorch, so it never runs the slow tests).

## Things that are easy to break by accident

1. **The slow tests are the correctness proof, and CI does not run them.** `uv run pytest -m slow` downloads the
   6.6 GB TabFM checkpoint once and checks that the context cache matches stock TabFM, and that the HTTP service
   predicts identically end to end. Run it before sending a change to `context_cache.py`, `model.py`, the service
   engine or the Triton model.

2. **`docker/requirements.txt` is generated.** The images install from it, not from `uv.lock`. After any
   dependency change (`uv lock`, `uv add`, a version bump), run `docker/requirements.sh` and commit both files.
   torch itself comes from the PyTorch index in each Dockerfile (`TORCH_VERSION`), so keep that in step with the
   torch version in `uv.lock`.

3. **The numbers in the docs come from `results/`.** Sweeps write `results/*.parquet` (Apple M4) and
   `results/cuda/*_cuda.parquet` (L4), load tests write `results/load/*.jsonl`, and `uv run ltm-serve plots`
   renders `docs/figures/` from them. The tables in `docs/WRITEUP.md` and `README.md` are written by hand from the
   same files: if you rerun something, commit the new results, regenerate the figures and update the tables to
   match. `results/logs/` is gitignored because raw logs contain cloud project ids, node names and local paths;
   keep those out of everything that is committed.

4. **The GKE scripts cost money.** `k8s/gke/up.sh` creates a cluster, an L4 node pool, an Artifact Registry
   repository and a bucket; `k8s/gke/down.sh` deletes all of it. The L4 pool scales to zero only when no pod needs
   a GPU node: the bench and load-generator pods sleep after they finish so their results can be copied out, and
   until you delete them the L4 node they run on stays up and billed.

5. **Single-GPU rollouts need `Recreate`.** The GPU InferenceServices set `deploymentStrategy: Recreate`. With the
   default rolling update, the new pod waits for a GPU that the old pod still holds and stays Pending.

6. **Image tags are reused.** Cloud Build pushes the same `:cu126` tags every time, so the GPU manifests use
   `imagePullPolicy: Always`; with `IfNotPresent` a node that already has the tag keeps running the old image.

7. **Nothing in the repo root may be called `triton`.** A `triton/` directory there shadows the `triton` package
   that `torch._dynamo` imports, which breaks model loading (hence `triton_models/`).

## Licences

Contributions are accepted under the repository's [Apache-2.0 licence](LICENSE). TabFM's pretrained weights are
not part of this repository and stay under their own non-commercial licence.
