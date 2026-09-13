#!/usr/bin/env bash
# Regenerate docker/requirements.txt from uv.lock: everything the service and benchmarks need, minus torch/triton and
# the CUDA wheels, which come from the variant-specific PyTorch index in each Dockerfile instead.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --frozen --no-dev --extra serve --extra bench --no-emit-project --no-hashes \
  | grep -v -E '^\s*#' \
  | grep -v -E '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' > docker/requirements.txt
