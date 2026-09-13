# FastAPI TabFM service. Weights are NOT baked in: they are mounted (hostPath / PVC) or pulled by KServe's
# storage initializer, so the image stays ~1-3 GB and a weight update does not require an image rebuild.
#
#   CPU (arm64 Mac / kind):  docker build -f docker/service.Dockerfile -t ltm-serve:cpu .
#   CUDA (x86_64 GKE):       docker buildx build --platform linux/amd64 --build-arg TORCH_VARIANT=cu126 ...
FROM python:3.12-slim

ARG TORCH_VERSION=2.14.0
ARG TORCH_VARIANT=cpu
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
# Dependency layer first so code edits do not reinstall torch. requirements.txt is generated from uv.lock with
# torch/CUDA wheels removed; torch then comes from the variant-specific PyTorch index.
COPY docker/requirements.txt ./requirements.txt
# requirements.txt excludes torch, triton and the CUDA 13 wheels (nvidia-*, cuda-*) that the lock pins for Linux from
# PyPI (~3.3 GB); the variant-specific torch wheel brings exactly the CUDA runtime it was built against.
# gcc: torch.compile generates Triton kernels and compiles small C launchers at runtime (CUDA images only).
RUN if [ "${TORCH_VARIANT}" != "cpu" ]; then \
      apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*; \
    fi \
    && uv pip install --system --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} torch==${TORCH_VERSION} \
    && uv pip install --system -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-deps .

# Non-root, as a k8s PodSecurity "restricted" profile expects.
RUN useradd --uid 10001 --create-home app
USER 10001
ENV HF_HOME=/tmp/hf
EXPOSE 8080
CMD ["uvicorn", "ltm_serve.service.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
