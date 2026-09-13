# Triton Inference Server + Python backend with TabFM's dependencies installed into the container interpreter.
# The NGC image is multi-arch: arm64 runs CPU-only on a Mac, amd64 uses the GPU on GKE.
#
#   docker build -f docker/triton.Dockerfile -t ltm-triton:cpu .
#   docker buildx build --platform linux/amd64 --build-arg TORCH_VARIANT=cu126 -f docker/triton.Dockerfile ...
ARG TRITON_TAG=26.08-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

ARG TORCH_VERSION=2.14.0
ARG TORCH_VARIANT=cpu
ENV PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1

COPY docker/requirements.txt /tmp/requirements.txt
# requirements.txt excludes torch and CUDA wheels (see service.Dockerfile); the NGC base already ships gcc.
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} torch==${TORCH_VERSION} \
    && python3 -m pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md /opt/ltm_serve/
COPY src /opt/ltm_serve/src
RUN python3 -m pip install --no-deps /opt/ltm_serve

COPY triton_models/model_repository /model_repository
# KIND_GPU on CUDA images: Triton pins one model instance per visible GPU.
ARG INSTANCE_KIND=KIND_CPU
RUN sed -i "s/kind: KIND_CPU/kind: ${INSTANCE_KIND}/" /model_repository/tabfm/config.pbtxt
ENV HF_HOME=/tmp/hf
CMD ["tritonserver", "--model-repository=/model_repository", "--log-verbose=0"]
