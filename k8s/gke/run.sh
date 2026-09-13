#!/usr/bin/env bash
# Render a GKE manifest with envsubst and apply it:  k8s/gke/run.sh <manifest.yaml> [KEY=VALUE ...]
# Defaults below are overridable per call, e.g. MAX_DELAY_MS=5 k8s/gke/run.sh k8s/gke/isvc-fastapi-gpu.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Your GCP project: $PROJECT, else the active gcloud project.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
export PROJECT="${PROJECT:?set PROJECT to your GCP project id (or run: gcloud config set project <id>)}"
REGION="${REGION:-us-central1}"
REGISTRY="$REGION-docker.pkg.dev/$PROJECT/ltm-serve"

export BUCKET="${BUCKET:-$PROJECT-ltm-serve}"
export MIN_REPLICAS="${MIN_REPLICAS:-1}" MAX_REPLICAS="${MAX_REPLICAS:-1}" TARGET_INFLIGHT="${TARGET_INFLIGHT:-4}"
export MAX_BATCH_ROWS="${MAX_BATCH_ROWS:-256}" MAX_DELAY_MS="${MAX_DELAY_MS:-0}"
export COMPILE_MODE="${COMPILE_MODE:-}"
export DEMO_CONTEXT="${DEMO_CONTEXT:-demo:n_train=1024,n_features=10,n_classes=2,n_estimators=8}"
export TRITON_CONTEXT="${TRITON_CONTEXT:-synthetic:n_train=1024,n_features=10,n_classes=2,n_estimators=8}"
export SWEEPS="${SWEEPS:-noise_floor stages context_scaling feature_scaling query_scaling ensemble member_batching overhead compile}"
export NAME="${NAME:-run}" URL="${URL:-http://tabfm-fastapi-predictor.ltm.svc.cluster.local}" MODEL="${MODEL:-demo}"
export INPUT_NAME="${INPUT_NAME:-rows}" ROWS="${ROWS:-1}" RATES="${RATES:-1,2,5,10,20}" DURATION="${DURATION:-30}"
export PROCS="${PROCS:-3}" CLIENT_CPU="${CLIENT_CPU:-2}" LOADGEN_POOL="${LOADGEN_POOL:-default-pool}"

manifest="$1"
shift
for kv in "$@"; do export "${kv?}"; done
case "$(basename "$manifest")" in
  *triton*) export IMAGE="${IMAGE:-$REGISTRY/ltm-triton:cu126}" ;;
  *) export IMAGE="${IMAGE:-$REGISTRY/ltm-serve:cu126}" ;;
esac
if [[ "$(basename "$manifest")" == job-loadgen.yaml ]]; then
  "$HERE/kctl" -n ltm create configmap loadgen --from-file="$HERE/../../src/ltm_serve/loadgen.py" \
    --dry-run=client -o yaml | "$HERE/kctl" apply -f - >/dev/null
fi
envsubst < "$manifest" | "$HERE/kctl" apply -f -
