#!/usr/bin/env bash
# Create the local kind cluster, install cert-manager + KServe (Standard mode) + KEDA + a minimal Prometheus,
# and load the locally built images. Every kubectl/helm call names the context explicitly, so whatever context
# your shell has selected is never touched.
set -euo pipefail
# A dedicated kubeconfig: creating these clusters never changes the context your shell has selected.
export KUBECONFIG="${LTM_KUBECONFIG:-$HOME/.kube/ltm-serve}"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/.cache/ltm_serve/checkpoints}"
KSERVE_VERSION="${KSERVE_VERSION:-v0.20.0}"
CTX=kind-ltm
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! kind get clusters | grep -qx ltm; then
  envsubst < "$HERE/cluster.yaml" | kind create cluster --config - --wait 120s
fi

K() { kubectl --context "$CTX" "$@"; }
H() { helm --kube-context "$CTX" "$@"; }

H repo add jetstack https://charts.jetstack.io >/dev/null
H repo add kedacore https://kedacore.github.io/charts >/dev/null
H repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
H repo update >/dev/null

H upgrade --install cert-manager jetstack/cert-manager -n cert-manager --create-namespace \
  --set crds.enabled=true --wait
H upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version "$KSERVE_VERSION" \
  -n kserve --create-namespace --wait
H upgrade --install kserve-resources oci://ghcr.io/kserve/charts/kserve-resources --version "$KSERVE_VERSION" \
  -n kserve --set kserve.controller.deploymentMode=Standard --wait
H upgrade --install keda kedacore/keda -n keda --create-namespace --wait
# Just the Prometheus server: scrapes pods annotated prometheus.io/scrape=true (our service and Triton).
H upgrade --install prometheus prometheus-community/prometheus -n monitoring --create-namespace \
  --set alertmanager.enabled=false --set prometheus-pushgateway.enabled=false \
  --set prometheus-node-exporter.enabled=false --set kube-state-metrics.enabled=false \
  --set server.persistentVolume.enabled=false --set server.global.scrape_interval=5s --set server.global.scrape_timeout=4s --wait

for image in ltm-serve:cpu ltm-triton:cpu; do
  if docker image inspect "$image" >/dev/null 2>&1; then kind load docker-image "$image" --name ltm; fi
done

K apply -f "$HERE/../kserve/namespace.yaml"
K apply -f "$HERE/weights-pv.yaml"
echo "kind cluster ready: kubectl --context $CTX get isvc -n ltm"
