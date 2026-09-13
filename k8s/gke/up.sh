#!/usr/bin/env bash
# GKE for the GPU phase: a zonal Standard cluster with one small system node and an L4 node pool that autoscales
# from 0, so GPU cost exists only while an L4 pod is scheduled. `down.sh` deletes everything this creates.
set -euo pipefail
# A dedicated kubeconfig: creating these clusters never changes the context your shell has selected.
export KUBECONFIG="${LTM_KUBECONFIG:-$HOME/.kube/ltm-serve}"

# Your GCP project: $PROJECT, else the active gcloud project.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
export PROJECT="${PROJECT:?set PROJECT to your GCP project id (or run: gcloud config set project <id>)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
CLUSTER="${CLUSTER:-ltm-serve}"
BUCKET="${BUCKET:-$PROJECT-ltm-serve}"
KSERVE_VERSION="${KSERVE_VERSION:-v0.20.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

gcloud services enable cloudbuild.googleapis.com container.googleapis.com artifactregistry.googleapis.com \
  --project "$PROJECT"

gcloud artifacts repositories describe ltm-serve --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud artifacts repositories create ltm-serve --repository-format=docker --location "$REGION" --project "$PROJECT"
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 ||
  gcloud storage buckets create "gs://$BUCKET" --location "$REGION" --uniform-bucket-level-access --project "$PROJECT"

if ! gcloud container clusters describe "$CLUSTER" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  # Image streaming lazily pulls image layers from Artifact Registry: the ~10 GB Triton image starts in seconds.
  gcloud container clusters create "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --release-channel regular --num-nodes 1 --machine-type e2-standard-4 \
    --workload-pool "$PROJECT.svc.id.goog" --enable-image-streaming
  gcloud container node-pools create l4 --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" \
    --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
    --enable-autoscaling --num-nodes 0 --min-nodes 0 --max-nodes 2 \
    --node-taints nvidia.com/gpu=present:NoSchedule --enable-image-streaming
fi
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT"
CTX="gke_${PROJECT}_${ZONE}_${CLUSTER}"
K() { kubectl --context "$CTX" "$@"; }
H() { helm --kube-context "$CTX" "$@"; }

H repo add jetstack https://charts.jetstack.io >/dev/null
H repo add kedacore https://kedacore.github.io/charts >/dev/null
H repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
H repo update >/dev/null
H upgrade --install cert-manager jetstack/cert-manager -n cert-manager --create-namespace --set crds.enabled=true --wait
H upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version "$KSERVE_VERSION" \
  -n kserve --create-namespace --wait
H upgrade --install kserve-resources oci://ghcr.io/kserve/charts/kserve-resources --version "$KSERVE_VERSION" \
  -n kserve --set kserve.controller.deploymentMode=Standard --wait
H upgrade --install keda kedacore/keda -n keda --create-namespace --wait
H upgrade --install prometheus prometheus-community/prometheus -n monitoring --create-namespace \
  --set alertmanager.enabled=false --set prometheus-pushgateway.enabled=false \
  --set prometheus-node-exporter.enabled=false --set kube-state-metrics.enabled=false \
  --set server.persistentVolume.enabled=false --set server.global.scrape_interval=5s --set server.global.scrape_timeout=4s --wait

K apply -f "$HERE/../kserve/namespace.yaml"
# Workload Identity, no service-account keys: pods running as ltm/default may read the weights bucket.
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --role roles/storage.objectViewer \
  --member "principal://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$PROJECT.svc.id.goog/subject/ns/ltm/sa/default" \
  >/dev/null

if ! gcloud storage ls "gs://$BUCKET/tabfm-bf16/model.safetensors" >/dev/null 2>&1; then
  gcloud builds submit "$HERE/../.." --project "$PROJECT" --config "$HERE/cloudbuild.yaml" \
    --substitutions "_REGION=$REGION,_BUCKET=$BUCKET"
fi
echo "GKE ready: kubectl --context $CTX get nodes"
