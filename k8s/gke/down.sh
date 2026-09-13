#!/usr/bin/env bash
# Delete every billable resource up.sh created: cluster (incl. node pools and load balancers), images, bucket.
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

gcloud container clusters delete "$CLUSTER" --zone "$ZONE" --project "$PROJECT" --quiet || true
gcloud artifacts repositories delete ltm-serve --location "$REGION" --project "$PROJECT" --quiet || true
gcloud storage rm --recursive "gs://$BUCKET" --project "$PROJECT" || true
# Cloud Build source tarballs (gcloud builds submit creates <project>_cloudbuild on first use).
gcloud storage rm --recursive "gs://${PROJECT}_cloudbuild/source" --project "$PROJECT" || true
kubectl config delete-context "gke_${PROJECT}_${ZONE}_${CLUSTER}" 2>/dev/null || true
echo "remaining clusters:"; gcloud container clusters list --project "$PROJECT"
