#!/usr/bin/env bash
# The "optimise the deployed system" experiment: redeploy FastAPI and Triton with LTM_COMPILE_MODE=reduce-overhead
# (torch.compile + CUDA graphs, row buckets warmed up before ready) and rerun the same in-cluster load test as the
# eager runs. Results: results/load/raw/gke_*_compiled.jsonl, startup logs in results/logs/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
# Your GCP project: $PROJECT, else the active gcloud project.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
export PROJECT="${PROJECT:?set PROJECT to your GCP project id (or run: gcloud config set project <id>)}"
K="$HERE/kctl"
RATES="${RATES:-80,160,240,320,480,640}"
LOGS="$ROOT/results/logs"
RAW="$ROOT/results/load/raw"
mkdir -p "$LOGS" "$RAW"

ts() { date -u +%H:%M:%S; }

wait_ready() { # wait_ready <isvc> <timeout_s>
  local t0; t0=$(date +%s)
  until [ "$($K -n ltm get isvc "$1" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = "True" ]; do
    if [ $(( $(date +%s) - t0 )) -gt "$2" ]; then echo "$(ts) TIMEOUT waiting for $1"; return 1; fi
    if $K -n ltm get pods -l "serving.kserve.io/inferenceservice=$1" --no-headers 2>/dev/null | grep -q -E "CrashLoop|Error|ImagePull"; then
      echo "$(ts) FAILED $1"; $K -n ltm get pods; return 1
    fi
    sleep 10
  done
  echo "$(ts) READY $1 after $(( $(date +%s) - t0 ))s"
}

load() { # load <name> <url> <model> <input>
  "$HERE/run.sh" "$HERE/job-loadgen.yaml" NAME="$1" URL="$2" MODEL="$3" INPUT_NAME="$4" ROWS=1 RATES="$RATES" \
    DURATION=30 PROCS=4 CLIENT_CPU=4 LOADGEN_POOL=l4 >/dev/null
  until $K -n ltm get pod "loadgen-$1" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q -E "Running|Failed"; do sleep 5; done
  until $K -n ltm logs "loadgen-$1" 2>/dev/null | grep -q -E "LOAD_DONE|Traceback"; do sleep 10; done
  $K -n ltm logs "loadgen-$1" | grep -E "rate=|Traceback" | sed "s/^/$(ts) $1 /"
  $K -n ltm cp "loadgen-$1:/tmp/load.jsonl" "$RAW/gke_$1.jsonl" >/dev/null
  $K -n ltm delete pod "loadgen-$1" --wait=true >/dev/null
}

if [ "${SKIP_FASTAPI:-0}" != "1" ]; then
echo "$(ts) === FastAPI, reduce-overhead ==="
"$HERE/run.sh" "$HERE/isvc-fastapi-gpu.yaml" COMPILE_MODE=reduce-overhead MIN_REPLICAS=1 MAX_REPLICAS=1 >/dev/null
wait_ready tabfm-fastapi 2400
P=$($K -n ltm get pods -l serving.kserve.io/inferenceservice=tabfm-fastapi -o name | head -1)
$K -n ltm logs "$P" -c kserve-container > "$LOGS/gke_fastapi_compiled_startup.log" 2>&1 || true
grep -E "model loaded|demo context" "$LOGS/gke_fastapi_compiled_startup.log" | sed "s/^/$(ts) /"
load fastapi-compiled http://tabfm-fastapi-predictor.ltm.svc.cluster.local demo rows
$K -n ltm exec "$P" -c kserve-container -- python -c "
import urllib.request
for l in urllib.request.urlopen('http://localhost:8080/metrics').read().decode().splitlines():
    if l.startswith(('ltm_batch_rows_sum','ltm_batch_rows_count','ltm_inference_seconds_sum')): print(l)" | sed "s/^/$(ts) metrics /"
$K -n ltm delete isvc tabfm-fastapi --wait=true >/dev/null
until [ -z "$($K -n ltm get pods -l serving.kserve.io/inferenceservice=tabfm-fastapi --no-headers 2>/dev/null)" ]; do sleep 5; done
fi

if [ "${SKIP_TRITON:-0}" != "1" ]; then

echo "$(ts) === Triton, reduce-overhead, 1 instance ==="
"$HERE/run.sh" "$HERE/isvc-triton-gpu.yaml" COMPILE_MODE=reduce-overhead >/dev/null
wait_ready tabfm-triton 2400
P=$($K -n ltm get pods -l serving.kserve.io/inferenceservice=tabfm-triton -o name | head -1)
$K -n ltm logs "$P" -c kserve-container > "$LOGS/gke_triton_compiled_startup.log" 2>&1 || true
grep -E "tabfm on" "$LOGS/gke_triton_compiled_startup.log" | sed "s/^/$(ts) /"
load triton-compiled-i1 http://tabfm-triton-predictor.ltm.svc.cluster.local tabfm ROWS

echo "$(ts) === Triton, reduce-overhead, 2 instances (repository API reload) ==="
$K -n ltm exec "$P" -c kserve-container -- python3 -c "
import json, time, urllib.request
cfg = json.load(urllib.request.urlopen('http://localhost:8080/v2/models/tabfm/config'))
cfg['instance_group'] = [{'count': 2, 'kind': 'KIND_GPU', 'gpus': [0]}]
body = json.dumps({'parameters': {'config': json.dumps(cfg)}}).encode()
t0 = time.time()
req = urllib.request.Request('http://localhost:8080/v2/repository/models/tabfm/load', data=body, method='POST', headers={'content-type': 'application/json'})
print('reload status', urllib.request.urlopen(req, timeout=3000).status, 'in %.0fs' % (time.time() - t0))" | sed "s/^/$(ts) /"
load triton-compiled-i2 http://tabfm-triton-predictor.ltm.svc.cluster.local tabfm ROWS
$K -n ltm exec "$P" -c kserve-container -- curl -s localhost:8002/metrics | grep -E "^nv_gpu_utilization|^nv_inference_(count|exec_count)\{" | sed "s/^/$(ts) metrics /"
fi
echo "$(ts) === DONE ==="
