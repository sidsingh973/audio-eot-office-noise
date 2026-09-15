#!/bin/bash
# Serving benchmark: model-only benchmark, a closed-loop sweep per server configuration, then open-loop runs.
# Each configuration gets a fresh server on port 8100.
# Results: reports/bench_model.md, reports/stress_*.md, reports/stress_results.jsonl.
#
#   bash scripts/run_stress_grid.sh                         # everything; open loop at fractions of the best peak
#   SKIP_BENCH=1 KEEP_RESULTS=1 CONFIGS="w2i4:2:1:4" OPEN_CONFIGS="w1i8:1:1:8 w2i4:2:1:4" \
#       OPEN_RATES="90 100 110 120" OPEN_TAG=_knee bash scripts/run_stress_grid.sh
#
# A config is tag:workers:ort_threads:inflight_per_worker.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
UV=.venv/bin/uvicorn
PORT=8100
ALL_CONFIGS="w1:1:1:1 w2:2:1:1 w4:4:1:1 w6:6:1:1 w8:8:1:1 w1t4:1:4:1 w1i4:1:1:4 w1i8:1:1:8 w2i4:2:1:4"
CONFIGS=${CONFIGS-$ALL_CONFIGS}
mkdir -p logs reports
[ -n "${KEEP_RESULTS:-}" ] || rm -f reports/stress_results.jsonl

serve() {  # tag workers threads inflight -> starts the server in the background, sets $SERVER_PID
    EOT_THREADS=$3 EOT_MAX_INFLIGHT=$4 $UV service.app:app --port $PORT --workers $2 --log-level warning \
        > "logs/serve_$1.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 60); do curl -sf "localhost:$PORT/health" > /dev/null && return 0; sleep 1; done
    echo "server $1 did not start"; cat "logs/serve_$1.log"; exit 1
}

run() {  # tag workers threads inflight [stress_test args...]
    local tag=$1 w=$2 t=$3 i=$4
    shift 4
    serve "$tag" "$w" "$t" "$i"
    echo "=== $(date +%H:%M:%S) $tag: $w worker(s) x $t ORT thread(s), $i in-flight per worker"
    $PY scripts/stress_test.py --url "http://localhost:$PORT" --tag "$tag" \
        --note "$w worker(s) x $t ORT thread(s), $i in-flight inference(s) per worker" "$@"
    echo "server memory (RSS, MB): $(ps -o rss= -p "$SERVER_PID" $(pgrep -P "$SERVER_PID") | awk '{s+=$1} END {printf "%.0f", s/1024}')"
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2> /dev/null
}

if [ -z "${SKIP_BENCH:-}" ]; then
    echo "=== $(date +%H:%M:%S) model-only benchmark"
    $PY scripts/bench_model.py
fi

for c in $CONFIGS; do
    IFS=: read -r tag w t i <<< "$c"
    run "$tag" "$w" "$t" "$i"
done

if [ -n "${OPEN_CONFIGS:-}" ]; then
    for c in $OPEN_CONFIGS; do
        IFS=: read -r tag w t i <<< "$c"
        run "${tag}_open${OPEN_TAG:-}" "$w" "$t" "$i" --open-loop ${OPEN_RATES:?set OPEN_RATES} --duration 30
    done
else
    read -r BEST W T I PEAK < <(ALL="$ALL_CONFIGS $CONFIGS" $PY -c '
import json, os
cfg = {c.split(":")[0]: c.split(":")[1:] for c in os.environ["ALL"].split()}
rows = [json.loads(line) for line in open("reports/stress_results.jsonl")]
best = max((r for r in rows if r["mode"] == "closed"), key=lambda r: r["rps"])
print(best["tag"], *cfg[best["tag"]], round(best["rps"]))
')
    RATES=${OPEN_RATES:-$($PY -c "print(' '.join(str(max(1, round($PEAK * f))) for f in (0.25, 0.5, 0.75, 0.9, 1.0)))")}
    echo "=== best closed-loop config: $BEST ($PEAK req/s peak); open loop at $RATES req/s"
    run "${BEST}_open${OPEN_TAG:-}" "$W" "$T" "$I" --open-loop $RATES --duration 30
fi
echo "=== $(date +%H:%M:%S) done"
