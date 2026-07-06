#!/usr/bin/env bash
# Throughput regression tripwire: fail if the q8_0 gen p50 in the fresh
# results dropped more than THRESHOLD_PCT below the recorded baseline.
# Usage: bench-compare.sh results.csv baseline.csv
set -euo pipefail
THRESHOLD_PCT="${THRESHOLD_PCT:-10}"
results="$1"; baseline="$2"
key='cima,Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0,gen p50'
extract() { grep -F "$key" "$1" | head -1 | awk -F, '{print $4}'; }
now=$(extract "$results"); base=$(extract "$baseline")
if [[ -z "$now" || -z "$base" ]]; then
  echo "bench-compare: key not found in one of the files — skipping tripwire"
  exit 0
fi
drop=$(awk -v n="$now" -v b="$base" 'BEGIN{printf "%.1f", (b-n)/b*100}')
echo "gen p50: baseline=$base now=$now (drop ${drop}%)"
awk -v d="$drop" -v t="$THRESHOLD_PCT" 'BEGIN{exit !(d>t)}' \
  && { echo "REGRESSION: >${THRESHOLD_PCT}% throughput drop"; exit 1; } \
  || { echo "within tolerance"; exit 0; }