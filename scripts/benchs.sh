#!/usr/bin/env bash
# =============================================================================
# benchs.sh — cross-engine benchmark, two comparisons only:
#   1) cima vs ollama            (same gguf weights, same wire protocol)
#   2) cima safetensors vs cima gguf   (format A/B on ONE engine)
#
# Host mode (default): compose up → benches inside the bench container →
# full teardown (down -v) even on Ctrl-C. NO human interaction required.
# Inner mode:  --inner   (executed inside the bench container)
#
# STATISTICS: every latency/throughput metric is sampled N times and
# reported as nearest-rank percentiles — p50 / p90 / p99 — with the sample
# count in the unit column. Warm metrics (prompt, gen) get GEN_RUNS
# samples (default 20); expensive cold cycles (load, unload) get
# LOAD_RUNS (default 5); media (image, audio) get MEDIA_RUNS (default 5).
# With n=5, p99 is effectively the max — the sample count is printed next
# to every number so nobody over-reads a tail from a thimble of data.
# Override with:  GEN_RUNS=50 LOAD_RUNS=10 MEDIA_RUNS=10 ./scripts/benchs.sh
#
# Metrics per engine/model:
#   load       wall-clock cold load: VERIFIED unload → empty-prompt load,
#              timed externally — identical yardstick for every engine
#   load(eng)  engine-reported load_duration (ollama field), N/A if absent
#   prompt     prompt_eval_duration on a fixed prompt (warm)
#   gen        eval tok/s on a fixed num_predict (engine-reported, warm)
#   gen(wall)  wall-clock tok/s over the same request (like-for-like
#              yardstick across engines; single request — labelled so)
#   image      wall time for a warm vision generate
#   audio      wall time for a warm audio generate — through the API
#              (`audio` is an additive parameter of /api/generate; engines
#              without it get an honest N/A)
#   unload     keep_alive:0 latency, verified via /api/ps empty
#
# Fairness notes baked in:
#   * one discarded warmup cycle per engine/model (first-touch pathologies
#     like GPU discovery watchdogs stay out of the percentiles)
#   * unloads are VERIFIED via /api/ps — a warm engine cannot masquerade
#     as cold
#   * warm-metric sampling happens on a resident model: the sequence is
#     load once, then GEN_RUNS identical requests back-to-back
# =============================================================================
set -euo pipefail

# Compose file lives under docker/ (run these scripts from the repo root).
# Exported so every `docker compose ...` below resolves it without -f.
export COMPOSE_FILE="${COMPOSE_FILE:-../docker/docker-compose.yml}"
cd "$(dirname "$0")" || exit 1

GGUF_TEXT_CIMA="Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0"
GGUF_TEXT_OLLAMA="qwen2.5:0.5b-instruct-q8_0"
MM_CIMA="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
MM_OLLAMA="gemma4:e2b"
ST_CIMA="Qwen/Qwen2.5-0.5B-Instruct"
# quant-matched 4-bit pair: same base model, ~4-bit in both containers.
# NOT bit-identical (Q4_K_M ~4.85 bpw, bnb NF4 ~4.5 bpw) — footnote it.
Q4_GGUF_CIMA="Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m"
BNB4_CIMA="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"

LOAD_RUNS="${LOAD_RUNS:-5}"           # cold load/unload cycles per model
GEN_RUNS="${GEN_RUNS:-20}"            # warm generate requests per model
MEDIA_RUNS="${MEDIA_RUNS:-5}"         # warm image/audio requests per model
NUM_PREDICT="${NUM_PREDICT:-64}"
NUM_PREDICT_IMAGE=96
PROMPT="Why is the sky blue? Answer in two sentences."
UNLOAD_POLLS=60                       # x0.5s = 30s drain budget

# ---------------------------------------------------------------------------
# HOST MODE
# ---------------------------------------------------------------------------
if [[ "${1:-}" != "--inner" ]]; then
  command -v docker >/dev/null || { echo "docker required in host mode"; exit 1; }
  export COMPOSE_PROFILES="tools"     # so down sees the bench service too
  LOGDIR=logs; mkdir -p "$LOGDIR"
  CLEANED=0
  cleanup() {
    [[ $CLEANED == 1 ]] && return; CLEANED=1
    ts=$(date +%Y%m%d-%H%M%S)
    docker compose logs --no-color --timestamps > "$LOGDIR/compose-$ts.log" 2>&1 || true
    docker compose ps -a >> "$LOGDIR/compose-$ts.log" 2>&1 || true
    echo "== cleanup: logs saved to $LOGDIR/compose-$ts.log; compose down -v =="
    docker compose down -v --remove-orphans || true
  }
  trap cleanup EXIT INT TERM

  docker compose up -d --build cima ollama
  # Preflight: a service that crashed at startup (e.g. GPU not injected)
  # otherwise turns into a silent liveness hang. Verify RUNNING; on
  # failure, print the dying container's own words and abort.
  sleep 3
  for s in cima ollama; do
    if ! docker compose ps --status running "$s" | grep -q "$s"; then
      echo "ERROR: service '$s' is not running — its log tail:"
      docker compose logs --no-color --tail 40 "$s" || true
      exit 1
    fi
  done
  docker compose build bench

  echo "== launching benches (cima vs ollama, then format A/B on cima) =="
  # The bench service mounts the repo root at /bench (working_dir); the
  # harness lives under scripts/, so invoke it by that path, not by CWD.
  docker compose run --rm -T \
    -e LOAD_RUNS="$LOAD_RUNS" -e GEN_RUNS="$GEN_RUNS" -e MEDIA_RUNS="$MEDIA_RUNS" -e NUM_PREDICT="$NUM_PREDICT" \
    bench scripts/benchs.sh --inner </dev/null
  echo "== all done — results in bench-results.csv =="
  exit 0
fi

# ---------------------------------------------------------------------------
# INNER MODE — pure HTTP against service DNS names
# ---------------------------------------------------------------------------
CIMA=http://cima:11435; OLLAMA=http://ollama:11434
ASSETS=/assets; mkdir -p "$ASSETS"
RESULTS=/bench/bench-results.csv

say() { printf '%-8s %-40s %-14s %10s %s\n' "$@"; }
rec() { echo "$1,$2,$3,$4,$5" >> "$RESULTS"; say "$1" "$2" "$3" "$4" "$5"; }
# 64-bit bash arithmetic — awk's %d overflowed on ns-epoch values and
# zeroed every wall-clock metric in an earlier revision.
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }

# Nearest-rank percentile of stdin values: pctl P [fmt]
pctl() {
  local p=$1 fmt="${2:-%.1f}"
  sort -n | awk -v p="$p" -v fmt="$fmt" \
    '{a[NR]=$1} END{ if(NR==0){printf "NA"; exit}
       r=int((p*NR+99)/100); if(r<1)r=1; if(r>NR)r=NR; printf fmt, a[r] }'
}

# Emit p50/p90/p99 rows for a sample list.
# rec_pcts engine model metric unit fmt v1 v2 ...
rec_pcts() {
  local name=$1 model=$2 metric=$3 unit=$4 fmt=$5; shift 5
  local n=$#
  if [[ $n -eq 0 ]]; then rec "$name" "$model" "$metric" "FAIL" "-"; return; fi
  local p
  for p in 50 90 99; do
    rec "$name" "$model" "${metric} p${p}" \
        "$(printf '%s\n' "$@" | pctl "$p" "$fmt")" "${unit} n=${n}"
  done
}

wait_up() { # url name [polls]
  local polls="${3:-120}"
  for _ in $(seq 1 "$polls"); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 2; done
  echo "ERROR: $2 never became ready at $1"; return 1
}

gen_assets() {
  [[ -f $ASSETS/photo.jpg ]] || ffmpeg -v error -f lavfi -i testsrc=size=320x240:rate=1 -frames:v 1 "$ASSETS/photo.jpg"
  [[ -f $ASSETS/speech.wav ]] || espeak-ng -w "$ASSETS/speech.wav" "a quick brown fox jumped over the lazy dog" || true
}

# ---- ollama-protocol helpers (work for cima AND ollama) --------------------
o_gen() { # base model [tag]
  # The optional tag PREFIXES the prompt with a unique marker: identical
  # repeated prompts hit engine prompt/session caches (prompt_eval ≈ 0,
  # and it exposed a cima gemma-4 KV-reuse bug), which is not what a
  # throughput bench should measure. A leading tag forces a full prefill
  # of comparable length on every engine, every request.
  local tagged="${3:+[sample $3] }$PROMPT"
  curl -sf "$1/api/generate" -d "{\"model\":\"$2\",\"prompt\":\"$tagged\",\"stream\":false,\"options\":{\"num_predict\":$NUM_PREDICT,\"temperature\":0}}"
}
o_load_only() { # base model — empty prompt = "load, generate nothing".
  # Echoes the body: the COLD engine-reported load_duration lives here.
  curl -sf "$1/api/generate" -d "{\"model\":\"$2\",\"stream\":false}"
}
o_unload() { # base model — unload and CONFIRM via /api/ps
  curl -sf "$1/api/generate" -d "{\"model\":\"$2\",\"keep_alive\":0}" >/dev/null 2>&1 || true
  for _ in $(seq 1 "$UNLOAD_POLLS"); do
    local n
    n=$(curl -sf "$1/api/ps" 2>/dev/null | jq -r '.models | length' 2>/dev/null || echo "?")
    [[ "$n" == "0" ]] && return 0
    sleep 0.5
  done
  echo "WARN: $1 still reports loaded models after keep_alive:0" >&2
  return 1
}

bench_ollama_proto() { # engine_name base model
  local name=$1 base=$2 model=$3 r out cold_ok=1
  local wall_loads=() eng_loads=() unloads=() prompts=() rates=() wall_rates=()

  # Warmup, discarded: absorbs first-touch costs (GPU discovery watchdog,
  # weight page-in) so they never pollute the percentiles.
  o_unload "$base" "$model" || true
  o_gen "$base" "$model" >/dev/null || true

  # --- Phase A: LOAD_RUNS verified-cold load/unload cycles ---
  for r in $(seq 1 "$LOAD_RUNS"); do
    local u0 u1
    u0=$(now_ms); o_unload "$base" "$model" || cold_ok=0; u1=$(now_ms)
    unloads+=($(( u1 - u0 )))
    sleep 1
    local t0 t1 cold_out
    t0=$(now_ms)
    cold_out=$(o_load_only "$base" "$model") || { rec "$name" "$model" "load" "FAIL" "-"; return; }
    t1=$(now_ms)
    wall_loads+=($(( t1 - t0 )))
    # engine-reported COLD load (read off the empty-prompt load, not a warm
    # generate, which correctly-but-uselessly reports ~0)
    eng_loads+=("$(jq -r '(.load_duration // 0) / 1e6' <<<"$cold_out")")
  done

  # --- Phase B: model resident, GEN_RUNS identical warm requests ---
  for r in $(seq 1 "$GEN_RUNS"); do
    local g0 g1
    g0=$(now_ms)
    out=$(o_gen "$base" "$model" "$r") || { rec "$name" "$model" "gen" "FAIL" "-"; return; }
    g1=$(now_ms)
    prompts+=("$(jq -r '(.prompt_eval_duration // 0) / 1e6' <<<"$out")")
    rates+=("$(jq -r 'if .eval_duration>0 then (.eval_count / (.eval_duration/1e9)) else 0 end' <<<"$out")")
    # wall-clock rate over the SAME request: the like-for-like yardstick
    # against engines that expose no eval_duration
    local ec
    ec=$(jq -r '.eval_count // 0' <<<"$out")
    wall_rates+=("$(awk -v n="$ec" -v ms="$(( g1 - g0 ))" 'BEGIN{ if (ms>0) printf "%.1f", n/(ms/1000); else print 0 }')")
  done

  if [[ $cold_ok == 1 ]]; then
    rec_pcts "$name" "$model" "load" "ms(wall)" "%.0f" "${wall_loads[@]}"
  else
    rec "$name" "$model" "load" "UNTRUSTED" "-"   # engine never verifiably cold
  fi
  # engine-reported load: N/A when the engine never fills the field
  local eng_p50
  eng_p50=$(printf '%s\n' "${eng_loads[@]}" | pctl 50 "%.0f")
  if awk -v v="$eng_p50" 'BEGIN{exit !(v>0)}'; then
    rec_pcts "$name" "$model" "load(eng)" "ms" "%.0f" "${eng_loads[@]}"
  else
    rec "$name" "$model" "load(eng)" "N/A" "-"
  fi
  # NB on reading percentiles: for LATENCIES (load/prompt/unload/image/
  # audio) p99 is the slow tail. For RATES (gen, tok/s) rank order runs
  # the other way — p99 is the FASTEST run and p50 is the headline.
  rec_pcts "$name" "$model" "prompt"    "ms"               "%.0f" "${prompts[@]}"
  rec_pcts "$name" "$model" "gen"       "tok/s"            "%.1f" "${rates[@]}"
  rec_pcts "$name" "$model" "gen(wall)" "tok/s(wall,1req)" "%.1f" "${wall_rates[@]}"
  rec_pcts "$name" "$model" "unload"    "ms(wall)"         "%.0f" "${unloads[@]}"
  o_unload "$base" "$model" || true
}

bench_media() { # engine base model metric json_key file num_predict
  local name=$1 base=$2 model=$3 metric=$4 key=$5 file=$6 np=$7
  local t0 t1 out r times=()
  [[ -f "$file" ]] || { rec "$name" "$model" "$metric" "NOASSET" "-"; return; }
  # Warm the model first so this measures encode+generate, not load.
  o_load_only "$base" "$model" >/dev/null 2>&1 || true
  local prompt="Describe this image in one sentence"
  [[ "$key" == "audio" ]] && prompt="Transcribe this audio"
  for r in $(seq 1 "$MEDIA_RUNS"); do
    # Body goes through a FILE: a wav's base64 (~176 KB) blows Linux's
    # 128 KB per-argument ceiling (MAX_ARG_STRLEN) as a curl -d literal —
    # field-observed as "curl: Argument list too long".
    # "think": false — thinking-parser models otherwise route every token
    # into .thinking and .response comes back empty.
    # The [sample N] tag defeats prompt caching (same reason as o_gen):
    # a cached repeat measures the cache, not the media encoder.
    jq -n --arg model "$model" --arg prompt "[sample $r] $prompt" \
          --arg key "$key" --rawfile b64 <(base64 -w0 "$file") --argjson np "$np" \
          '{model:$model, prompt:$prompt, stream:false, think:false, options:{num_predict:$np}} + {($key): [$b64 | rtrimstr("\n")]}' \
      > /tmp/media-req.json
    t0=$(now_ms)
    out=$(curl -sf "$base/api/generate" -d @/tmp/media-req.json) \
      || { rec "$name" "$model" "$metric" "FAIL" "-"; return; }
    t1=$(now_ms)
    jq -e '((.response // "") + (.thinking // "")) | length > 0' <<<"$out" >/dev/null \
      || { rec "$name" "$model" "$metric" "EMPTY" "-"; return; }
    times+=($(( t1 - t0 )))
  done
  rec_pcts "$name" "$model" "$metric" "ms(wall)" "%.0f" "${times[@]}"
}

# ---------------------------------------------------------------------------
echo "engine,model,metric,value,unit" > "$RESULTS"

echo "== waiting for engines =="
wait_up "$CIMA/api/version" cima
wait_up "$OLLAMA/api/version" ollama
gen_assets

echo "== pulling models (same weights per comparison) =="
curl -sf "$CIMA/api/pull"   -d "{\"model\":\"$GGUF_TEXT_CIMA\",\"stream\":false}" >/dev/null
curl -sf "$CIMA/api/pull"   -d "{\"model\":\"$MM_CIMA\",\"stream\":false}" >/dev/null
curl -sf "$CIMA/api/pull"   -d "{\"model\":\"$ST_CIMA\",\"stream\":false}" >/dev/null
curl -sf "$CIMA/api/pull"   -d "{\"model\":\"$Q4_GGUF_CIMA\",\"stream\":false}" >/dev/null
curl -sf "$CIMA/api/pull"   -d "{\"model\":\"$BNB4_CIMA\",\"stream\":false}" >/dev/null || echo "(bnb-4bit pull failed — quant-matched section will FAIL visibly)"
curl -sf "$OLLAMA/api/pull" -d "{\"model\":\"$GGUF_TEXT_OLLAMA\",\"stream\":false}" >/dev/null
curl -sf "$OLLAMA/api/pull" -d "{\"model\":\"$MM_OLLAMA\",\"stream\":false}" >/dev/null || echo "(ollama pull $MM_OLLAMA failed — its benches will be skipped)"

echo; echo "== 1) cima vs ollama — text (gguf, same q8_0 weights) =="
say ENGINE MODEL METRIC VALUE UNIT
bench_ollama_proto cima   "$CIMA"   "$GGUF_TEXT_CIMA"
bench_ollama_proto ollama "$OLLAMA" "$GGUF_TEXT_OLLAMA"

echo; echo "== 1b) cima vs ollama — multimodal (gemma-4 E2B) =="
bench_ollama_proto cima "$CIMA" "$MM_CIMA"
bench_media cima "$CIMA" "$MM_CIMA" "image" "images" "$ASSETS/photo.jpg"  "$NUM_PREDICT_IMAGE"
# audio through the API: `audio` is an additive parameter of /api/generate
# (base64 array, same shape as `images`) — Ollama-protocol compatible
# because absent keys change nothing.
bench_media cima "$CIMA" "$MM_CIMA" "audio" "audio"  "$ASSETS/speech.wav" "$NUM_PREDICT_IMAGE"
if curl -sf "$OLLAMA/api/show" -d "{\"model\":\"$MM_OLLAMA\"}" >/dev/null 2>&1; then
  bench_ollama_proto ollama "$OLLAMA" "$MM_OLLAMA"
  bench_media ollama "$OLLAMA" "$MM_OLLAMA" "image" "images" "$ASSETS/photo.jpg" "$NUM_PREDICT_IMAGE"
fi
# audio: cima-only capability among the contenders — honest N/A elsewhere.
rec ollama "$MM_OLLAMA" "audio" "N/A" "-"

echo; echo "== 2) format A/B on ONE engine (cima): safetensors vs gguf =="
o_unload "$CIMA" "$MM_CIMA" || true    # drain the 5.7GB model first
echo "(same base model, full precision: safetensors bf16 vs gguf q8_0)"
bench_ollama_proto cima "$CIMA" "$ST_CIMA"
bench_ollama_proto cima "$CIMA" "$GGUF_TEXT_CIMA"
echo "(same base model, ~4-bit: gguf Q4_K_M ~4.85 bpw vs bnb NF4 ~4.5 bpw — near-matched, not identical)"
bench_ollama_proto cima "$CIMA" "$Q4_GGUF_CIMA"
bench_ollama_proto cima "$CIMA" "$BNB4_CIMA"
o_unload "$CIMA" "$BNB4_CIMA" || true

echo; echo "== done — results in bench-results.csv =="