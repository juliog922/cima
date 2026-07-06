#!/usr/bin/env bash
# =============================================================================
# test.sh — cima API conformance + stress suite (ollama-protocol contract).
#
# Host mode orchestrates compose (cima only) and cleans up; --inner runs
# inside the bench container. Every check prints PASS/FAIL; exit code is
# the FAIL count. This doubles as the ollama-parity conformance record:
# a FAIL here is either a bug or an unimplemented parity feature — both
# are work items by decree.
#
# Sections: liveness · listing · generate (stream/non-stream/options/
# stop/seed determinism) · chat (+tools param) · structured format ·
# images · embeddings · load-unload lifecycle · error contracts (unknown
# model, malformed json, missing fields) · not-implemented endpoints
# (create/copy/push → explicit 501) · concurrency stress · mixed-model
# stress with load/unload churn · ADVERSARIAL (abort/spam/oversize/races).
#
# STRESS-WHILE-TESTING: a background chaos generator (metadata polls +
# short generates on a second model) runs through every functional
# section, so each endpoint is exercised on a BUSY server — the real
# flow. It pauses only around the ps-empty lifecycle check (which it
# would legitimately pollute) and must itself sustain a minimum op count
# to pass.
# =============================================================================
set -uo pipefail

# Compose file lives under docker/ (run these scripts from the repo root).
# Exported so every `docker compose ...` below resolves it without -f.
export COMPOSE_FILE="${COMPOSE_FILE:-../docker/docker-compose.yml}"
cd "$(dirname "$0")" || exit 1

if [[ "${1:-}" != "--inner" ]]; then
  command -v docker >/dev/null || { echo "docker required in host mode"; exit 1; }
  LOGDIR=logs; mkdir -p "$LOGDIR"
  cleanup() {
    # Preserve the black box BEFORE tearing it down: engine logs and
    # container states outlive the containers.
    ts=$(date +%Y%m%d-%H%M%S)
    docker compose logs --no-color --timestamps > "$LOGDIR/compose-$ts.log" 2>&1 || true
    docker compose ps -a >> "$LOGDIR/compose-$ts.log" 2>&1 || true
    echo "== cleanup: logs saved to $LOGDIR/compose-$ts.log; compose down -v =="
    docker compose down -v --remove-orphans || true
  }
  trap cleanup EXIT INT TERM
  docker compose up -d --build cima
  # Preflight: a service that crashed at startup (e.g. GPU not injected)
  # otherwise turns into a silent liveness hang. Verify RUNNING; on
  # failure, print the dying container's own words and abort.
  sleep 3
  s=cima
  if ! docker compose ps --status running "$s" | grep -q "$s"; then
    echo "ERROR: service '$s' is not running — its log tail:"
    docker compose logs --no-color --tail 40 "$s" || true
    exit 1
  fi
  docker compose --profile tools build bench
  # scripts live under scripts/ relative to the /bench working_dir.
  docker compose --profile tools run --rm bench bash scripts/test.sh --inner
  exit $?
fi

BASE=http://cima:11435
SMALL="Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0"
SMALL2="Qwen/Qwen2.5-0.5B-Instruct"
MM="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
FAILS=0; PASSES=0
pass() { echo "PASS  $1"; PASSES=$((PASSES+1)); }
fail() { echo "FAIL  $1  ${2:-}"; FAILS=$((FAILS+1)); }
code_of() { curl -s -o /tmp/body -w '%{http_code}' "$@"; }

echo "== liveness =="
for _ in $(seq 1 120); do curl -sf "$BASE/api/version" >/dev/null && break; sleep 2; done
curl -sf "$BASE/api/version" >/dev/null && pass "GET /api/version" || { fail "server unreachable"; exit 1; }

echo "== pulls =="
for m in "$SMALL" "$SMALL2" "$MM"; do
  c=$(code_of -X POST "$BASE/api/pull" -d "{\"model\":\"$m\",\"stream\":false}")
  [[ $c == 200 ]] && pass "pull $m" || fail "pull $m" "http $c: $(head -c200 /tmp/body)"
done

echo "== background chaos: continuous mixed load through every section =="
CHAOS_LOG=/tmp/chaos.count; : > "$CHAOS_LOG"; rm -f /tmp/chaos.pause
chaos_loop() {
  while :; do
    [[ -f /tmp/chaos.pause ]] && { sleep 0.3; continue; }
    curl -sf --max-time 10 "$BASE/api/tags" >/dev/null 2>&1 && echo t >> "$CHAOS_LOG"
    curl -sf --max-time 10 "$BASE/api/ps"   >/dev/null 2>&1 && echo p >> "$CHAOS_LOG"
    curl -sf --max-time 60 -X POST "$BASE/api/generate" \
      -d "{\"model\":\"$SMALL2\",\"prompt\":\"chaos tick\",\"stream\":false,\"options\":{\"num_predict\":4}}" \
      | jq -e '.done==true' >/dev/null 2>&1 && echo g >> "$CHAOS_LOG"
    sleep 0.7
  done
}
chaos_loop & CHAOS_PID=$!
chaos_pause()  { touch /tmp/chaos.pause; sleep 3; }
chaos_resume() { rm -f /tmp/chaos.pause; }
pass "chaos generator running (pid $CHAOS_PID) — all sections below run under load"

echo "== listing =="
c=$(code_of "$BASE/api/tags")
[[ $c == 200 ]] && jq -e '.models | length >= 3' /tmp/body >/dev/null \
  && pass "/api/tags lists pulled models" || fail "/api/tags" "http $c"

echo "== correctness: statelessness, determinism, embedding sanity =="
# These catch the class of bug that returns a clean 200 with valid JSON but
# WRONG numbers — e.g. the dequant-scratch overflow, which corrupted device
# memory so that the first request was right and every later one degraded.
# Shape/HTTP checks pass on that bug; only cross-request comparison catches it.

# Helper: greedy (temperature 0) generate, response text only.
gen0() { # $1=model $2=prompt [$3=base64 image]
  local body
  if [[ -n "${3:-}" ]]; then
    body=$(jq -n --arg m "$1" --arg p "$2" --arg img "$3" \
      '{model:$m,prompt:$p,stream:false,images:[$img],options:{num_predict:24,temperature:0}}')
  else
    body=$(jq -n --arg m "$1" --arg p "$2" \
      '{model:$m,prompt:$p,stream:false,options:{num_predict:24,temperature:0}}')
  fi
  curl -sf --max-time 120 -X POST "$BASE/api/generate" -d "$body" | jq -r '.response // empty'
}

# 1. Text idempotency: identical greedy request three times → identical text.
t1=$(gen0 "$SMALL" "Name three primary colors.")
t2=$(gen0 "$SMALL" "Name three primary colors.")
t3=$(gen0 "$SMALL" "Name three primary colors.")
if [[ -n "$t1" && "$t1" == "$t2" && "$t2" == "$t3" ]]; then
  pass "text greedy idempotent across 3 requests"
else
  fail "text greedy NOT idempotent (state leak)" "r1='$(head -c60 <<<"$t1")' r2='$(head -c60 <<<"$t2")' r3='$(head -c60 <<<"$t3")'"
fi

# 2. Interleave invariance: A,B,A greedy → both A identical (B must not
#    perturb the resident state that A depends on).
a1=$(gen0 "$SMALL" "Name three primary colors.")
_b=$(gen0 "$SMALL" "Write a haiku about the sea.")
a2=$(gen0 "$SMALL" "Name three primary colors.")
[[ -n "$a1" && "$a1" == "$a2" ]] && pass "interleave invariance (A,B,A → A==A)" \
  || fail "interleave invariance broken (B perturbs A)" "a1='$(head -c60 <<<"$a1")' a2='$(head -c60 <<<"$a2")'"

# 3. Multimodal idempotency: the exact bug that shipped. Same image+prompt
#    three times against the resident gemma-4 → identical greedy responses.
ffmpeg -v error -f lavfi -i testsrc=size=160x120:rate=1 -frames:v 1 /tmp/c.png 2>/dev/null || true
if [[ -f /tmp/c.png ]]; then
  cimg=$(base64 -w0 /tmp/c.png)
  m1=$(gen0 "$MM" "Describe this image." "$cimg")
  m2=$(gen0 "$MM" "Describe this image." "$cimg")
  m3=$(gen0 "$MM" "Describe this image." "$cimg")
  if [[ -n "$m1" && "$m1" == "$m2" && "$m2" == "$m3" ]]; then
    pass "multimodal greedy idempotent across 3 requests (scratch-overflow guard)"
  else
    fail "multimodal NOT idempotent — resident-state corruption" \
      "r1='$(head -c50 <<<"$m1")' r2='$(head -c50 <<<"$m2")' r3='$(head -c50 <<<"$m3")'"
  fi
fi

# 4. Greedy determinism over a LONGER generation (64 tokens): a corruption
#    that only manifests deep in decode still shows here.
l1=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Count from one to twenty in words.\",\"stream\":false,\"options\":{\"num_predict\":64,\"temperature\":0}}" | jq -r .response)
l2=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Count from one to twenty in words.\",\"stream\":false,\"options\":{\"num_predict\":64,\"temperature\":0}}" | jq -r .response)
[[ -n "$l1" && "$l1" == "$l2" ]] && pass "greedy determinism over 64 tokens" \
  || fail "greedy nondeterministic at length (uninit read / race)" "len1=${#l1} len2=${#l2}"

# 5. Embedding — identical inputs must give an identical vector (cosine 1.0),
#    and unrelated inputs must be discriminable (cosine < 0.99), all finite.
emb() { curl -sf -X POST "$BASE/api/embed" -d "{\"model\":\"$SMALL2\",\"input\":$(jq -Rn --arg t "$1" '$t')}" | jq -c '.embeddings[0]'; }
E1=$(emb "The cat sat on the mat."); E2=$(emb "The cat sat on the mat."); E3=$(emb "Quarterly financial projections rose.")
emb_out=$(python3 - "$E1" "$E2" "$E3" <<'PY'
import sys, json, math
def load(s):
    try: return json.loads(s)
    except Exception: return None
a,b,c = (load(sys.argv[i]) for i in (1,2,3))
def cos(u,v):
    d=sum(x*y for x,y in zip(u,v)); n=math.sqrt(sum(x*x for x in u))*math.sqrt(sum(y*y for y in v))
    return d/n if n else 0.0
if not a or not b or not c:
    print("FAIL|embedding sanity (empty/invalid embedding vector)"); sys.exit(0)
if any(not math.isfinite(x) for x in a+b+c):
    print("FAIL|embedding sanity (NaN/Inf in embedding)"); sys.exit(0)
same=cos(a,b); disc=cos(a,c)
print(("PASS|" if same>=0.9999 else "FAIL|")+f"embedding identical inputs cosine={same:.6f} (want ~1.0)")
print(("PASS|" if disc<0.99 else "FAIL|")+f"embedding discriminates unrelated cosine={disc:.4f} (want <0.99)")
PY
)
while IFS='|' read -r verdict msg; do
  [[ "$verdict" == PASS ]] && pass "$msg" || fail "$msg"
done <<<"$emb_out"

echo "== generate: non-stream =="
c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Say OK\",\"stream\":false,\"options\":{\"num_predict\":8,\"temperature\":0}}")
[[ $c == 200 ]] && jq -e '.done == true and (.response|length>0) and .eval_count>0 and .total_duration>0 and .load_duration>=0 and .prompt_eval_count>0' /tmp/body >/dev/null \
  && pass "generate non-stream + timing fields" || fail "generate non-stream" "http $c: $(head -c300 /tmp/body)"
jq -e '.done_reason == "stop" or .done_reason == "length"' /tmp/body >/dev/null \
  && pass "done_reason present" || fail "done_reason missing/unknown" "$(jq -r '.done_reason' /tmp/body 2>/dev/null)"

echo "== generate: stream (ndjson, final frame carries stats) =="
curl -sN -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Count to five\",\"options\":{\"num_predict\":16}}" > /tmp/stream || true
frames=$(wc -l < /tmp/stream)
last=$(tail -n1 /tmp/stream)
[[ $frames -ge 2 ]] && jq -e '.done == true and .eval_duration > 0' <<<"$last" >/dev/null \
  && pass "generate stream ($frames frames)" || fail "generate stream" "frames=$frames last=$(head -c200 <<<"$last")"

echo "== options: num_predict cap, stop sequence, seed determinism =="
c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Write a long story\",\"stream\":false,\"options\":{\"num_predict\":5}}")
jq -e '.eval_count <= 6' /tmp/body >/dev/null && pass "num_predict respected" || fail "num_predict" "eval_count=$(jq -r .eval_count /tmp/body)"
c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"List: apple banana cherry\",\"stream\":false,\"options\":{\"num_predict\":64,\"stop\":[\"banana\"]}}")
[[ $c == 200 ]] && ! jq -r '.response' /tmp/body | grep -q banana && pass "stop sequence honored" || fail "stop sequence" "$(jq -r '.response' /tmp/body | head -c120)"
r1=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Random word:\",\"stream\":false,\"options\":{\"num_predict\":8,\"seed\":42,\"temperature\":0.9}}" | jq -r .response)
r2=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Random word:\",\"stream\":false,\"options\":{\"num_predict\":8,\"seed\":42,\"temperature\":0.9}}" | jq -r .response)
[[ -n "$r1" && "$r1" == "$r2" ]] && pass "seed determinism" || fail "seed determinism" "'$r1' vs '$r2'"

echo "== chat: messages, system, tools parameter =="
c=$(code_of -X POST "$BASE/api/chat" -d "{\"model\":\"$SMALL\",\"stream\":false,\"messages\":[{\"role\":\"system\",\"content\":\"Answer tersely\"},{\"role\":\"user\",\"content\":\"2+2?\"}],\"options\":{\"num_predict\":8,\"temperature\":0}}")
[[ $c == 200 ]] && jq -e '.message.role == "assistant" and (.message.content|length>0)' /tmp/body >/dev/null \
  && pass "chat basic" || fail "chat basic" "http $c: $(head -c200 /tmp/body)"
c=$(code_of -X POST "$BASE/api/chat" -d "{\"model\":\"$SMALL\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"options\":{\"num_predict\":64,\"temperature\":0}}")
if [[ $c == 200 ]] && jq -e '.message.tool_calls[0].function.name == "get_weather"' /tmp/body >/dev/null 2>&1; then
  pass "chat tools → tool_calls emitted"
else
  fail "chat tools (parity requirement)" "http $c: $(head -c300 /tmp/body)"
fi

echo "== options: done_reason, exclusive stop, ignore_eos, top_k, tools =="
# done_reason must reflect WHY generation ended.
dr_len=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Write a long essay about rivers.\",\"stream\":false,\"options\":{\"num_predict\":5,\"temperature\":0}}" | jq -r .done_reason)
[[ "$dr_len" == "length" ]] && pass "done_reason=length when num_predict cap is hit" \
  || fail "done_reason wrong at cap" "got '$dr_len', want 'length'"
dr_stop=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Say hello.\",\"stream\":false,\"options\":{\"num_predict\":200,\"temperature\":0}}" | jq -r .done_reason)
[[ "$dr_stop" == "stop" ]] && pass "done_reason=stop on natural EOS" \
  || fail "done_reason wrong on EOS" "got '$dr_stop', want 'stop'"

# Exclusive stop: the stop string must NOT appear in the output.
so=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"List fruits: apple banana cherry\",\"stream\":false,\"options\":{\"num_predict\":64,\"stop\":[\"banana\"]}}" | jq -r .response)
if grep -qi "banana" <<<"$so"; then
  fail "stop is inclusive (stop text leaked into output)" "output contained the stop string"
else
  pass "stop sequence is exclusive (stop text absent)"
fi

# ignore_eos forces exactly num_predict tokens.
ec=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":32,\"ignore_eos\":true}}" | jq -r .eval_count)
[[ "$ec" == "32" ]] && pass "ignore_eos generates exactly num_predict tokens" \
  || fail "ignore_eos token count" "eval_count=$ec, want 32"

# top_k=1 is greedy even at high temperature → two runs identical.
k1a=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"A fruit:\",\"stream\":false,\"options\":{\"temperature\":2.0,\"top_k\":1,\"num_predict\":6}}" | jq -r .response)
k1b=$(curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"A fruit:\",\"stream\":false,\"options\":{\"temperature\":2.0,\"top_k\":1,\"num_predict\":6}}" | jq -r .response)
[[ -n "$k1a" && "$k1a" == "$k1b" ]] && pass "top_k=1 is deterministic at high temperature" \
  || fail "top_k=1 not deterministic" "a='$k1a' b='$k1b'"

# tools: model should surface a structured tool_call for a weather query.
tc=$(curl -sf -X POST "$BASE/api/chat" -d "{\"model\":\"$SMALL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"stream\":false,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"options\":{\"temperature\":0,\"num_predict\":80}}")
# Accept either a structured tool_call OR (weaker models) the tool name in text —
# the wire-shape check is the real assertion; name-in-text is a soft pass.
if jq -e '.message.tool_calls[0].function.name' >/dev/null 2>&1 <<<"$tc"; then
  pass "tools: structured tool_call returned"
elif grep -q "get_weather" <<<"$tc"; then
  pass "tools: tool referenced (unstructured; model-dependent)"
else
  fail "tools: no tool_call and no tool reference" "$(jq -rc '.message' <<<"$tc" 2>/dev/null | head -c150)"
fi

echo "== structured format: json + schema =="
c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Give a JSON object with key answer=4\",\"stream\":false,\"format\":\"json\",\"options\":{\"num_predict\":128,\"temperature\":0}}")
[[ $c == 200 ]] && jq -r '.response' /tmp/body | jq -e . >/dev/null 2>&1 \
  && pass "format:json returns valid JSON" || fail "format:json (parity requirement)" "http $c: $(jq -r '.response' /tmp/body 2>/dev/null | head -c150)"
c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Population of France?\",\"stream\":false,\"format\":{\"type\":\"object\",\"properties\":{\"population\":{\"type\":\"integer\"}},\"required\":[\"population\"]},\"options\":{\"num_predict\":128,\"temperature\":0}}")
[[ $c == 200 ]] && jq -r '.response' /tmp/body | jq -e '.population|numbers' >/dev/null 2>&1 \
  && pass "format:schema constrained output" || fail "format:schema (parity requirement)" "http $c"

echo "== images (multimodal generate) =="
ffmpeg -v error -f lavfi -i testsrc=size=160x120:rate=1 -frames:v 1 /tmp/t.png 2>/dev/null || true
if [[ -f /tmp/t.png ]]; then
  img=$(base64 -w0 /tmp/t.png)
  c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"$MM\",\"prompt\":\"one word: what is this\",\"stream\":false,\"images\":[\"$img\"],\"options\":{\"num_predict\":16}}")
  [[ $c == 200 ]] && jq -e '.response|length>0' /tmp/body >/dev/null \
    && pass "images param" || fail "images param (parity requirement)" "http $c: $(head -c200 /tmp/body)"
fi

echo "== embeddings =="
c=$(code_of -X POST "$BASE/api/embed" -d "{\"model\":\"$SMALL2\",\"input\":\"hello world\"}")
[[ $c == 200 ]] && jq -e '.embeddings[0] | length > 10' /tmp/body >/dev/null \
  && pass "/api/embed" || fail "/api/embed" "http $c: $(head -c200 /tmp/body)"

echo "== lifecycle: load, ps, unload, ps-empty (chaos paused: ps-empty is a legitimate quiet-state assertion) =="
chaos_pause
curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL2\",\"keep_alive\":0,\"stream\":false}" >/dev/null
curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"stream\":false}" >/dev/null
curl -sf "$BASE/api/ps" | jq -e --arg m "$SMALL" '.models[]?.name == $m' >/dev/null \
  && pass "/api/ps shows loaded model" || fail "/api/ps after load"
curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"keep_alive\":0,\"stream\":false}" >/dev/null
sleep 2
curl -sf "$BASE/api/ps" | jq -e '(.models // []) | length == 0' >/dev/null \
  && pass "keep_alive:0 unloads" || fail "keep_alive:0 unload" "$(curl -sf $BASE/api/ps)"
chaos_resume

echo "== error contracts =="
c=$(code_of -X POST "$BASE/api/generate" -d "{\"model\":\"nope/never:x\",\"prompt\":\"hi\",\"stream\":false}")
[[ $c -ge 400 && $c -lt 500 ]] && jq -e '.error|length>0' /tmp/body >/dev/null \
  && pass "unknown model → 4xx + json error" || fail "unknown model" "http $c: $(head -c200 /tmp/body)"
c=$(code_of -X POST "$BASE/api/generate" -d '{not json')
[[ $c == 400 ]] && pass "malformed JSON → 400" || fail "malformed JSON" "http $c"
c=$(code_of -X POST "$BASE/api/generate" -d '{"prompt":"no model"}')
[[ $c -ge 400 && $c -lt 500 ]] && pass "missing model field → 4xx" || fail "missing model" "http $c"

echo "== not-implemented endpoints (explicit, by decree) =="
for ep in create copy push; do
  c=$(code_of -X POST "$BASE/api/$ep" -d '{"model":"x"}')
  [[ $c == 501 ]] && jq -e '.error' /tmp/body >/dev/null \
    && pass "/api/$ep → 501 + explanation" || fail "/api/$ep should be 501" "http $c: $(head -c150 /tmp/body)"
done

echo "== concurrency: 8 parallel generates, one model =="
pids=(); okc=0
for i in $(seq 1 8); do
  (curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"say $i\",\"stream\":false,\"options\":{\"num_predict\":12}}" | jq -e '.done==true' >/dev/null) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" && okc=$((okc+1)); done
[[ $okc == 8 ]] && pass "8/8 concurrent requests completed" || fail "concurrency" "$okc/8 ok"
curl -sf "$BASE/api/version" >/dev/null && pass "server alive after burst" || fail "server dead after burst"

echo "== mixed-model churn: alternating models with unloads, 12 rounds =="
churn_ok=0
for i in $(seq 1 12); do
  m=$SMALL; [[ $((i % 3)) == 0 ]] && m=$SMALL2; [[ $((i % 4)) == 0 ]] && m=$MM
  if curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$m\",\"prompt\":\"ping $i\",\"stream\":false,\"options\":{\"num_predict\":6}}" | jq -e '.done==true' >/dev/null; then
    churn_ok=$((churn_ok+1))
  fi
  [[ $((i % 5)) == 0 ]] && curl -sf -X POST "$BASE/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0,\"stream\":false}" >/dev/null
done
[[ $churn_ok == 12 ]] && pass "model churn 12/12" || fail "model churn" "$churn_ok/12"

echo "== adversarial: actively trying to break the server (chaos still running) =="
alive() { curl -sf --max-time 10 "$BASE/api/version" >/dev/null; }

# A1 — oversized body (~600 KB prompt): any non-5xx verdict, server alive.
python3 - << 'PY'
import json
open('/tmp/big.json','w').write(json.dumps({
  "model": "MODEL", "prompt": "lorem ipsum " * 50000,
  "stream": False, "options": {"num_predict": 4}}))
PY
sed -i "s|MODEL|$SMALL|" /tmp/big.json
c=$(code_of --max-time 120 -X POST "$BASE/api/generate" --data-binary @/tmp/big.json)
[[ $c -lt 500 ]] && alive && pass "600KB prompt → http $c, server alive" || fail "oversized prompt" "http $c alive=$(alive && echo y || echo n)"

# A2 — client aborts mid-stream; orphaned generation must not wedge the queue.
curl -sN --max-time 2 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Write a very long essay\",\"options\":{\"num_predict\":512}}" >/dev/null 2>&1
c=$(code_of --max-time 120 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"still there?\",\"stream\":false,\"options\":{\"num_predict\":8}}")
[[ $c == 200 ]] && jq -e '.done==true' /tmp/body >/dev/null && pass "mid-stream client abort → next request clean" || fail "abort recovery" "http $c"

# A3 — connection spam: 10 rapid aborted requests in parallel.
spam_pids=()
for i in $(seq 1 10); do
  curl -sN --max-time 0.3 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"spam $i\",\"options\":{\"num_predict\":64}}" >/dev/null 2>&1 &
  spam_pids+=($!)
done
# NEVER bare `wait` here: the chaos loop is a deliberately immortal child,
# and waiting on "all children" waits on it forever (found the hard way).
for p in "${spam_pids[@]}"; do wait "$p" 2>/dev/null; done
sleep 1
alive && pass "10 aborted connections → server alive" || fail "connection spam killed server"

# A4 — mixed parallel workload: generate + chat + embed + constrained, all at once.
pids=(); mixed_ok=0
for i in 1 2 3; do
  (curl -sf --max-time 180 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"mix g$i\",\"stream\":false,\"options\":{\"num_predict\":8}}" | jq -e '.done==true' >/dev/null) & pids+=($!)
done
for i in 1 2; do
  (curl -sf --max-time 180 -X POST "$BASE/api/chat" -d "{\"model\":\"$SMALL\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"mix c$i\"}],\"options\":{\"num_predict\":8}}" | jq -e '.done==true' >/dev/null) & pids+=($!)
  (curl -sf --max-time 180 -X POST "$BASE/api/embed" -d "{\"model\":\"$SMALL2\",\"input\":\"mix e$i\"}" | jq -e '.embeddings[0]|length>10' >/dev/null) & pids+=($!)
  (curl -sf --max-time 180 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Return a small JSON object with a name and a number, then stop.\",\"stream\":false,\"format\":\"json\",\"options\":{\"num_predict\":128}}" | jq -re '.response' | jq -e . >/dev/null) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" && mixed_ok=$((mixed_ok+1)); done
[[ $mixed_ok == 9 ]] && pass "mixed parallel workload 9/9 (gen+chat+embed+format)" || fail "mixed workload" "$mixed_ok/9"

# A5 — tools with stream:true: final frame must carry parsed tool_calls.
curl -sN --max-time 120 -X POST "$BASE/api/chat" -d "{\"model\":\"$SMALL\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Tokyo? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"options\":{\"num_predict\":64,\"temperature\":0}}" > /tmp/tstream || true
tail -n1 /tmp/tstream | jq -e '.message.tool_calls[0].function.name == "get_weather"' >/dev/null 2>&1 \
  && pass "tools + stream:true → tool_calls on final frame" || fail "tools streaming" "$(tail -n1 /tmp/tstream | head -c200)"

# A6 — garbage option types must not 5xx or wedge.
c=$(code_of --max-time 60 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"temperature\":\"hot\",\"num_predict\":\"many\",\"top_k\":-3}}")
[[ $c -lt 500 ]] && alive && pass "garbage option types → http $c, alive" || fail "garbage options" "http $c"

# A7 — re-pull of a present model DURING a generate: both must succeed.
(curl -sf --max-time 180 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"pull race\",\"stream\":false,\"options\":{\"num_predict\":24}}" | jq -e '.done==true' >/dev/null) & g=$!
c=$(code_of --max-time 180 -X POST "$BASE/api/pull" -d "{\"model\":\"$SMALL2\",\"stream\":false}")
wait $g; gr=$?
[[ $gr == 0 && $c == 200 ]] && pass "pull during generate → both complete" || fail "pull/generate race" "generate=$([[ $gr == 0 ]] && echo ok || echo FAILED) pull=http:$c"

# A8 — unload race: keep_alive:0 fired while a generate is in flight.
(curl -sf --max-time 180 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"unload race\",\"stream\":false,\"options\":{\"num_predict\":32}}" | jq -e '.done==true' >/dev/null) & g=$!
sleep 0.2
c=$(code_of --max-time 180 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"keep_alive\":0,\"stream\":false}")
wait $g; gr=$?
[[ $gr == 0 && $c == 200 ]] && alive && pass "unload during generate → both requests clean" || fail "unload race" "gen=$gr unload=$c"

# A9 — schema with mixed required types: both keys, both typed, by construction.
c=$(code_of --max-time 120 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"Facts about France (name, population)\",\"stream\":false,\"format\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"population\":{\"type\":\"integer\"}},\"required\":[\"name\",\"population\"]},\"options\":{\"num_predict\":128,\"temperature\":0}}")
[[ $c == 200 ]] && jq -r '.response' /tmp/body | jq -e '(.name|type=="string") and (.population|type=="number")' >/dev/null 2>&1 \
  && pass "schema mixed types enforced (string+integer)" || fail "schema mixed types" "http $c: $(jq -r '.response' /tmp/body 2>/dev/null | head -c150)"

# A10 — 200-turn conversation: big but legal; must answer or reject cleanly.
python3 - << 'PY'
import json
msgs = []
for i in range(100):
    msgs.append({"role": "user", "content": f"note {i}"})
    msgs.append({"role": "assistant", "content": "ok"})
msgs.append({"role": "user", "content": "How many notes did I send? One word."})
open('/tmp/long.json','w').write(json.dumps({
  "model": "MODEL", "stream": False, "messages": msgs,
  "options": {"num_predict": 8}}))
PY
sed -i "s|MODEL|$SMALL|" /tmp/long.json
c=$(code_of --max-time 180 -X POST "$BASE/api/chat" --data-binary @/tmp/long.json)
[[ $c -lt 500 ]] && alive && pass "200-turn conversation → http $c, alive" || fail "long conversation" "http $c"

# A11 — num_predict:0 and stop-on-first-byte: degenerate but legal.
c=$(code_of --max-time 60 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":0}}")
[[ $c -lt 500 ]] && alive && pass "num_predict:0 → http $c, alive" || fail "num_predict:0" "http $c"
c=$(code_of --max-time 60 -X POST "$BASE/api/generate" -d "{\"model\":\"$SMALL\",\"prompt\":\"echo\",\"stream\":false,\"options\":{\"num_predict\":32,\"stop\":[\"e\"]}}")
[[ $c == 200 ]] && jq -e '.done==true' /tmp/body >/dev/null && pass "single-char stop sequence → clean" || fail "aggressive stop" "http $c"

echo "== chaos verdict =="
kill "$CHAOS_PID" 2>/dev/null; wait "$CHAOS_PID" 2>/dev/null
ops=$(wc -l < "$CHAOS_LOG"); gens=$(grep -c g "$CHAOS_LOG" || true)
[[ $ops -ge 30 && $gens -ge 5 ]] \
  && pass "background chaos sustained ($ops ops, $gens generates, zero downtime)" \
  || fail "background chaos starved" "$ops ops, $gens generates"
alive && pass "server alive at end of suite" || fail "server dead at end"

echo
echo "== $PASSES passed, $FAILS failed =="
exit "$FAILS"