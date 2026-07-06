# Benchmarks

`scripts/benchs.sh` measures cima against ollama on identical GGUF weights,
and cima-vs-itself across container formats (safetensors bf16 vs GGUF q8_0;
GGUF Q4_K_M vs bitsandbytes NF4 on the same base model).

## Methodology

Every metric is sampled N times and reported as nearest-rank p50/p90/p99
with the sample count printed beside each number (`GEN_RUNS`, `LOAD_RUNS`,
`MEDIA_RUNS` control N). Cold loads are verified cold — the unload is
confirmed via `/api/ps` before timing. One discarded warmup absorbs
first-touch costs. Each warm request carries a unique prompt prefix so
prompt caches measure nothing.

Two flavors of most numbers, and the distinction matters:

* engine-reported (`gen`, `prompt`, `load(eng)`) — the engine timing its
  own GPU loop; excludes everything around it.
* wall-clock (`load`, `gen(wall)`, `image`, `audio`, `unload`) — externally
  timed from request to full response: what a client experiences, and the
  only yardstick that is identical across engines.

`gen(wall)` is single-request throughput and is labelled so; batching
servers shine at concurrency and should not be quoted against this number.
Short generations amplify fixed per-request overhead — run with
`NUM_PREDICT=512` as well to show the asymptotic rate.

## Reading percentiles

For latencies, p99 is the slow tail. For rates (tok/s) rank order inverts:
p99 is the fastest run and p50 is the headline. With n=5, p99 is
effectively the max — hence the printed sample counts.
