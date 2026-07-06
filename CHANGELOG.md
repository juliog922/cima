# Changelog

All notable changes to cima are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- GGUF legacy 32-grain block formats Q4_0, Q4_1, Q5_0, Q5_1 (dequant and
  fused GEMV kernels). Checkpoints whose row lengths are not multiples of
  256 — which llama.cpp quantizes with these fallbacks inside K-quant
  files — now load and execute.
- bitsandbytes 4-bit execution on the standard transformer pipeline via
  host dequantization to f16 (gemma-4 keeps native packed residency).
- `audio` parameter benchmark coverage; audio was already an additive
  parameter of `/api/generate` and `/api/chat`.
- `CIMA_KEEP_ALIVE` — default model residency window.
- `CIMA_MODELS_DIR` — weight store location.
- `CIMA_LOG_FORMAT=json` — one JSON object per log line.
- `CIMA_MAX_QUEUE` — bounded admission; excess requests receive HTTP 429.
- `cima vet --preflight` — metadata-only certification of a checkpoint
  (architecture, tokenizer family, complete tensor table: dtypes and block
  grain) over HTTP Range requests, without downloading weights.
- Startup banner logging version, GPU, and resolved configuration; warning
  on unrecognized `CIMA_*` environment variables.
- Benchmark suite reports p50/p90/p99 percentiles with sample counts.

### Fixed
- gemma-4 emitted a single stop token on any request reusing a KV prefix
  (repeated prompts, multi-turn chat). Incremental prefill is restricted to
  the standard architecture until the gemma-4 offset-prefill path is
  validated.

## [0.1.0] - unreleased
Initial public release.
