# cima

A minimalist, zero-dependency CUDA inference engine for NVIDIA GPUs on
Linux, with an Ollama-compatible HTTP API. It pulls models directly from
the Hugging Face Hub and serves GGUF and safetensors checkpoints — but only
the ones it has been certified against. The goal is not "runs anything";
it is a controlled, tested set with no surprises in production.

## Quickstart

    docker run --gpus all -p 11435:11435 \
      -v cima-models:/data/models \
      ghcr.io/OWNER/cima:latest

Then, against the running server:

    curl http://localhost:11435/api/pull -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0"}'
    curl http://localhost:11435/api/generate -d '{
      "model":"Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0",
      "prompt":"Why is the sky blue?","stream":false
    }'

The API is the Ollama surface, so existing Ollama clients work unchanged.

## Why cima

- **Zero runtime dependencies.** Links libc, libcurl, and the CUDA driver
  stack via FFI; nothing else. The whole engine is a handful of Rust files.
- **Controlled model support.** Every served model passes `cima vet` on
  real hardware or a metadata preflight that proves architectural identity
  to a certified family member. Unsupported checkpoints fail in seconds
  with a precise reason, never garbage output. See [docs/models.md](docs/models.md).
- **Fast where it counts.** Fused dequant-GEMV kernels for GGUF quants,
  CUDA-graph decode, device-resident token pipeline, incremental prefill.
  See [docs/benchmarks.md](docs/benchmarks.md).
- **Operable.** Crash-surviving log ring, JSON log mode, request telemetry,
  bounded admission, VRAM forecasting, health checks.

## Documentation

- [Configuration](docs/configuration.md) — every environment variable
- [Operations](docs/operations.md) — running, logging, VRAM, networking
- [API](docs/api.md) — endpoint matrix and the `audio` extension
- [Supported models](docs/models.md) — the certified registry
- [Benchmarks](docs/benchmarks.md) — methodology and results

## Building from source

    cargo build --release        # target/release/cima
    cargo test                   # CPU unit suites (no GPU needed to build)

CUDA kernels compile at runtime via NVRTC. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the test tiers and the process for
proposing a new model.

## License

Apache-2.0. See [LICENSE](LICENSE).
