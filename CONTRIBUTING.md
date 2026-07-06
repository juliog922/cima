# Contributing to cima

## Ground rules

**Zero runtime dependencies is a feature.** The engine links libc, libcurl,
and the CUDA driver stack via FFI and nothing else. A pull request adding a
crate dependency needs a written justification of why the functionality
cannot live in-tree; "convenience" is not one.

**Every supported model is a tested model.** cima deliberately refuses
checkpoints outside its certified set. Do not weaken a loader gate to make
a model "work" — register the capability properly (see "Proposing a new
model" below) or leave the precise rejection in place.

## Building

    cargo build --release        # binary at target/release/cima
    cargo test                   # CPU-only unit suites, no GPU required

CUDA kernels compile at runtime through NVRTC; a GPU is needed to serve,
not to build.

## Test tiers

1. `cargo test` — unit suites (quantization codecs, JSON, tokenizer,
   container formats). Runs everywhere; required for every PR.
2. `cima selftest` — GPU-vs-CPU numerical battery over every kernel.
   Required when touching `kernels.cu`, `cuda.rs`, or `quant.rs`.
3. `scripts/test.sh` — API conformance and adversarial stress against a
   compose deployment. Required when touching `api.rs`.
4. `scripts/benchs.sh` — performance percentiles. Attach before/after
   numbers to any PR claiming a performance change.

## Proposing a new model

New models enter through the registry, never ad hoc:

1. Run `cima vet ORG/REPO` on real hardware; every announced capability is
   exercised. Attach the full output to the PR.
2. For large variants of an already-certified family, run
   `cima vet ORG/REPO:TAG --preflight` and attach the report — it proves
   the architecture, tokenizer, and complete tensor table match what the
   engine executes, without downloading weights.
3. Add the row to `registry.toml` with capabilities, minimum VRAM, and the
   vet date. `docs/models.md` is generated from that file.

## Code standards

`cargo fmt` and `cargo clippy --all-targets -- -D warnings` are enforced
in CI. Comments state contracts and invariants, not development history;
incident details belong in commit messages. Public items carry doc
comments (`missing_docs` is denied on the library crate).
