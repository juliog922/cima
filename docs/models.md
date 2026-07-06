# Supported models

Generated from `registry.toml` — do not edit by hand
(`scripts/gen-models-doc.sh`). Every row's metadata is re-preflighted
nightly.

**Certification:** *verified* = full `cima vet` on hardware;
*verified-by-family* = metadata preflight clean and a smaller family
member is verified; *avoid* = known-defective, listed as a warning.

| Model | Family | Format | Capabilities | Min VRAM | Status | Vetted |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0` | qwen2 | gguf | generate | 2 GiB | verified | 2026-07-04 |
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m` | qwen2 | gguf | generate | 2 GiB | verified | 2026-07-04 |
| `Qwen/Qwen2.5-0.5B-Instruct` | qwen2 | safetensors | generate | 2 GiB | verified | 2026-07-04 |
| `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` | qwen2 | safetensors+bnb | generate | 2 GiB | verified | 2026-07-04 |
| `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` | gemma4 | gguf | generate,  vision,  audio | 6 GiB | verified | 2026-07-04 |
| `bartowski/Qwen2.5-7B-Instruct-GGUF:IQ4_XS` | qwen2 | gguf | generate | 6 GiB | verified-by-family | 2026-07-04 |
