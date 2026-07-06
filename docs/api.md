# API

cima implements the Ollama HTTP surface 1:1 so existing Ollama clients work
unchanged. `scripts/test.sh` is the executable form of this contract; a
mismatch is a bug in one of the two.

## Implemented

| Endpoint | Notes |
|---|---|
| `POST /api/generate` | Streaming and non-streaming; `options`, `stop`, `seed`, `format` (JSON mode and JSON-schema constrained decoding), `keep_alive`, `images`, **`audio`** |
| `POST /api/chat` | Multi-turn; per-message `images` and **`audio`**; `tools` parameter accepted |
| `POST /api/embed`, `POST /api/embeddings` | Single and batch |
| `POST /api/pull` | Streaming progress; `ORG/REPO:TAG` selects one GGUF quantization; preflight gate before download (`"force": true` overrides) |
| `POST /api/show`, `POST /api/delete` | |
| `GET /api/tags`, `GET /api/ps`, `GET /api/version` | |

## Extension: `audio`

`audio` is an additive parameter shaped exactly like `images` — an array of
base64 payloads — on `/api/generate` (top level) and `/api/chat` (per
message). WAV is decoded natively (other containers are rejected with the
ffmpeg one-liner to convert). Clients that never send the key are
unaffected, which is what keeps the protocol Ollama-compatible.

## Not implemented

`/api/create`, `/api/copy`, `/api/push`, `/api/blobs` return an explicit
501 with a reason — cima's model store is the Hugging Face Hub plus the
certified registry, not a Modelfile build system.

## Backpressure

With `CIMA_MAX_QUEUE` set, requests beyond the cap receive HTTP 429 with a
JSON error body. Clients should retry with backoff.
