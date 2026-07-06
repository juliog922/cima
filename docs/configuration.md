# Configuration

cima is configured entirely through environment variables. On startup the
server logs the resolved value of every public variable and warns about any
`CIMA_*` variable it does not recognize.

## Public configuration

These form a stable contract across releases.

| Variable | Default | Meaning |
|---|---|---|
| `CIMA_HOST` | `127.0.0.1` | Bind address. The container image sets `0.0.0.0`. |
| `CIMA_PORT` | `11435` | Bind port. |
| `CIMA_MODELS_DIR` | `./models` | Weight store root. Set to a mounted volume in containers (`/data/models` in the official image). |
| `CIMA_KEEP_ALIVE` | `5m` | Model residency after a request; every served request resets the clock. Accepts seconds (`600`), suffixed durations (`90s`, `10m`, `2h`), `0` (release immediately after each request), or `-1`/`forever`. Per-request `keep_alive` overrides per call. |
| `CIMA_MAX_SEQ` | model default | Cap the context window; the KV cache is sized from it. The primary VRAM lever on small GPUs. |
| `CIMA_MAX_QUEUE` | unbounded | Maximum queued + running requests. At the cap, new requests receive HTTP 429 instead of waiting — fast failure over invisible latency. |
| `CIMA_GPU` | `0` | CUDA device ordinal. |
| `CIMA_LOG` | warnings+errors | `info` or `debug` add detail on stderr; the full stream always lands in the crash-surviving ring (`cima logs`). |
| `CIMA_LOG_FORMAT` | `text` | `json` emits one JSON object per line (`ts`, `level`, `file`, `line`, `msg`) for log collectors. |
| `HF_TOKEN` | none | Hugging Face token for gated or private repositories. |

## Diagnostics (unstable)

Escape hatches and probes for debugging; semantics may change between
versions. Each disables one performance path, so a misbehaving request can
be bisected without a rebuild: `CIMA_NO_GRAPH` (CUDA-graph decode),
`CIMA_NO_PIPELINE` (device-resident token pipeline), `CIMA_NO_INCR`
(incremental prefill), `CIMA_CUBLAS` (opt into cuBLAS prefill; default is the
deterministic native kernel) / `CIMA_NO_CUBLAS` (force native), `CIMA_PIN`
(pinned-memory staging), `CIMA_SHM` (log ring location/size),
`CIMA_INCR_CHECK` (audit the incremental-prefill token chain), plus the
`CIMA_G4_*`, `CIMA_DUMP_*`, and `CIMA_TRACE_*` families of gemma-4 and
tensor-level probes.