# Operations

## Running

    docker run --gpus all -p 11435:11435 \
      -v cima-models:/data/models \
      ghcr.io/OWNER/cima:latest

The image runs as a non-root user, stores weights in the `/data/models`
volume, and answers `GET /api/version` for liveness (wired as the image
HEALTHCHECK).

## Network posture

The API ships no authentication or TLS by design: it trusts its network.
Bind to loopback or a private network, and put a reverse proxy
(nginx/caddy/traefik) in front for TLS, access control, and request size
limits when exposing it further. See SECURITY.md.

## Logs

Runtime telemetry goes to a crash-surviving shared-memory ring readable
with `cima logs` (follow with `-f`) — it outlives engine crashes, which is
where it earns its keep. `CIMA_LOG=info` mirrors the stream to stderr for
`docker logs`; `CIMA_LOG_FORMAT=json` switches stderr to JSON lines for
collectors.

Every request logs an admission line (model, queue wait, queue depth) and
a completion `METRIC` line (prompt/generated tokens, ttft, tok/s, VRAM
delta). Load failures name the exact tensor and check that rejected the
checkpoint.

## Capacity and VRAM

One model is resident at a time; loading a different model evicts the
current one deterministically. Before building a model the engine forecasts
weights + KV cache + workspace against free VRAM and refuses with the
numbers when it cannot fit — `CIMA_MAX_SEQ` shrinks the KV term. Registry
rows in `docs/models.md` carry a minimum-VRAM column measured on real
hardware.

Under concurrency the GPU queue is strict FIFO; set `CIMA_MAX_QUEUE` to
convert overload into fast 429s your client can retry.

## Model lifecycle

    cima pull ORG/REPO:TAG        # download one quantization
    cima vet ORG/REPO --preflight # certify metadata without downloading
    cima run MODEL "prompt"       # CLI inference
    cima stop MODEL               # release GPU residency now
    cima rm MODEL                 # delete local weights

Idle models unload after `CIMA_KEEP_ALIVE`; a request's `keep_alive` field
overrides per call, exactly as in Ollama.
