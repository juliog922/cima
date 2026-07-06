# syntax=docker/dockerfile:1
# =============================================================================
# cima — production image
#
# Build stage runs on the CUDA *devel* image so the driver stub
# (libcuda.so in stubs/), libnvrtc, and libnvidia-ml are present for the
# link step; Rust is installed on top. cuBLAS is NOT linked — it is
# dlopened at runtime, so the engine runs with or without it.
#
# Runtime stage is the CUDA *runtime* image (carries libnvrtc, libnvidia-ml,
# and optionally libcublas). libcuda.so itself is injected by the NVIDIA
# container runtime at `docker run --gpus all`.
#
#   docker build -t cima .
#   docker run --gpus all -p 11435:11435 -v cima-models:/data/models cima
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG RUST_VERSION=1.83.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*
# Rust toolchain (pinned) via rustup.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
# The driver stub lives under stubs/; build.rs already adds that search
# path. Make it discoverable to the linker invocation as well.
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
RUN cargo build --release --locked

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
LABEL org.opencontainers.image.title="cima" \
      org.opencontainers.image.description="Minimalist CUDA inference engine with an Ollama-compatible API" \
      org.opencontainers.image.source="https://github.com/OWNER/cima" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libcurl4 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 cima \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /app cima \
 && mkdir -p /data/models && chown -R cima:cima /data/models

COPY --from=build /build/target/release/cima /usr/local/bin/cima
# NVRTC compiles the kernels from source at startup and needs the toolkit
# headers (cuda_fp16.h and friends). The runtime base ships the libraries
# but not the headers, so copy just the include tree from the devel stage
# (~tens of MB) rather than pulling the full devel image (~GBs). Copy into
# the versioned prefix that the /usr/local/cuda symlink resolves to.
COPY --from=build /usr/local/cuda/include /usr/local/cuda-12.4/include
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# NOTE: the image intentionally runs as root so the entrypoint can fix
# ownership of a mounted models volume (which may arrive root-owned), then
# it drops to the unprivileged cima user (uid 10001) via setpriv before
# exec'ing the server. Run with `--user` to skip that and stay non-root.
WORKDIR /app
ENV CIMA_HOST=0.0.0.0 \
    CIMA_PORT=11435 \
    CIMA_MODELS_DIR=/data/models \
    CUDA_HOME=/usr/local/cuda-12.4
VOLUME /data/models
EXPOSE 11435
# /api/ready returns 503 until the server is live AND every model listed in
# CIMA_PULL_AT_STARTUP is on disk, so `curl -sf` (which fails on non-2xx)
# keeps the container `unhealthy` until pulls finish. A generous
# start-period covers multi-GB first pulls without flapping; tune it (or
# CIMA_PULL_AT_STARTUP) for your largest model. Dependents gate on this via
# `depends_on: { cima: { condition: service_healthy } }`.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30m --retries=3 \
  CMD curl -sf http://127.0.0.1:11435/api/ready || exit 1
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh", "cima"]
CMD ["serve"]