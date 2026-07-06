#!/bin/sh
# cima container entrypoint.
#
# A mounted models volume (named volume or host bind) can arrive owned by
# root — Docker only seeds ownership when it first creates an *empty* named
# volume, and never for bind mounts or pre-existing volumes. cima runs
# unprivileged and would then fail to write. So: if we are root, make the
# models dir writable by the cima user and drop to it; if we are already
# non-root (someone ran with --user), just exec.
set -e

MODELS_DIR="${CIMA_MODELS_DIR:-/data/models}"
CIMA_UID=10001
CIMA_GID=10001

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$MODELS_DIR"
    # Only chown when needed — a large warm volume shouldn't pay a recursive
    # chown on every boot.
    if [ "$(stat -c '%u' "$MODELS_DIR")" != "$CIMA_UID" ]; then
        chown -R "$CIMA_UID:$CIMA_GID" "$MODELS_DIR"
    fi
    # Drop privileges to the cima user for the actual server (setpriv ships
    # with util-linux in the base image — no extra dependency).
    exec setpriv --reuid "$CIMA_UID" --regid "$CIMA_GID" --init-groups "$@"
fi

# Already non-root (explicit --user): run as-is.
exec "$@"