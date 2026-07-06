# Security Policy

## Supported versions

Only the most recent minor release receives security fixes.

## Reporting a vulnerability

Report privately via GitHub Security Advisories ("Report a vulnerability"
on the repository's Security tab). Do not open public issues for
exploitable problems. Reports are acknowledged within 72 hours.

## Deployment guidance

cima binds to `127.0.0.1` by default and ships no authentication or TLS —
the API trusts its network. Expose it only behind a reverse proxy that
terminates TLS and enforces access control, and treat the models directory
as trusted input: checkpoints are parsed with bounds-checked readers, but
serving arbitrary third-party files is not a supported configuration.
`cima vet` and the curated registry exist precisely to keep unvetted
checkpoints out of production.
