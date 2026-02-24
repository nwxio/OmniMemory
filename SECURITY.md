# Security policy

## Supported versions

Security updates are provided for the latest active branch.

| Version | Supported |
|---------|-----------|
| latest `main` | yes |
| older tags/branches | best effort |

## Reporting a vulnerability

If you discover a security issue, do not open a public issue with exploit details.

Preferred disclosure channels:

1. GitHub Security Advisory (private report) for this repository.
2. If private advisory is unavailable, open a minimal issue asking maintainers for a private channel.

Please include:

- affected component/file
- impact and attack scenario
- reproduction steps or proof-of-concept
- suggested mitigation (if available)

## Response expectations

- Initial triage target: within 5 business days.
- Valid reports receive status updates during investigation.
- Fixes are released based on severity and operational risk.

## Security hardening guidance

- Never commit secrets (`.env`, API keys, database passwords).
- Rotate credentials used in examples before production use.
- Restrict database/cache network exposure.
- Use TLS and secret management in production environments.
- Keep dependencies and runtime patched.
