# Security Policy

## Supported versions

Only the latest commit on `main` is supported. Fixes are not backported.

## Reporting a vulnerability

Please report vulnerabilities **privately**. Do not open a public issue.

1. Preferred: use GitHub's private reporting ([**Report a vulnerability**](https://github.com/norenlivingston/Portfolio/security/advisories/new) on the Security tab).
2. Alternatively, email **norenlivingston@gmail.com** with the subject line `SECURITY: Portfolio`.

Please include the affected file or endpoint, steps to reproduce, and the impact you expect. You should get an acknowledgement within 7 days. Confirmed issues are fixed on `main` and disclosed in a GitHub Security Advisory once a fix is available.

## Scope notes

- The inference API (`projects/03_mlops/serve.py`) is a local demo. It has **no authentication, rate limiting, or TLS** and should not be exposed to the public internet as-is. Reports that only restate this are out of scope. Bypasses of its input validation are in scope.
- `model_registry.py` loads the model with `joblib`, which uses pickle. Only load model files the pipeline produced itself; never load untrusted `.pkl` files.
- Never commit API keys. `ANTHROPIC_API_KEY` is read from the environment only.
