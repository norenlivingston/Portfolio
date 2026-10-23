# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Input validation on `/predict` and `/explain`: missing, unknown, non-numeric, NaN, and infinite feature values return `422` with a list of problems.
- Categorical-feature support in explanations: SHAP contributions and the feature-importance plot report raw feature names, with one-hot columns summed back to their source.
- Pinned, cross-platform lockfiles (`requirements*.txt`, generated from `requirements*.in`) and a declared Python range (3.11–3.13) in `pyproject.toml`.
- CI: ruff lint, a 3.11–3.13 test matrix with an 85% coverage floor, a README quick-start replay, an agent dependency check, a Docker Compose smoke test, dependency review, CodeQL, and Dependabot.
- Governance files: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, and PR/issue templates.

### Security
- Upgraded `fastapi` 0.115 → 0.162 and `starlette` 0.46.2 → 1.9.0. `starlette` 0.46.x has three high-severity advisories (GHSA-7f5h-v6xp-fcq8, GHSA-wqp7-x3pw-xc5r, GHSA-82w8-qh3p-5jfq), and the old `fastapi` cap kept it below 0.47. `requirements.in` now sets an explicit `starlette` floor.

### Changed
- **Breaking:** invalid inference requests now return `422` instead of `400`. `400` is now only used for valid requests the model can't serve (e.g. `/explain` on a linear model).
- Agent/MCP packages moved to the optional `requirements-agents.txt`; `shap` moved to the core requirements, since `/explain` needs it.
- Docker Compose now runs a one-shot `train` service; `api` and `mlflow` start only after it succeeds, with health checks on both. The image itself only serves.
- With `outlier_strategy: remove`, rows are dropped from the training split only. The test split keeps every row (clipped to train-fitted bounds), so the hold-out evaluation population never changes.

## [0.0.0] – 2026-10-23

Baseline before the maturity cleanup: config-driven synthetic regression pipeline, leakage-safe EDA, MLflow tracking, FastAPI serving with SHAP, and the agent/MCP layer.
