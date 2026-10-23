# Contributing

Thanks for your interest in improving this project. It's a personal portfolio
repository, but issues and pull requests are welcome.

## Setup

Supported Python versions: **3.11, 3.12, 3.13** (declared in `projects/pyproject.toml`, tested in CI).

```bash
cd projects
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt  # core + test/lint tools
pip install -r requirements-agents.txt   # optional: only for 05_agents/
```

## Before opening a pull request

Run the same checks CI runs, from `projects/`:

```bash
ruff check .                 # lint
pytest --cov                 # tests + coverage (fails under 85%)
python run_pipeline.py       # full pipeline smoke test
docker compose config --quiet   # if you touched Docker files
```

## Branches and pull requests

- Branch from `main`; keep each PR focused on one change.
- Fill in the pull request template: problem, approach, tests, results, risks.
- Add or update tests for any behaviour change. A bug fix should come with a test that fails without it.
- Update `CHANGELOG.md` under **Unreleased** for user-visible changes.
- CI must be green before merge.

## Dependencies

Dependencies are pinned in lockfiles generated with [uv](https://docs.astral.sh/uv/):

| Edit this (loose ranges) | Regenerates this (exact pins) | Contents |
|---|---|---|
| `requirements.in` | `requirements.txt` | Core pipeline + API |
| `requirements-agents.in` | `requirements-agents.txt` | Optional agent / MCP layer |
| `requirements-dev.in` | `requirements-dev.txt` | Core + test/lint tools |

After editing an `.in` file (or merging a Dependabot bump), regenerate **all three**, core first, so the other two stay constrained to the same pins:

```bash
cd projects
for f in requirements requirements-agents requirements-dev; do
  uv pip compile $f.in --universal --python-version 3.11 -o $f.txt
done
```

## Code style

- `ruff check .` must pass; configuration lives in `projects/pyproject.toml`.
- Every tunable parameter belongs in `config.yaml`, not in code. If you add one, add it to `config.docker.yaml` too (a test enforces they match).
- Fit anything learned from data (feature selection, outlier bounds, scalers) on the training split only. `tests/test_eda.py` has a leakage regression test; extend it rather than weakening it.

## Reporting bugs and security issues

- Bugs and feature ideas: open an issue using the templates.
- Security vulnerabilities: **do not open a public issue**. See [SECURITY.md](SECURITY.md).

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
