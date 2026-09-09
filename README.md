# End-to-End ML Pipeline

![CI](https://github.com/norenlivingston/Portfolio/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A clean, reproducible, **tested** machine learning pipeline covering the full data science workflow — from raw data generation through EDA, model training, experiment tracking, containerized serving, and an LLM agent layer on top.

Built as a portfolio project to demonstrate practical ML engineering skills.

---

## Architecture

```
projects/
├── config.yaml              ← single source of truth for all parameters
├── run_pipeline.py          ← runs the full pipeline with one command
├── Dockerfile / docker-compose.yml
├── tests/                   ← pytest suite covering every stage
│
├── 00_dataset_build/
│   └── dataset_build.py     ← generates synthetic regression data
│
├── 01_eda/
│   ├── eda.py               ← feature selection, VIF, outlier treatment
│   └── eda_visualizations/  ← correlation heatmap, scatter plots, distributions
│
├── 02_ml_pipeline/
│   └── pipeline.py          ← cross-validation, model selection, MLflow tracking
│
├── 03_mlops/
│   ├── model_registry.py    ← shared config/model loading, predict + SHAP explain
│   └── serve.py             ← FastAPI inference endpoint
│
└── 05_agents/
    ├── tools.py              ← tool definitions shared by the agent and MCP server
    ├── mcp_server.py         ← MCP server exposing the pipeline as LLM tools
    └── agent.py               ← tool-calling agent — Claude API or local Ollama
```

`model_registry.py` is the single place that loads config/model and implements predict/explain — `serve.py`, `agent.py`, and `mcp_server.py` all call into it rather than duplicating logic.

---

## Quick Start

**macOS / Linux**
```bash
git clone https://github.com/norenlivingston/Portfolio.git
cd Portfolio/projects
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/norenlivingston/Portfolio.git
cd Portfolio\projects
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe run_pipeline.py
```
(The rest of this README uses bare `python`/`pip` for brevity — on Windows, either activate the venv first with `.venv\Scripts\Activate.ps1` typed at a PowerShell prompt — not double-clicked in Explorer, which just opens it as text — or substitute `.venv\Scripts\python.exe` directly in every command below.)

Sample output:

```
2026-04-09 12:00:00  INFO     -------------------------------------------------------
2026-04-09 12:00:00  INFO       STAGE 1 / 3 — Dataset Build
2026-04-09 12:00:01  INFO     Saved raw dataset → data/synthetic_regression_dataset.csv
2026-04-09 12:00:01  INFO       STAGE 2 / 3 — Exploratory Data Analysis
2026-04-09 12:00:03  INFO     Top 8 features selected: ['Feature_5', ...]
2026-04-09 12:00:04  INFO       STAGE 3 / 3 — Model Training & Evaluation
2026-04-09 12:00:08  INFO       RandomForest          R² = 0.9991 ± 0.0002
2026-04-09 12:00:08  INFO       LinearRegression      R² = 0.9988 ± 0.0003
2026-04-09 12:00:08  INFO     Selected: RandomForest
2026-04-09 12:00:09  INFO     Hold-out → MAE=1.8  RMSE=2.4  R²=0.9991
2026-04-09 12:00:09  INFO     MLflow run logged → a1b2c3d4... (experiment: ml-pipeline)
```

Which model wins CV can flip between `RandomForest` and `LinearRegression` depending on installed library versions — this synthetic data is close to linear, so both score within a hair of each other. Seeing `LinearRegression` selected on your machine isn't a bug.

---

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| **Dataset Build** | `00_dataset_build/dataset_build.py` | `data/synthetic_regression_dataset.csv` |
| **EDA** | `01_eda/eda.py` | `data/cleaned_synthetic_regression_dataset.csv`, plots |
| **Model Training** | `02_ml_pipeline/pipeline.py` | `best_model.pkl`, MLflow run, feature importance plot |
| **Serving** | `03_mlops/serve.py` | REST API at `localhost:8000` |

Each stage can be run independently as well as part of the full pipeline.

---

## Testing & CI

A pytest suite ([`projects/tests/`](projects/tests/)) exercises every stage end-to-end against a tiny synthetic config (200 rows, 3-fold CV) so the whole suite runs in seconds: dataset generation determinism, feature selection/VIF/outlier-treatment logic, model training + MLflow logging, the FastAPI endpoints (via `TestClient`), and the agent/MCP tool implementations.

```bash
cd projects
pip install -r requirements-dev.txt      # Windows: .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
pytest -v                                # Windows: .venv\Scripts\python.exe -m pytest -v
```

[GitHub Actions](.github/workflows/ci.yml) runs the test suite and a full pipeline smoke test on every push and pull request to `main`.

---

## Configuration

All parameters are in `projects/config.yaml`. No code changes needed to tune the pipeline:

```yaml
pipeline:
  cv_folds: 5           # k-fold cross-validation
  test_size: 0.2
  models:
    random_forest:
      n_estimators: 100
    linear_regression: {}

eda:
  top_n_features: 8
  outlier_method: "iqr"
  outlier_strategy: "iqr_bound"

mlops:
  mlflow_tracking_uri: "sqlite:///mlflow_store/mlruns.db"
  mlflow_experiment: "ml-pipeline"

agents:
  anthropic_model: "claude-opus-5"
  ollama_model: "qwen2.5"
```

Tests point every path at an isolated `tmp_path` via a `PIPELINE_CONFIG` environment variable override — see [`model_registry.py`](projects/03_mlops/model_registry.py).

---

## MLOps

### Experiment Tracking (MLflow)

Every training run logs params, metrics, the feature-importance plot, and the serialized model to MLflow — not just a flat JSON file:

```bash
cd projects
mlflow ui --backend-store-uri sqlite:///mlflow_store/mlruns.db
```

Then open `http://localhost:5000` to compare runs, inspect hyperparameters, and download logged model artifacts. (A lightweight JSON log at `03_mlops/metrics_log.json` is also kept — the agent/MCP tools read it directly without needing an MLflow client round-trip.)

### Containerized Serving

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). On Windows, Docker Desktop needs WSL2, and the underlying Windows features it depends on (`Microsoft-Windows-Subsystem-Linux`, `VirtualMachinePlatform`) are disabled by default on a lot of machines even when BIOS-level virtualization is on — if Docker Desktop reports "virtualization support not detected," check both with `dism.exe /online /get-featureinfo /featurename:VirtualMachinePlatform` (and the WSL one) before assuming it's a hardware issue.

```bash
cd projects
docker compose up --build
```

This starts two services: `api` (trains the pipeline, then serves predictions on `:8000`) and `mlflow` (tracking UI on `:5000`) sharing the same MLflow store via named volumes. Or run just the API:

```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 ml-pipeline
```

### Inference API

```bash
cd projects
python 03_mlops/serve.py     # Windows: .venv\Scripts\python.exe 03_mlops\serve.py
```

Leave that running in its own terminal — it's a server, not a one-off command. Then, in a **second** terminal:

**macOS / Linux / curl**
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Feature_3": 1.2, "Feature_7": -0.5}}'

# Explain a prediction (SHAP contributions, tree-based models only)
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"features": {"Feature_3": 1.2, "Feature_7": -0.5}}'
```

**Windows (PowerShell)** — `curl` is aliased to `Invoke-WebRequest` here with different flags, so use `Invoke-RestMethod` instead:
```powershell
Invoke-RestMethod http://localhost:8000/health

$body = @{ features = @{ Feature_3 = 1.2; Feature_7 = -0.5 } } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
Invoke-RestMethod -Uri http://localhost:8000/explain -Method Post -Body $body -ContentType "application/json"
```

Use whatever feature names your own `/health` response shows — `select_top_features` picks different columns each pipeline run.

Easiest of all, on any OS: open `http://localhost:8000/docs` in a browser for the interactive Swagger UI — click "Try it out" on any endpoint, no shell syntax required.

---

## AI Agents & MCP

### Agent — Claude API or local Ollama

The agent answers natural-language questions about the trained model by calling tools (`list_features`, `predict`, `explain_prediction`, `get_latest_metrics`, `get_run_history`) — the same tools exposed by the MCP server below. It supports two backends:

```bash
cd projects

# Claude API (default) — needs ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-...                                      # Windows: $env:ANTHROPIC_API_KEY = "sk-..."
python 05_agents/agent.py --question "Which model was selected and why?"

# Local, free, no API key — needs Ollama running
ollama pull qwen2.5
python 05_agents/agent.py --provider ollama --question "Which model was selected and why?"
```

Calls to `--provider anthropic` cost real (small) money — a few tool calls for one question is a fraction of a cent, but it's not free like everything else in this repo. `--provider ollama` has no per-use cost, just a one-time multi-GB model download.

Sample output:

```
-------------------------------------------------------
Q: What features does the model need? Make a prediction with all of them
   set to 1.0, then explain which features drove that prediction.
-------------------------------------------------------
  [tool]   list_features()
  [result] ["Feature_3", "Feature_7", ...]

  [tool]   predict({"features": {"Feature_3": 1.0, "Feature_7": 1.0, ...}})
  [result] {"prediction": 284.73}

  [tool]   explain_prediction({"features": {"Feature_3": 1.0, ...}})
  [result] {"prediction": 284.73, "base_value": 12.4, "contributions": {"Feature_7": 156.2, ...}}

A: The model requires 8 features. With all set to 1.0, the predicted
   target value is 284.73 — driven mostly by Feature_7 (+156.2) and
   Feature_3 (+89.1) relative to the model's baseline of 12.4.
```

### Explainability (SHAP)

`explain_prediction` returns per-feature SHAP contributions for a single prediction — not just global feature importance, but *why this specific prediction* came out the way it did. Supported for tree-based models (e.g. RandomForest); returns a clear error for linear models where it doesn't apply.

### MCP Server

The MCP server exposes the same tools to any MCP-compatible client.

```bash
# Test in browser — no API key needed
npx @modelcontextprotocol/inspector python 05_agents/mcp_server.py
```

**Windows (PowerShell):** plain `npx` fails under PowerShell's default script-execution policy (`npx.ps1 cannot be loaded because running scripts is disabled`). Use `npx.cmd` instead, and point at the venv's Python explicitly rather than bare `python` so the subprocess actually has `mcp`/`shap`/etc. installed:
```powershell
npx.cmd @modelcontextprotocol/inspector .venv\Scripts\python.exe 05_agents\mcp_server.py
```
This opens a local browser UI — click the "Disconnected" toggle to connect, then the **Tools** tab to call `list_features`, `predict`, etc. directly against your trained model. The first tool call after connecting can take a while (Windows Defender scanning newly-loaded files) — if it times out, just retry.

Connect to **Claude Desktop** by adding this to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ml-pipeline": {
      "command": "python",
      "args": ["/absolute/path/to/projects/05_agents/mcp_server.py"]
    }
  }
}
```

Tools exposed: `list_features` · `get_latest_metrics` · `get_run_history` · `predict` · `explain_prediction`

---

## Skills Demonstrated

| Area | Tools / Techniques |
|---|---|
| **Data Engineering** | pandas, NumPy, reproducible data generation |
| **EDA** | Correlation analysis, VIF (multicollinearity), IQR outlier treatment, Seaborn / Matplotlib |
| **Machine Learning** | scikit-learn Pipelines, k-fold cross-validation, model comparison, feature importance |
| **MLOps** | MLflow experiment tracking, model serialization (joblib), FastAPI + Uvicorn serving, Docker / Compose |
| **AI Agents & GenAI** | Claude API tool use (manual agentic loop), MCP server + client, Ollama local LLM, SHAP explainability |
| **Software Engineering** | Config-driven design, pytest suite + GitHub Actions CI, DRY shared modules, structured logging, clean repo |

---

## Stack

Python · scikit-learn · pandas · NumPy · FastAPI · Uvicorn · Pydantic · MLflow · SHAP · Matplotlib · Seaborn · statsmodels · PyYAML · MCP · Anthropic Claude API · Ollama · pytest · Docker · GitHub Actions

---

Noren Livingston, M.S.
[norenlivingston@gmail.com](mailto:norenlivingston@gmail.com)
