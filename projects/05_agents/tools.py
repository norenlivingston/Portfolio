"""
Shared tool definitions for the AI agent layer.

Both agent.py (manual tool-calling loop, Ollama or Claude) and mcp_server.py
(MCP server) wrap the same underlying implementations from model_registry.py,
so the pipeline's serving logic lives in exactly one place.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_mlops"))

from model_registry import (  # noqa: E402
    explain_one,
    get_features,
    get_latest_metrics,
    get_run_history,
    predict_one,
)

# ── Tool implementations ──────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "list_features":      lambda args: get_features(),
    "get_latest_metrics": lambda args: get_latest_metrics(),
    "get_run_history":    lambda args: get_run_history(),
    "predict":             lambda args: predict_one(args["features"]),
    "explain_prediction":  lambda args: explain_one(args["features"]),
}

# ── Tool schemas (Anthropic `input_schema` format — the canonical version) ────

ANTHROPIC_TOOLS = [
    {
        "name": "list_features",
        "description": "Return the feature names required to make a prediction.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_latest_metrics",
        "description": (
            "Return performance metrics from the most recent pipeline run: "
            "best model name, CV R², test R², RMSE, MAE, and timestamp."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_run_history",
        "description": "Return metrics from all previous pipeline runs in chronological order.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "predict",
        "description": (
            "Make a regression prediction given feature values. "
            "Call list_features first if you don't know the required feature names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Dict of {feature_name: float_value} for all required features.",
                }
            },
            "required": ["features"],
        },
    },
    {
        "name": "explain_prediction",
        "description": (
            "Explain a prediction via per-feature SHAP contributions, showing which "
            "features pushed the prediction up or down and by how much. "
            "Only supported for tree-based models (e.g. RandomForest)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Dict of {feature_name: float_value} for all required features.",
                }
            },
            "required": ["features"],
        },
    },
]


def to_ollama_tools(tools: list = ANTHROPIC_TOOLS) -> list:
    """Adapt the canonical Anthropic tool schemas to Ollama's OpenAI-style format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]
