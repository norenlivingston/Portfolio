"""
MCP Server — ML Pipeline Tools
Exposes the trained regression pipeline as tools any MCP-compatible
client can call (Claude Desktop, MCP Inspector, custom agents).

Usage (from projects/ directory):
    python 05_agents/mcp_server.py

Test without any API key using MCP Inspector:
    npx @modelcontextprotocol/inspector python 05_agents/mcp_server.py

Connect to Claude Desktop — add this to your claude_desktop_config.json:
    {
      "mcpServers": {
        "ml-pipeline": {
          "command": "python",
          "args": ["/absolute/path/to/projects/05_agents/mcp_server.py"]
        }
      }
    }
"""
from mcp.server.fastmcp import FastMCP

from tools import explain_one, get_features
from tools import get_latest_metrics as _get_latest_metrics
from tools import get_run_history as _get_run_history
from tools import predict_one

mcp = FastMCP("ml-pipeline")


@mcp.tool()
def list_features() -> list:
    """Return the feature names required to make a prediction."""
    return get_features()


@mcp.tool()
def get_latest_metrics() -> dict:
    """Return performance metrics from the most recent pipeline run."""
    return _get_latest_metrics()


@mcp.tool()
def get_run_history() -> list:
    """Return metrics from all previous pipeline runs in chronological order."""
    return _get_run_history()


@mcp.tool()
def predict(features: dict) -> dict:
    """
    Make a regression prediction given feature values.
    Call list_features() first to see which features are required.
    Pass a dict of {feature_name: float_value} for every required feature.
    """
    return predict_one(features)


@mcp.tool()
def explain_prediction(features: dict) -> dict:
    """
    Explain a prediction via per-feature SHAP contributions.
    Pass a dict of {feature_name: float_value} for every required feature.
    """
    return explain_one(features)


if __name__ == "__main__":
    mcp.run()
