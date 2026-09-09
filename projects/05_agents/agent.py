"""
AI Agent — ML Pipeline Assistant
A tool-calling agent that answers natural language questions about the
trained model by invoking the same tools exposed by the MCP server.
Supports two backends:

    --provider anthropic   Claude API (needs ANTHROPIC_API_KEY) — default
    --provider ollama      Local, free, no API key (needs Ollama running)

Requirements:
    export ANTHROPIC_API_KEY=sk-...          # for --provider anthropic
    ollama pull qwen2.5                      # for --provider ollama

Usage (from projects/ directory):
    python 05_agents/agent.py
    python 05_agents/agent.py --question "Which model was selected?"
    python 05_agents/agent.py --provider ollama --question "Predict with all features set to 1.0"
"""
import argparse
import json

from tools import ANTHROPIC_TOOLS, TOOL_REGISTRY, to_ollama_tools  # noqa: E402 (adds model_registry to sys.path)
from model_registry import get_config

SYSTEM_PROMPT = (
    "You are a helpful assistant for an ML pipeline project. "
    "You have tools to inspect a trained regression model, run predictions, "
    "and explain individual predictions via SHAP feature contributions. "
    "Always use tools to answer accurately rather than guessing."
)

DEMO_QUESTIONS = [
    "How did the model perform on the test set?",
    "Which model was selected and what does the CV score tell us?",
    "What features does the model need? Make a prediction with all of them set to 1.0, "
    "then explain which features drove that prediction.",
]


def _log_tool_call(name: str, args: dict, result) -> None:
    print(f"  [tool]   {name}({json.dumps(args) if args else ''})")
    print(f"  [result] {json.dumps(result)}\n")


# ── Anthropic (Claude API) backend ────────────────────────────────────────────

def _run_agent_anthropic(question: str, model: str, verbose: bool) -> str:
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    try:
        while True:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=ANTHROPIC_TOOLS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                return next((b.text for b in response.content if b.type == "text"), "")

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = TOOL_REGISTRY[block.name](block.input)
                if verbose:
                    _log_tool_call(block.name, block.input, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_results})

    except (anthropic.AuthenticationError, TypeError):
        return (
            "Error: no valid Anthropic API key found. "
            "Set ANTHROPIC_API_KEY, or run with --provider ollama for a free local alternative."
        )
    except anthropic.APIConnectionError:
        return "Error: could not reach the Anthropic API. Check your network connection."


# ── Ollama (local, free) backend ──────────────────────────────────────────────

def _run_agent_ollama(question: str, model: str, verbose: bool) -> str:
    import ollama

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = to_ollama_tools()

    while True:
        response = ollama.chat(model=model, messages=messages, tools=tools)
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            args = call.function.arguments or {}
            result = TOOL_REGISTRY[call.function.name](args)
            if verbose:
                _log_tool_call(call.function.name, args, result)
            messages.append({"role": "tool", "content": json.dumps(result)})


# ── Entry point ───────────────────────────────────────────────────────────────

def _default_model(provider: str) -> str:
    agents_cfg = get_config().get("agents", {})
    return agents_cfg.get(f"{provider}_model") or {"anthropic": "claude-opus-5", "ollama": "qwen2.5"}[provider]


def run_agent(question: str, provider: str = "anthropic", model: str = None, verbose: bool = True) -> str:
    model = model or _default_model(provider)
    if provider == "anthropic":
        return _run_agent_anthropic(question, model, verbose)
    if provider == "ollama":
        return _run_agent_ollama(question, model, verbose)
    raise ValueError(f"Unknown provider: {provider!r}")


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline AI Agent")
    parser.add_argument("--question", "-q", default=None, help="Ask the agent a question")
    parser.add_argument("--provider", "-p", default="anthropic", choices=["anthropic", "ollama"])
    parser.add_argument("--model", "-m", default=None, help="Override the default model for the chosen provider")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEMO_QUESTIONS

    for question in questions:
        print(f"\n{'-' * 55}")
        print(f"Q: {question}")
        print(f"{'-' * 55}")
        answer = run_agent(question, provider=args.provider, model=args.model)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
