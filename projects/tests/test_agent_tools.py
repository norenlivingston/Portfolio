from tools import ANTHROPIC_TOOLS, TOOL_REGISTRY, to_ollama_tools


def test_tool_registry_matches_schemas():
    schema_names = {t["name"] for t in ANTHROPIC_TOOLS}
    assert schema_names == set(TOOL_REGISTRY)


def test_list_features_tool(trained_env):
    import model_registry
    result = TOOL_REGISTRY["list_features"]({})
    assert result == model_registry.get_features()


def test_predict_tool(trained_env):
    import model_registry
    features = {f: 1.0 for f in model_registry.get_features()}
    result = TOOL_REGISTRY["predict"]({"features": features})
    assert "prediction" in result


def test_to_ollama_tools_preserves_names_and_schema():
    ollama_tools = to_ollama_tools()

    assert len(ollama_tools) == len(ANTHROPIC_TOOLS)
    for anthropic_tool, ollama_tool in zip(ANTHROPIC_TOOLS, ollama_tools):
        assert ollama_tool["type"] == "function"
        assert ollama_tool["function"]["name"] == anthropic_tool["name"]
        assert ollama_tool["function"]["parameters"] == anthropic_tool["input_schema"]
