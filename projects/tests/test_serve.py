from fastapi.testclient import TestClient


def _client(trained_env):
    # Import after trained_env sets PIPELINE_CONFIG so the app's module-level
    # get_config()/get_model() calls resolve to the tiny trained pipeline.
    import serve
    return TestClient(serve.app)


def test_health(trained_env):
    client = _client(trained_env)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["expected_features"]) == set(
        __import__("model_registry").get_features()
    )


def test_predict_success(trained_env):
    import model_registry
    client = _client(trained_env)

    features = {f: 1.0 for f in model_registry.get_features()}
    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)


def test_predict_missing_feature_returns_400(trained_env):
    client = _client(trained_env)
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 400


def test_explain_endpoint(trained_env):
    import model_registry
    client = _client(trained_env)

    features = {f: 1.0 for f in model_registry.get_features()}
    response = client.post("/explain", json={"features": features})

    is_tree_model = hasattr(model_registry.get_model().named_steps["regressor"], "feature_importances_")
    if is_tree_model:
        assert response.status_code == 200
        assert "contributions" in response.json()
    else:
        assert response.status_code == 400
