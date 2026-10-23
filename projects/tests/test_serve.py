import json

import pytest
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


def test_predict_using_health_reported_features(trained_env):
    """Mirrors the README's documented flow: read /health, build a full
    feature payload from its response, then POST /predict — rather than a
    hard-coded feature list, which would break whenever feature selection
    picks a different set of columns."""
    client = _client(trained_env)

    expected_features = client.get("/health").json()["expected_features"]
    features = {f: 1.0 for f in expected_features}
    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_missing_feature_returns_422(trained_env):
    client = _client(trained_env)
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422


def test_predict_partial_features_returns_422(trained_env):
    client = _client(trained_env)
    expected_features = client.get("/health").json()["expected_features"]
    features = {f: 1.0 for f in expected_features[:-1]}  # omit one required feature
    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert any("Missing features" in e for e in response.json()["detail"])


def _valid_features(client):
    return {f: 1.0 for f in client.get("/health").json()["expected_features"]}


@pytest.mark.parametrize("endpoint", ["/predict", "/explain"])
@pytest.mark.parametrize("mutate, expected", [
    (lambda f: f.update({"Not_A_Feature": 1.0}), "Unknown features"),
    (lambda f: f.update({next(iter(f)): "abc"}),  "expected a number"),
    (lambda f: f.update({next(iter(f)): None}),   "expected a number"),
    (lambda f: f.update({next(iter(f)): True}),   "expected a number"),
])
def test_malformed_features_return_422(trained_env, endpoint, mutate, expected):
    client = _client(trained_env)
    features = _valid_features(client)
    mutate(features)

    response = client.post(endpoint, json={"features": features})

    assert response.status_code == 422
    assert any(expected in e for e in response.json()["detail"])


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_return_422(trained_env, literal):
    # Standard JSON can't express these, but Python's json (and many clients)
    # emit them as bare literals — send the raw body to exercise that path.
    client = _client(trained_env)
    features = _valid_features(client)
    first = next(iter(features))
    body = json.dumps({"features": features}).replace(f'"{first}": 1.0', f'"{first}": {literal}')

    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    assert any("must be finite" in e for e in response.json()["detail"])


def test_features_not_an_object_returns_422(trained_env):
    client = _client(trained_env)
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


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
