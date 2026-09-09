import model_registry


def test_predict_one_success(trained_env):
    features = {f: 1.0 for f in model_registry.get_features()}
    result = model_registry.predict_one(features)

    assert "error" not in result
    assert isinstance(result["prediction"], float)


def test_predict_one_missing_feature(trained_env):
    features = {f: 1.0 for f in model_registry.get_features()}
    features.pop(next(iter(features)))

    result = model_registry.predict_one(features)
    assert "error" in result


def test_explain_one_matches_prediction(trained_env):
    features = {f: 1.0 for f in model_registry.get_features()}
    prediction = model_registry.predict_one(features)["prediction"]
    explanation = model_registry.explain_one(features)

    is_tree_model = hasattr(model_registry.get_model().named_steps["regressor"], "feature_importances_")
    if not is_tree_model:
        # CV can select LinearRegression on this tiny dataset — explainability is tree-only.
        assert "error" in explanation
        return

    assert "error" not in explanation
    assert explanation["prediction"] == prediction
    assert set(explanation["contributions"]) == set(model_registry.get_features())
    # base value + contributions should reconstruct the prediction (SHAP's additivity property)
    reconstructed = explanation["base_value"] + sum(explanation["contributions"].values())
    assert abs(reconstructed - prediction) < 1e-6


def test_get_latest_metrics_and_history(trained_env):
    history = model_registry.get_run_history()
    latest = model_registry.get_latest_metrics()

    assert len(history) >= 1
    assert latest == history[-1]


def test_get_latest_metrics_missing_log(use_config):
    assert model_registry.get_latest_metrics() == {"error": "No metrics log found. Run the pipeline first."}
    assert model_registry.get_run_history() == []
