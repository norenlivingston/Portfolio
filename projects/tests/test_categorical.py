"""Categorical-feature coverage: the pipeline's preprocessor one-hot encodes
object columns, so predictions, validation, explanations, and importances
must all speak in raw input feature names rather than encoded columns."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import model_registry
from pipeline import _build_preprocessor, _save_feature_importance, source_feature_names


@pytest.fixture
def categorical_model(use_config, tiny_config, tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "Num_A": rng.standard_normal(n),
        "Num_B": rng.standard_normal(n),
        "Color": rng.choice(["red", "green", "blue"], size=n),
    })
    y = 3 * df["Num_A"] + df["Color"].map({"red": 5.0, "green": 0.0, "blue": -5.0})

    pipe = Pipeline([
        ("preprocessor", _build_preprocessor(["Num_A", "Num_B"], ["Color"])),
        ("regressor",    RandomForestRegressor(n_estimators=20, random_state=0)),
    ]).fit(df, y)
    joblib.dump(pipe, tiny_config["mlops"]["model_path"])
    model_registry.clear_cache()
    return pipe


def test_source_feature_names_maps_one_hot_columns_back(categorical_model):
    sources = source_feature_names(categorical_model.named_steps["preprocessor"])
    assert sources == ["Num_A", "Num_B", "Color", "Color", "Color"]


def test_predict_and_explain_with_categorical_feature(categorical_model):
    features = {"Num_A": 0.5, "Num_B": -1.0, "Color": "red"}

    prediction = model_registry.predict_one(features)
    explanation = model_registry.explain_one(features)

    assert "error" not in prediction
    assert "error" not in explanation
    assert set(explanation["contributions"]) == {"Num_A", "Num_B", "Color"}
    reconstructed = explanation["base_value"] + sum(explanation["contributions"].values())
    assert abs(reconstructed - prediction["prediction"]) < 1e-6


def test_categorical_feature_must_be_a_string(categorical_model):
    errors = model_registry.validate_features({"Num_A": 0.5, "Num_B": -1.0, "Color": 1.0})
    assert errors == ["Color: expected a string category, got 1.0"]


def test_feature_importance_plot_uses_source_names(categorical_model, tmp_path, monkeypatch):
    captured = {}
    import pipeline
    monkeypatch.setattr(pipeline.plt, "xticks", lambda ticks, labels, **kw: captured.update(labels=labels))

    _save_feature_importance(categorical_model, str(tmp_path / "importances.png"))

    assert sorted(captured["labels"]) == ["Color", "Num_A", "Num_B"]
