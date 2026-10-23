"""
Shared model/config access + prediction logic for the MLOps and agent layers.

Config resolution order:
    1. PIPELINE_CONFIG env var (explicit path — used by tests and containers)
    2. config.yaml in the current working directory
    3. config.yaml in the parent of the current working directory

All loads are cached; call `clear_cache()` after swapping PIPELINE_CONFIG
(e.g. between test cases) to force a reload.
"""
import json
import math
import numbers
import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_ml_pipeline"))


def _config_path() -> Path:
    override = os.environ.get("PIPELINE_CONFIG")
    if override:
        return Path(override)

    for search in (Path.cwd(), Path.cwd().parent):
        p = search / "config.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(
        "config.yaml not found. Run from the projects/ directory, "
        "or set PIPELINE_CONFIG to an explicit path."
    )


@lru_cache
def get_config() -> dict:
    with open(_config_path()) as f:
        return yaml.safe_load(f)


@lru_cache
def get_model():
    return joblib.load(get_config()["mlops"]["model_path"])


def get_features() -> list:
    # The preprocessor (ColumnTransformer) was fitted on a named DataFrame and
    # reliably stores input feature names. Pipeline.__getattr__ proxies to the
    # last step (the regressor), which was fit on a numpy array and may not.
    return list(get_model().named_steps["preprocessor"].feature_names_in_)


def clear_cache() -> None:
    get_config.cache_clear()
    get_model.cache_clear()


def get_categorical_features() -> list:
    preprocessor = get_model().named_steps["preprocessor"]
    return [c for name, _, cols in preprocessor.transformers_ if name == "cat" for c in cols]


# ── Input validation ─────────────────────────────────────────────────────────

def validate_features(features) -> list:
    """Every problem with a feature payload, as human-readable strings (empty
    list = valid). Required: exactly the model's features, numeric ones finite
    numbers, categorical ones strings."""
    if not isinstance(features, dict):
        return ["features must be an object mapping feature name -> value"]

    expected    = get_features()
    categorical = set(get_categorical_features())
    errors      = []

    missing = sorted(set(expected) - set(features))
    unknown = sorted(set(features) - set(expected))
    if missing:
        errors.append(f"Missing features: {missing}")
    if unknown:
        errors.append(f"Unknown features: {unknown}")

    for name in expected:
        if name not in features:
            continue
        value = features[name]
        if name in categorical:
            if not isinstance(value, str):
                errors.append(f"{name}: expected a string category, got {value!r}")
        elif isinstance(value, bool) or not isinstance(value, numbers.Real):
            errors.append(f"{name}: expected a number, got {value!r}")
        elif not math.isfinite(value):
            errors.append(f"{name}: must be finite, got {value!r}")

    return errors


# ── Prediction & explainability ────────────────────────────────────────────

def predict_one(features: dict) -> dict:
    errors = validate_features(features)
    if errors:
        return {"error": "; ".join(errors)}

    X = pd.DataFrame([{f: features[f] for f in get_features()}])
    pred = float(get_model().predict(X)[0])
    return {"prediction": pred}


def explain_one(features: dict) -> dict:
    """Per-feature SHAP contributions for a single prediction (tree models only).
    Contributions of one-hot columns are summed back onto their source feature,
    so every key is a raw input feature name."""
    errors = validate_features(features)
    if errors:
        return {"error": "; ".join(errors)}

    model = get_model()
    regressor = model.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        return {"error": f"Explainability is not supported for {type(regressor).__name__}."}

    import shap  # heavy import — deferred until actually needed

    from pipeline import aggregate_by_source, source_feature_names

    preprocessor = model.named_steps["preprocessor"]
    X = pd.DataFrame([{f: features[f] for f in get_features()}])
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):  # one-hot output can be sparse
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(regressor)
    shap_values = np.asarray(explainer.shap_values(X_transformed))[0]
    base_value = explainer.expected_value
    if not np.isscalar(base_value):
        base_value = base_value[0]

    contributions = dict(sorted(
        aggregate_by_source(shap_values, source_feature_names(preprocessor)).items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    ))

    return {
        "prediction": predict_one(features)["prediction"],
        "base_value": float(base_value),
        "contributions": contributions,
    }


# ── Metrics log ──────────────────────────────────────────────────────────────

def get_latest_metrics() -> dict:
    log_path = Path(get_config()["mlops"]["metrics_log"])
    if not log_path.exists():
        return {"error": "No metrics log found. Run the pipeline first."}
    history = json.loads(log_path.read_text())
    return history[-1]


def get_run_history() -> list:
    log_path = Path(get_config()["mlops"]["metrics_log"])
    if not log_path.exists():
        return []
    return json.loads(log_path.read_text())
