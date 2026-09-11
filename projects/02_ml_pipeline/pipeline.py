"""
Stage 3 — Model Training & Evaluation
Compares candidate models via k-fold cross-validation, selects the best,
evaluates on a hold-out test set, and persists the pipeline + a metrics log.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer,     numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def _candidate_models(config: dict) -> dict:
    rf = config["pipeline"]["models"]["random_forest"]
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=rf["n_estimators"],
            max_depth=rf.get("max_depth"),
            random_state=config["pipeline"]["random_seed"],
        ),
        "LinearRegression": LinearRegression(),
    }


def _save_feature_importance(pipeline: Pipeline, feature_names: list, out_path: str) -> None:
    model = pipeline.named_steps["regressor"]
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[idx], color="steelblue")
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    log.info("Feature importance plot → %s", out_path)


def _append_metrics(log_path: str, entry: dict) -> None:
    path    = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2))
    log.info("Metrics appended → %s", log_path)


# ── Main entry ────────────────────────────────────────────────────────────────

def train_pipeline(config: dict) -> dict:
    cfg      = config["pipeline"]
    cfg_mlfl = config["mlops"]

    mlflow.set_tracking_uri(cfg_mlfl.get("mlflow_tracking_uri", "sqlite:///mlflow_store/mlruns.db"))
    mlflow.set_experiment(cfg_mlfl.get("mlflow_experiment", "ml-pipeline"))

    # The train/test split and all leakage-sensitive fitting (feature selection,
    # outlier bounds) already happened in the EDA stage — these files are the
    # untouched result, so the test rows below have never influenced training.
    df_train = pd.read_csv(config["data"]["processed_train_path"])
    df_test  = pd.read_csv(config["data"]["processed_test_path"])
    log.info("Loaded processed dataset: train=%d rows, test=%d rows", len(df_train), len(df_test))

    X_train = df_train.drop(columns=["Target"])
    y_train = df_train["Target"]
    X_test  = df_test.drop(columns=["Target"])
    y_test  = df_test["Target"]

    numeric_features     = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    cv = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_seed"])

    # ── Cross-validate all candidates ────────────────────────────────────────
    log.info("Cross-validating candidates (%d-fold)...", cfg["cv_folds"])
    cv_results = {}
    for name, estimator in _candidate_models(config).items():
        pipe   = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_results[name] = scores
        log.info("  %-20s  R² = %.4f ± %.4f", name, scores.mean(), scores.std())

    best_name   = max(cv_results, key=lambda k: cv_results[k].mean())
    best_scores = cv_results[best_name]
    log.info("Selected: %s", best_name)

    with mlflow.start_run(run_name=best_name) as run:
        # ── Final fit on full training set ───────────────────────────────────
        best_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    _candidate_models(config)[best_name]),
        ])
        best_pipeline.fit(X_train, y_train)

        # ── Hold-out evaluation ───────────────────────────────────────────────
        y_pred = best_pipeline.predict(X_test)
        mae    = float(mean_absolute_error(y_test, y_pred))
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2     = float(r2_score(y_test, y_pred))
        log.info("Hold-out → MAE=%.4f  RMSE=%.4f  R²=%.4f", mae, rmse, r2)

        # ── Persist model ───────────────────────────────────────────────────────
        model_path = Path(config["mlops"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path)
        log.info("Model saved → %s", model_path)

        # ── Feature importance plot ─────────────────────────────────────────────
        importance_path = "02_ml_pipeline/feature_importances.png"
        _save_feature_importance(best_pipeline, numeric_features, importance_path)

        # ── MLflow tracking ──────────────────────────────────────────────────────
        mlflow.log_params({
            "model":        best_name,
            "cv_folds":     cfg["cv_folds"],
            "test_size":    cfg["test_size"],
            "n_features":   len(numeric_features) + len(categorical_features),
            **{f"model__{k}": v for k, v in _candidate_models(config)[best_name].get_params().items()},
        })
        mlflow.log_metrics({
            "cv_r2_mean": float(best_scores.mean()),
            "cv_r2_std":  float(best_scores.std()),
            "test_mae":   mae,
            "test_rmse":  rmse,
            "test_r2":    r2,
        })
        if Path(importance_path).exists():
            mlflow.log_artifact(importance_path)
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        log.info("MLflow run logged → %s (experiment: %s)", run.info.run_id, cfg_mlfl.get("mlflow_experiment", "ml-pipeline"))

        # ── Append run to metrics log (used by the agent/MCP tools) ─────────────
        metrics = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "best_model":    best_name,
            "cv_r2_mean":    float(best_scores.mean()),
            "cv_r2_std":     float(best_scores.std()),
            "test_mae":      mae,
            "test_rmse":     rmse,
            "test_r2":       r2,
            "mlflow_run_id": run.info.run_id,
        }
        _append_metrics(config["mlops"]["metrics_log"], metrics)

    return metrics


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    for search in [Path.cwd(), Path.cwd().parent]:
        cfg_path = search / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            break
    else:
        raise FileNotFoundError("config.yaml not found. Run from projects/ directory.")

    train_pipeline(cfg)
