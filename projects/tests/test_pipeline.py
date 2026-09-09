import json
from pathlib import Path

import mlflow

from dataset_build import build_dataset
from eda import run_eda
from pipeline import train_pipeline


def _run(tiny_config):
    build_dataset(tiny_config)
    run_eda(tiny_config)
    return train_pipeline(tiny_config)


def test_train_pipeline_returns_expected_metrics(tiny_config):
    metrics = _run(tiny_config)

    for key in ("best_model", "cv_r2_mean", "cv_r2_std", "test_mae", "test_rmse", "test_r2", "mlflow_run_id"):
        assert key in metrics

    assert metrics["best_model"] in {"RandomForest", "LinearRegression"}
    assert metrics["test_mae"] >= 0
    assert metrics["test_rmse"] >= 0
    assert -1.0 <= metrics["test_r2"] <= 1.0


def test_train_pipeline_saves_model(tiny_config):
    _run(tiny_config)
    assert Path(tiny_config["mlops"]["model_path"]).exists()


def test_train_pipeline_appends_metrics_log(tiny_config):
    _run(tiny_config)
    _run(tiny_config)  # second run should append, not overwrite

    history = json.loads(Path(tiny_config["mlops"]["metrics_log"]).read_text())
    assert len(history) == 2


def test_train_pipeline_logs_to_mlflow(tiny_config):
    metrics = _run(tiny_config)

    mlflow.set_tracking_uri(tiny_config["mlops"]["mlflow_tracking_uri"])
    run = mlflow.get_run(metrics["mlflow_run_id"])
    assert run.data.metrics["test_r2"] == metrics["test_r2"]
