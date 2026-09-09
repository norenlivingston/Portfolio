import sys
from pathlib import Path

import pytest
import yaml

PROJECTS_DIR = Path(__file__).resolve().parent.parent
for _p in ("00_dataset_build", "01_eda", "02_ml_pipeline", "03_mlops", "05_agents"):
    sys.path.insert(0, str(PROJECTS_DIR / _p))


@pytest.fixture
def tiny_config(tmp_path):
    """A full config pointing every path into an isolated tmp directory, sized
    small so build -> EDA -> train runs in well under a second."""
    return {
        "data": {
            "raw_path":       str(tmp_path / "raw.csv"),
            "processed_path": str(tmp_path / "processed.csv"),
            "n_samples":      200,
            "n_features":     10,
            "random_seed":    42,
        },
        "eda": {
            "output_dir":        str(tmp_path / "eda_plots"),
            "top_n_features":    4,
            "outlier_method":    "iqr",
            "outlier_threshold": 1.5,
            "outlier_strategy":  "iqr_bound",
        },
        "pipeline": {
            "test_size":   0.2,
            "cv_folds":    3,
            "random_seed": 42,
            "models": {
                "random_forest": {"n_estimators": 10, "max_depth": 5},
                "linear_regression": {},
            },
        },
        "mlops": {
            "model_path":          str(tmp_path / "model.pkl"),
            "metrics_log":         str(tmp_path / "metrics_log.json"),
            "host":                "0.0.0.0",
            "port":                8000,
            "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}",
            "mlflow_experiment":   "test-experiment",
        },
        "agents": {
            "anthropic_model": "claude-opus-5",
            "ollama_model":    "qwen2.5",
        },
    }


@pytest.fixture
def config_path(tmp_path, tiny_config):
    """Write tiny_config to disk (model_registry reads config from a file)."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    return path


@pytest.fixture
def use_config(monkeypatch, config_path):
    """Point PIPELINE_CONFIG at config_path and clear model_registry's caches."""
    monkeypatch.setenv("PIPELINE_CONFIG", str(config_path))
    import model_registry
    model_registry.clear_cache()
    yield config_path
    model_registry.clear_cache()


@pytest.fixture
def trained_env(use_config, tiny_config):
    """Run the real pipeline stages end-to-end against tiny_config, then hand
    back the config so tests can exercise serving/agent code against a real,
    tiny, trained model."""
    from dataset_build import build_dataset
    from eda import run_eda
    from pipeline import train_pipeline

    build_dataset(tiny_config)
    run_eda(tiny_config)
    train_pipeline(tiny_config)

    import model_registry
    model_registry.clear_cache()

    return tiny_config
