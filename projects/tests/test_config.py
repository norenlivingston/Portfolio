from pathlib import Path

import yaml

PROJECTS_DIR = Path(__file__).resolve().parent.parent

# Keys allowed to differ between config.yaml and config.docker.yaml: where
# artifacts are written. Everything that affects modelling must match.
PATH_KEYS = {
    ("data", "raw_path"),
    ("data", "processed_train_path"),
    ("data", "processed_test_path"),
    ("eda", "output_dir"),
    ("mlops", "model_path"),
    ("mlops", "metrics_log"),
    ("mlops", "mlflow_tracking_uri"),
}


def _flatten(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _flatten(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def test_docker_config_only_differs_in_artifact_paths():
    local  = dict(_flatten(yaml.safe_load((PROJECTS_DIR / "config.yaml").read_text())))
    docker = dict(_flatten(yaml.safe_load((PROJECTS_DIR / "config.docker.yaml").read_text())))

    assert local.keys() == docker.keys()
    differing = {k for k in local if local[k] != docker[k]}
    assert differing <= PATH_KEYS, f"Non-path settings drifted: {differing - PATH_KEYS}"
