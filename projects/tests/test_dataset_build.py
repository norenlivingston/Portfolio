from pathlib import Path

from dataset_build import build_dataset


def test_shape_and_columns(tiny_config):
    df = build_dataset(tiny_config)

    n_samples = tiny_config["data"]["n_samples"]
    n_features = tiny_config["data"]["n_features"]
    assert df.shape == (n_samples, n_features + 1)
    assert "Target" in df.columns
    assert list(df.columns[:-1]) == [f"Feature_{i+1}" for i in range(n_features)]


def test_writes_csv(tiny_config):
    build_dataset(tiny_config)
    assert Path(tiny_config["data"]["raw_path"]).exists()


def test_deterministic_with_seed(tiny_config):
    df1 = build_dataset(tiny_config)
    df2 = build_dataset(tiny_config)
    assert df1.equals(df2)


def test_different_seed_changes_output(tiny_config):
    df1 = build_dataset(tiny_config)
    tiny_config["data"]["random_seed"] = 7
    df2 = build_dataset(tiny_config)
    assert not df1.equals(df2)
