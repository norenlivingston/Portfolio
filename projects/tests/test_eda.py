from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset_build import build_dataset
from eda import compute_vif, replace_outliers, run_eda, select_top_features


def test_select_top_features_ranks_by_absolute_correlation():
    rng = np.random.default_rng(0)
    n = 500
    strong = rng.standard_normal(n)
    weak = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    target = 10 * strong - 8 * weak + 0.01 * noise  # strong/weak matter, noise doesn't

    df = pd.DataFrame({"Strong": strong, "Weak": weak, "Noise": noise, "Target": target})
    top2 = select_top_features(df, n=2)

    assert set(top2) == {"Strong", "Weak"}


def test_compute_vif_returns_one_row_per_feature():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "A": rng.standard_normal(200),
        "B": rng.standard_normal(200),
    })
    vif = compute_vif(df)

    assert set(vif["Feature"]) == {"A", "B"}
    assert (vif["VIF"] > 0).all()


@pytest.mark.parametrize("strategy", ["mean", "median", "iqr_bound", "remove"])
def test_replace_outliers_strategies(strategy):
    values = [1.0] * 20 + [1000.0]  # one obvious outlier
    df = pd.DataFrame({"Feature_1": values, "Target": range(21)})

    cleaned = replace_outliers(df, method="iqr", threshold=1.5, strategy=strategy)

    if strategy == "remove":
        assert len(cleaned) == 20
        assert 1000.0 not in cleaned["Feature_1"].values
    else:
        assert len(cleaned) == 21
        assert 1000.0 not in cleaned["Feature_1"].values


def test_replace_outliers_rejects_unknown_method():
    df = pd.DataFrame({"Feature_1": [1.0, 2.0, 3.0], "Target": [0, 1, 2]})
    with pytest.raises(ValueError):
        replace_outliers(df, method="bogus", threshold=1.5, strategy="mean")


def test_run_eda_end_to_end(tiny_config):
    build_dataset(tiny_config)
    cleaned = run_eda(tiny_config)

    n = tiny_config["eda"]["top_n_features"]
    assert cleaned.shape[1] == n + 1  # selected features + Target
    assert Path(tiny_config["data"]["processed_path"]).exists()
    assert Path(tiny_config["eda"]["output_dir"], "correlation_heatmap.png").exists()
