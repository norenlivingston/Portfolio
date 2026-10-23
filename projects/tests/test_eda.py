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
    train, test = run_eda(tiny_config)

    n = tiny_config["eda"]["top_n_features"]
    assert train.shape[1] == n + 1  # selected features + Target
    assert test.shape[1] == n + 1
    assert Path(tiny_config["data"]["processed_train_path"]).exists()
    assert Path(tiny_config["data"]["processed_test_path"]).exists()
    assert Path(tiny_config["eda"]["output_dir"], "correlation_heatmap.png").exists()


def test_run_eda_test_set_does_not_influence_feature_selection_or_outlier_bounds(tiny_config):
    """Regression test for train/test leakage: corrupting the rows that land in
    the test split must not change which features get selected or how the
    training split gets cleaned, since both are fit before the test set is
    ever touched."""
    import numpy as np
    import pandas as pd

    build_dataset(tiny_config)
    train_a, _ = run_eda(tiny_config)

    raw_path = Path(tiny_config["data"]["raw_path"])
    df = pd.read_csv(raw_path)

    # Recreate the same split eda.py will make, then corrupt only the rows
    # that will land in the test partition with extreme, target-correlated
    # values that would shift correlations and outlier bounds if leaked.
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(
        df.index,
        test_size=tiny_config["pipeline"]["test_size"],
        random_state=tiny_config["pipeline"]["random_seed"],
    )
    rng = np.random.default_rng(0)
    feature_cols = [c for c in df.columns if c != "Target"]
    df.loc[test_idx, feature_cols] = rng.uniform(1e6, 2e6, size=(len(test_idx), len(feature_cols)))
    df.loc[test_idx, "Target"] = rng.uniform(1e6, 2e6, size=len(test_idx))
    df.to_csv(raw_path, index=False)

    train_b, _ = run_eda(tiny_config)

    assert list(train_a.columns) == list(train_b.columns)
    pd.testing.assert_frame_equal(
        train_a.reset_index(drop=True), train_b.reset_index(drop=True)
    )


def test_remove_strategy_never_drops_test_rows(tiny_config):
    """Under `remove`, outlier rows are dropped from training only. The test
    split keeps every row (clipped to train-fitted bounds) so the hold-out
    evaluation population is not changed by cleaning."""
    tiny_config["eda"]["outlier_strategy"] = "remove"
    tiny_config["eda"]["outlier_threshold"] = 0.5  # tight bounds so outliers certainly exist
    build_dataset(tiny_config)
    train, test = run_eda(tiny_config)

    n_test = round(tiny_config["data"]["n_samples"] * tiny_config["pipeline"]["test_size"])
    assert len(test) == n_test
    assert len(train) < tiny_config["data"]["n_samples"] - n_test


def test_outlier_transform_without_row_drop_clips_to_train_bounds():
    from eda import OutlierTreatment

    train = pd.DataFrame({"Feature_1": [1.0, 2.0, 3.0, 4.0, 5.0], "Target": range(5)})
    test = pd.DataFrame({"Feature_1": [-100.0, 3.0, 100.0], "Target": range(3)})

    treatment = OutlierTreatment("iqr", 1.5, "remove").fit(train)
    lo, hi = treatment.bounds_["Feature_1"]
    cleaned = treatment.transform(test, allow_row_drop=False)

    assert len(cleaned) == 3
    assert cleaned["Feature_1"].tolist() == [lo, 3.0, hi]
