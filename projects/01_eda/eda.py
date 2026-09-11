"""
Stage 2 — Exploratory Data Analysis
Selects the top correlated features, treats outliers, and saves the cleaned
dataset. Outputs visualisations to the configured directory.
"""
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

log = logging.getLogger(__name__)


# ── Feature selection ─────────────────────────────────────────────────────────

def select_top_features(df: pd.DataFrame, n: int) -> list:
    corr = df.corr()["Target"].drop("Target").abs()
    return corr.nlargest(n).index.tolist()


# ── Multicollinearity check ───────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = add_constant(df)
    vif = pd.DataFrame({
        "Feature": X.columns,
        "VIF":     [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif[vif["Feature"] != "const"].reset_index(drop=True)


# ── Outlier treatment ─────────────────────────────────────────────────────────

class OutlierTreatment:
    """Learns per-column outlier bounds (and replacement values) from a single
    fit set, then applies them unchanged to any dataframe. Fitting on the
    training split only and reusing those bounds for the test split keeps the
    test set from influencing its own cleaning."""

    def __init__(self, method: str, threshold: float, strategy: str):
        self.method    = method
        self.threshold = threshold
        self.strategy  = strategy
        self.bounds_   = {}
        self.medians_  = {}
        self.means_    = {}

    def fit(self, df: pd.DataFrame) -> "OutlierTreatment":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["Target"])

        for col in numeric_cols:
            if self.method == "iqr":
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr     = q3 - q1
                lo, hi  = q1 - self.threshold * iqr, q3 + self.threshold * iqr
            elif self.method == "zscore":
                mean, std = df[col].mean(), df[col].std()
                lo, hi    = mean - self.threshold * std, mean + self.threshold * std
            else:
                raise ValueError(f"Unknown outlier method: {self.method!r}")

            self.bounds_[col]  = (lo, hi)
            self.medians_[col] = df[col].median()
            self.means_[col]   = df[col].mean()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        for col, (lo, hi) in self.bounds_.items():
            if col not in df_out.columns:
                continue

            mask  = (df_out[col] < lo) | (df_out[col] > hi)
            n_out = int(mask.sum())
            if n_out == 0:
                continue

            if self.strategy == "median":
                df_out.loc[mask, col] = self.medians_[col]
            elif self.strategy == "mean":
                df_out.loc[mask, col] = self.means_[col]
            elif self.strategy == "iqr_bound":
                df_out.loc[df_out[col] < lo, col] = lo
                df_out.loc[df_out[col] > hi, col] = hi
            elif self.strategy == "remove":
                df_out = df_out[~mask]
            else:
                raise ValueError(f"Unknown outlier strategy: {self.strategy!r}")

            log.debug("  %-15s  treated %d outliers", col, n_out)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def replace_outliers(df: pd.DataFrame, method: str, threshold: float, strategy: str) -> pd.DataFrame:
    """Fits and applies outlier treatment to the same dataframe. Kept for callers
    that only have one dataset; pipelines with a train/test split should fit
    `OutlierTreatment` on the training data and transform each split separately."""
    return OutlierTreatment(method, threshold, strategy).fit_transform(df)


# ── Visualisations ────────────────────────────────────────────────────────────

def save_plots(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=100)
    plt.close()

    # Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Target"], bins=30, kde=True)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=100)
    plt.close()

    # Top-4 feature–target scatter
    top4 = df.corr()["Target"].drop("Target").abs().nlargest(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flat, top4):
        ax.scatter(df[feat], df["Target"], alpha=0.3, s=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("Target")
    fig.suptitle("Top 4 Features vs Target")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_target_scatter.png"), dpi=100)
    plt.close()

    log.info("Plots saved → %s", output_dir)


# ── Main entry ────────────────────────────────────────────────────────────────

def run_eda(config: dict) -> tuple:
    cfg_data = config["data"]
    cfg_eda  = config["eda"]
    cfg_pipe = config["pipeline"]

    df = pd.read_csv(cfg_data["raw_path"])
    log.info("Loaded raw dataset: %d rows × %d cols", *df.shape)
    log.info("Missing values: %d", int(df.isnull().sum().sum()))

    # Split immediately — every statistic below is fit on the training split
    # only, so the test set stays untouched until final evaluation.
    df_train, df_test = train_test_split(
        df, test_size=cfg_pipe["test_size"], random_state=cfg_pipe["random_seed"]
    )
    log.info("Split → train=%d, test=%d", len(df_train), len(df_test))

    # Feature selection — fit on training data only
    top_features = select_top_features(df_train, cfg_eda["top_n_features"])
    log.info("Top %d features selected (train only): %s", len(top_features), top_features)
    cols     = top_features + ["Target"]
    df_train = df_train[cols].copy()
    df_test  = df_test[cols].copy()

    # Multicollinearity — diagnostic only, computed on training data
    vif = compute_vif(df_train.drop(columns=["Target"]))
    log.info("VIF results (train):\n%s", vif.to_string(index=False))

    # Outlier treatment — bounds fit on training data, then applied to both splits
    outliers = OutlierTreatment(
        method=cfg_eda["outlier_method"],
        threshold=cfg_eda["outlier_threshold"],
        strategy=cfg_eda["outlier_strategy"],
    ).fit(df_train)
    df_train_clean = outliers.transform(df_train)
    df_test_clean  = outliers.transform(df_test)
    log.info("After outlier treatment → train=%d rows, test=%d rows", len(df_train_clean), len(df_test_clean))

    # Visualisations — training data only, so plots never reveal test-set shape
    save_plots(df_train_clean, cfg_eda["output_dir"])

    # Save cleaned splits
    train_out = cfg_data["processed_train_path"]
    test_out  = cfg_data["processed_test_path"]
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(test_out).parent.mkdir(parents=True, exist_ok=True)
    df_train_clean.to_csv(train_out, index=False)
    df_test_clean.to_csv(test_out, index=False)
    log.info("Saved cleaned train/test datasets → %s , %s", train_out, test_out)

    return df_train_clean, df_test_clean


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

    run_eda(cfg)
