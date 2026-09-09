"""
End-to-end ML pipeline — single entry point.

Usage (from the projects/ directory):
    python run_pipeline.py
    python run_pipeline.py --config config.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Make sub-packages importable without installing them as packages.
for _p in ("00_dataset_build", "01_eda", "02_ml_pipeline"):
    sys.path.insert(0, _p)

from dataset_build import build_dataset  # noqa: E402
from eda import run_eda                  # noqa: E402
from pipeline import train_pipeline      # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _section(title: str) -> None:
    log.info("-" * 55)
    log.info("  %s", title)
    log.info("-" * 55)


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    Path("data").mkdir(exist_ok=True)

    _section("STAGE 1 / 3 — Dataset Build")
    build_dataset(config)

    _section("STAGE 2 / 3 — Exploratory Data Analysis")
    run_eda(config)

    _section("STAGE 3 / 3 — Model Training & Evaluation")
    metrics = train_pipeline(config)

    _section("Pipeline complete")
    log.info("  Best model  : %s",         metrics["best_model"])
    log.info("  CV R²       : %.4f ± %.4f", metrics["cv_r2_mean"], metrics["cv_r2_std"])
    log.info("  Test R²     : %.4f",        metrics["test_r2"])
    log.info("  Test RMSE   : %.4f",        metrics["test_rmse"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end ML pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
