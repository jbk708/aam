"""Random Forest baseline for comparison with AAM model predictions."""

import click
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import biom


def load_sample_ids(path: str) -> list[str]:
    """Load sample IDs from a text file (one per line)."""
    raise NotImplementedError


def load_biom_as_dataframe(path: str) -> pd.DataFrame:
    """Load BIOM table and convert to DataFrame (samples x features)."""
    raise NotImplementedError


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    max_features: str = "sqrt",
    random_seed: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train a Random Forest regressor."""
    raise NotImplementedError


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics (R², MAE, RMSE)."""
    raise NotImplementedError


@click.command()
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--metadata", required=True, type=click.Path(exists=True), help="Path to metadata TSV file")
@click.option("--metadata-column", required=True, type=str, help="Column name for regression target")
@click.option(
    "--train-samples", required=True, type=click.Path(exists=True), help="Path to training sample IDs (one per line)"
)
@click.option(
    "--val-samples", required=True, type=click.Path(exists=True), help="Path to validation sample IDs (one per line)"
)
@click.option("--output", required=True, type=click.Path(), help="Output file for predictions")
@click.option("--n-estimators", default=500, type=int, help="Number of trees in the forest")
@click.option("--max-features", default="sqrt", type=str, help="Max features per tree (sqrt, log2, or float)")
@click.option("--random-seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--n-jobs", default=-1, type=int, help="Number of parallel jobs (-1 for all cores)")
def rf_baseline(
    table: str,
    metadata: str,
    metadata_column: str,
    train_samples: str,
    val_samples: str,
    output: str,
    n_estimators: int,
    max_features: str,
    random_seed: int,
    n_jobs: int,
) -> None:
    """Run Random Forest regression as a baseline comparison for AAM."""
    raise NotImplementedError


if __name__ == "__main__":
    rf_baseline()
