"""Random Forest baseline for AAM comparison."""

import logging
import math
from pathlib import Path

import biom
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from aam.cli.utils import validate_file_path


def load_sample_ids(path: str) -> list[str]:
    """Load sample IDs from a text file (one per line)."""
    validate_file_path(path, "Sample IDs file")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_biom_as_dataframe(path: str) -> pd.DataFrame:
    """Load BIOM table and convert to DataFrame (samples x features)."""
    table = biom.load_table(path)
    sample_ids = table.ids(axis="sample")
    feature_ids = table.ids(axis="observation")
    data = table.matrix_data.toarray()
    df = pd.DataFrame(data.T, index=sample_ids, columns=feature_ids)
    return df


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    max_features: str = "sqrt",
    random_seed: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train a Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics (R², MAE, RMSE)."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
    }


def _filter_samples(
    sample_ids: list[str],
    valid_samples: set[str],
    split_name: str,
    logger: logging.Logger,
) -> list[str]:
    """Filter sample IDs to only include those present in valid_samples."""
    missing = [s for s in sample_ids if s not in valid_samples]
    if missing:
        logger.warning(f"{len(missing)} {split_name} samples not found in BIOM/metadata: {missing[:5]}...")
    return [s for s in sample_ids if s in valid_samples]


def create_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    mae: float,
    rmse: float,
) -> plt.Figure:
    """Create prediction vs actual scatter plot matching AAM's TensorBoard style."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Perfect Prediction", alpha=0.5)

    metrics_str = f"R² = {r2:.4f}, MAE = {mae:.4f}"
    if len(y_true) > 1:
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), "b-", linewidth=2, label=f"Linear Fit, {metrics_str}")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Random Forest Baseline ({metrics_str}, RMSE = {rmse:.4f})")
    ax.legend(loc="upper left")

    plt.tight_layout()
    return fig


@click.command()
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--metadata", required=True, type=click.Path(exists=True), help="Path to metadata TSV file")
@click.option("--metadata-column", required=True, type=str, help="Column name for regression target")
@click.option("--train-samples", required=True, type=click.Path(exists=True), help="Path to training sample IDs (one per line)")
@click.option("--val-samples", required=True, type=click.Path(exists=True), help="Path to validation sample IDs (one per line)")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    train_ids = load_sample_ids(train_samples)
    val_ids = load_sample_ids(val_samples)
    logger.info(f"Loaded {len(train_ids)} training samples, {len(val_ids)} validation samples")

    metadata_df = pd.read_csv(metadata, sep="\t")
    if "sample_id" not in metadata_df.columns:
        raise click.ClickException(f"Metadata must have 'sample_id' column. Found: {list(metadata_df.columns)}")
    if metadata_column not in metadata_df.columns:
        raise click.ClickException(f"Metadata column '{metadata_column}' not found. Available: {list(metadata_df.columns)}")
    metadata_df = metadata_df.set_index("sample_id")

    biom_df = load_biom_as_dataframe(table)
    logger.info(f"Features: {biom_df.shape[1]} ASVs")

    biom_samples = set(biom_df.index)
    metadata_samples = set(metadata_df.index)
    valid_samples = biom_samples & metadata_samples

    train_ids = _filter_samples(train_ids, valid_samples, "training", logger)
    val_ids = _filter_samples(val_ids, valid_samples, "validation", logger)

    X_train = biom_df.loc[train_ids]
    y_train = metadata_df.loc[train_ids, metadata_column].astype(float)
    X_val = biom_df.loc[val_ids]
    y_val = metadata_df.loc[val_ids, metadata_column].astype(float)

    nonzero_cols = X_train.columns[X_train.sum(axis=0) > 0]
    X_train = X_train[nonzero_cols]
    X_val = X_val[nonzero_cols]
    logger.info(f"After filtering zero-sum features: {len(nonzero_cols)} ASVs")

    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    model = train_random_forest(
        X_train, y_train, n_estimators=n_estimators, max_features=max_features, random_seed=random_seed, n_jobs=n_jobs
    )

    y_pred = model.predict(X_val)
    y_pred_series = pd.Series(y_pred, index=val_ids)

    metrics = compute_metrics(y_val, y_pred_series)
    click.echo("Validation Metrics:")
    click.echo(f"  R²:   {metrics['r2']:.3f}")
    click.echo(f"  MAE:  {metrics['mae']:.3f}")
    click.echo(f"  RMSE: {metrics['rmse']:.3f}")

    output_df = pd.DataFrame({"sample_id": val_ids, "actual": y_val.values, "predicted": y_pred})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Predictions saved to: {output}")

    plot_path = output_path.with_suffix(".png")
    fig = create_prediction_plot(y_val.values, y_pred, metrics["r2"], metrics["mae"], metrics["rmse"])
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    rf_baseline()
