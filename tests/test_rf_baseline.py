"""Unit tests for Random Forest baseline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from aam.cli.rf_baseline import (
    compute_metrics,
    load_biom_as_dataframe,
    load_sample_ids,
    rf_baseline,
    train_random_forest,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ids_file(temp_dir):
    """Create a sample IDs file."""
    ids_file = temp_dir / "sample_ids.txt"
    ids_file.write_text("sample1\nsample2\nsample3\n")
    return str(ids_file)


@pytest.fixture
def train_samples_file(temp_dir):
    """Create training sample IDs file."""
    ids_file = temp_dir / "train_samples.txt"
    ids_file.write_text("sample1\nsample2\nsample3\nsample4\n")
    return str(ids_file)


@pytest.fixture
def val_samples_file(temp_dir):
    """Create validation sample IDs file."""
    ids_file = temp_dir / "val_samples.txt"
    ids_file.write_text("sample5\nsample6\n")
    return str(ids_file)


@pytest.fixture
def metadata_file(temp_dir):
    """Create a metadata file with target column."""
    metadata_file = temp_dir / "metadata.tsv"
    metadata_file.write_text(
        "sample_id\ttarget\tother_col\n"
        "sample1\t10.5\tA\n"
        "sample2\t20.3\tB\n"
        "sample3\t15.2\tA\n"
        "sample4\t18.7\tB\n"
        "sample5\t12.1\tA\n"
        "sample6\t22.8\tB\n"
    )
    return str(metadata_file)


def _create_mock_biom_table(
    sample_ids: list[str],
    feature_ids: list[str],
    data: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock BIOM table with the given sample and feature IDs."""
    mock_table = MagicMock()
    mock_table.ids = lambda axis=None: (np.array(sample_ids) if axis == "sample" else np.array(feature_ids))
    if data is None:
        data = np.random.rand(len(feature_ids), len(sample_ids)) * 100
    mock_table.matrix_data = MagicMock()
    mock_table.matrix_data.toarray.return_value = data
    return mock_table


@pytest.fixture
def mock_biom_table():
    """Create a mock BIOM table with 6 samples and 4 features."""
    sample_ids = ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"]
    feature_ids = ["ASV1", "ASV2", "ASV3", "ASV4"]
    data = np.array(
        [
            [100, 200, 50, 150, 80, 120],
            [50, 100, 200, 80, 150, 90],
            [200, 50, 100, 120, 60, 180],
            [75, 150, 75, 90, 110, 70],
        ]
    )
    return _create_mock_biom_table(sample_ids, feature_ids, data)


class TestLoadSampleIds:
    """Tests for load_sample_ids function."""

    def test_load_sample_ids_basic(self, sample_ids_file):
        """Test loading sample IDs from file."""
        ids = load_sample_ids(sample_ids_file)
        assert ids == ["sample1", "sample2", "sample3"]

    def test_load_sample_ids_strips_whitespace(self, temp_dir):
        """Test that whitespace is stripped from sample IDs."""
        ids_file = temp_dir / "ids_whitespace.txt"
        ids_file.write_text("  sample1  \nsample2\t\n  sample3\n")
        ids = load_sample_ids(str(ids_file))
        assert ids == ["sample1", "sample2", "sample3"]

    def test_load_sample_ids_skips_empty_lines(self, temp_dir):
        """Test that empty lines are skipped."""
        ids_file = temp_dir / "ids_empty.txt"
        ids_file.write_text("sample1\n\nsample2\n\n\nsample3\n")
        ids = load_sample_ids(str(ids_file))
        assert ids == ["sample1", "sample2", "sample3"]

    def test_load_sample_ids_file_not_found(self, temp_dir):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_sample_ids(str(temp_dir / "nonexistent.txt"))


class TestLoadBiomAsDataframe:
    """Tests for load_biom_as_dataframe function."""

    def test_load_biom_as_dataframe_shape(self, temp_dir, mock_biom_table):
        """Test that BIOM is loaded with correct shape (samples x features)."""
        with patch("biom.load_table", return_value=mock_biom_table):
            biom_file = temp_dir / "test.biom"
            biom_file.touch()
            df = load_biom_as_dataframe(str(biom_file))
            assert df.shape == (6, 4)
            assert list(df.index) == ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"]
            assert list(df.columns) == ["ASV1", "ASV2", "ASV3", "ASV4"]

    def test_load_biom_as_dataframe_values(self, temp_dir, mock_biom_table):
        """Test that values are correctly transposed."""
        with patch("biom.load_table", return_value=mock_biom_table):
            biom_file = temp_dir / "test.biom"
            biom_file.touch()
            df = load_biom_as_dataframe(str(biom_file))
            assert df.loc["sample1", "ASV1"] == 100
            assert df.loc["sample1", "ASV2"] == 50


class TestTrainRandomForest:
    """Tests for train_random_forest function."""

    def test_train_random_forest_returns_model(self):
        """Test that training returns a fitted RandomForestRegressor."""
        from sklearn.ensemble import RandomForestRegressor

        X_train = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
        y_train = pd.Series([10, 20, 15, 25])

        model = train_random_forest(X_train, y_train, n_estimators=10, random_seed=42)

        assert isinstance(model, RandomForestRegressor)
        assert hasattr(model, "predict")

    def test_train_random_forest_can_predict(self):
        """Test that trained model can make predictions."""
        X_train = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
        y_train = pd.Series([10, 20, 15, 25])

        model = train_random_forest(X_train, y_train, n_estimators=10, random_seed=42)
        predictions = model.predict(X_train)

        assert len(predictions) == 4
        assert all(isinstance(p, (int, float, np.floating)) for p in predictions)

    def test_train_random_forest_respects_params(self):
        """Test that training respects hyperparameters."""
        X_train = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
        y_train = pd.Series([10, 20, 15, 25])

        model = train_random_forest(X_train, y_train, n_estimators=50, max_features="log2", random_seed=123)

        assert model.n_estimators == 50
        assert model.max_features == "log2"
        assert model.random_state == 123


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = pd.Series([1.0, 2.0, 3.0, 4.0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_compute_metrics_with_error(self):
        """Test metrics with some prediction error."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = pd.Series([1.5, 2.5, 3.5, 4.5])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == pytest.approx(0.5)
        assert metrics["rmse"] == pytest.approx(0.5)
        assert metrics["r2"] < 1.0

    def test_compute_metrics_returns_all_keys(self):
        """Test that all expected metric keys are returned."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 2.1, 3.1])

        metrics = compute_metrics(y_true, y_pred)

        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics


class TestRFBaselineCLI:
    """Tests for rf_baseline CLI command."""

    def _run_cli(
        self,
        temp_dir: Path,
        metadata_file: str,
        train_samples_file: str,
        val_samples_file: str,
        mock_table: MagicMock,
    ) -> tuple[Path, any]:
        """Run the rf_baseline CLI with standard arguments."""
        biom_file = temp_dir / "test.biom"
        biom_file.touch()
        output_file = temp_dir / "predictions.tsv"

        with patch("biom.load_table", return_value=mock_table):
            runner = CliRunner()
            result = runner.invoke(
                rf_baseline,
                [
                    "--table",
                    str(biom_file),
                    "--metadata",
                    metadata_file,
                    "--metadata-column",
                    "target",
                    "--train-samples",
                    train_samples_file,
                    "--val-samples",
                    val_samples_file,
                    "--output",
                    str(output_file),
                    "--n-estimators",
                    "10",
                ],
            )
        return output_file, result

    def test_cli_missing_required_args(self):
        """Test that CLI fails with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(rf_baseline, [])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_cli_with_all_required_args(self, temp_dir, metadata_file, train_samples_file, val_samples_file):
        """Test CLI with all required arguments."""
        sample_ids = ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"]
        mock_table = _create_mock_biom_table(sample_ids, ["ASV1", "ASV2", "ASV3"])

        output_file, result = self._run_cli(temp_dir, metadata_file, train_samples_file, val_samples_file, mock_table)

        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        assert output_file.exists()

    def test_cli_output_format(self, temp_dir, metadata_file, train_samples_file, val_samples_file):
        """Test that output file has correct format."""
        sample_ids = ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"]
        mock_table = _create_mock_biom_table(sample_ids, ["ASV1", "ASV2", "ASV3"])

        output_file, result = self._run_cli(temp_dir, metadata_file, train_samples_file, val_samples_file, mock_table)

        assert result.exit_code == 0
        df = pd.read_csv(output_file, sep="\t")
        assert list(df.columns) == ["sample_id", "actual", "predicted"]
        assert len(df) == 2

    def test_cli_prints_metrics(self, temp_dir, metadata_file, train_samples_file, val_samples_file):
        """Test that CLI prints metrics to console."""
        sample_ids = ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"]
        mock_table = _create_mock_biom_table(sample_ids, ["ASV1", "ASV2", "ASV3"])

        _, result = self._run_cli(temp_dir, metadata_file, train_samples_file, val_samples_file, mock_table)

        assert result.exit_code == 0
        assert "R²" in result.output or "R2" in result.output
        assert "MAE" in result.output
        assert "RMSE" in result.output

    def test_cli_warns_missing_samples(self, temp_dir, metadata_file, caplog):
        """Test CLI warns when samples not found in BIOM."""
        import logging

        train_file = temp_dir / "train.txt"
        train_file.write_text("sample1\nsample2\nmissing_sample\n")
        val_file = temp_dir / "val.txt"
        val_file.write_text("sample3\n")

        mock_table = _create_mock_biom_table(["sample1", "sample2", "sample3"], ["ASV1", "ASV2"])

        with caplog.at_level(logging.WARNING):
            _, result = self._run_cli(temp_dir, metadata_file, str(train_file), str(val_file), mock_table)

        assert "missing_sample" in caplog.text.lower() or "not found" in caplog.text.lower()
