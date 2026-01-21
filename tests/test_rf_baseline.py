"""Unit tests for Random Forest baseline script."""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from aam.cli.rf_baseline import (
    rf_baseline,
    load_sample_ids,
    load_biom_as_dataframe,
    train_random_forest,
    compute_metrics,
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


@pytest.fixture
def mock_biom_table():
    """Create a mock BIOM table."""
    mock_table = MagicMock()

    def mock_ids(axis=None):
        if axis == "sample":
            return np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"])
        elif axis == "observation":
            return np.array(["ASV1", "ASV2", "ASV3", "ASV4"])
        return np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"])

    mock_table.ids = mock_ids

    data = np.array(
        [
            [100, 200, 50, 150, 80, 120],
            [50, 100, 200, 80, 150, 90],
            [200, 50, 100, 120, 60, 180],
            [75, 150, 75, 90, 110, 70],
        ]
    )
    mock_table.matrix_data = MagicMock()
    mock_table.matrix_data.toarray.return_value = data
    return mock_table


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

    def test_cli_missing_required_args(self):
        """Test that CLI fails with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(rf_baseline, [])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_cli_with_all_required_args(self, temp_dir, metadata_file, train_samples_file, val_samples_file):
        """Test CLI with all required arguments."""
        biom_file = temp_dir / "test.biom"
        biom_file.touch()
        output_file = temp_dir / "predictions.tsv"

        mock_table = MagicMock()
        mock_table.ids = lambda axis=None: (
            np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"])
            if axis == "sample"
            else np.array(["ASV1", "ASV2", "ASV3"])
        )
        data = np.random.rand(3, 6) * 100
        mock_table.matrix_data = MagicMock()
        mock_table.matrix_data.toarray.return_value = data

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

            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert output_file.exists()

    def test_cli_output_format(self, temp_dir, metadata_file, train_samples_file, val_samples_file):
        """Test that output file has correct format."""
        biom_file = temp_dir / "test.biom"
        biom_file.touch()
        output_file = temp_dir / "predictions.tsv"

        mock_table = MagicMock()
        mock_table.ids = lambda axis=None: (
            np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"])
            if axis == "sample"
            else np.array(["ASV1", "ASV2", "ASV3"])
        )
        data = np.random.rand(3, 6) * 100
        mock_table.matrix_data = MagicMock()
        mock_table.matrix_data.toarray.return_value = data

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

            assert result.exit_code == 0
            df = pd.read_csv(output_file, sep="\t")
            assert "sample_id" in df.columns
            assert "actual" in df.columns
            assert "predicted" in df.columns
            assert len(df) == 2

    def test_cli_prints_metrics(self, temp_dir, metadata_file, train_samples_file, val_samples_file):
        """Test that CLI prints metrics to console."""
        biom_file = temp_dir / "test.biom"
        biom_file.touch()
        output_file = temp_dir / "predictions.tsv"

        mock_table = MagicMock()
        mock_table.ids = lambda axis=None: (
            np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"])
            if axis == "sample"
            else np.array(["ASV1", "ASV2", "ASV3"])
        )
        data = np.random.rand(3, 6) * 100
        mock_table.matrix_data = MagicMock()
        mock_table.matrix_data.toarray.return_value = data

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

            assert result.exit_code == 0
            assert "R²" in result.output or "R2" in result.output
            assert "MAE" in result.output
            assert "RMSE" in result.output

    def test_cli_warns_missing_samples(self, temp_dir, metadata_file):
        """Test CLI warns when samples not found in BIOM."""
        biom_file = temp_dir / "test.biom"
        biom_file.touch()
        output_file = temp_dir / "predictions.tsv"

        train_file = temp_dir / "train.txt"
        train_file.write_text("sample1\nsample2\nmissing_sample\n")
        val_file = temp_dir / "val.txt"
        val_file.write_text("sample3\n")

        mock_table = MagicMock()
        mock_table.ids = lambda axis=None: (
            np.array(["sample1", "sample2", "sample3"]) if axis == "sample" else np.array(["ASV1", "ASV2"])
        )
        data = np.random.rand(2, 3) * 100
        mock_table.matrix_data = MagicMock()
        mock_table.matrix_data.toarray.return_value = data

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
                    str(train_file),
                    "--val-samples",
                    str(val_file),
                    "--output",
                    str(output_file),
                    "--n-estimators",
                    "10",
                ],
            )

            assert "missing_sample" in result.output.lower() or "warning" in result.output.lower()
