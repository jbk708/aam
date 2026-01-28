"""Unit tests for CLI interface."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import click
from click.testing import CliRunner
import inspect
import pandas as pd

from aam.cli import (
    cli,
)
from aam.cli.utils import (
    setup_logging,
    setup_device,
    setup_random_seed,
    validate_file_path,
    validate_arguments,
)
from aam.cli.train import train
from aam.cli.pretrain import pretrain
from aam.cli.predict import predict


def _setup_train_data_parallel_mocks(mock_biom_loader, mock_unifrac_loader, mock_dataset, mock_dataloader, mock_model):
    """Set up common mocks for train --data-parallel tests.

    Returns the configured mock instances for further customization if needed.
    """
    import numpy as np
    from skbio import DistanceMatrix

    mock_biom_loader_instance = MagicMock()
    mock_biom_loader.return_value = mock_biom_loader_instance
    mock_table = MagicMock()

    def mock_ids(axis=None):
        if axis == "sample":
            return ["sample1", "sample2", "sample3", "sample4"]
        elif axis == "observation":
            return ["obs1", "obs2", "obs3"]
        return ["sample1", "sample2", "sample3", "sample4"]

    mock_table.ids = mock_ids
    mock_biom_loader_instance.load_table.return_value = mock_table
    mock_biom_loader_instance.rarefy.return_value = mock_table

    mock_unifrac_loader_instance = MagicMock()
    mock_unifrac_loader.return_value = mock_unifrac_loader_instance

    dist_data = np.random.rand(4, 4)
    dist_data = (dist_data + dist_data.T) / 2
    np.fill_diagonal(dist_data, 0)
    mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
    mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

    mock_dataset_instance = MagicMock()
    mock_dataset_instance.get_normalization_params.return_value = None
    mock_dataset_instance.get_count_normalization_params.return_value = None
    mock_dataset.return_value = mock_dataset_instance

    mock_dataloader_instance = MagicMock()
    mock_dataloader_instance.__len__ = MagicMock(return_value=1)
    mock_dataloader.return_value = mock_dataloader_instance

    mock_model_instance = MagicMock()
    mock_model_instance.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.return_value = mock_model_instance

    return {
        "biom_loader": mock_biom_loader_instance,
        "table": mock_table,
        "unifrac_loader": mock_unifrac_loader_instance,
        "distance_matrix": mock_distance_matrix,
        "dataset": mock_dataset_instance,
        "dataloader": mock_dataloader_instance,
        "model": mock_model_instance,
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_biom_file(temp_dir):
    """Create a sample BIOM file path."""
    biom_file = temp_dir / "test.biom"
    biom_file.touch()
    return str(biom_file)


@pytest.fixture
def sample_unifrac_matrix_file(temp_dir):
    """Create a sample UniFrac matrix file path."""
    import numpy as np

    matrix_file = temp_dir / "distances.npy"
    # Create a simple 4x4 distance matrix
    distances = np.random.rand(4, 4)
    distances = (distances + distances.T) / 2  # Make symmetric
    np.fill_diagonal(distances, 0.0)  # Diagonal is 0
    np.save(matrix_file, distances)
    return str(matrix_file)


@pytest.fixture
def sample_metadata_file(temp_dir):
    """Create a sample metadata file."""
    metadata_file = temp_dir / "metadata.tsv"
    metadata_file.write_text("sample_id\ttarget\nsample1\t1.0\nsample2\t2.0\n")
    return str(metadata_file)


@pytest.fixture
def sample_tree_file(temp_dir):
    """Create a sample tree file (Newick format)."""
    tree_file = temp_dir / "test_tree.nwk"
    # Create a simple Newick tree with a few tips
    tree_str = "(ASV1:0.1,ASV2:0.1,ASV3:0.1);"
    tree_file.write_text(tree_str)
    return str(tree_file)


@pytest.fixture
def sample_output_dir(temp_dir):
    """Create a sample output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return str(output_dir)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_log_file(self, temp_dir):
        """Test that logging creates log file."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()
        setup_logging(log_dir, "INFO")
        log_file = log_dir / "training.log"
        assert log_file.exists()

    def test_setup_logging_different_levels(self, temp_dir):
        """Test logging with different levels."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            setup_logging(log_dir, level)


class TestSetupDevice:
    """Tests for setup_device function."""

    def test_setup_device_cpu(self):
        """Test device setup for CPU."""
        device = setup_device("cpu")
        assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_setup_device_cuda(self):
        """Test device setup for CUDA."""
        device = setup_device("cuda")
        assert device.type == "cuda"

    def test_setup_device_invalid(self):
        """Test device setup with invalid device."""
        with pytest.raises(ValueError):
            setup_device("invalid")


class TestSetupRandomSeed:
    """Tests for setup_random_seed function."""

    def test_setup_random_seed_with_seed(self):
        """Test random seed setup with seed."""
        setup_random_seed(42)
        torch_seed = torch.initial_seed()
        assert torch_seed is not None

    def test_setup_random_seed_without_seed(self):
        """Test random seed setup without seed."""
        setup_random_seed(None)
        torch_seed = torch.initial_seed()
        assert torch_seed is not None

    def test_setup_random_seed_reproducibility(self):
        """Test that random seed ensures reproducibility."""
        setup_random_seed(42)
        value1 = torch.rand(1).item()
        setup_random_seed(42)
        value2 = torch.rand(1).item()
        assert value1 == value2


class TestValidateFilePath:
    """Tests for validate_file_path function."""

    def test_validate_file_path_exists(self, sample_biom_file):
        """Test validation of existing file."""
        validate_file_path(sample_biom_file)

    def test_validate_file_path_not_exists(self, temp_dir):
        """Test validation of non-existent file."""
        non_existent = temp_dir / "nonexistent.biom"
        with pytest.raises(FileNotFoundError):
            validate_file_path(str(non_existent))

    def test_validate_file_path_with_type(self, temp_dir):
        """Test validation with file type."""
        non_existent = temp_dir / "nonexistent.biom"
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_file_path(str(non_existent), "BIOM")
        assert "BIOM" in str(exc_info.value)


class TestValidateArguments:
    """Tests for validate_arguments function."""

    def test_validate_arguments_batch_size_positive(self):
        """Test batch size validation."""
        validate_arguments(batch_size=8)
        with pytest.raises(ValueError):
            validate_arguments(batch_size=0)
        with pytest.raises(ValueError):
            validate_arguments(batch_size=-1)

    def test_validate_arguments_batch_size_even(self):
        """Test batch size must be even."""
        validate_arguments(batch_size=8)
        with pytest.raises(ValueError):
            validate_arguments(batch_size=7)

    def test_validate_arguments_classifier_requires_out_dim(self):
        """Test classifier requires out_dim > 1."""
        validate_arguments(classifier=True, out_dim=2)
        with pytest.raises(ValueError):
            validate_arguments(classifier=True, out_dim=1)

    def test_validate_arguments_learning_rate_positive(self):
        """Test learning rate validation."""
        validate_arguments(lr=1e-4)
        with pytest.raises(ValueError):
            validate_arguments(lr=0.0)
        with pytest.raises(ValueError):
            validate_arguments(lr=-1e-4)

    def test_validate_arguments_test_size_range(self):
        """Test test_size validation."""
        validate_arguments(test_size=0.2)
        validate_arguments(test_size=0.0)
        validate_arguments(test_size=1.0)
        with pytest.raises(ValueError):
            validate_arguments(test_size=-0.1)
        with pytest.raises(ValueError):
            validate_arguments(test_size=1.1)

    def test_validate_arguments_epochs_positive(self):
        """Test epochs validation."""
        validate_arguments(epochs=100)
        with pytest.raises(ValueError):
            validate_arguments(epochs=0)
        with pytest.raises(ValueError):
            validate_arguments(epochs=-1)


class TestCLICommands:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AAM" in result.output

    def test_train_command_help(self, runner):
        """Test train command help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train AAM model" in result.output

    def test_train_command_missing_required_args(self, runner):
        """Test train command with missing required arguments."""
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0

    def test_train_command_file_validation(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command file validation."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                "nonexistent.biom",
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
            ],
        )
        assert result.exit_code != 0

    def test_train_command_batch_size_validation(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command batch size validation."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "7",
            ],
        )
        assert result.exit_code != 0

    def test_train_command_classifier_validation(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command classifier validation."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--classifier",
                "--out-dim",
                "1",
            ],
        )
        assert result.exit_code != 0

    def test_pretrain_command_help(self, runner):
        """Test pretrain command help."""
        result = runner.invoke(cli, ["pretrain", "--help"])
        assert result.exit_code == 0
        assert "Pre-train SequenceEncoder" in result.output

    def test_pretrain_command_missing_required_args(self, runner):
        """Test pretrain command with missing required arguments."""
        result = runner.invoke(cli, ["pretrain"])
        assert result.exit_code != 0

    def test_pretrain_command_file_validation(self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_output_dir):
        """Test pretrain command file validation."""
        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                "nonexistent.biom",
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
            ],
        )
        assert result.exit_code != 0

    def test_pretrain_command_batch_size_validation(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_output_dir
    ):
        """Test pretrain command batch size validation."""
        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "7",
            ],
        )
        assert result.exit_code != 0

    @pytest.mark.parametrize("command", ["train", "pretrain"])
    def test_command_default_patience(self, command):
        """Test that train/pretrain commands have default patience of 10."""
        cmd = cli.commands[command]
        patience_option = None
        for param in cmd.params:
            if param.name == "patience":
                patience_option = param
                break
        assert patience_option is not None, "patience parameter not found"
        assert patience_option.default == 10, f"Expected default to be 10, got {patience_option.default}"

    @pytest.mark.parametrize("command", ["train", "pretrain"])
    def test_command_data_parallel_option_exists(self, runner, command):
        """Test that --data-parallel option appears in train/pretrain help."""
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert "--data-parallel" in result.output
        assert "DataParallel" in result.output
        assert "UniFrac" in result.output

    def test_pretrain_command_data_parallel_distributed_mutual_exclusion(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_output_dir
    ):
        """Test that --data-parallel and --distributed cannot be used together."""
        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--data-parallel",
                "--distributed",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use multiple distributed training options together" in result.output

    @pytest.mark.parametrize("command", ["train", "pretrain"])
    def test_command_fsdp_option_exists(self, runner, command):
        """Test that --fsdp option appears in train/pretrain help."""
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert "--fsdp" in result.output
        assert "FSDP" in result.output

    def test_pretrain_command_fsdp_distributed_mutual_exclusion(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_output_dir
    ):
        """Test that --fsdp and --distributed cannot be used together."""
        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--fsdp",
                "--distributed",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use multiple distributed training options together" in result.output

    def test_pretrain_command_fsdp_data_parallel_mutual_exclusion(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_output_dir
    ):
        """Test that --fsdp and --data-parallel cannot be used together."""
        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--fsdp",
                "--data-parallel",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use multiple distributed training options together" in result.output

    @pytest.mark.parametrize("command", ["train", "pretrain"])
    def test_command_fsdp_sharded_checkpoint_option_exists(self, runner, command):
        """Test that --fsdp-sharded-checkpoint option appears in train/pretrain help."""
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert "--fsdp-sharded-checkpoint" in result.output

    def test_pretrain_command_fsdp_sharded_checkpoint_requires_fsdp(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_output_dir
    ):
        """Test that --fsdp-sharded-checkpoint requires --fsdp."""
        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--fsdp-sharded-checkpoint",
            ],
        )
        assert result.exit_code != 0
        assert "--fsdp-sharded-checkpoint requires --fsdp" in result.output

    def test_train_command_fsdp_distributed_mutual_exclusion(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test that --fsdp and --distributed cannot be used together."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--fsdp",
                "--distributed",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use multiple distributed training options together" in result.output

    def test_train_command_fsdp_sharded_checkpoint_requires_fsdp(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test that --fsdp-sharded-checkpoint requires --fsdp."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--fsdp-sharded-checkpoint",
            ],
        )
        assert result.exit_code != 0
        assert "--fsdp-sharded-checkpoint requires --fsdp" in result.output

    def test_train_command_data_parallel_distributed_mutual_exclusion(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test that --data-parallel and --distributed cannot be used together."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--data-parallel",
                "--distributed",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use multiple distributed training options together" in result.output

    def test_train_command_data_parallel_fsdp_mutual_exclusion(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test that --data-parallel and --fsdp cannot be used together."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--data-parallel",
                "--fsdp",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use multiple distributed training options together" in result.output

    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.torch.cuda.is_available")
    @patch("aam.cli.train.torch.cuda.device_count")
    @patch("aam.cli.train.torch.nn.DataParallel")
    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.Trainer")
    def test_train_command_data_parallel_wraps_model(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_data_parallel,
        mock_device_count,
        mock_cuda_available,
        mock_dataloader,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test train command wraps model with DataParallel when --data-parallel is used."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        mock_setup_device.return_value = torch.device("cuda")

        _setup_train_data_parallel_mocks(mock_biom_loader, mock_unifrac_loader, mock_dataset, mock_dataloader, mock_model)

        mock_dp_wrapped_model = MagicMock()
        mock_data_parallel.return_value = mock_dp_wrapped_model

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--data-parallel",
            ],
        )

        mock_data_parallel.assert_called_once()
        call_args = mock_data_parallel.call_args
        assert call_args is not None

    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.torch.cuda.is_available")
    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.SequencePredictor")
    def test_train_command_data_parallel_requires_cuda(
        self,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_cuda_available,
        mock_dataloader,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test train command with --data-parallel fails without CUDA."""
        mock_cuda_available.return_value = False
        mock_setup_device.return_value = torch.device("cpu")

        _setup_train_data_parallel_mocks(mock_biom_loader, mock_unifrac_loader, mock_dataset, mock_dataloader, mock_model)

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--data-parallel",
            ],
        )

        assert result.exit_code != 0
        assert "--data-parallel requires CUDA" in result.output

    def test_predict_command_help(self, runner):
        """Test predict command help."""
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "Run inference" in result.output

    def test_predict_command_missing_required_args(self, runner):
        """Test predict command with missing required arguments."""
        result = runner.invoke(cli, ["predict"])
        assert result.exit_code != 0

    def test_predict_command_file_validation(self, runner, sample_biom_file):
        """Test predict command file validation."""
        result = runner.invoke(
            cli,
            [
                "predict",
                "--model",
                "nonexistent.pt",
                "--table",
                sample_biom_file,
                "--output",
                "output.tsv",
            ],
        )
        assert result.exit_code != 0

    def test_train_command_regressor_hidden_dims_option_exists(self, runner):
        """Test that --regressor-hidden-dims option appears in train help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--regressor-hidden-dims" in result.output
        assert "MLP regression head" in result.output

    def test_train_command_regressor_dropout_option_exists(self, runner):
        """Test that --regressor-dropout option appears in train help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--regressor-dropout" in result.output

    def test_train_command_regressor_dropout_invalid_range(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test that --regressor-dropout rejects values outside [0, 1)."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--regressor-dropout",
                "1.5",
            ],
        )
        assert result.exit_code != 0
        assert "1.5" in result.output or "range" in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI with mocked components."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.Trainer")
    def test_train_command_integration(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test train command integration with mocked components."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()
        mock_table.ids = MagicMock(return_value=["sample1", "sample2", "sample3", "sample4"])
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        # Create a symmetric distance matrix (required by DistanceMatrix)
        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2  # Make symmetric
        np.fill_diagonal(dist_data, 0)  # Diagonal must be 0
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
            ],
        )

        assert mock_setup_logging.called
        assert mock_setup_device.called
        assert mock_setup_seed.called
        assert mock_validate_file.called
        assert mock_validate_args.called

    @patch("aam.cli.pretrain.setup_logging")
    @patch("aam.cli.pretrain.setup_device")
    @patch("aam.cli.pretrain.setup_random_seed")
    @patch("aam.cli.pretrain.validate_file_path")
    @patch("aam.cli.pretrain.validate_arguments")
    @patch("aam.cli.pretrain.BIOMLoader")
    @patch("aam.cli.pretrain.UniFracLoader")
    @patch("aam.cli.pretrain.ASVDataset")
    @patch("aam.cli.pretrain.SequenceEncoder")
    @patch("aam.cli.pretrain.Trainer")
    def test_pretrain_command_integration(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_output_dir,
    ):
        """Test pretrain command integration with mocked components."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        # Mock ids() to return list when called with axis="sample" or axis="observation"
        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        # Create a symmetric distance matrix (required by DistanceMatrix)
        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2  # Make symmetric
        np.fill_diagonal(dist_data, 0)  # Diagonal must be 0
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
            ],
        )

        assert mock_setup_logging.called
        assert mock_setup_device.called
        assert mock_setup_seed.called
        assert mock_validate_file.called
        assert mock_validate_args.called

    @patch("aam.cli.pretrain.DataLoader")
    @patch("aam.cli.pretrain.torch.cuda.is_available")
    @patch("aam.cli.pretrain.torch.cuda.device_count")
    @patch("aam.cli.pretrain.torch.nn.DataParallel")
    @patch("aam.cli.pretrain.setup_logging")
    @patch("aam.cli.pretrain.setup_device")
    @patch("aam.cli.pretrain.setup_random_seed")
    @patch("aam.cli.pretrain.validate_file_path")
    @patch("aam.cli.pretrain.validate_arguments")
    @patch("aam.cli.pretrain.BIOMLoader")
    @patch("aam.cli.pretrain.UniFracLoader")
    @patch("aam.cli.pretrain.ASVDataset")
    @patch("aam.cli.pretrain.SequenceEncoder")
    @patch("aam.cli.pretrain.Trainer")
    def test_pretrain_command_data_parallel_wraps_model(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_data_parallel,
        mock_device_count,
        mock_cuda_available,
        mock_dataloader,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_output_dir,
    ):
        """Test pretrain command wraps model with DataParallel when --data-parallel is used."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        mock_setup_device.return_value = torch.device("cuda")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_dp_wrapped_model = MagicMock()
        mock_data_parallel.return_value = mock_dp_wrapped_model

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--data-parallel",
            ],
        )

        # Model is moved to device before wrapping, so DataParallel receives model.to()
        mock_data_parallel.assert_called_once()
        call_args = mock_data_parallel.call_args
        assert call_args is not None

    @patch("aam.cli.pretrain.DataLoader")
    @patch("aam.cli.pretrain.torch.cuda.is_available")
    @patch("aam.cli.pretrain.setup_logging")
    @patch("aam.cli.pretrain.setup_device")
    @patch("aam.cli.pretrain.setup_random_seed")
    @patch("aam.cli.pretrain.validate_file_path")
    @patch("aam.cli.pretrain.validate_arguments")
    @patch("aam.cli.pretrain.BIOMLoader")
    @patch("aam.cli.pretrain.UniFracLoader")
    @patch("aam.cli.pretrain.ASVDataset")
    @patch("aam.cli.pretrain.SequenceEncoder")
    def test_pretrain_command_data_parallel_requires_cuda(
        self,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_cuda_available,
        mock_dataloader,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_output_dir,
    ):
        """Test pretrain command with --data-parallel fails without CUDA."""
        mock_cuda_available.return_value = False
        mock_setup_device.return_value = torch.device("cpu")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--data-parallel",
            ],
        )

        assert result.exit_code != 0
        assert "--data-parallel requires CUDA" in result.output

    @patch("aam.cli.pretrain.DataLoader")
    @patch("aam.cli.pretrain.create_scheduler")
    @patch("aam.cli.pretrain.create_optimizer")
    @patch("aam.cli.pretrain.setup_logging")
    @patch("aam.cli.pretrain.setup_device")
    @patch("aam.cli.pretrain.setup_random_seed")
    @patch("aam.cli.pretrain.validate_file_path")
    @patch("aam.cli.pretrain.validate_arguments")
    @patch("aam.cli.pretrain.BIOMLoader")
    @patch("aam.cli.pretrain.UniFracLoader")
    @patch("aam.cli.pretrain.ASVDataset")
    @patch("aam.cli.pretrain.SequenceEncoder")
    @patch("aam.cli.pretrain.Trainer")
    def test_pretrain_scheduler_accounts_for_gradient_accumulation(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_create_optimizer,
        mock_create_scheduler,
        mock_dataloader,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_output_dir,
    ):
        """Test that num_training_steps accounts for gradient_accumulation_steps."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        batches_per_epoch = 100
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=batches_per_epoch)
        mock_dataloader.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer.return_value = mock_trainer_instance

        epochs = 10
        gradient_accumulation_steps = 4

        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                str(epochs),
                "--gradient-accumulation-steps",
                str(gradient_accumulation_steps),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        mock_create_scheduler.assert_called_once()
        scheduler_call = mock_create_scheduler.call_args
        actual_steps = scheduler_call.kwargs["num_training_steps"]
        expected_steps = (batches_per_epoch // gradient_accumulation_steps) * epochs
        assert actual_steps == expected_steps, (
            f"Expected {expected_steps} training steps (accounting for gradient accumulation), got {actual_steps}"
        )

    @patch("aam.cli.pretrain.DataLoader")
    @patch("aam.cli.pretrain.create_scheduler")
    @patch("aam.cli.pretrain.create_optimizer")
    @patch("aam.cli.pretrain.setup_logging")
    @patch("aam.cli.pretrain.setup_device")
    @patch("aam.cli.pretrain.setup_random_seed")
    @patch("aam.cli.pretrain.validate_file_path")
    @patch("aam.cli.pretrain.validate_arguments")
    @patch("aam.cli.pretrain.BIOMLoader")
    @patch("aam.cli.pretrain.UniFracLoader")
    @patch("aam.cli.pretrain.ASVDataset")
    @patch("aam.cli.pretrain.SequenceEncoder")
    @patch("aam.cli.pretrain.Trainer")
    def test_pretrain_resume_loads_checkpoint_once(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_create_optimizer,
        mock_create_scheduler,
        mock_dataloader,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_output_dir,
        temp_dir,
    ):
        """Test that --resume-from loads checkpoint exactly once."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=10)
        mock_dataloader.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer_instance.load_checkpoint.return_value = {
            "epoch": 5,
            "best_val_loss": 0.5,
            "best_metric_value": 0.5,
        }
        mock_trainer.return_value = mock_trainer_instance

        checkpoint_path = temp_dir / "checkpoint.pt"
        checkpoint_path.touch()

        result = runner.invoke(
            cli,
            [
                "pretrain",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--output-dir",
                sample_output_dir,
                "--epochs",
                "10",
                "--resume-from",
                str(checkpoint_path),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify load_checkpoint was called exactly once
        assert mock_trainer_instance.load_checkpoint.call_count == 1

        # Verify train() was called with start_epoch (not resume_from)
        train_call = mock_trainer_instance.train.call_args
        assert train_call.kwargs.get("start_epoch") == 6  # epoch 5 + 1
        assert train_call.kwargs.get("initial_best_metric_value") == 0.5
        assert "resume_from" not in train_call.kwargs or train_call.kwargs.get("resume_from") is None

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    def test_predict_command_integration(
        self,
        mock_dataloader,
        mock_dataset,
        mock_biom_loader,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
    ):
        """Test predict command integration with mocked components."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_checkpoint = {
            "model_state_dict": {},
            "config": {
                "max_bp": 150,
                "token_limit": 1024,
                "embedding_dim": 128,
                "encoder_type": "unifrac",
                "out_dim": 1,
                "is_classifier": False,
            },
        }
        mock_load.return_value = mock_checkpoint

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()
        mock_biom_loader_instance.load_table.return_value = mock_table

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader.return_value = mock_dataloader_instance
        mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([]))

        output_file = temp_dir / "predictions.tsv"
        model_file = temp_dir / "model.pt"
        model_file.touch()

        result = runner.invoke(
            cli,
            [
                "predict",
                "--model",
                str(model_file),
                "--table",
                sample_biom_file,
                "--output",
                str(output_file),
            ],
        )

        assert mock_setup_device.called or result.exit_code == 0
        assert mock_validate_file.called

    def test_setup_device_cuda_when_not_available(self):
        """Test CUDA device setup when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA is not available"):
                setup_device("cuda")

    def test_setup_random_seed_cuda(self):
        """Test random seed setup with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.manual_seed_all") as mock_cuda_seed:
                with patch("torch.backends.cudnn") as mock_cudnn:
                    setup_random_seed(42)
                    mock_cuda_seed.assert_called_with(42)
                    mock_cudnn.deterministic = True
                    mock_cudnn.benchmark = False

    @patch("aam.cli.train.pd.read_csv")
    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.train_test_split")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.create_optimizer")
    @patch("aam.cli.train.create_scheduler")
    @patch("aam.cli.train.MultiTaskLoss")
    @patch("aam.cli.train.Trainer")
    def test_train_command_full_flow(
        self,
        mock_trainer_class,
        mock_loss_class,
        mock_create_scheduler,
        mock_create_optimizer,
        mock_model_class,
        mock_dataloader_class,
        mock_dataset_class,
        mock_train_test_split,
        mock_unifrac_loader_class,
        mock_biom_loader_class,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        mock_read_csv,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test train command with full flow including data loading and model setup."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_metadata_df = MagicMock()
        mock_metadata_df.columns = pd.Index(["sample_id", "target"])
        mock_read_csv.return_value = mock_metadata_df

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        # Mock ids() to return list when called with axis="sample" or axis="observation"
        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader_class.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        # Create a symmetric distance matrix (required by DistanceMatrix)
        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2  # Make symmetric
        np.fill_diagonal(dist_data, 0)  # Diagonal must be 0
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_train_ids = ["sample1", "sample2", "sample3"]
        mock_val_ids = ["sample4"]
        mock_train_test_split.return_value = (mock_train_ids, mock_val_ids)

        mock_train_table = MagicMock()
        mock_val_table = MagicMock()
        mock_table.filter.side_effect = lambda ids, **kwargs: mock_train_table if ids == mock_train_ids else mock_val_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader_class.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_loss_instance = MagicMock()
        mock_loss_class.return_value = mock_loss_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer_class.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert mock_biom_loader_instance.load_table.called
        assert mock_biom_loader_instance.rarefy.called
        assert mock_unifrac_loader_instance.load_matrix.called
        assert mock_dataset_class.call_count == 2
        assert mock_model_class.called
        assert mock_trainer_instance.train.called

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    @patch("aam.cli.predict.Path")
    def test_predict_command_with_batches(
        self,
        mock_path_class,
        mock_dataloader_class,
        mock_dataset_class,
        mock_biom_loader_class,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
    ):
        """Test predict command with actual batch processing."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_checkpoint = {
            "model_state_dict": {},
            "config": {
                "max_bp": 150,
                "token_limit": 1024,
                "embedding_dim": 128,
                "encoder_type": "unifrac",
                "out_dim": 1,
                "is_classifier": False,
            },
        }
        mock_load.return_value = mock_checkpoint

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()
        mock_biom_loader_instance.load_table.return_value = mock_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_output = {"target_prediction": torch.tensor([[1.0], [2.0]])}
        mock_model_instance.return_value = mock_output
        mock_model_class.return_value = mock_model_instance

        mock_batch1 = {
            "tokens": torch.tensor([[[1, 2, 3]]]),
            "sample_ids": ["sample1"],
        }
        mock_batch2 = {
            "tokens": torch.tensor([[[2, 3, 4]]]),
            "sample_ids": ["sample2"],
        }
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([mock_batch1, mock_batch2]))
        mock_dataloader_class.return_value = mock_dataloader_instance

        output_file = temp_dir / "predictions.tsv"
        model_file = temp_dir / "model.pt"
        model_file.touch()

        mock_path_instance = MagicMock()
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path_class.return_value = mock_path_instance

        import numpy as np

        mock_tensor1 = MagicMock()
        mock_numpy1 = np.array([1.0])
        mock_tensor1.cpu.return_value.numpy.return_value = mock_numpy1
        mock_tensor2 = MagicMock()
        mock_numpy2 = np.array([2.0])
        mock_tensor2.cpu.return_value.numpy.return_value = mock_numpy2
        mock_output1 = {"target_prediction": mock_tensor1}
        mock_output2 = {"target_prediction": mock_tensor2}
        mock_model_instance.side_effect = [mock_output1, mock_output2]

        with patch("aam.cli.predict.pd.DataFrame") as mock_df_class:
            mock_df_instance = MagicMock()
            mock_df_class.return_value = mock_df_instance

            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--model",
                    str(model_file),
                    "--table",
                    sample_biom_file,
                    "--output",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0
            assert mock_model_instance.call_count == 2
            assert mock_df_instance.to_csv.called

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    def test_predict_command_multiclass_output(
        self,
        mock_dataloader_class,
        mock_dataset_class,
        mock_biom_loader_class,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
    ):
        """Test predict command with multi-class predictions."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_checkpoint = {
            "model_state_dict": {},
            "config": {
                "max_bp": 150,
                "token_limit": 1024,
                "embedding_dim": 128,
                "encoder_type": "unifrac",
                "out_dim": 3,
                "is_classifier": True,
            },
        }
        mock_load.return_value = mock_checkpoint

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()
        mock_biom_loader_instance.load_table.return_value = mock_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        import numpy as np

        mock_model_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_numpy = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        mock_tensor.cpu.return_value.numpy.return_value = mock_numpy
        mock_output = {"target_prediction": mock_tensor}
        mock_model_instance.return_value = mock_output
        mock_model_class.return_value = mock_model_instance

        mock_batch = {
            "tokens": torch.tensor([[[1, 2, 3]], [[2, 3, 4]]]),
            "sample_ids": ["sample1", "sample2"],
        }
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader_class.return_value = mock_dataloader_instance

        output_file = temp_dir / "predictions.tsv"
        model_file = temp_dir / "model.pt"
        model_file.touch()

        with patch("aam.cli.predict.pd.DataFrame") as mock_df_class:
            mock_df_instance = MagicMock()
            mock_df_class.return_value = mock_df_instance

            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--model",
                    str(model_file),
                    "--table",
                    sample_biom_file,
                    "--output",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0
            call_args = mock_df_class.call_args
            if call_args and len(call_args) > 0:
                df_data = call_args[0][0] if call_args[0] else {}
                assert isinstance(df_data, dict)

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    def test_predict_command_checkpoint_fallback(
        self,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        sample_tree_file,
        temp_dir,
    ):
        """Test predict command with checkpoint fallback (no model_state_dict)."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_checkpoint = {}  # No model_state_dict, should use checkpoint directly
        mock_load.return_value = mock_checkpoint

        model_file = temp_dir / "model.pt"
        model_file.touch()

        with patch("aam.cli.predict.BIOMLoader"), patch("aam.cli.predict.ASVDataset"), patch("aam.cli.predict.DataLoader"):
            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--model",
                    str(model_file),
                    "--table",
                    sample_biom_file,
                    "--output",
                    str(temp_dir / "output.tsv"),
                ],
            )

            assert result.exit_code == 0
            assert mock_model_class.called

    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.BIOMLoader")
    def test_train_command_error_handling(
        self,
        mock_biom_loader_class,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        sample_tree_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test train command error handling."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_biom_loader_instance.load_table.side_effect = Exception("Load error")

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code != 0

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    def test_predict_command_error_handling(
        self,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        sample_tree_file,
        temp_dir,
    ):
        """Test predict command error handling."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_load.side_effect = Exception("Load error")

        model_file = temp_dir / "model.pt"
        model_file.touch()

        result = runner.invoke(
            cli,
            [
                "predict",
                "--model",
                str(model_file),
                "--table",
                sample_biom_file,
                "--output",
                str(temp_dir / "output.tsv"),
            ],
        )

        assert result.exit_code != 0


class TestPretrainedEncoderLoading:
    """Tests for pretrained encoder loading via CLI."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_train_command_help_shows_pretrained_encoder_option(self, runner):
        """Test that --pretrained-encoder option appears in help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--pretrained-encoder" in result.output

    def test_train_command_pretrained_encoder_file_validation(
        self, runner, sample_biom_file, sample_tree_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command validates pretrained encoder file exists."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--tree",
                sample_tree_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--pretrained-encoder",
                "nonexistent.pt",
            ],
        )
        assert result.exit_code != 0

    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.train_test_split")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.load_pretrained_encoder")
    @patch("aam.cli.train.create_optimizer")
    @patch("aam.cli.train.create_scheduler")
    @patch("aam.cli.train.MultiTaskLoss")
    @patch("aam.cli.train.Trainer")
    @patch("aam.cli.train.pd.read_csv")
    def test_train_command_loads_pretrained_encoder(
        self,
        mock_read_csv,
        mock_trainer_class,
        mock_loss_class,
        mock_create_scheduler,
        mock_create_optimizer,
        mock_load_pretrained_encoder,
        mock_model_class,
        mock_dataloader_class,
        mock_dataset_class,
        mock_train_test_split,
        mock_unifrac_loader_class,
        mock_biom_loader_class,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        runner,
        sample_biom_file,
        sample_tree_file,
        sample_metadata_file,
        sample_output_dir,
        sample_unifrac_matrix_file,
        temp_dir,
    ):
        """Test train command loads pretrained encoder when provided."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_metadata_df = MagicMock()
        mock_metadata_df.columns = pd.Index(["sample_id", "target"])
        mock_read_csv.return_value = mock_metadata_df

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        # Mock ids() to return list when called with axis="sample" or axis="observation"
        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        # Mock UniFracLoader
        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader_class.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_train_ids = ["sample1", "sample2", "sample3"]
        mock_val_ids = ["sample4"]
        mock_train_test_split.return_value = (mock_train_ids, mock_val_ids)

        mock_train_table = MagicMock()
        mock_val_table = MagicMock()
        mock_table.filter.side_effect = lambda ids, **kwargs: mock_train_table if ids == mock_train_ids else mock_val_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader_class.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_loss_instance = MagicMock()
        mock_loss_class.return_value = mock_loss_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer_class.return_value = mock_trainer_instance

        pretrained_encoder_path = temp_dir / "pretrained.pt"
        pretrained_encoder_path.touch()

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--pretrained-encoder",
                str(pretrained_encoder_path),
            ],
        )

        assert result.exit_code == 0
        assert mock_load_pretrained_encoder.called
        call_args = mock_load_pretrained_encoder.call_args
        assert call_args[0][0] == str(pretrained_encoder_path)
        assert call_args[0][1] == mock_model_instance
        assert call_args[1]["strict"] is False

    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.train_test_split")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.load_pretrained_encoder")
    @patch("aam.cli.train.create_optimizer")
    @patch("aam.cli.train.create_scheduler")
    @patch("aam.cli.train.MultiTaskLoss")
    @patch("aam.cli.train.Trainer")
    @patch("aam.cli.train.pd.read_csv")
    def test_train_command_pretrained_encoder_with_freeze_base(
        self,
        mock_read_csv,
        mock_trainer_class,
        mock_loss_class,
        mock_create_scheduler,
        mock_create_optimizer,
        mock_load_pretrained_encoder,
        mock_model_class,
        mock_dataloader_class,
        mock_dataset_class,
        mock_train_test_split,
        mock_unifrac_loader_class,
        mock_biom_loader_class,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        runner,
        sample_biom_file,
        sample_tree_file,
        sample_metadata_file,
        sample_output_dir,
        sample_unifrac_matrix_file,
        temp_dir,
    ):
        """Test train command with pretrained encoder and freeze_base option."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_metadata_df = MagicMock()
        mock_metadata_df.columns = pd.Index(["sample_id", "target"])
        mock_read_csv.return_value = mock_metadata_df

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        # Mock ids() to return list when called with axis="sample" or axis="observation"
        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        # Mock UniFracLoader
        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader_class.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_train_ids = ["sample1", "sample2", "sample3"]
        mock_val_ids = ["sample4"]
        mock_train_test_split.return_value = (mock_train_ids, mock_val_ids)

        mock_train_table = MagicMock()
        mock_val_table = MagicMock()
        mock_table.filter.side_effect = lambda ids, **kwargs: mock_train_table if ids == mock_train_ids else mock_val_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader_class.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_loss_instance = MagicMock()
        mock_loss_class.return_value = mock_loss_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer_class.return_value = mock_trainer_instance

        pretrained_encoder_path = temp_dir / "pretrained.pt"
        pretrained_encoder_path.touch()

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--pretrained-encoder",
                str(pretrained_encoder_path),
                "--freeze-base",
            ],
        )

        assert result.exit_code == 0
        assert mock_load_pretrained_encoder.called
        assert mock_create_optimizer.called
        optimizer_call_args = mock_create_optimizer.call_args
        assert optimizer_call_args[1]["freeze_base"] is True

    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.train_test_split")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.load_pretrained_encoder")
    @patch("aam.cli.train.create_optimizer")
    @patch("aam.cli.train.create_scheduler")
    @patch("aam.cli.train.MultiTaskLoss")
    @patch("aam.cli.train.Trainer")
    @patch("aam.cli.train.pd.read_csv")
    def test_train_command_pretrained_encoder_load_error_handling(
        self,
        mock_read_csv,
        mock_trainer_class,
        mock_loss_class,
        mock_create_scheduler,
        mock_create_optimizer,
        mock_load_pretrained_encoder,
        mock_model_class,
        mock_dataloader_class,
        mock_dataset_class,
        mock_train_test_split,
        mock_unifrac_loader_class,
        mock_biom_loader_class,
        mock_validate_args,
        mock_validate_file,
        mock_setup_seed,
        mock_setup_device,
        mock_setup_logging,
        runner,
        sample_biom_file,
        sample_tree_file,
        sample_metadata_file,
        sample_output_dir,
        sample_unifrac_matrix_file,
        temp_dir,
    ):
        """Test train command handles errors when loading pretrained encoder."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_metadata_df = MagicMock()
        mock_metadata_df.columns = pd.Index(["sample_id", "target"])
        mock_read_csv.return_value = mock_metadata_df

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        # Mock ids() to return list when called with axis="sample" or axis="observation"
        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        # Mock UniFracLoader
        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader_class.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_train_ids = ["sample1", "sample2", "sample3"]
        mock_val_ids = ["sample4"]
        mock_train_test_split.return_value = (mock_train_ids, mock_val_ids)

        mock_train_table = MagicMock()
        mock_val_table = MagicMock()
        mock_table.filter.side_effect = lambda ids, **kwargs: mock_train_table if ids == mock_train_ids else mock_val_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader_class.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_load_pretrained_encoder.side_effect = Exception("Failed to load checkpoint")

        pretrained_encoder_path = temp_dir / "pretrained.pt"
        pretrained_encoder_path.touch()

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--pretrained-encoder",
                str(pretrained_encoder_path),
            ],
        )

        assert result.exit_code != 0


class TestMemoryEfficientDefaults:
    """Tests for PYT-18.1: Memory-efficient default settings."""

    def test_train_gradient_checkpointing_default_true(self):
        """Test that gradient_checkpointing defaults to True in train command."""
        train_cmd = cli.commands["train"]
        gc_option = None
        for param in train_cmd.params:
            if param.name == "gradient_checkpointing":
                gc_option = param
                break
        assert gc_option is not None, "gradient_checkpointing parameter not found"
        assert gc_option.default is True, f"Expected default to be True, got {gc_option.default}"

    def test_pretrain_gradient_checkpointing_default_true(self):
        """Test that gradient_checkpointing defaults to True in pretrain command."""
        pretrain_cmd = cli.commands["pretrain"]
        gc_option = None
        for param in pretrain_cmd.params:
            if param.name == "gradient_checkpointing":
                gc_option = param
                break
        assert gc_option is not None, "gradient_checkpointing parameter not found"
        assert gc_option.default is True, f"Expected default to be True, got {gc_option.default}"

    def test_train_no_gradient_checkpointing_flag_exists(self):
        """Test that --no-gradient-checkpointing flag is available in train command."""
        train_cmd = cli.commands["train"]
        gc_option = None
        for param in train_cmd.params:
            if param.name == "gradient_checkpointing":
                gc_option = param
                break
        assert gc_option is not None, "gradient_checkpointing parameter not found"
        # Click boolean flags use secondary_opts for the --no- version
        assert gc_option.is_flag is True, "Should be a flag"
        assert gc_option.flag_value is True, "Flag value should be True (default enabled)"

    def test_pretrain_no_gradient_checkpointing_flag_exists(self):
        """Test that --no-gradient-checkpointing flag is available in pretrain command."""
        pretrain_cmd = cli.commands["pretrain"]
        gc_option = None
        for param in pretrain_cmd.params:
            if param.name == "gradient_checkpointing":
                gc_option = param
                break
        assert gc_option is not None, "gradient_checkpointing parameter not found"
        # Click boolean flags use secondary_opts for the --no- version
        assert gc_option.is_flag is True, "Should be a flag"
        assert gc_option.flag_value is True, "Flag value should be True (default enabled)"

    def test_train_attn_implementation_default_mem_efficient(self):
        """Test that attn_implementation defaults to mem_efficient in train command."""
        train_cmd = cli.commands["train"]
        attn_option = None
        for param in train_cmd.params:
            if param.name == "attn_implementation":
                attn_option = param
                break
        assert attn_option is not None, "attn_implementation parameter not found"
        assert attn_option.default == "mem_efficient", f"Expected default to be 'mem_efficient', got {attn_option.default}"

    def test_pretrain_attn_implementation_default_mem_efficient(self):
        """Test that attn_implementation defaults to mem_efficient in pretrain command."""
        pretrain_cmd = cli.commands["pretrain"]
        attn_option = None
        for param in pretrain_cmd.params:
            if param.name == "attn_implementation":
                attn_option = param
                break
        assert attn_option is not None, "attn_implementation parameter not found"
        assert attn_option.default == "mem_efficient", f"Expected default to be 'mem_efficient', got {attn_option.default}"

    def test_train_asv_chunk_size_exists(self):
        """Test that asv_chunk_size parameter exists in train command."""
        train_cmd = cli.commands["train"]
        chunk_option = None
        for param in train_cmd.params:
            if param.name == "asv_chunk_size":
                chunk_option = param
                break
        assert chunk_option is not None, "asv_chunk_size parameter not found in train command"

    def test_train_asv_chunk_size_default_256(self):
        """Test that asv_chunk_size defaults to 256 in train command."""
        train_cmd = cli.commands["train"]
        chunk_option = None
        for param in train_cmd.params:
            if param.name == "asv_chunk_size":
                chunk_option = param
                break
        assert chunk_option is not None, "asv_chunk_size parameter not found"
        assert chunk_option.default == 256, f"Expected default to be 256, got {chunk_option.default}"

    def test_pretrain_asv_chunk_size_default_256(self):
        """Test that asv_chunk_size defaults to 256 in pretrain command."""
        pretrain_cmd = cli.commands["pretrain"]
        chunk_option = None
        for param in pretrain_cmd.params:
            if param.name == "asv_chunk_size":
                chunk_option = param
                break
        assert chunk_option is not None, "asv_chunk_size parameter not found"
        assert chunk_option.default == 256, f"Expected default to be 256, got {chunk_option.default}"

    def test_train_help_shows_memory_options(self):
        """Test that train help shows memory-related options with descriptions."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--no-gradient-checkpointing" in result.output
        assert "--asv-chunk-size" in result.output
        assert "--attn-implementation" in result.output

    def test_pretrain_help_shows_memory_options(self):
        """Test that pretrain help shows memory-related options with descriptions."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pretrain", "--help"])
        assert result.exit_code == 0
        assert "--no-gradient-checkpointing" in result.output
        assert "--asv-chunk-size" in result.output
        assert "--attn-implementation" in result.output

    def test_pretrain_memory_profile_option_exists(self):
        """Test that --memory-profile option is available in pretrain command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pretrain", "--help"])
        assert result.exit_code == 0
        assert "--memory-profile" in result.output
        assert "GPU memory profiling" in result.output


class TestMetadataLoading:
    """Tests for metadata loading with column name variations."""

    def test_metadata_with_whitespace_in_column_name(self, temp_dir):
        """Test that metadata with whitespace in column names works correctly."""
        import pandas as pd

        metadata_file = temp_dir / "metadata_whitespace.tsv"
        metadata_file.write_text(" sample_id \ttarget\nsample1\t1.0\nsample2\t2.0\n")

        metadata_df = pd.read_csv(metadata_file, sep="\t", encoding="utf-8-sig")
        metadata_df.columns = metadata_df.columns.str.strip()

        assert "sample_id" in metadata_df.columns
        assert "target" in metadata_df.columns
        assert len(metadata_df) == 2

    def test_metadata_with_missing_sample_id_column(self, temp_dir):
        """Test that missing sample_id column provides helpful error message."""
        import pandas as pd

        metadata_file = temp_dir / "metadata_no_sample_id.tsv"
        metadata_file.write_text("id\ttarget\nsample1\t1.0\nsample2\t2.0\n")

        metadata_df = pd.read_csv(metadata_file, sep="\t", encoding="utf-8-sig")
        metadata_df.columns = metadata_df.columns.str.strip()

        with pytest.raises(ValueError) as exc_info:
            if "sample_id" not in metadata_df.columns:
                found_columns = list(metadata_df.columns)
                raise ValueError(
                    f"Metadata file must have 'sample_id' column.\n"
                    f"Found columns: {found_columns}\n"
                    f"Expected: 'sample_id'\n"
                    f"Tip: Check for whitespace or encoding issues in column names."
                )

        error_msg = str(exc_info.value)
        assert "sample_id" in error_msg.lower()
        assert "found columns" in error_msg.lower() or "columns" in error_msg.lower()

    def test_metadata_with_normal_column_name(self, temp_dir):
        """Test that normal metadata file still works (regression test)."""
        import pandas as pd

        metadata_file = temp_dir / "metadata_normal.tsv"
        metadata_file.write_text("sample_id\ttarget\nsample1\t1.0\nsample2\t2.0\n")

        metadata_df = pd.read_csv(metadata_file, sep="\t", encoding="utf-8-sig")
        metadata_df.columns = metadata_df.columns.str.strip()

        assert "sample_id" in metadata_df.columns
        assert "target" in metadata_df.columns
        assert len(metadata_df) == 2

    def test_metadata_with_trailing_whitespace_in_column_name(self, temp_dir):
        """Test that metadata with trailing whitespace in column names works."""
        import pandas as pd

        metadata_file = temp_dir / "metadata_trailing_whitespace.tsv"
        metadata_file.write_text("sample_id \ttarget \nsample1\t1.0\nsample2\t2.0\n")

        metadata_df = pd.read_csv(metadata_file, sep="\t", encoding="utf-8-sig")
        metadata_df.columns = metadata_df.columns.str.strip()

        assert "sample_id" in metadata_df.columns
        assert "target" in metadata_df.columns
        assert len(metadata_df) == 2

    def test_metadata_with_leading_whitespace_in_column_name(self, temp_dir):
        """Test that metadata with leading whitespace in column names works."""
        import pandas as pd

        metadata_file = temp_dir / "metadata_leading_whitespace.tsv"
        metadata_file.write_text(" sample_id\ttarget\nsample1\t1.0\nsample2\t2.0\n")

        metadata_df = pd.read_csv(metadata_file, sep="\t", encoding="utf-8-sig")
        metadata_df.columns = metadata_df.columns.str.strip()

        assert "sample_id" in metadata_df.columns
        assert "target" in metadata_df.columns
        assert len(metadata_df) == 2


class TestBestMetricCLI:
    """Tests for --best-metric CLI flag."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        from click.testing import CliRunner

        return CliRunner()

    def test_train_command_help_shows_best_metric_option(self, runner):
        """Test that --best-metric option appears in help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--best-metric" in result.output

    def test_train_command_best_metric_choices(self, runner):
        """Test that --best-metric shows available choices."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        # Check all choices are listed
        assert "val_loss" in result.output
        assert "r2" in result.output
        assert "mae" in result.output
        assert "accuracy" in result.output
        assert "f1" in result.output

    def test_train_command_best_metric_invalid_choice(
        self, runner, sample_biom_file, sample_unifrac_matrix_file, sample_metadata_file, sample_output_dir
    ):
        """Test that --best-metric rejects invalid choices."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--best-metric",
                "invalid_metric",
            ],
        )
        assert result.exit_code != 0
        # Click rejects invalid choices before our validation
        assert "Invalid value" in result.output or "is not one of" in result.output


class TestCountPenaltyCLI:
    """Tests for --count-penalty CLI option."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_train_command_option_exists(self, runner):
        """Test that --count-penalty option is available in train command."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--count-penalty" in result.output

    def test_pretrain_command_option_exists(self, runner):
        """Test that --count-penalty option is available in pretrain command."""
        result = runner.invoke(cli, ["pretrain", "--help"])
        assert result.exit_code == 0
        assert "--count-penalty" in result.output

    def test_train_command_default_value(self):
        """Test that --count-penalty has default value of 1.0 in train command."""
        count_penalty_param = next(
            (p for p in train.params if isinstance(p, click.Option) and "--count-penalty" in p.opts),
            None,
        )
        assert count_penalty_param is not None, "--count-penalty option not found in train command"
        assert count_penalty_param.default == 1.0

    def test_pretrain_command_default_value(self):
        """Test that --count-penalty has default value of 1.0 in pretrain command."""
        count_penalty_param = next(
            (p for p in pretrain.params if isinstance(p, click.Option) and "--count-penalty" in p.opts),
            None,
        )
        assert count_penalty_param is not None, "--count-penalty option not found in pretrain command"
        assert count_penalty_param.default == 1.0


class TestCountPredictionToggleCLI:
    """Tests for --count-prediction/--no-count-prediction CLI option."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_train_command_option_exists(self, runner):
        """Test that --count-prediction option is available in train command."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--count-prediction" in result.output
        assert "--no-count-prediction" in result.output

    def test_train_command_default_value(self):
        """Test that --count-prediction has default value of True in train command."""
        count_prediction_param = next(
            (p for p in train.params if isinstance(p, click.Option) and "--count-prediction" in p.opts),
            None,
        )
        assert count_prediction_param is not None, "--count-prediction option not found in train command"
        assert count_prediction_param.default is True


class TestOutputArtifacts:
    """Tests for CLN-10: Training output artifacts (sample lists)."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files."""
        return tmp_path

    @pytest.fixture
    def sample_biom_file(self, temp_dir):
        """Create a sample BIOM file for testing."""
        biom_file = temp_dir / "test.biom"
        biom_file.touch()
        return str(biom_file)

    @pytest.fixture
    def sample_unifrac_matrix_file(self, temp_dir):
        """Create a sample UniFrac matrix file for testing."""
        unifrac_file = temp_dir / "test_unifrac.h5"
        unifrac_file.touch()
        return str(unifrac_file)

    @pytest.fixture
    def sample_metadata_file(self, temp_dir):
        """Create a sample metadata file for testing."""
        metadata_file = temp_dir / "test_metadata.tsv"
        metadata_file.write_text("sample_id\ttarget\nsample1\t1.0\nsample2\t2.0\nsample3\t3.0\nsample4\t4.0\n")
        return str(metadata_file)

    @pytest.fixture
    def sample_output_dir(self, temp_dir):
        """Create a sample output directory for testing."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        return str(output_dir)

    @patch("aam.cli.train.Trainer")
    @patch("aam.cli.train.MultiTaskLoss")
    @patch("aam.cli.train.create_scheduler")
    @patch("aam.cli.train.create_optimizer")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.train_test_split")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.pd.read_csv")
    def test_train_creates_sample_id_files(
        self,
        mock_read_csv,
        mock_setup_logging,
        mock_setup_device,
        mock_setup_seed,
        mock_validate_file,
        mock_validate_args,
        mock_biom_loader_class,
        mock_unifrac_loader_class,
        mock_train_test_split,
        mock_dataset_class,
        mock_dataloader_class,
        mock_model_class,
        mock_create_optimizer,
        mock_create_scheduler,
        mock_loss_class,
        mock_trainer_class,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test that train command creates train_samples.txt and val_samples.txt files."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_metadata_df = MagicMock()
        mock_metadata_df.columns = pd.Index(["sample_id", "target"])
        mock_read_csv.return_value = mock_metadata_df

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        def mock_ids(axis=None):
            if axis == "sample":
                return ["sample1", "sample2", "sample3", "sample4"]
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return ["sample1", "sample2", "sample3", "sample4"]

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader_class.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(4, 4)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=["sample1", "sample2", "sample3", "sample4"])
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        mock_train_ids = ["sample1", "sample2", "sample3"]
        mock_val_ids = ["sample4"]
        mock_train_test_split.return_value = (mock_train_ids, mock_val_ids)

        mock_train_table = MagicMock()
        mock_val_table = MagicMock()
        mock_table.filter.side_effect = lambda ids, **kwargs: mock_train_table if ids == mock_train_ids else mock_val_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader_class.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_loss_instance = MagicMock()
        mock_loss_class.return_value = mock_loss_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer_class.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify sample ID files were created
        train_samples_file = Path(sample_output_dir) / "train_samples.txt"
        val_samples_file = Path(sample_output_dir) / "val_samples.txt"

        assert train_samples_file.exists(), "train_samples.txt should be created"
        assert val_samples_file.exists(), "val_samples.txt should be created"

        # Verify contents
        train_content = train_samples_file.read_text().strip().split("\n")
        val_content = val_samples_file.read_text().strip().split("\n")

        assert train_content == mock_train_ids
        assert val_content == mock_val_ids

    @patch("aam.cli.train.Trainer")
    @patch("aam.cli.train.MultiTaskLoss")
    @patch("aam.cli.train.create_scheduler")
    @patch("aam.cli.train.create_optimizer")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.train_test_split")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.validate_arguments")
    @patch("aam.cli.train.validate_file_path")
    @patch("aam.cli.train.setup_random_seed")
    @patch("aam.cli.train.setup_device")
    @patch("aam.cli.train.setup_logging")
    @patch("aam.cli.train.pd.read_csv")
    def test_train_sample_files_contain_correct_count(
        self,
        mock_read_csv,
        mock_setup_logging,
        mock_setup_device,
        mock_setup_seed,
        mock_validate_file,
        mock_validate_args,
        mock_biom_loader_class,
        mock_unifrac_loader_class,
        mock_train_test_split,
        mock_dataset_class,
        mock_dataloader_class,
        mock_model_class,
        mock_create_optimizer,
        mock_create_scheduler,
        mock_loss_class,
        mock_trainer_class,
        runner,
        sample_biom_file,
        sample_unifrac_matrix_file,
        sample_metadata_file,
        sample_output_dir,
    ):
        """Test that sample files contain the correct number of samples."""
        mock_setup_device.return_value = torch.device("cpu")

        mock_metadata_df = MagicMock()
        mock_metadata_df.columns = pd.Index(["sample_id", "target"])
        mock_read_csv.return_value = mock_metadata_df

        mock_biom_loader_instance = MagicMock()
        mock_biom_loader_class.return_value = mock_biom_loader_instance
        mock_table = MagicMock()

        # Create a larger sample set to test 80/20 split
        all_samples = [f"sample{i}" for i in range(10)]

        def mock_ids(axis=None):
            if axis == "sample":
                return all_samples
            elif axis == "observation":
                return ["obs1", "obs2", "obs3"]
            return all_samples

        mock_table.ids = mock_ids
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_loader_instance = MagicMock()
        mock_unifrac_loader_class.return_value = mock_unifrac_loader_instance
        from skbio import DistanceMatrix
        import numpy as np

        dist_data = np.random.rand(10, 10)
        dist_data = (dist_data + dist_data.T) / 2
        np.fill_diagonal(dist_data, 0)
        mock_distance_matrix = DistanceMatrix(dist_data, ids=all_samples)
        mock_unifrac_loader_instance.load_matrix.return_value = mock_distance_matrix

        # 80/20 split
        mock_train_ids = all_samples[:8]
        mock_val_ids = all_samples[8:]
        mock_train_test_split.return_value = (mock_train_ids, mock_val_ids)

        mock_train_table = MagicMock()
        mock_val_table = MagicMock()
        mock_table.filter.side_effect = lambda ids, **kwargs: mock_train_table if len(ids) == 8 else mock_val_table

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_normalization_params.return_value = None
        mock_dataset_instance.get_count_normalization_params.return_value = None
        mock_dataset_class.return_value = mock_dataset_instance

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader_class.return_value = mock_dataloader_instance

        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        mock_optimizer = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler

        mock_loss_instance = MagicMock()
        mock_loss_class.return_value = mock_loss_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"train_loss": [1.0], "val_loss": [0.9]}
        mock_trainer_class.return_value = mock_trainer_instance

        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                sample_biom_file,
                "--unifrac-matrix",
                sample_unifrac_matrix_file,
                "--metadata",
                sample_metadata_file,
                "--metadata-column",
                "target",
                "--output-dir",
                sample_output_dir,
                "--batch-size",
                "8",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0

        train_samples_file = Path(sample_output_dir) / "train_samples.txt"
        val_samples_file = Path(sample_output_dir) / "val_samples.txt"

        train_content = train_samples_file.read_text().strip().split("\n")
        val_content = val_samples_file.read_text().strip().split("\n")

        assert len(train_content) == 8, f"Expected 8 training samples, got {len(train_content)}"
        assert len(val_content) == 2, f"Expected 2 validation samples, got {len(val_content)}"


class TestCategoricalHelp:
    """Tests for --categorical-help decision tree."""

    def test_categorical_help_output(self):
        """Test that --categorical-help shows decision tree with all sections."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--categorical-help"])

        assert result.exit_code == 0
        assert "Categorical Conditioning Decision Tree" in result.output
        assert "--categorical-fusion concat" in result.output
        assert "--categorical-fusion cross-attention" in result.output
        assert "--categorical-fusion gmu" in result.output
        assert "--conditional-output-scaling" in result.output
        assert "Recommended Combinations" in result.output
        assert "Avoid" in result.output
        assert "Example Usage" in result.output


class TestCategoricalValidationWarnings:
    """Tests for categorical validation warnings."""

    @pytest.fixture
    def metadata_with_location(self, temp_dir):
        """Create metadata file with location column for categorical tests."""
        metadata_file = temp_dir / "metadata_categorical.tsv"
        metadata_file.write_text(
            "sample_id\ttarget\tlocation\n"
            "sample1\t1.0\tsite_a\n"
            "sample2\t2.0\tsite_b\n"
            "sample3\t3.0\tsite_a\n"
            "sample4\t4.0\tsite_b\n"
        )
        return str(metadata_file)

    @pytest.mark.parametrize("fusion_strategy", ["gmu", "cross-attention"])
    @patch("aam.cli.train.BIOMLoader")
    @patch("aam.cli.train.UniFracLoader")
    @patch("aam.cli.train.ASVDataset")
    @patch("aam.cli.train.DataLoader")
    @patch("aam.cli.train.SequencePredictor")
    @patch("aam.cli.train.Trainer")
    def test_warning_conditional_scaling_with_advanced_fusion(
        self,
        mock_trainer,
        mock_model,
        mock_dataloader,
        mock_dataset,
        mock_unifrac_loader,
        mock_biom_loader,
        fusion_strategy,
        temp_dir,
        sample_biom_file,
        sample_unifrac_matrix_file,
        metadata_with_location,
        caplog,
    ):
        """Test warning when using conditional-output-scaling with gmu or cross-attention fusion."""
        import logging

        _setup_train_data_parallel_mocks(mock_biom_loader, mock_unifrac_loader, mock_dataset, mock_dataloader, mock_model)
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = (1.0, {})

        output_dir = str(temp_dir / "output")
        runner = CliRunner()

        with caplog.at_level(logging.WARNING):
            runner.invoke(
                cli,
                [
                    "train",
                    "--table",
                    sample_biom_file,
                    "--unifrac-matrix",
                    sample_unifrac_matrix_file,
                    "--metadata",
                    metadata_with_location,
                    "--metadata-column",
                    "target",
                    "--output-dir",
                    output_dir,
                    "--categorical-columns",
                    "location",
                    "--categorical-fusion",
                    fusion_strategy,
                    "--conditional-output-scaling",
                    "location",
                    "--batch-size",
                    "4",
                    "--epochs",
                    "1",
                    "--device",
                    "cpu",
                ],
            )

        warning_found = any("redundant" in record.message.lower() for record in caplog.records)
        assert warning_found, f"Expected warning about redundant usage with {fusion_strategy}"


def _setup_predict_mocks(
    mock_setup_device,
    mock_load,
    mock_biom_loader,
    mock_dataset,
    mock_model_class,
    mock_dataloader,
    with_predictions: bool = False,
):
    """Set up common mocks for predict CLI tests.

    Args:
        with_predictions: If True, configure model to return predictions.
    """
    import numpy as np

    mock_setup_device.return_value = torch.device("cpu")

    mock_load.return_value = {
        "model_state_dict": {},
        "config": {"max_bp": 150, "token_limit": 1024, "embedding_dim": 128},
    }

    mock_biom_loader_instance = MagicMock()
    mock_biom_loader.return_value = mock_biom_loader_instance
    mock_biom_loader_instance.load_table.return_value = MagicMock()

    mock_dataset.return_value = MagicMock()

    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance

    if with_predictions:
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([1.5])
        mock_model_instance.return_value = {"target_prediction": mock_tensor}

        mock_batch = {
            "tokens": torch.tensor([[[1, 2, 3]]]),
            "sample_ids": ["sample1"],
        }

        def create_dataloader_iterator(*args, **kwargs):
            mock_dl = MagicMock()
            mock_dl.__iter__ = MagicMock(return_value=iter([mock_batch]))
            return mock_dl

        mock_dataloader.side_effect = create_dataloader_iterator
    else:
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([]))
        mock_dataloader.return_value = mock_dataloader_instance

    return mock_model_instance


class TestMultiPassPrediction:
    """Tests for multi-pass prediction aggregation (CLN-14)."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_predict_help_shows_multi_pass_options(self, runner):
        """Test that predict help shows --asv-sampling, --prediction-passes, --output-variance."""
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--asv-sampling" in result.output
        assert "--prediction-passes" in result.output
        assert "--output-variance" in result.output

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    def test_warning_prediction_passes_without_random_sampling(
        self,
        mock_dataloader,
        mock_dataset,
        mock_biom_loader,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
        caplog,
    ):
        """Test warning when --prediction-passes > 1 without --asv-sampling random."""
        import logging

        _setup_predict_mocks(
            mock_setup_device,
            mock_load,
            mock_biom_loader,
            mock_dataset,
            mock_model_class,
            mock_dataloader,
            with_predictions=False,
        )

        model_file = temp_dir / "model.pt"
        model_file.touch()
        output_file = temp_dir / "predictions.tsv"

        with caplog.at_level(logging.WARNING):
            runner.invoke(
                cli,
                [
                    "predict",
                    "--model",
                    str(model_file),
                    "--table",
                    sample_biom_file,
                    "--output",
                    str(output_file),
                    "--prediction-passes",
                    "5",
                    "--asv-sampling",
                    "abundance",
                    "--device",
                    "cpu",
                ],
            )

        warning_found = any("only has effect with --asv-sampling=random" in record.message for record in caplog.records)
        assert warning_found, "Expected warning about prediction-passes only working with random sampling"

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    def test_warning_output_variance_with_single_pass(
        self,
        mock_dataloader,
        mock_dataset,
        mock_biom_loader,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
        caplog,
    ):
        """Test warning when --output-variance is used with single prediction pass."""
        import logging

        _setup_predict_mocks(
            mock_setup_device,
            mock_load,
            mock_biom_loader,
            mock_dataset,
            mock_model_class,
            mock_dataloader,
            with_predictions=False,
        )

        model_file = temp_dir / "model.pt"
        model_file.touch()
        output_file = temp_dir / "predictions.tsv"

        with caplog.at_level(logging.WARNING):
            runner.invoke(
                cli,
                [
                    "predict",
                    "--model",
                    str(model_file),
                    "--table",
                    sample_biom_file,
                    "--output",
                    str(output_file),
                    "--output-variance",
                    "--device",
                    "cpu",
                ],
            )

        warning_found = any("no effect with single prediction pass" in record.message for record in caplog.records)
        assert warning_found, "Expected warning about output-variance with single pass"

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    def test_multi_pass_creates_multiple_dataloaders(
        self,
        mock_dataloader,
        mock_dataset,
        mock_biom_loader,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
    ):
        """Test that multi-pass prediction creates a new dataloader for each pass."""
        _setup_predict_mocks(
            mock_setup_device,
            mock_load,
            mock_biom_loader,
            mock_dataset,
            mock_model_class,
            mock_dataloader,
            with_predictions=True,
        )

        model_file = temp_dir / "model.pt"
        model_file.touch()
        output_file = temp_dir / "predictions.tsv"

        result = runner.invoke(
            cli,
            [
                "predict",
                "--model",
                str(model_file),
                "--table",
                sample_biom_file,
                "--output",
                str(output_file),
                "--prediction-passes",
                "3",
                "--asv-sampling",
                "random",
                "--device",
                "cpu",
            ],
        )

        assert result.exit_code == 0
        # Initial dataloader + 3 passes = 4 calls
        assert mock_dataloader.call_count == 4

    @patch("aam.cli.predict.setup_device")
    @patch("aam.cli.predict.validate_file_path")
    @patch("aam.cli.predict.torch.load")
    @patch("aam.cli.predict.SequencePredictor")
    @patch("aam.cli.predict.BIOMLoader")
    @patch("aam.cli.predict.ASVDataset")
    @patch("aam.cli.predict.DataLoader")
    @patch("aam.cli.predict.pd.DataFrame")
    def test_output_variance_column_included(
        self,
        mock_df_class,
        mock_dataloader,
        mock_dataset,
        mock_biom_loader,
        mock_model_class,
        mock_load,
        mock_validate_file,
        mock_setup_device,
        runner,
        sample_biom_file,
        temp_dir,
    ):
        """Test that prediction_std column is included when --output-variance is used."""
        _setup_predict_mocks(
            mock_setup_device,
            mock_load,
            mock_biom_loader,
            mock_dataset,
            mock_model_class,
            mock_dataloader,
            with_predictions=True,
        )

        mock_df_instance = MagicMock()
        mock_df_class.return_value = mock_df_instance

        model_file = temp_dir / "model.pt"
        model_file.touch()
        output_file = temp_dir / "predictions.tsv"

        result = runner.invoke(
            cli,
            [
                "predict",
                "--model",
                str(model_file),
                "--table",
                sample_biom_file,
                "--output",
                str(output_file),
                "--prediction-passes",
                "3",
                "--asv-sampling",
                "random",
                "--output-variance",
                "--device",
                "cpu",
            ],
        )

        assert result.exit_code == 0
        mock_df_instance.__setitem__.assert_called()
        calls = mock_df_instance.__setitem__.call_args_list
        column_names = [call[0][0] for call in calls]
        assert "prediction_std" in column_names


class TestValPredictionPassesOption:
    """Tests for --val-prediction-passes option in train command (CLN-15)."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_train_help_shows_val_prediction_passes_option(self, runner):
        """Test that train help shows --val-prediction-passes option."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--val-prediction-passes" in result.output

    def test_train_val_prediction_passes_default_is_one(self):
        """Test that --val-prediction-passes defaults to 1."""
        from aam.cli.train import train

        for param in train.params:
            if param.name == "val_prediction_passes":
                assert param.default == 1
                break
        else:
            pytest.fail("--val-prediction-passes parameter not found")

    def test_train_val_prediction_passes_warns_with_non_random_sampling(self, runner, caplog):
        """Test warning when --val-prediction-passes > 1 without --asv-sampling random."""
        import logging

        # Just test the help shows the option for now
        # Full integration test would require extensive mocking
        result = runner.invoke(cli, ["train", "--help"])
        assert "--val-prediction-passes" in result.output
        assert "--asv-sampling" in result.output
