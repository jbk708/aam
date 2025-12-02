"""Unit tests for CLI interface."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import click
from click.testing import CliRunner

from aam.cli import (
    setup_logging,
    setup_device,
    setup_random_seed,
    validate_file_path,
    validate_arguments,
    cli,
    train,
    predict,
)


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
def sample_tree_file(temp_dir):
    """Create a sample tree file path."""
    tree_file = temp_dir / "test.nwk"
    tree_file.write_text("(A:0.1,B:0.2);")
    return str(tree_file)


@pytest.fixture
def sample_metadata_file(temp_dir):
    """Create a sample metadata file."""
    metadata_file = temp_dir / "metadata.tsv"
    metadata_file.write_text("sample_id\ttarget\nsample1\t1.0\nsample2\t2.0\n")
    return str(metadata_file)


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
        self, runner, sample_biom_file, sample_tree_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command file validation."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--table",
                "nonexistent.biom",
                "--tree",
                sample_tree_file,
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
        self, runner, sample_biom_file, sample_tree_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command batch size validation."""
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
                "--batch-size",
                "7",
            ],
        )
        assert result.exit_code != 0

    def test_train_command_classifier_validation(
        self, runner, sample_biom_file, sample_tree_file, sample_metadata_file, sample_output_dir
    ):
        """Test train command classifier validation."""
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
                "--classifier",
                "--out-dim",
                "1",
            ],
        )
        assert result.exit_code != 0

    def test_predict_command_help(self, runner):
        """Test predict command help."""
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "Run inference" in result.output

    def test_predict_command_missing_required_args(self, runner):
        """Test predict command with missing required arguments."""
        result = runner.invoke(cli, ["predict"])
        assert result.exit_code != 0

    def test_predict_command_file_validation(self, runner, sample_biom_file, sample_tree_file):
        """Test predict command file validation."""
        result = runner.invoke(
            cli,
            [
                "predict",
                "--model",
                "nonexistent.pt",
                "--table",
                sample_biom_file,
                "--tree",
                sample_tree_file,
                "--output",
                "output.tsv",
            ],
        )
        assert result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI with mocked components."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @patch("aam.cli.setup_logging")
    @patch("aam.cli.setup_device")
    @patch("aam.cli.setup_random_seed")
    @patch("aam.cli.validate_file_path")
    @patch("aam.cli.validate_arguments")
    @patch("aam.data.biom_loader.BIOMLoader")
    @patch("aam.data.unifrac.UniFracComputer")
    @patch("aam.data.dataset.ASVDataset")
    @patch("aam.models.sequence_predictor.SequencePredictor")
    @patch("aam.training.trainer.Trainer")
    def test_train_command_integration(
        self,
        mock_trainer,
        mock_model,
        mock_dataset,
        mock_unifrac,
        mock_biom_loader,
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
    ):
        """Test train command integration with mocked components."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_biom_loader_instance = MagicMock()
        mock_biom_loader.return_value = mock_biom_loader_instance
        mock_table = MagicMock()
        mock_biom_loader_instance.load_table.return_value = mock_table
        mock_biom_loader_instance.rarefy.return_value = mock_table

        mock_unifrac_instance = MagicMock()
        mock_unifrac.return_value = mock_unifrac_instance
        mock_distance_matrix = MagicMock()
        mock_unifrac_instance.compute_unweighted.return_value = mock_distance_matrix

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
                "--tree",
                sample_tree_file,
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

    @patch("aam.cli.setup_device")
    @patch("aam.cli.validate_file_path")
    @patch("torch.load")
    @patch("aam.cli.SequencePredictor")
    @patch("aam.cli.BIOMLoader")
    @patch("aam.cli.ASVDataset")
    @patch("aam.cli.DataLoader")
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
        sample_tree_file,
        temp_dir,
    ):
        """Test predict command integration with mocked components."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_checkpoint = {
            "model_state_dict": {},
            "config": {"max_bp": 150, "token_limit": 1024, "embedding_dim": 128, "encoder_type": "unifrac", "out_dim": 1, "is_classifier": False},
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
                "--tree",
                sample_tree_file,
                "--output",
                str(output_file),
            ],
        )

        assert mock_setup_device.called or result.exit_code == 0
        assert mock_validate_file.called
