"""Tests for shared training utilities."""

import logging
from unittest.mock import MagicMock, patch

import click
import pytest
import torch
import torch.nn as nn

from aam.cli.training_utils import (
    build_scheduler_kwargs,
    setup_ddp,
    setup_fsdp,
    validate_distributed_options,
    wrap_data_parallel,
)


class TestValidateDistributedOptions:
    """Tests for validate_distributed_options."""

    def test_no_options_enabled(self):
        """No error when no distributed options are enabled."""
        validate_distributed_options(
            distributed=False,
            data_parallel=False,
            fsdp=False,
            fsdp_sharded_checkpoint=False,
        )

    def test_single_option_distributed(self):
        """No error when only distributed is enabled."""
        validate_distributed_options(
            distributed=True,
            data_parallel=False,
            fsdp=False,
            fsdp_sharded_checkpoint=False,
        )

    def test_single_option_data_parallel(self):
        """No error when only data_parallel is enabled."""
        validate_distributed_options(
            distributed=False,
            data_parallel=True,
            fsdp=False,
            fsdp_sharded_checkpoint=False,
        )

    def test_single_option_fsdp(self):
        """No error when only fsdp is enabled."""
        validate_distributed_options(
            distributed=False,
            data_parallel=False,
            fsdp=True,
            fsdp_sharded_checkpoint=False,
        )

    def test_fsdp_with_sharded_checkpoint(self):
        """No error when fsdp and fsdp_sharded_checkpoint are enabled."""
        validate_distributed_options(
            distributed=False,
            data_parallel=False,
            fsdp=True,
            fsdp_sharded_checkpoint=True,
        )

    def test_multiple_options_raises(self):
        """Error when multiple distributed options are enabled."""
        with pytest.raises(click.ClickException) as exc_info:
            validate_distributed_options(
                distributed=True,
                data_parallel=True,
                fsdp=False,
                fsdp_sharded_checkpoint=False,
            )
        assert "Cannot use multiple distributed training options" in str(exc_info.value)

    def test_all_options_raises(self):
        """Error when all distributed options are enabled."""
        with pytest.raises(click.ClickException) as exc_info:
            validate_distributed_options(
                distributed=True,
                data_parallel=True,
                fsdp=True,
                fsdp_sharded_checkpoint=False,
            )
        assert "Cannot use multiple distributed training options" in str(exc_info.value)

    def test_fsdp_sharded_without_fsdp_raises(self):
        """Error when fsdp_sharded_checkpoint is enabled without fsdp."""
        with pytest.raises(click.ClickException) as exc_info:
            validate_distributed_options(
                distributed=False,
                data_parallel=False,
                fsdp=False,
                fsdp_sharded_checkpoint=True,
            )
        assert "--fsdp-sharded-checkpoint requires --fsdp" in str(exc_info.value)


class TestBuildSchedulerKwargs:
    """Tests for build_scheduler_kwargs."""

    def test_cosine_restarts_all_options(self):
        """Cosine restarts with all options."""
        kwargs = build_scheduler_kwargs(
            scheduler="cosine_restarts",
            scheduler_t0=100,
            scheduler_t_mult=2,
            scheduler_eta_min=1e-6,
            scheduler_patience=None,
            scheduler_factor=None,
            scheduler_min_lr=None,
        )
        assert kwargs == {"T_0": 100, "T_mult": 2, "eta_min": 1e-6}

    def test_cosine_restarts_partial_options(self):
        """Cosine restarts with only some options."""
        kwargs = build_scheduler_kwargs(
            scheduler="cosine_restarts",
            scheduler_t0=50,
            scheduler_t_mult=None,
            scheduler_eta_min=None,
            scheduler_patience=None,
            scheduler_factor=None,
            scheduler_min_lr=None,
        )
        assert kwargs == {"T_0": 50}

    def test_cosine_with_eta_min(self):
        """Cosine with eta_min."""
        kwargs = build_scheduler_kwargs(
            scheduler="cosine",
            scheduler_t0=None,
            scheduler_t_mult=None,
            scheduler_eta_min=1e-7,
            scheduler_patience=None,
            scheduler_factor=None,
            scheduler_min_lr=None,
        )
        assert kwargs == {"eta_min": 1e-7}

    def test_plateau_all_options(self):
        """Plateau with all options."""
        kwargs = build_scheduler_kwargs(
            scheduler="plateau",
            scheduler_t0=None,
            scheduler_t_mult=None,
            scheduler_eta_min=None,
            scheduler_patience=5,
            scheduler_factor=0.3,
            scheduler_min_lr=1e-8,
        )
        assert kwargs == {"patience": 5, "factor": 0.3, "min_lr": 1e-8}

    def test_plateau_partial_options(self):
        """Plateau with only some options."""
        kwargs = build_scheduler_kwargs(
            scheduler="plateau",
            scheduler_t0=None,
            scheduler_t_mult=None,
            scheduler_eta_min=None,
            scheduler_patience=10,
            scheduler_factor=None,
            scheduler_min_lr=None,
        )
        assert kwargs == {"patience": 10}

    def test_warmup_cosine_no_options(self):
        """Warmup cosine returns empty dict."""
        kwargs = build_scheduler_kwargs(
            scheduler="warmup_cosine",
            scheduler_t0=None,
            scheduler_t_mult=None,
            scheduler_eta_min=None,
            scheduler_patience=None,
            scheduler_factor=None,
            scheduler_min_lr=None,
        )
        assert kwargs == {}

    def test_onecycle_no_options(self):
        """Onecycle returns empty dict."""
        kwargs = build_scheduler_kwargs(
            scheduler="onecycle",
            scheduler_t0=None,
            scheduler_t_mult=None,
            scheduler_eta_min=None,
            scheduler_patience=None,
            scheduler_factor=None,
            scheduler_min_lr=None,
        )
        assert kwargs == {}


class TestWrapDataParallel:
    """Tests for wrap_data_parallel."""

    def test_no_cuda_raises(self):
        """Error when CUDA is not available."""
        model = nn.Linear(10, 5)
        logger = logging.getLogger("test")

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(click.ClickException) as exc_info:
                wrap_data_parallel(model, logger)
            assert "--data-parallel requires CUDA" in str(exc_info.value)

    def test_single_gpu_warning(self):
        """Warning when only one GPU is available."""
        model = nn.Linear(10, 5)
        logger = MagicMock(spec=logging.Logger)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch.object(nn.Module, "to", return_value=model):
                    with patch("torch.nn.DataParallel", return_value=model):
                        result = wrap_data_parallel(model, logger)

        logger.warning.assert_called()
        assert "only 1 GPU" in str(logger.warning.call_args)

    def test_multi_gpu_success(self):
        """Successful wrapping with multiple GPUs."""
        model = nn.Linear(10, 5)
        logger = MagicMock(spec=logging.Logger)
        wrapped_model = MagicMock(spec=nn.DataParallel)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                with patch.object(nn.Module, "to", return_value=model):
                    with patch("torch.nn.DataParallel", return_value=wrapped_model) as dp_mock:
                        result = wrap_data_parallel(model, logger)

        dp_mock.assert_called_once()
        call_args = dp_mock.call_args
        assert call_args[1]["device_ids"] == [0, 1, 2, 3]
        assert result is wrapped_model


class TestSetupFsdp:
    """Tests for setup_fsdp."""

    def test_no_cuda_raises(self):
        """Error when CUDA is not available."""
        model = nn.Linear(10, 5)
        logger = logging.getLogger("test")

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(click.ClickException) as exc_info:
                setup_fsdp(model, sync_batchnorm=False, logger=logger)
            assert "FSDP requires CUDA" in str(exc_info.value)

    def test_fsdp_wrapping_failure_raises(self):
        """Error when FSDP wrapping fails."""
        model = nn.Linear(10, 5)
        logger = MagicMock(spec=logging.Logger)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("aam.cli.training_utils.wrap_model_fsdp", side_effect=RuntimeError("FSDP error")):
                with patch("aam.cli.training_utils.is_main_process", return_value=True):
                    with pytest.raises(click.ClickException) as exc_info:
                        setup_fsdp(model, sync_batchnorm=False, logger=logger)
            assert "FSDP model wrapping failed" in str(exc_info.value)

    def test_sync_batchnorm_conversion(self):
        """SyncBatchNorm conversion when requested."""
        model = nn.Linear(10, 5)
        logger = MagicMock(spec=logging.Logger)
        wrapped_model = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("aam.cli.training_utils.wrap_model_fsdp", return_value=wrapped_model):
                with patch("aam.cli.training_utils.sync_batch_norm") as sync_bn_mock:
                    with patch("aam.cli.training_utils.is_main_process", return_value=True):
                        sync_bn_mock.return_value = model
                        result = setup_fsdp(model, sync_batchnorm=True, logger=logger)

        sync_bn_mock.assert_called_once_with(model)


class TestSetupDdp:
    """Tests for setup_ddp."""

    def test_basic_ddp_setup(self):
        """Basic DDP setup without sync batchnorm."""
        model = nn.Linear(10, 5)
        device = torch.device("cuda:0")
        logger = MagicMock(spec=logging.Logger)
        wrapped_model = MagicMock()

        with patch.object(nn.Module, "to", return_value=model):
            with patch("aam.cli.training_utils.wrap_model_ddp", return_value=wrapped_model) as ddp_mock:
                with patch("aam.cli.training_utils.get_local_rank", return_value=0):
                    with patch("aam.cli.training_utils.is_main_process", return_value=True):
                        result = setup_ddp(
                            model,
                            device=device,
                            sync_batchnorm=False,
                            find_unused_parameters=False,
                            logger=logger,
                        )

        ddp_mock.assert_called_once()
        assert result is wrapped_model

    def test_ddp_with_sync_batchnorm(self):
        """DDP setup with sync batchnorm."""
        model = nn.Linear(10, 5)
        device = torch.device("cuda:0")
        logger = MagicMock(spec=logging.Logger)
        wrapped_model = MagicMock()

        with patch.object(nn.Module, "to", return_value=model):
            with patch("aam.cli.training_utils.wrap_model_ddp", return_value=wrapped_model):
                with patch("aam.cli.training_utils.sync_batch_norm") as sync_bn_mock:
                    with patch("aam.cli.training_utils.get_local_rank", return_value=0):
                        with patch("aam.cli.training_utils.is_main_process", return_value=True):
                            sync_bn_mock.return_value = model
                            result = setup_ddp(
                                model,
                                device=device,
                                sync_batchnorm=True,
                                find_unused_parameters=False,
                                logger=logger,
                            )

        sync_bn_mock.assert_called_once_with(model)

    def test_ddp_with_find_unused_parameters(self):
        """DDP setup with find_unused_parameters."""
        model = nn.Linear(10, 5)
        device = torch.device("cuda:0")
        logger = MagicMock(spec=logging.Logger)
        wrapped_model = MagicMock()

        with patch.object(nn.Module, "to", return_value=model):
            with patch("aam.cli.training_utils.wrap_model_ddp", return_value=wrapped_model) as ddp_mock:
                with patch("aam.cli.training_utils.get_local_rank", return_value=0):
                    with patch("aam.cli.training_utils.is_main_process", return_value=True):
                        result = setup_ddp(
                            model,
                            device=device,
                            sync_batchnorm=False,
                            find_unused_parameters=True,
                            logger=logger,
                        )

        ddp_mock.assert_called_once()
        call_kwargs = ddp_mock.call_args[1]
        assert call_kwargs["find_unused_parameters"] is True
