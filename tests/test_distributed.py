"""Unit tests for distributed training utilities."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import os

from aam.training.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    create_distributed_dataloader,
    reduce_tensor,
    print_rank0,
    DistributedTrainer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestDistributedStateQueries:
    """Test distributed state query functions."""

    def test_is_distributed_false_when_not_initialized(self):
        """Test is_distributed returns False when not initialized."""
        # Clean up any existing distributed state
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        assert is_distributed() is False

    def test_get_rank_returns_zero_when_not_distributed(self):
        """Test get_rank returns 0 when not in distributed mode."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        assert get_rank() == 0

    def test_get_world_size_returns_one_when_not_distributed(self):
        """Test get_world_size returns 1 when not in distributed mode."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        assert get_world_size() == 1

    def test_is_main_process_true_when_not_distributed(self):
        """Test is_main_process returns True when not in distributed mode."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        assert is_main_process() is True


class TestSetupDistributed:
    """Test distributed setup and cleanup."""

    def test_setup_requires_environment_variables(self):
        """Test that setup_distributed fails gracefully without env vars."""
        # Clear any distributed env vars
        env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
        old_values = {k: os.environ.pop(k, None) for k in env_vars}

        try:
            # Should handle missing env vars gracefully
            with pytest.raises((RuntimeError, ValueError, KeyError)):
                setup_distributed()
        finally:
            # Restore env vars
            for k, v in old_values.items():
                if v is not None:
                    os.environ[k] = v

    def test_cleanup_is_safe_when_not_initialized(self):
        """Test cleanup_distributed is safe to call when not initialized."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        # Should not raise
        cleanup_distributed()


class TestWrapModelDDP:
    """Test DDP model wrapping."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wrap_model_ddp_requires_distributed(self):
        """Test wrap_model_ddp raises when not in distributed mode."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        model = SimpleModel().cuda()
        with pytest.raises((RuntimeError, AssertionError)):
            wrap_model_ddp(model, device_id=0)

    def test_wrap_model_ddp_returns_ddp_type(self):
        """Test that wrapped model has correct type annotation."""
        # This is a compile-time type check, just verify the function signature
        import inspect
        sig = inspect.signature(wrap_model_ddp)
        # Check return type annotation exists
        assert "DDP" in str(sig.return_annotation) or sig.return_annotation != inspect.Parameter.empty


class TestDistributedDataLoader:
    """Test distributed dataloader creation."""

    def test_create_distributed_dataloader_returns_tuple(self):
        """Test create_distributed_dataloader returns DataLoader and Sampler."""
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(100, 10))

        # Mock distributed state
        with patch("aam.training.distributed.is_distributed", return_value=True), \
             patch("aam.training.distributed.get_rank", return_value=0), \
             patch("aam.training.distributed.get_world_size", return_value=2):

            dataloader, sampler = create_distributed_dataloader(
                dataset, batch_size=8, shuffle=True
            )

            assert dataloader is not None
            assert sampler is not None

    def test_create_distributed_dataloader_batch_size_per_gpu(self):
        """Test that batch_size is per-GPU (not total)."""
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(100, 10))
        batch_size = 8

        with patch("aam.training.distributed.is_distributed", return_value=True), \
             patch("aam.training.distributed.get_rank", return_value=0), \
             patch("aam.training.distributed.get_world_size", return_value=2):

            dataloader, _ = create_distributed_dataloader(
                dataset, batch_size=batch_size
            )

            assert dataloader.batch_size == batch_size


class TestReduceTensor:
    """Test tensor reduction across processes."""

    def test_reduce_tensor_noop_when_not_distributed(self):
        """Test reduce_tensor returns input unchanged when not distributed."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_tensor(tensor)
        torch.testing.assert_close(result, tensor)

    def test_reduce_tensor_preserves_device(self):
        """Test reduce_tensor preserves tensor device."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_tensor(tensor)
        assert result.device == tensor.device


class TestPrintRank0:
    """Test rank-0 only printing."""

    def test_print_rank0_prints_when_main_process(self, capsys):
        """Test print_rank0 prints when on main process."""
        with patch("aam.training.distributed.is_main_process", return_value=True):
            print_rank0("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out

    def test_print_rank0_silent_when_not_main(self, capsys):
        """Test print_rank0 is silent when not on main process."""
        with patch("aam.training.distributed.is_main_process", return_value=False):
            print_rank0("test message")
            captured = capsys.readouterr()
            assert "test message" not in captured.out


class TestDistributedTrainer:
    """Test DistributedTrainer wrapper."""

    def test_distributed_trainer_init_accepts_trainer_kwargs(self):
        """Test DistributedTrainer passes kwargs to Trainer."""
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        # Should not raise during init (actual distributed setup happens later)
        with patch("aam.training.distributed.setup_distributed") as mock_setup:
            mock_setup.return_value = (0, 1, torch.device("cpu"))

            trainer = DistributedTrainer(
                model=model,
                loss_fn=loss_fn,
                backend="gloo",  # Use gloo for CPU testing
                max_grad_norm=1.0,
            )

            assert trainer is not None

    def test_distributed_trainer_cleanup(self):
        """Test DistributedTrainer cleanup method."""
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        with patch("aam.training.distributed.setup_distributed") as mock_setup, \
             patch("aam.training.distributed.cleanup_distributed") as mock_cleanup:
            mock_setup.return_value = (0, 1, torch.device("cpu"))

            trainer = DistributedTrainer(
                model=model,
                loss_fn=loss_fn,
                backend="gloo",
            )
            trainer.cleanup()

            mock_cleanup.assert_called_once()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Multi-GPU not available"
)
class TestMultiGPUIntegration:
    """Integration tests requiring multiple GPUs."""

    def test_can_move_tensors_between_gpus(self):
        """Test basic multi-GPU tensor operations."""
        t0 = torch.randn(10, 10, device="cuda:0")
        t1 = t0.to("cuda:1")
        assert t1.device == torch.device("cuda:1")
        torch.testing.assert_close(t0.cpu(), t1.cpu())

    def test_nccl_backend_available(self):
        """Test NCCL/RCCL backend is available."""
        assert torch.distributed.is_nccl_available()
