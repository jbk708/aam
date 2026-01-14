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
    wrap_model_fsdp,
    get_fsdp_wrap_policy,
    is_fsdp_model,
    is_ddp_model,
    unwrap_model,
    create_distributed_dataloader,
    reduce_tensor,
    print_rank0,
    DistributedTrainer,
    get_fsdp_state_dict,
    set_fsdp_state_dict,
    get_fsdp_optimizer_state_dict,
    set_fsdp_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


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
        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.get_rank", return_value=0),
            patch("aam.training.distributed.get_world_size", return_value=2),
        ):
            dataloader, sampler = create_distributed_dataloader(dataset, batch_size=8, shuffle=True)

            assert dataloader is not None
            assert sampler is not None

    def test_create_distributed_dataloader_batch_size_per_gpu(self):
        """Test that batch_size is per-GPU (not total)."""
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.randn(100, 10))
        batch_size = 8

        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.get_rank", return_value=0),
            patch("aam.training.distributed.get_world_size", return_value=2),
        ):
            dataloader, _ = create_distributed_dataloader(dataset, batch_size=batch_size)

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

        # Mock all distributed state for testing (DDP constructor needs process group)
        with (
            patch("aam.training.distributed.setup_distributed") as mock_setup,
            patch("aam.training.distributed.wrap_model_ddp") as mock_wrap,
            patch("aam.training.distributed.is_main_process", return_value=True),
            patch("aam.training.distributed.get_local_rank", return_value=0),
        ):
            mock_setup.return_value = (0, 1, torch.device("cpu"))
            # wrap_model_ddp returns a mock that delegates to the real model
            mock_ddp = MagicMock(wraps=model)
            mock_ddp.module = model
            mock_ddp.parameters = model.parameters
            mock_wrap.return_value = mock_ddp

            trainer = DistributedTrainer(
                model=model,
                loss_fn=loss_fn,
                backend="gloo",  # Use gloo for CPU testing
                max_grad_norm=1.0,
            )

            assert trainer is not None
            mock_setup.assert_called_once()
            mock_wrap.assert_called_once()

    def test_distributed_trainer_cleanup(self):
        """Test DistributedTrainer cleanup method."""
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        # Mock all distributed state for testing (DDP constructor needs process group)
        with (
            patch("aam.training.distributed.setup_distributed") as mock_setup,
            patch("aam.training.distributed.cleanup_distributed") as mock_cleanup,
            patch("aam.training.distributed.wrap_model_ddp") as mock_wrap,
            patch("aam.training.distributed.is_main_process", return_value=True),
            patch("aam.training.distributed.get_local_rank", return_value=0),
        ):
            mock_setup.return_value = (0, 1, torch.device("cpu"))
            # wrap_model_ddp returns a mock that delegates to the real model
            mock_ddp = MagicMock(wraps=model)
            mock_ddp.module = model
            mock_ddp.parameters = model.parameters
            mock_wrap.return_value = mock_ddp

            trainer = DistributedTrainer(
                model=model,
                loss_fn=loss_fn,
                backend="gloo",
            )
            trainer.cleanup()

            mock_cleanup.assert_called_once()


class TestModelTypeChecks:
    """Test model type checking functions."""

    def test_is_fsdp_model_false_for_plain_model(self):
        """Test is_fsdp_model returns False for unwrapped model."""
        model = SimpleModel()
        assert is_fsdp_model(model) is False

    def test_is_ddp_model_false_for_plain_model(self):
        """Test is_ddp_model returns False for unwrapped model."""
        model = SimpleModel()
        assert is_ddp_model(model) is False

    def test_is_fsdp_model_true_for_fsdp_wrapped(self):
        """Test is_fsdp_model returns True for FSDP-wrapped model."""
        model = SimpleModel()
        # Create a mock FSDP model
        mock_fsdp = MagicMock(spec=FSDP)
        mock_fsdp.__class__ = FSDP
        assert is_fsdp_model(mock_fsdp) is True

    def test_is_ddp_model_true_for_ddp_wrapped(self):
        """Test is_ddp_model returns True for DDP-wrapped model."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = SimpleModel()
        mock_ddp = MagicMock(spec=DDP)
        mock_ddp.__class__ = DDP
        assert is_ddp_model(mock_ddp) is True


class TestUnwrapModel:
    """Test model unwrapping utility."""

    def test_unwrap_plain_model_returns_same(self):
        """Test unwrap_model returns plain model unchanged."""
        model = SimpleModel()
        result = unwrap_model(model)
        assert result is model

    def test_unwrap_ddp_model_returns_module(self):
        """Test unwrap_model returns .module for DDP model."""
        inner_model = SimpleModel()
        mock_ddp = MagicMock()
        mock_ddp.__class__ = type("DistributedDataParallel", (nn.Module,), {})
        mock_ddp.module = inner_model

        with (
            patch("aam.training.distributed.is_ddp_model", return_value=True),
            patch("aam.training.distributed.is_fsdp_model", return_value=False),
        ):
            result = unwrap_model(mock_ddp)
            assert result is inner_model

    def test_unwrap_fsdp_model_returns_module(self):
        """Test unwrap_model returns .module for FSDP model."""
        inner_model = SimpleModel()
        mock_fsdp = MagicMock()
        mock_fsdp.__class__ = FSDP
        mock_fsdp.module = inner_model

        # Patch is_fsdp_model to return True for our mock
        with patch("aam.training.distributed.is_fsdp_model", return_value=True):
            result = unwrap_model(mock_fsdp)
            assert result is inner_model


class TestWrapModelFSDP:
    """Test FSDP model wrapping."""

    def test_wrap_model_fsdp_requires_distributed(self):
        """Test wrap_model_fsdp raises RuntimeError when not in distributed mode."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        model = SimpleModel()
        with pytest.raises(RuntimeError, match="distributed training not initialized"):
            wrap_model_fsdp(model)

    def test_wrap_model_fsdp_accepts_sharding_strategy(self):
        """Test wrap_model_fsdp signature accepts sharding_strategy parameter."""
        import inspect

        sig = inspect.signature(wrap_model_fsdp)
        params = list(sig.parameters.keys())
        assert "sharding_strategy" in params
        assert "mixed_precision" in params
        assert "cpu_offload" in params

    def test_wrap_model_fsdp_returns_fsdp_type(self):
        """Test wrap_model_fsdp has correct return type annotation."""
        import inspect

        sig = inspect.signature(wrap_model_fsdp)
        # Return type is FullyShardedDataParallel (which is FSDP)
        assert "FullyShardedDataParallel" in str(sig.return_annotation)

    def test_wrap_model_fsdp_default_sharding_strategy(self):
        """Test wrap_model_fsdp defaults to FULL_SHARD strategy."""
        import inspect

        sig = inspect.signature(wrap_model_fsdp)
        sharding_param = sig.parameters["sharding_strategy"]
        assert sharding_param.default == ShardingStrategy.FULL_SHARD

    def test_wrap_model_fsdp_requires_cuda(self):
        """Test wrap_model_fsdp raises RuntimeError when CUDA is not available."""
        model = SimpleModel()

        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                wrap_model_fsdp(model)

    def test_wrap_model_fsdp_calls_fsdp_constructor(self):
        """Test wrap_model_fsdp calls FSDP with correct arguments."""
        model = SimpleModel()

        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.torch.cuda.is_available", return_value=True),
            patch("aam.training.distributed.torch.cuda.current_device", return_value=0),
            patch("aam.training.distributed.FSDP") as mock_fsdp,
        ):
            mock_fsdp.return_value = MagicMock(spec=FSDP)
            result = wrap_model_fsdp(model)

            # Verify FSDP was called
            mock_fsdp.assert_called_once()

            # Verify key arguments
            call_kwargs = mock_fsdp.call_args[1]
            assert call_kwargs["sharding_strategy"] == ShardingStrategy.FULL_SHARD
            assert call_kwargs["device_id"] == 0
            assert call_kwargs["cpu_offload"] is None

            # Verify result is the FSDP-wrapped model
            assert result is mock_fsdp.return_value

    def test_wrap_model_fsdp_with_cpu_offload(self):
        """Test wrap_model_fsdp enables CPU offload when requested."""
        model = SimpleModel()

        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.torch.cuda.is_available", return_value=True),
            patch("aam.training.distributed.torch.cuda.current_device", return_value=0),
            patch("aam.training.distributed.FSDP") as mock_fsdp,
        ):
            mock_fsdp.return_value = MagicMock(spec=FSDP)
            wrap_model_fsdp(model, cpu_offload=True)

            # Verify cpu_offload is set
            call_kwargs = mock_fsdp.call_args[1]
            assert call_kwargs["cpu_offload"] is not None
            assert call_kwargs["cpu_offload"].offload_params is True


class TestGetFSDPWrapPolicy:
    """Test FSDP wrap policy generation."""

    def test_get_fsdp_wrap_policy_returns_module_wrap_policy(self):
        """Test get_fsdp_wrap_policy returns a ModuleWrapPolicy."""
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        policy = get_fsdp_wrap_policy()
        assert isinstance(policy, ModuleWrapPolicy)

    def test_get_fsdp_wrap_policy_default_wraps_transformer_encoder_layer(self):
        """Test get_fsdp_wrap_policy wraps TransformerEncoderLayer by default."""
        policy = get_fsdp_wrap_policy()
        # ModuleWrapPolicy stores the classes in _module_classes
        assert nn.TransformerEncoderLayer in policy._module_classes

    def test_get_fsdp_wrap_policy_accepts_custom_classes(self):
        """Test get_fsdp_wrap_policy can wrap custom module classes."""
        custom_classes = {nn.Linear, nn.LayerNorm}
        policy = get_fsdp_wrap_policy(transformer_layer_cls=custom_classes)
        assert nn.Linear in policy._module_classes
        assert nn.LayerNorm in policy._module_classes

    def test_get_fsdp_wrap_policy_accepts_transformer_layer_cls(self):
        """Test get_fsdp_wrap_policy signature accepts transformer_layer_cls parameter."""
        import inspect

        sig = inspect.signature(get_fsdp_wrap_policy)
        params = list(sig.parameters.keys())
        assert "transformer_layer_cls" in params


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Multi-GPU not available")
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


class TestFSDPCheckpointFunctions:
    """Test FSDP checkpoint utility functions."""

    def test_get_fsdp_state_dict_requires_fsdp_model(self):
        """Test get_fsdp_state_dict raises TypeError for non-FSDP model."""
        model = SimpleModel()
        with pytest.raises(TypeError, match="Expected FSDP model"):
            get_fsdp_state_dict(model)

    def test_set_fsdp_state_dict_requires_fsdp_model(self):
        """Test set_fsdp_state_dict raises TypeError for non-FSDP model."""
        model = SimpleModel()
        with pytest.raises(TypeError, match="Expected FSDP model"):
            set_fsdp_state_dict(model, {})

    def test_get_fsdp_optimizer_state_dict_requires_fsdp_model(self):
        """Test get_fsdp_optimizer_state_dict raises TypeError for non-FSDP model."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        with pytest.raises(TypeError, match="Expected FSDP model"):
            get_fsdp_optimizer_state_dict(model, optimizer)

    def test_set_fsdp_optimizer_state_dict_requires_fsdp_model(self):
        """Test set_fsdp_optimizer_state_dict raises TypeError for non-FSDP model."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        with pytest.raises(TypeError, match="Expected FSDP model"):
            set_fsdp_optimizer_state_dict(model, optimizer, {})

    def test_get_fsdp_state_dict_function_signature(self):
        """Test get_fsdp_state_dict has correct parameters."""
        import inspect

        sig = inspect.signature(get_fsdp_state_dict)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "sharded" in params
        assert "cpu_offload" in params
        assert "rank0_only" in params

    def test_set_fsdp_state_dict_function_signature(self):
        """Test set_fsdp_state_dict has correct parameters."""
        import inspect

        sig = inspect.signature(set_fsdp_state_dict)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "state_dict" in params
        assert "sharded" in params
        assert "strict" in params

    def test_get_fsdp_optimizer_state_dict_function_signature(self):
        """Test get_fsdp_optimizer_state_dict has correct parameters."""
        import inspect

        sig = inspect.signature(get_fsdp_optimizer_state_dict)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "optimizer" in params
        assert "sharded" in params

    def test_set_fsdp_optimizer_state_dict_function_signature(self):
        """Test set_fsdp_optimizer_state_dict has correct parameters."""
        import inspect

        sig = inspect.signature(set_fsdp_optimizer_state_dict)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "optimizer" in params
        assert "optim_state_dict" in params
        assert "sharded" in params

    def test_get_fsdp_state_dict_defaults_to_full_state_dict(self):
        """Test get_fsdp_state_dict defaults sharded=False."""
        import inspect

        sig = inspect.signature(get_fsdp_state_dict)
        sharded_param = sig.parameters["sharded"]
        assert sharded_param.default is False

    def test_get_fsdp_state_dict_defaults_to_cpu_offload(self):
        """Test get_fsdp_state_dict defaults cpu_offload=True."""
        import inspect

        sig = inspect.signature(get_fsdp_state_dict)
        cpu_offload_param = sig.parameters["cpu_offload"]
        assert cpu_offload_param.default is True

    def test_get_fsdp_state_dict_defaults_to_rank0_only(self):
        """Test get_fsdp_state_dict defaults rank0_only=True."""
        import inspect

        sig = inspect.signature(get_fsdp_state_dict)
        rank0_only_param = sig.parameters["rank0_only"]
        assert rank0_only_param.default is True

    def test_get_fsdp_state_dict_calls_fsdp_state_dict_type(self):
        """Test get_fsdp_state_dict uses FSDP.state_dict_type context manager."""
        from torch.distributed.fsdp import StateDictType

        mock_fsdp_model = MagicMock(spec=FSDP)
        mock_fsdp_model.__class__ = FSDP
        mock_state_dict = {"weight": torch.randn(10, 5)}
        mock_fsdp_model.state_dict.return_value = mock_state_dict

        with (
            patch("aam.training.distributed.is_fsdp_model", return_value=True),
            patch("aam.training.distributed.FSDP.state_dict_type") as mock_context,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            result = get_fsdp_state_dict(mock_fsdp_model)

            # Verify state_dict_type was called
            mock_context.assert_called_once()
            # Verify FULL_STATE_DICT is used by default
            call_args = mock_context.call_args
            assert call_args[0][1] == StateDictType.FULL_STATE_DICT

    def test_get_fsdp_state_dict_sharded_mode(self):
        """Test get_fsdp_state_dict with sharded=True."""
        from torch.distributed.fsdp import StateDictType

        mock_fsdp_model = MagicMock(spec=FSDP)
        mock_fsdp_model.__class__ = FSDP
        mock_fsdp_model.state_dict.return_value = {}

        with (
            patch("aam.training.distributed.is_fsdp_model", return_value=True),
            patch("aam.training.distributed.FSDP.state_dict_type") as mock_context,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            get_fsdp_state_dict(mock_fsdp_model, sharded=True)

            # Verify SHARDED_STATE_DICT is used when sharded=True
            call_args = mock_context.call_args
            assert call_args[0][1] == StateDictType.SHARDED_STATE_DICT

    def test_set_fsdp_state_dict_calls_load_state_dict(self):
        """Test set_fsdp_state_dict calls model.load_state_dict."""
        mock_fsdp_model = MagicMock(spec=FSDP)
        mock_fsdp_model.__class__ = FSDP
        state_dict = {"weight": torch.randn(10, 5)}

        with (
            patch("aam.training.distributed.is_fsdp_model", return_value=True),
            patch("aam.training.distributed.FSDP.state_dict_type") as mock_context,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            set_fsdp_state_dict(mock_fsdp_model, state_dict)

            mock_fsdp_model.load_state_dict.assert_called_once_with(state_dict, strict=True)

    def test_set_fsdp_state_dict_respects_strict_parameter(self):
        """Test set_fsdp_state_dict passes strict parameter to load_state_dict."""
        mock_fsdp_model = MagicMock(spec=FSDP)
        mock_fsdp_model.__class__ = FSDP
        state_dict = {"weight": torch.randn(10, 5)}

        with (
            patch("aam.training.distributed.is_fsdp_model", return_value=True),
            patch("aam.training.distributed.FSDP.state_dict_type") as mock_context,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            set_fsdp_state_dict(mock_fsdp_model, state_dict, strict=False)

            mock_fsdp_model.load_state_dict.assert_called_once_with(state_dict, strict=False)

    def test_get_fsdp_optimizer_state_dict_calls_fsdp_optim_state_dict(self):
        """Test get_fsdp_optimizer_state_dict uses FSDP.optim_state_dict."""
        mock_fsdp_model = MagicMock(spec=FSDP)
        mock_fsdp_model.__class__ = FSDP
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with (
            patch("aam.training.distributed.is_fsdp_model", return_value=True),
            patch("aam.training.distributed.FSDP.state_dict_type") as mock_context,
            patch("aam.training.distributed.FSDP.optim_state_dict") as mock_optim_state_dict,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)
            mock_optim_state_dict.return_value = {"state": {}, "param_groups": []}

            result = get_fsdp_optimizer_state_dict(mock_fsdp_model, optimizer)

            mock_optim_state_dict.assert_called_once_with(mock_fsdp_model, optimizer)

    def test_set_fsdp_optimizer_state_dict_calls_optim_state_dict_to_load(self):
        """Test set_fsdp_optimizer_state_dict uses FSDP.optim_state_dict_to_load."""
        mock_fsdp_model = MagicMock(spec=FSDP)
        mock_fsdp_model.__class__ = FSDP
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        optim_state_dict = {"state": {}, "param_groups": []}

        with (
            patch("aam.training.distributed.is_fsdp_model", return_value=True),
            patch("aam.training.distributed.FSDP.state_dict_type") as mock_context,
            patch("aam.training.distributed.FSDP.optim_state_dict_to_load") as mock_optim_to_load,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock(return_value=False)
            mock_optim_to_load.return_value = {"state": {}, "param_groups": []}

            set_fsdp_optimizer_state_dict(mock_fsdp_model, optimizer, optim_state_dict)

            mock_optim_to_load.assert_called_once_with(mock_fsdp_model, optimizer, optim_state_dict)
            optimizer.load_state_dict.assert_called_once()


class TestFSDPCheckpointTrainerIntegration:
    """Test FSDP checkpoint integration with Trainer."""

    def test_save_checkpoint_stores_world_size_for_sharded(self):
        """Test save_checkpoint stores world size when using sharded checkpoints."""
        from aam.training.trainer import Trainer
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        with (
            patch("aam.training.trainer.is_fsdp_model", return_value=True),
            patch("aam.training.trainer.get_fsdp_state_dict", return_value={"weight": torch.randn(10, 5)}),
            patch("aam.training.trainer.get_fsdp_optimizer_state_dict", return_value={"state": {}}),
            patch("aam.training.trainer.get_world_size", return_value=4),
            patch("torch.save") as mock_save,
        ):
            trainer = Trainer(model=model, loss_fn=loss_fn, use_sharded_checkpoint=True)
            trainer.save_checkpoint("/tmp/test.pt", epoch=1, best_val_loss=0.5)

            # Verify torch.save was called with checkpoint containing world size
            mock_save.assert_called_once()
            saved_checkpoint = mock_save.call_args[0][0]
            assert saved_checkpoint["fsdp_sharded"] is True
            assert saved_checkpoint["fsdp_world_size"] == 4

    def test_load_checkpoint_validates_world_size_mismatch(self):
        """Test load_checkpoint raises error when world size doesn't match."""
        from aam.training.trainer import Trainer
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        # Create checkpoint saved with world_size=4
        checkpoint = {
            "epoch": 1,
            "best_val_loss": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "fsdp_sharded": True,
            "fsdp_world_size": 4,
        }

        with (
            patch("aam.training.trainer.is_fsdp_model", return_value=True),
            patch("aam.training.trainer.get_world_size", return_value=2),  # Current world size is 2
            patch("torch.load", return_value=checkpoint),
        ):
            trainer = Trainer(model=model, loss_fn=loss_fn)

            with pytest.raises(RuntimeError, match="world_size=4.*current world_size=2"):
                trainer.load_checkpoint("/tmp/test.pt")

    def test_load_checkpoint_accepts_matching_world_size(self):
        """Test load_checkpoint succeeds when world size matches."""
        from aam.training.trainer import Trainer
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        checkpoint = {
            "epoch": 1,
            "best_val_loss": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "fsdp_sharded": True,
            "fsdp_world_size": 4,
        }

        with (
            patch("aam.training.trainer.is_fsdp_model", return_value=True),
            patch("aam.training.trainer.get_world_size", return_value=4),  # Matching world size
            patch("aam.training.trainer.set_fsdp_state_dict"),
            patch("aam.training.trainer.set_fsdp_optimizer_state_dict"),
            patch("torch.load", return_value=checkpoint),
        ):
            trainer = Trainer(model=model, loss_fn=loss_fn)
            result = trainer.load_checkpoint("/tmp/test.pt")

            assert result["epoch"] == 1
            assert result["best_val_loss"] == 0.5

    def test_load_checkpoint_validates_required_keys(self):
        """Test load_checkpoint raises error for missing required keys."""
        from aam.training.trainer import Trainer
        from aam.training.losses import MultiTaskLoss

        model = SimpleModel()
        loss_fn = MultiTaskLoss()

        # Missing model_state_dict
        checkpoint = {
            "epoch": 1,
            "best_val_loss": 0.5,
        }

        with patch("torch.load", return_value=checkpoint):
            trainer = Trainer(model=model, loss_fn=loss_fn)

            with pytest.raises(ValueError, match="missing required keys.*model_state_dict"):
                trainer.load_checkpoint("/tmp/test.pt")


class TestGatherEmbeddingsForUnifrac:
    """Test embedding gathering for UniFrac pairwise distance computation."""

    def test_gather_embeddings_function_exists(self):
        """Test gather_embeddings_for_unifrac function is importable."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        assert callable(gather_embeddings_for_unifrac)

    def test_gather_embeddings_returns_input_when_not_distributed(self):
        """Test gather_embeddings_for_unifrac returns input unchanged when not in distributed mode."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        embeddings = torch.randn(4, 128)  # batch_size=4, embed_dim=128
        result = gather_embeddings_for_unifrac(embeddings)

        # When not distributed, should return the input unchanged
        torch.testing.assert_close(result, embeddings)

    def test_gather_embeddings_preserves_shape_when_not_distributed(self):
        """Test gather_embeddings_for_unifrac preserves tensor shape when not distributed."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        embeddings = torch.randn(8, 256)
        result = gather_embeddings_for_unifrac(embeddings)

        assert result.shape == embeddings.shape

    def test_gather_embeddings_preserves_device(self):
        """Test gather_embeddings_for_unifrac preserves tensor device."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        embeddings = torch.randn(4, 64)
        result = gather_embeddings_for_unifrac(embeddings)

        assert result.device == embeddings.device

    def test_gather_embeddings_preserves_requires_grad(self):
        """Test gather_embeddings_for_unifrac preserves gradient tracking."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        embeddings = torch.randn(4, 64, requires_grad=True)
        result = gather_embeddings_for_unifrac(embeddings)

        assert result.requires_grad == embeddings.requires_grad

    def test_gather_embeddings_function_signature(self):
        """Test gather_embeddings_for_unifrac has correct function signature."""
        import inspect
        from aam.training.distributed import gather_embeddings_for_unifrac

        sig = inspect.signature(gather_embeddings_for_unifrac)
        params = list(sig.parameters.keys())

        assert "embeddings" in params
        # Return type should be Tensor
        assert "Tensor" in str(sig.return_annotation)

    def test_gather_embeddings_concatenates_in_distributed_mode(self):
        """Test gather_embeddings_for_unifrac concatenates across ranks in distributed mode."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        local_batch_size = 4
        embed_dim = 64
        world_size = 2

        embeddings = torch.randn(local_batch_size, embed_dim)

        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.get_world_size", return_value=world_size),
            patch("aam.training.distributed.get_rank", return_value=0),
            patch("aam.training.distributed.dist.all_gather") as mock_all_gather,
        ):
            # Simulate all_gather filling the gathered list
            def fill_gathered(gathered_list, tensor):
                for i in range(len(gathered_list)):
                    gathered_list[i].copy_(torch.randn_like(tensor))

            mock_all_gather.side_effect = fill_gathered

            result = gather_embeddings_for_unifrac(embeddings)

            # Result should have global_batch_size = local_batch_size * world_size
            assert result.shape[0] == local_batch_size * world_size
            assert result.shape[1] == embed_dim

    def test_gather_embeddings_calls_all_gather_in_distributed(self):
        """Test gather_embeddings_for_unifrac uses all_gather in distributed mode."""
        from aam.training.distributed import gather_embeddings_for_unifrac

        embeddings = torch.randn(4, 64)

        with (
            patch("aam.training.distributed.is_distributed", return_value=True),
            patch("aam.training.distributed.get_world_size", return_value=2),
            patch("aam.training.distributed.get_rank", return_value=0),
            patch("aam.training.distributed.dist.all_gather") as mock_all_gather,
        ):
            # Just make it not crash
            def fill_gathered(gathered_list, tensor):
                for i in range(len(gathered_list)):
                    gathered_list[i].copy_(tensor)

            mock_all_gather.side_effect = fill_gathered

            gather_embeddings_for_unifrac(embeddings)

            # Verify all_gather was called
            mock_all_gather.assert_called_once()


class TestDistributedValidationMetrics:
    """Tests for distributed validation metric synchronization."""

    def test_streaming_regression_has_sync_method(self):
        """Test StreamingRegressionMetrics has sync_distributed method."""
        from aam.training.metrics import StreamingRegressionMetrics

        streaming = StreamingRegressionMetrics()
        assert hasattr(streaming, "sync_distributed")
        assert callable(streaming.sync_distributed)

    def test_streaming_classification_has_sync_method(self):
        """Test StreamingClassificationMetrics has sync_distributed method."""
        from aam.training.metrics import StreamingClassificationMetrics

        streaming = StreamingClassificationMetrics()
        assert hasattr(streaming, "sync_distributed")
        assert callable(streaming.sync_distributed)

    def test_streaming_count_has_sync_method(self):
        """Test StreamingCountMetrics has sync_distributed method."""
        from aam.training.metrics import StreamingCountMetrics

        streaming = StreamingCountMetrics()
        assert hasattr(streaming, "sync_distributed")
        assert callable(streaming.sync_distributed)

    def test_streaming_regression_has_merge_method(self):
        """Test StreamingRegressionMetrics has _merge_from method for combining stats."""
        from aam.training.metrics import StreamingRegressionMetrics

        streaming = StreamingRegressionMetrics()
        assert hasattr(streaming, "_merge_from")
        assert callable(streaming._merge_from)

    def test_streaming_classification_has_merge_method(self):
        """Test StreamingClassificationMetrics has _merge_from method."""
        from aam.training.metrics import StreamingClassificationMetrics

        streaming = StreamingClassificationMetrics()
        assert hasattr(streaming, "_merge_from")
        assert callable(streaming._merge_from)

    def test_streaming_count_has_merge_method(self):
        """Test StreamingCountMetrics has _merge_from method."""
        from aam.training.metrics import StreamingCountMetrics

        streaming = StreamingCountMetrics()
        assert hasattr(streaming, "_merge_from")
        assert callable(streaming._merge_from)

    def test_regression_sync_uses_all_reduce_in_distributed(self):
        """Test that sync_distributed uses all_reduce for distributed stats."""
        from aam.training.metrics import StreamingRegressionMetrics

        streaming = StreamingRegressionMetrics()
        streaming.update(torch.randn(10), torch.randn(10))

        with (
            patch("aam.training.metrics.dist.is_initialized", return_value=True),
            patch("aam.training.metrics.dist.all_reduce") as mock_all_reduce,
            patch("aam.training.metrics.dist.get_world_size", return_value=4),
        ):
            streaming.sync_distributed()
            # Should call all_reduce to sync statistics
            assert mock_all_reduce.called

    def test_classification_sync_uses_all_reduce_in_distributed(self):
        """Test that sync_distributed uses all_reduce for confusion matrix."""
        from aam.training.metrics import StreamingClassificationMetrics

        streaming = StreamingClassificationMetrics()
        streaming.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))

        with (
            patch("aam.training.metrics.dist.is_initialized", return_value=True),
            patch("aam.training.metrics.dist.all_reduce") as mock_all_reduce,
            patch("aam.training.metrics.dist.get_world_size", return_value=4),
        ):
            streaming.sync_distributed()
            # Should call all_reduce to sync confusion matrix
            assert mock_all_reduce.called
