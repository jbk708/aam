"""Distributed training utilities for multi-GPU training with DDP and FSDP.

Supports both NCCL (NVIDIA) and RCCL (AMD ROCm) backends.

Usage:
    # Single-node multi-GPU with torchrun (DDP):
    torchrun --nproc_per_node=4 train.py --distributed

    # Single-node multi-GPU with torchrun (FSDP):
    torchrun --nproc_per_node=4 train.py --fsdp

    # Multi-node:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\
        --master_addr=<master_ip> --master_port=29500 train.py --distributed
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig
from torch.distributed.fsdp import FullOptimStateDictConfig, ShardedOptimStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Tuple, Any, List, Set, Type
import torch.nn as nn


def is_distributed() -> bool:
    """Check if running in distributed mode.

    Returns:
        True if distributed training is initialized, False otherwise.
    """
    return dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process.

    Returns:
        Process rank, or 0 if not in distributed mode.
    """
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes.

    Returns:
        World size, or 1 if not in distributed mode.
    """
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node).

    Returns:
        Local rank from LOCAL_RANK env var, or 0 if not set.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Returns:
        True if rank 0 or not in distributed mode.
    """
    return get_rank() == 0


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> Tuple[int, int, torch.device]:
    """Initialize distributed training environment.

    This function should be called at the start of training when using
    torchrun or similar launchers that set the required environment variables.

    Environment variables used:
        - RANK: Global process rank
        - WORLD_SIZE: Total number of processes
        - LOCAL_RANK: GPU index on this node
        - MASTER_ADDR: Address of rank 0 process
        - MASTER_PORT: Port for communication

    Args:
        backend: Distributed backend. Use 'nccl' for GPU (works with both
            NVIDIA CUDA and AMD ROCm via RCCL). Use 'gloo' for CPU.
        init_method: URL for process group initialization. Defaults to 'env://'
            which reads from environment variables.

    Returns:
        Tuple of (rank, world_size, device)

    Raises:
        RuntimeError: If required environment variables are not set.
    """
    # Check for required environment variables
    required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing = [v for v in required_vars if v not in os.environ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables for distributed training: {missing}. "
            "Use torchrun to launch distributed training, e.g.: "
            "torchrun --nproc_per_node=4 train.py --distributed"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set device before initializing process group
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize process group
    if init_method is None:
        init_method = "env://"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )

    # Synchronize all processes
    dist.barrier()

    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed training environment.

    Safe to call even if not in distributed mode.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
    gradient_as_bucket_view: bool = True,
) -> DDP:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap. Should already be on the correct device.
        device_id: Local GPU device ID (typically LOCAL_RANK).
        find_unused_parameters: Set True if model has unused parameters in
            some forward passes. Slower but necessary for some architectures.
        broadcast_buffers: Whether to broadcast buffers on each forward pass.
        gradient_as_bucket_view: Memory optimization for gradients.

    Returns:
        DDP-wrapped model.

    Raises:
        RuntimeError: If distributed training is not initialized.
    """
    if not is_distributed():
        raise RuntimeError("Cannot wrap model with DDP: distributed training not initialized. Call setup_distributed() first.")

    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def get_fsdp_wrap_policy(
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
) -> ModuleWrapPolicy:
    """Get the FSDP auto-wrap policy for transformer models.

    FSDP wraps model modules to shard their parameters. This function returns
    a policy that wraps transformer encoder layers, which is the standard
    approach for transformer models.

    Args:
        transformer_layer_cls: Set of module classes to wrap. If None, uses
            default transformer layer classes (nn.TransformerEncoderLayer).

    Returns:
        ModuleWrapPolicy for FSDP auto-wrapping.
    """
    if transformer_layer_cls is None:
        transformer_layer_cls = {nn.TransformerEncoderLayer}
    return ModuleWrapPolicy(transformer_layer_cls)


def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    mixed_precision: Optional[MixedPrecision] = None,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    cpu_offload: bool = False,
) -> FSDP:
    """Wrap model with FullyShardedDataParallel (FSDP).

    FSDP shards model parameters, gradients, and optimizer states across GPUs,
    enabling training of models larger than single-GPU memory. Unlike DDP which
    replicates the full model on each GPU, FSDP only materializes full parameters
    during forward/backward passes.

    Args:
        model: Model to wrap. Should NOT be moved to device yet (FSDP handles this).
        sharding_strategy: How to shard the model:
            - FULL_SHARD: Shard parameters, gradients, and optimizer states (most memory efficient)
            - SHARD_GRAD_OP: Shard gradients and optimizer states only
            - NO_SHARD: Don't shard (equivalent to DDP)
            - HYBRID_SHARD: Shard within node, replicate across nodes
        mixed_precision: Optional mixed precision policy for FSDP.
        transformer_layer_cls: Set of module classes to wrap individually.
            If None, uses default transformer layer classes.
        cpu_offload: Whether to offload parameters to CPU when not in use.
            Saves GPU memory but slower due to CPU-GPU transfers.

    Returns:
        FSDP-wrapped model.

    Raises:
        RuntimeError: If distributed training is not initialized.

    Example:
        >>> model = SequencePredictor(...)
        >>> fsdp_model = wrap_model_fsdp(
        ...     model,
        ...     sharding_strategy=ShardingStrategy.FULL_SHARD,
        ... )
    """
    if not is_distributed():
        raise RuntimeError("Cannot wrap model with FSDP: distributed training not initialized. Call setup_distributed() first.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cannot wrap model with FSDP: CUDA is not available. "
            "FSDP requires GPU training. Ensure CUDA drivers are installed "
            "and GPUs are visible (run 'nvidia-smi' to verify)."
        )

    from torch.distributed.fsdp import CPUOffload

    wrap_policy = get_fsdp_wrap_policy(transformer_layer_cls)

    cpu_offload_config = CPUOffload(offload_params=cpu_offload) if cpu_offload else None

    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
    )


def is_fsdp_model(model: nn.Module) -> bool:
    """Check if a model is wrapped with FSDP.

    Args:
        model: Model to check.

    Returns:
        True if model is an FSDP instance, False otherwise.
    """
    return isinstance(model, FSDP)


def is_ddp_model(model: nn.Module) -> bool:
    """Check if a model is wrapped with DDP.

    Args:
        model: Model to check.

    Returns:
        True if model is a DDP instance, False otherwise.
    """
    return isinstance(model, DDP)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap a model from DDP or FSDP wrapper.

    Args:
        model: Model that may be wrapped with DDP or FSDP.

    Returns:
        The underlying model without the distributed wrapper.
    """
    if is_fsdp_model(model) or is_ddp_model(model):
        # DDP and FSDP store the wrapped module in .module attribute
        inner: nn.Module = model.module  # type: ignore[union-attr]
        return inner
    return model


def get_fsdp_state_dict(
    model: FSDP,
    sharded: bool = False,
    cpu_offload: bool = True,
    rank0_only: bool = True,
) -> dict:
    """Get state dict from an FSDP model.

    Uses FSDP's state_dict_type context manager to properly gather sharded
    parameters. By default, gathers full state dict on rank 0 for checkpoint
    compatibility with non-FSDP models.

    Args:
        model: FSDP-wrapped model.
        sharded: If True, return sharded state dict (each rank saves its shard).
            If False, gather full state dict on rank 0.
        cpu_offload: Offload state dict to CPU to save GPU memory.
        rank0_only: Only populate state dict on rank 0 (others get empty dict).
            Only used when sharded=False.

    Returns:
        State dict. When sharded=False and rank0_only=True, only rank 0 gets
        the full state dict; other ranks get an empty dict.

    Raises:
        TypeError: If model is not an FSDP instance.
        RuntimeError: If state dict gathering fails (e.g., NCCL timeout, OOM).
    """
    if not is_fsdp_model(model):
        raise TypeError(f"Expected FSDP model, got {type(model).__name__}")

    if sharded:
        state_dict_config = ShardedStateDictConfig(offload_to_cpu=cpu_offload)
        state_dict_type = StateDictType.SHARDED_STATE_DICT
    else:
        state_dict_config = FullStateDictConfig(
            offload_to_cpu=cpu_offload,
            rank0_only=rank0_only,
        )
        state_dict_type = StateDictType.FULL_STATE_DICT

    try:
        with FSDP.state_dict_type(model, state_dict_type, state_dict_config):
            return model.state_dict()
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "nccl" in error_msg or "timeout" in error_msg:
            raise RuntimeError(
                f"FSDP state dict gathering failed due to communication error. "
                f"This is often caused by GPU memory exhaustion or process desynchronization. "
                f"Try reducing model size or batch size. Original error: {e}"
            ) from e
        raise RuntimeError(
            f"Failed to get FSDP state dict (sharded={sharded}). "
            f"This may indicate corrupted FSDP state. Original error: {e}"
        ) from e


def set_fsdp_state_dict(
    model: FSDP,
    state_dict: dict,
    sharded: bool = False,
    strict: bool = True,
) -> None:
    """Load state dict into an FSDP model.

    Uses FSDP's state_dict_type context manager for proper loading. Supports
    loading both full state dicts (from non-FSDP or gathered checkpoints) and
    sharded state dicts.

    Args:
        model: FSDP-wrapped model to load state dict into.
        state_dict: State dict to load.
        sharded: If True, expect sharded state dict format.
            If False, expect full state dict format.
        strict: Whether to strictly enforce that the keys in state_dict match.

    Raises:
        TypeError: If model is not an FSDP instance.
        RuntimeError: If state dict loading fails (e.g., shape mismatch, key mismatch).
    """
    if not is_fsdp_model(model):
        raise TypeError(f"Expected FSDP model, got {type(model).__name__}")

    if sharded:
        state_dict_config = ShardedStateDictConfig()
        state_dict_type = StateDictType.SHARDED_STATE_DICT
    else:
        state_dict_config = FullStateDictConfig()
        state_dict_type = StateDictType.FULL_STATE_DICT

    try:
        with FSDP.state_dict_type(model, state_dict_type, state_dict_config):
            model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            raise RuntimeError(
                f"FSDP state dict shape mismatch. This usually means the checkpoint "
                f"was saved with different model architecture (embedding dim, layers, etc.). "
                f"Ensure checkpoint and model configurations match. Original error: {e}"
            ) from e
        if "Missing key" in error_msg or "Unexpected key" in error_msg:
            raise RuntimeError(
                f"FSDP state dict key mismatch (sharded={sharded}, strict={strict}). "
                f"For transfer learning, try strict=False. Original error: {e}"
            ) from e
        raise RuntimeError(
            f"Failed to load FSDP state dict (sharded={sharded}). Original error: {e}"
        ) from e


def get_fsdp_optimizer_state_dict(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    sharded: bool = False,
) -> dict:
    """Get optimizer state dict for an FSDP model.

    FSDP requires special handling for optimizer state dicts because optimizer
    states are also sharded across ranks.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer to get state dict from.
        sharded: If True, return sharded optimizer state dict.
            If False, gather full optimizer state dict.

    Returns:
        Optimizer state dict. When sharded=False, only rank 0 gets the full
        state dict; other ranks may receive empty or partial dicts.

    Raises:
        TypeError: If model is not an FSDP instance.
        RuntimeError: If optimizer state dict gathering fails.
    """
    if not is_fsdp_model(model):
        raise TypeError(f"Expected FSDP model, got {type(model).__name__}")

    if sharded:
        optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
        state_dict_type = StateDictType.SHARDED_STATE_DICT
    else:
        optim_state_dict_config = FullOptimStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        state_dict_type = StateDictType.FULL_STATE_DICT

    try:
        with FSDP.state_dict_type(model, state_dict_type, optim_state_dict_config=optim_state_dict_config):
            return FSDP.optim_state_dict(model, optimizer)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to get FSDP optimizer state dict (sharded={sharded}). "
            f"Ensure optimizer was created with FSDP model parameters. Original error: {e}"
        ) from e


def set_fsdp_optimizer_state_dict(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    optim_state_dict: dict,
    sharded: bool = False,
) -> None:
    """Load optimizer state dict into an FSDP model's optimizer.

    FSDP requires special handling for optimizer state dicts because optimizer
    states are also sharded across ranks. When loading a full optimizer state dict
    (sharded=False), it will be scattered to all ranks automatically.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer to load state dict into.
        optim_state_dict: Optimizer state dict to load. When sharded=False,
            this should be the full gathered state dict (typically only available
            on rank 0); it will be scattered to all ranks automatically.
        sharded: If True, expect sharded optimizer state dict format.
            If False, expect full optimizer state dict format.

    Raises:
        TypeError: If model is not an FSDP instance.
        RuntimeError: If optimizer state dict loading fails.
    """
    if not is_fsdp_model(model):
        raise TypeError(f"Expected FSDP model, got {type(model).__name__}")

    if sharded:
        optim_state_dict_config = ShardedOptimStateDictConfig()
        state_dict_type = StateDictType.SHARDED_STATE_DICT
    else:
        optim_state_dict_config = FullOptimStateDictConfig()
        state_dict_type = StateDictType.FULL_STATE_DICT

    try:
        with FSDP.state_dict_type(model, state_dict_type, optim_state_dict_config=optim_state_dict_config):
            sharded_optim_state = FSDP.optim_state_dict_to_load(model, optimizer, optim_state_dict)
            optimizer.load_state_dict(sharded_optim_state)
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to load FSDP optimizer state dict (sharded={sharded}). "
            f"This may be caused by: (1) optimizer type mismatch, (2) model architecture change, "
            f"or (3) corrupted checkpoint. Try setting load_optimizer=False. Original error: {e}"
        ) from e


def create_distributed_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Optional[Any] = None,
) -> Tuple[DataLoader, DistributedSampler]:
    """Create a DataLoader with DistributedSampler for DDP training.

    The sampler automatically partitions the dataset across processes.
    Note: batch_size is per-GPU, so effective batch size = batch_size * world_size.

    Args:
        dataset: Dataset to load from.
        batch_size: Batch size per GPU.
        shuffle: Whether to shuffle. Note: shuffling is handled by the sampler,
            not the DataLoader, so this sets sampler.shuffle.
        num_workers: Number of data loading workers per process.
        pin_memory: Whether to pin memory (recommended for GPU training).
        drop_last: Whether to drop the last incomplete batch. Recommended
            for DDP to ensure all processes have same number of batches.
        collate_fn: Custom collate function.

    Returns:
        Tuple of (DataLoader, DistributedSampler). The sampler is returned
        so you can call sampler.set_epoch(epoch) at the start of each epoch
        to ensure proper shuffling.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    return dataloader, sampler


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce. Will be modified in-place.
        average: If True, divide by world_size after sum.

    Returns:
        Reduced tensor (same object, modified in-place).
    """
    if not is_distributed():
        return tensor

    world_size = get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if average:
        tensor = tensor / world_size

    return tensor


def gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes.

    Args:
        tensor: Local tensor to gather.

    Returns:
        List of tensors from all processes (only valid on rank 0).
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    return gathered


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src rank to all other ranks.

    Args:
        obj: Object to broadcast (only used on src rank).
        src: Source rank.

    Returns:
        Broadcasted object on all ranks.
    """
    if not is_distributed():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)

    return obj_list[0]


def print_rank0(message: str) -> None:
    """Print message only on rank 0.

    Args:
        message: Message to print.
    """
    if is_main_process():
        print(message)


def gather_embeddings_for_unifrac(embeddings: torch.Tensor) -> torch.Tensor:
    """Gather embeddings from all processes for UniFrac pairwise distance computation.

    In distributed training (DDP/FSDP), each GPU only sees its local batch. UniFrac
    loss requires pairwise distances across ALL samples, so we need to gather
    embeddings from all GPUs before computing pairwise distances.

    Without gathering:
        GPU 0: samples [0,1,2,3] -> local pairwise distances only
        GPU 1: samples [4,5,6,7] -> local pairwise distances only
        Missing: cross-GPU pairs (0,4), (0,5), (1,4), etc.

    With gathering:
        All GPUs get samples [0,1,2,3,4,5,6,7] -> full pairwise distances

    Args:
        embeddings: Local embeddings tensor [local_batch_size, embedding_dim]

    Returns:
        Gathered embeddings tensor [global_batch_size, embedding_dim] where
        global_batch_size = local_batch_size * world_size.
        If not in distributed mode, returns the input unchanged.

    Note:
        This operation performs an all-gather, so all ranks receive the full
        gathered tensor. The gradient flows back correctly during backward pass.
    """
    raise NotImplementedError("gather_embeddings_for_unifrac not yet implemented")


def sync_batch_norm(model: nn.Module) -> nn.Module:
    """Convert BatchNorm layers to SyncBatchNorm for distributed training.

    SyncBatchNorm synchronizes batch statistics across all processes,
    which is important when using small per-GPU batch sizes.

    Args:
        model: Model with BatchNorm layers.

    Returns:
        Model with SyncBatchNorm layers.
    """
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


class DistributedTrainer:
    """Wrapper for Trainer that handles distributed training setup.

    This class wraps the existing Trainer to add DDP support without
    modifying the core Trainer implementation.

    Example:
        # Launch with: torchrun --nproc_per_node=4 train.py
        trainer = DistributedTrainer(
            model=model,
            loss_fn=loss_fn,
            backend="nccl",
        )
        trainer.train(train_loader, val_loader, epochs=100)
        trainer.cleanup()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        sync_batchnorm: bool = False,
        **trainer_kwargs,
    ):
        """Initialize DistributedTrainer.

        Args:
            model: Model to train.
            loss_fn: Loss function.
            backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU).
            find_unused_parameters: Whether to find unused parameters in DDP.
            sync_batchnorm: Whether to convert BatchNorm to SyncBatchNorm.
            **trainer_kwargs: Additional arguments passed to Trainer.
        """
        from aam.training.trainer import Trainer

        # Setup distributed environment
        self.rank, self.world_size, self.device = setup_distributed(backend=backend)
        self.local_rank = get_local_rank()

        # Move model to device
        model = model.to(self.device)

        # Convert BatchNorm if requested
        if sync_batchnorm:
            model = sync_batch_norm(model)

        # Wrap with DDP
        self.ddp_model = wrap_model_ddp(
            model,
            device_id=self.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

        # Update trainer kwargs with device
        trainer_kwargs["device"] = self.device

        # Only log on main process
        if not is_main_process():
            trainer_kwargs["tensorboard_dir"] = None

        # Create underlying trainer with DDP model
        self.trainer = Trainer(
            model=self.ddp_model,
            loss_fn=loss_fn,
            **trainer_kwargs,
        )

        self.backend = backend

    @property
    def model(self) -> nn.Module:
        """Get the underlying model (unwrapped from DDP)."""
        return self.ddp_model.module

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        **train_kwargs,
    ):
        """Run distributed training.

        Args:
            train_loader: Training data loader (should use DistributedSampler).
            val_loader: Validation data loader (optional).
            epochs: Number of epochs.
            **train_kwargs: Additional arguments passed to Trainer.train().

        Returns:
            Training results from Trainer.train().
        """
        return self.trainer.train(
            train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            **train_kwargs,
        )

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        cleanup_distributed()

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save checkpoint (only on main process).

        Args:
            path: Path to save checkpoint.
            **kwargs: Additional data to save.
        """
        if is_main_process():
            self.trainer.save_checkpoint(path, **kwargs)

        # Ensure all processes wait for save to complete
        if is_distributed():
            dist.barrier()

    def load_checkpoint(self, path: str, **kwargs):
        """Load checkpoint on all processes.

        Args:
            path: Path to checkpoint.
            **kwargs: Additional arguments to load_checkpoint.
        """
        return self.trainer.load_checkpoint(path, **kwargs)
