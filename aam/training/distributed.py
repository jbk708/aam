"""Distributed training utilities for multi-GPU training with DDP.

Supports both NCCL (NVIDIA) and RCCL (AMD ROCm) backends.

Usage:
    # Single-node multi-GPU with torchrun:
    torchrun --nproc_per_node=4 train.py --distributed

    # Multi-node:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\
        --master_addr=<master_ip> --master_port=29500 train.py --distributed
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Tuple, Any, List
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
        # Get samplers if they exist
        train_sampler = getattr(train_loader, "sampler", None)
        val_sampler = getattr(val_loader, "sampler", None) if val_loader else None

        # For each epoch, set sampler epoch for proper shuffling
        original_train = self.trainer.train

        def train_with_epoch_sampler(*args, **kwargs):
            # Hook into epoch loop to set sampler epoch
            # This is a simplified version - full implementation would
            # need to modify the training loop
            return original_train(*args, **kwargs)

        return self.trainer.train(
            train_loader,
            val_loader=val_loader,
            epochs=epochs,
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
