"""Shared training utilities for AAM CLI commands.

This module extracts common patterns from pretrain.py and train.py to reduce
code duplication and ensure consistent behavior across training commands.
"""

import logging
from typing import Any, Optional

import click
import torch
import torch.nn as nn

from aam.training.distributed import (
    get_local_rank,
    is_main_process,
    sync_batch_norm,
    wrap_model_ddp,
    wrap_model_fsdp,
)


def validate_distributed_options(
    distributed: bool,
    data_parallel: bool,
    fsdp: bool,
    fsdp_sharded_checkpoint: bool,
) -> None:
    """Validate mutual exclusivity of distributed training options.

    Args:
        distributed: Whether DDP is enabled.
        data_parallel: Whether DataParallel is enabled.
        fsdp: Whether FSDP is enabled.
        fsdp_sharded_checkpoint: Whether FSDP sharded checkpoints are enabled.

    Raises:
        click.ClickException: If invalid combination of options.
    """
    num_distributed_options = sum([distributed, data_parallel, fsdp])
    if num_distributed_options > 1:
        raise click.ClickException(
            "Cannot use multiple distributed training options together. Choose one of:\n"
            "  --distributed: DDP (multi-node, but UniFrac has local pairwise issue)\n"
            "  --data-parallel: DataParallel (single-node, full pairwise UniFrac)\n"
            "  --fsdp: FSDP (memory-efficient, full pairwise UniFrac via gathering)"
        )

    if fsdp_sharded_checkpoint and not fsdp:
        raise click.ClickException("--fsdp-sharded-checkpoint requires --fsdp to be enabled.")


def build_scheduler_kwargs(
    scheduler: str,
    scheduler_t0: Optional[int],
    scheduler_t_mult: Optional[int],
    scheduler_eta_min: Optional[float],
    scheduler_patience: Optional[int],
    scheduler_factor: Optional[float],
    scheduler_min_lr: Optional[float],
) -> dict[str, Any]:
    """Build scheduler kwargs based on scheduler type and provided options.

    Args:
        scheduler: Scheduler type (cosine_restarts, cosine, plateau, etc.)
        scheduler_t0: Initial restart period for cosine_restarts.
        scheduler_t_mult: Restart period multiplier for cosine_restarts.
        scheduler_eta_min: Minimum learning rate for cosine schedulers.
        scheduler_patience: Patience for plateau scheduler.
        scheduler_factor: LR reduction factor for plateau scheduler.
        scheduler_min_lr: Minimum learning rate for plateau scheduler.

    Returns:
        Dictionary of scheduler kwargs.
    """
    scheduler_kwargs: dict[str, Any] = {}

    if scheduler == "cosine_restarts":
        if scheduler_t0 is not None:
            scheduler_kwargs["T_0"] = scheduler_t0
        if scheduler_t_mult is not None:
            scheduler_kwargs["T_mult"] = scheduler_t_mult
        if scheduler_eta_min is not None:
            scheduler_kwargs["eta_min"] = scheduler_eta_min
    elif scheduler == "cosine":
        if scheduler_eta_min is not None:
            scheduler_kwargs["eta_min"] = scheduler_eta_min
    elif scheduler == "plateau":
        if scheduler_patience is not None:
            scheduler_kwargs["patience"] = scheduler_patience
        if scheduler_factor is not None:
            scheduler_kwargs["factor"] = scheduler_factor
        if scheduler_min_lr is not None:
            scheduler_kwargs["min_lr"] = scheduler_min_lr

    return scheduler_kwargs


def wrap_data_parallel(
    model: nn.Module,
    logger: logging.Logger,
) -> nn.Module:
    """Wrap model with DataParallel for single-node multi-GPU training.

    Unlike DDP, DataParallel gathers outputs to GPU 0 before loss computation,
    preserving full pairwise comparisons for UniFrac loss.

    Args:
        model: The model to wrap.
        logger: Logger for informational messages.

    Returns:
        DataParallel-wrapped model.

    Raises:
        click.ClickException: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise click.ClickException(
            "--data-parallel requires CUDA but no GPU is available. "
            "Use --device cpu for CPU training, or ensure CUDA is properly installed."
        )

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        logger.warning(f"--data-parallel specified but only {num_gpus} GPU(s) available. Running on single GPU.")

    device_ids = list(range(num_gpus))
    logger.info(f"Using GPUs: {device_ids}")

    model = model.to(f"cuda:{device_ids[0]}")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(f"Model wrapped with DataParallel across {num_gpus} GPU(s)")
    logger.info("Note: GPU 0 has higher memory usage as it gathers all outputs for loss computation")

    return model


def setup_fsdp(
    model: nn.Module,
    sync_batchnorm: bool,
    logger: logging.Logger,
) -> nn.Module:
    """Setup FSDP (Fully Sharded Data Parallel) for distributed training.

    FSDP handles device placement internally and includes cross-GPU embedding
    gathering for correct UniFrac pairwise distances.

    Args:
        model: The model to wrap.
        sync_batchnorm: Whether to convert BatchNorm to SyncBatchNorm.
        logger: Logger for informational messages.

    Returns:
        FSDP-wrapped model.

    Raises:
        click.ClickException: If CUDA is not available or wrapping fails.
    """
    if not torch.cuda.is_available():
        raise click.ClickException(
            "FSDP requires CUDA but no GPU is available. "
            "Use --distributed for DDP (which supports CPU via gloo backend), "
            "or ensure CUDA is properly installed (run 'nvidia-smi' to check)."
        )

    if sync_batchnorm:
        model = sync_batch_norm(model)
        if is_main_process():
            logger.info("Converted BatchNorm to SyncBatchNorm for FSDP training")

    try:
        model = wrap_model_fsdp(model)
    except RuntimeError as e:
        logger.error(f"Failed to wrap model with FSDP: {e}", exc_info=True)
        raise click.ClickException(
            f"FSDP model wrapping failed: {e}\n"
            "Common causes:\n"
            "  - Model contains unsupported layer types\n"
            "  - Insufficient GPU memory for FSDP initialization\n"
            "  - Distributed process group not properly initialized\n"
            "Try using --distributed (DDP) instead, or reduce model size."
        )

    if is_main_process():
        logger.info("Model wrapped with FullyShardedDataParallel")
        logger.info("FSDP gathers embeddings across GPUs for correct UniFrac pairwise distances")

    return model


def setup_ddp(
    model: nn.Module,
    device: torch.device,
    sync_batchnorm: bool,
    find_unused_parameters: bool,
    logger: logging.Logger,
) -> nn.Module:
    """Setup DDP (Distributed Data Parallel) for distributed training.

    Args:
        model: The model to wrap.
        device: The device to use.
        sync_batchnorm: Whether to convert BatchNorm to SyncBatchNorm.
        find_unused_parameters: Whether to find unused parameters in DDP.
            Set to True when freeze_base=True or when using categorical features.
        logger: Logger for informational messages.

    Returns:
        DDP-wrapped model.
    """
    model = model.to(device)

    if sync_batchnorm:
        model = sync_batch_norm(model)
        if is_main_process():
            logger.info("Converted BatchNorm to SyncBatchNorm for distributed training")

    model = wrap_model_ddp(
        model,
        device_id=get_local_rank(),
        find_unused_parameters=find_unused_parameters,
    )

    if is_main_process():
        logger.info("Model wrapped with DistributedDataParallel")

    return model
