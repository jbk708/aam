"""Pretrain command for AAM CLI."""

import click
import sys
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, cast
from functools import partial
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader
from aam.data.dataset import ASVDataset, collate_fn
from skbio import DistanceMatrix
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.transformer import AttnImplementation
from aam.training.losses import MultiTaskLoss
from aam.training.trainer import Trainer, create_optimizer, create_scheduler
from aam.training.batch_size_finder import BatchSizeFinder
from aam.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    create_distributed_dataloader,
    is_main_process,
    get_local_rank,
    sync_batch_norm,
    wrap_model_ddp,
)
from aam.training.memory_profiler import MemoryProfiler, log_gpu_memory_stats
from aam.models.model_summary import log_model_summary
from aam.cli.utils import (
    setup_logging,
    setup_device,
    setup_expandable_segments,
    setup_random_seed,
    validate_file_path,
    validate_arguments,
)


@click.command()
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option(
    "--unifrac-matrix",
    required=True,
    type=click.Path(exists=True),
    help="Path to pre-computed UniFrac distance matrix (.npy, .h5, or .csv format)",
)
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory for checkpoints and logs")
@click.option("--epochs", default=100, type=int, help="Number of training epochs")
@click.option("--batch-size", default=8, type=int, help="Batch size (ignored if --auto-batch-size finds a different optimal size)")
@click.option(
    "--auto-batch-size/--no-auto-batch-size",
    default=True,
    help="Automatically find optimal batch size for GPU memory (default: enabled). "
    "Disable with --no-auto-batch-size to use --batch-size directly.",
)
@click.option(
    "--max-memory-fraction",
    default=0.8,
    type=float,
    help="Maximum GPU memory fraction to use for auto batch size finding (default: 0.8 = 80%%)",
)
@click.option(
    "--target-effective-batch-size",
    default=None,
    type=int,
    help="Target effective batch size for gradient accumulation tuning. "
    "If set and larger than found batch size, gradient accumulation steps are auto-computed.",
)
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--patience", default=10, type=int, help="Early stopping patience")
@click.option("--warmup-steps", default=10000, type=int, help="Learning rate warmup steps")
@click.option("--weight-decay", default=0.01, type=float, help="Weight decay for AdamW")
@click.option("--embedding-dim", default=128, type=int, help="Embedding dimension")
@click.option("--attention-heads", default=4, type=int, help="Number of attention heads")
@click.option("--attention-layers", default=4, type=int, help="Number of transformer layers")
@click.option("--max-bp", default=150, type=int, help="Maximum base pairs per sequence")
@click.option("--token-limit", default=1024, type=int, help="Maximum ASVs per sample")
@click.option(
    "--asv-sampling",
    default="first",
    type=click.Choice(["first", "abundance", "random"]),
    help="ASV selection strategy when sample exceeds token-limit: "
    "first (default, by matrix order), "
    "abundance (top N by count), "
    "random (random N each batch, acts as data augmentation)",
)
@click.option("--rarefy-depth", default=5000, type=int, help="Rarefaction depth")
@click.option("--test-size", default=0.2, type=float, help="Validation split size")
@click.option(
    "--unifrac-metric",
    default="unifrac",
    type=click.Choice(["unifrac", "faith_pd"]),
    help="UniFrac metric type (unifrac for pairwise, faith_pd for per-sample values)",
)
@click.option("--penalty", default=1.0, type=float, help="Weight for base/UniFrac loss")
@click.option("--nuc-penalty", default=1.0, type=float, help="Weight for nucleotide loss")
@click.option("--count-penalty", default=1.0, type=float, help="Weight for count loss (default: 1.0)")
@click.option(
    "--nuc-mask-ratio",
    default=0.15,
    type=float,
    help="Fraction of nucleotide positions to mask for MAE training (0.0 to disable, default: 0.15)",
)
@click.option(
    "--nuc-mask-strategy",
    default="random",
    type=click.Choice(["random", "span"]),
    help="Masking strategy: random (default) or span",
)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option(
    "--num-workers",
    default=4,
    type=int,
    help="Number of DataLoader worker processes (default: 4, use 0 to disable multiprocessing)",
)
@click.option("--resume-from", default=None, type=click.Path(exists=True), help="Path to checkpoint to resume from")
@click.option("--gradient-accumulation-steps", default=1, type=int, help="Number of gradient accumulation steps")
@click.option("--use-expandable-segments", is_flag=True, help="Enable PyTorch CUDA expandable segments for memory optimization")
@click.option("--max-grad-norm", default=None, type=float, help="Maximum gradient norm for clipping (None to disable)")
@click.option("--optimizer", default="adamw", type=click.Choice(["adamw", "adam", "sgd"]), help="Optimizer type")
@click.option(
    "--scheduler",
    default="warmup_cosine",
    type=click.Choice(["warmup_cosine", "cosine", "cosine_restarts", "plateau", "onecycle"]),
    help="Learning rate scheduler type (warmup_cosine: warmup+cosine decay, cosine: cosine annealing, cosine_restarts: cosine with warm restarts, plateau: reduce on plateau, onecycle: one cycle policy)",
)
@click.option(
    "--scheduler-t0",
    default=None,
    type=int,
    help="Initial restart period for cosine_restarts scheduler (default: num_training_steps // 4)",
)
@click.option(
    "--scheduler-t-mult", default=None, type=int, help="Restart period multiplier for cosine_restarts scheduler (default: 2)"
)
@click.option(
    "--scheduler-eta-min",
    default=None,
    type=float,
    help="Minimum learning rate for cosine/cosine_restarts schedulers (default: 0.0)",
)
@click.option(
    "--scheduler-patience",
    default=None,
    type=int,
    help="Patience for plateau scheduler (epochs to wait before reducing LR, default: 5)",
)
@click.option("--scheduler-factor", default=None, type=float, help="LR reduction factor for plateau scheduler (default: 0.3)")
@click.option("--scheduler-min-lr", default=None, type=float, help="Minimum learning rate for plateau scheduler (default: 0.0)")
@click.option(
    "--mixed-precision",
    default=None,
    type=click.Choice(["fp16", "bf16", "none"]),
    help="Mixed precision training mode (fp16, bf16, or none)",
)
@click.option("--compile-model", is_flag=True, help="Compile model with torch.compile() for optimization (PyTorch 2.0+)")
@click.option(
    "--tf32", is_flag=True, help="Enable TensorFloat32 for faster matrix multiplication on Ampere+ GPUs (RTX 30xx, A100, etc.)"
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=True,
    help="Use gradient checkpointing to reduce memory (default: enabled, use --no-gradient-checkpointing to disable)",
)
@click.option(
    "--attn-implementation",
    default="mem_efficient",
    type=click.Choice(["sdpa", "flash", "mem_efficient", "math"]),
    help="Attention implementation: mem_efficient (default), sdpa (auto-select), flash (Flash Attention), or math",
)
@click.option(
    "--asv-chunk-size", default=256, type=int, help="Process ASVs in chunks to reduce memory (default: 256, use 0 to disable)"
)
@click.option(
    "--no-sequence-cache",
    is_flag=True,
    help="Disable sequence tokenization cache (enabled by default for faster training)",
)
@click.option(
    "--distributed",
    is_flag=True,
    help="Enable distributed training with DDP. Use with torchrun: torchrun --nproc_per_node=4 -m aam.cli pretrain --distributed ...",
)
@click.option(
    "--data-parallel",
    is_flag=True,
    help="Enable DataParallel for multi-GPU pretraining on a single node. Unlike DDP, DataParallel preserves full pairwise comparisons for UniFrac loss. Note: GPU 0 has higher memory usage as it gathers all outputs.",
)
@click.option(
    "--sync-batchnorm",
    is_flag=True,
    help="Convert BatchNorm to SyncBatchNorm for distributed training (recommended for small batch sizes)",
)
@click.option(
    "--fsdp",
    is_flag=True,
    help="Enable FSDP (Fully Sharded Data Parallel) for memory-efficient distributed pretraining. Includes cross-GPU embedding gathering for correct UniFrac pairwise distances. Use with torchrun: torchrun --nproc_per_node=4 -m aam.cli pretrain --fsdp ...",
)
@click.option(
    "--fsdp-sharded-checkpoint",
    is_flag=True,
    help="Save FSDP checkpoints in sharded format (each rank saves its own shard). Faster for large models but requires same world size to load. Default: save full state dict on rank 0 for checkpoint compatibility.",
)
@click.option(
    "--memory-profile",
    is_flag=True,
    help="Enable GPU memory profiling. Logs peak memory usage per epoch and provides optimization recommendations.",
)
def pretrain(
    table: str,
    unifrac_matrix: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    auto_batch_size: bool,
    max_memory_fraction: float,
    target_effective_batch_size: Optional[int],
    lr: float,
    patience: int,
    warmup_steps: int,
    weight_decay: float,
    embedding_dim: int,
    attention_heads: int,
    attention_layers: int,
    max_bp: int,
    token_limit: int,
    asv_sampling: str,
    rarefy_depth: int,
    test_size: float,
    unifrac_metric: str,
    penalty: float,
    nuc_penalty: float,
    count_penalty: float,
    nuc_mask_ratio: float,
    nuc_mask_strategy: str,
    device: str,
    seed: Optional[int],
    num_workers: int,
    resume_from: Optional[str],
    gradient_accumulation_steps: int,
    use_expandable_segments: bool,
    max_grad_norm: Optional[float],
    optimizer: str,
    scheduler: str,
    scheduler_t0: Optional[int],
    scheduler_t_mult: Optional[int],
    scheduler_eta_min: Optional[float],
    scheduler_patience: Optional[int],
    scheduler_factor: Optional[float],
    scheduler_min_lr: Optional[float],
    mixed_precision: Optional[str],
    compile_model: bool,
    tf32: bool,
    gradient_checkpointing: bool,
    attn_implementation: str,
    asv_chunk_size: Optional[int],
    no_sequence_cache: bool,
    distributed: bool,
    data_parallel: bool,
    sync_batchnorm: bool,
    fsdp: bool,
    fsdp_sharded_checkpoint: bool,
    memory_profile: bool,
):
    """Pre-train SequenceEncoder on UniFrac and nucleotide prediction (self-supervised)."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_logging(output_path)
        logger = logging.getLogger(__name__)

        if tf32:
            torch.set_float32_matmul_precision("high")
            logger.info("TensorFloat32 enabled for faster matrix multiplication")

        logger.info("Starting SequenceEncoder pre-training")
        logger.info(f"Command: {' '.join(sys.argv)}")
        logger.info(f"Arguments: table={table}, unifrac_matrix={unifrac_matrix}")

        validate_file_path(table, "BIOM table")
        validate_file_path(unifrac_matrix, "UniFrac matrix")

        validate_arguments(
            batch_size=batch_size,
            lr=lr,
            test_size=test_size,
            epochs=epochs,
        )

        # Validate mutual exclusivity of distributed training options
        num_distributed_options = sum([distributed, data_parallel, fsdp])
        if num_distributed_options > 1:
            raise click.ClickException(
                "Cannot use multiple distributed training options together. Choose one of:\n"
                "  --distributed: DDP (multi-node, but UniFrac has local pairwise issue)\n"
                "  --data-parallel: DataParallel (single-node, full pairwise UniFrac)\n"
                "  --fsdp: FSDP (memory-efficient, full pairwise UniFrac via gathering)"
            )

        # Validate --fsdp-sharded-checkpoint requires --fsdp
        if fsdp_sharded_checkpoint and not fsdp:
            raise click.ClickException("--fsdp-sharded-checkpoint requires --fsdp to be enabled.")

        setup_expandable_segments(use_expandable_segments)

        # Determine if using any distributed training mode
        use_distributed = fsdp or distributed

        # Setup distributed training if enabled
        train_sampler = None
        val_sampler = None
        if use_distributed:
            rank, world_size, device_obj = setup_distributed(backend="nccl")
            mode = "FSDP" if fsdp else "Distributed"
            if is_main_process():
                logger.info(f"{mode} training enabled: rank {rank}/{world_size}")

            # Validate batch size for distributed training
            # UniFrac pairwise distances require at least 2 samples per GPU
            per_gpu_batch_size = batch_size // world_size
            if per_gpu_batch_size < 2:
                raise click.ClickException(
                    f"batch_size={batch_size} is too small for {world_size} GPUs. "
                    f"Each GPU would only get {per_gpu_batch_size} sample(s), but UniFrac "
                    f"requires at least 2 samples per GPU for pairwise distances. "
                    f"Use --batch-size {world_size * 2} or higher."
                )
        else:
            device_obj = setup_device(device)

        setup_random_seed(seed)

        if gradient_accumulation_steps < 1:
            raise click.ClickException("gradient_accumulation_steps must be >= 1")

        logger.info("Loading data...")
        biom_loader = BIOMLoader()
        table_obj = biom_loader.load_table(table)
        table_obj = biom_loader.rarefy(table_obj, depth=rarefy_depth, random_seed=seed)

        logger.info("Loading pre-computed UniFrac distance matrix...")
        unifrac_loader = UniFracLoader()

        # Get sample IDs from table for validation
        sample_ids = list(table_obj.ids(axis="sample"))
        logger.info(f"Total samples: {len(sample_ids)}")

        # Determine matrix format based on metric
        matrix_format = "pairwise" if unifrac_metric == "unifrac" else "faith_pd"

        # Load the matrix
        unifrac_distances = unifrac_loader.load_matrix(
            unifrac_matrix,
            sample_ids=sample_ids,
            matrix_format=matrix_format,
        )

        if unifrac_metric == "unifrac":
            unifrac_metric_name = "unweighted"
            encoder_type = "unifrac"
            base_output_dim = None
        else:
            unifrac_metric_name = "faith_pd"
            encoder_type = "faith_pd"
            base_output_dim = 1

        logger.info(
            f"Loaded UniFrac matrix: {type(unifrac_distances).__name__}, shape: {getattr(unifrac_distances, 'shape', 'N/A')}"
        )

        logger.info("Splitting data...")
        train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)
        logger.info(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

        # Log sample IDs for reproducibility
        train_ids_file = output_path / "train_sample_ids.txt"
        val_ids_file = output_path / "val_sample_ids.txt"
        train_ids_file.write_text("\n".join(train_ids))
        val_ids_file.write_text("\n".join(val_ids))
        logger.info(f"Train sample IDs saved to: {train_ids_file}")
        logger.info(f"Validation sample IDs saved to: {val_ids_file}")

        logger.info("Filtering tables for train/val splits...")
        train_table = table_obj.filter(train_ids, axis="sample", inplace=False)
        val_table = table_obj.filter(val_ids, axis="sample", inplace=False)
        logger.info("Table filtering complete")

        # Extract train/val distance matrices
        train_distance_matrix = None
        val_distance_matrix = None
        if unifrac_metric_name == "unweighted":
            if isinstance(unifrac_distances, DistanceMatrix):
                train_distance_matrix = unifrac_distances.filter(train_ids)
                val_distance_matrix = unifrac_distances.filter(val_ids)
            elif isinstance(unifrac_distances, np.ndarray):
                train_indices = [sample_ids.index(sid) for sid in train_ids]
                val_indices = [sample_ids.index(sid) for sid in val_ids]
                train_distance_matrix = unifrac_distances[np.ix_(train_indices, train_indices)]
                val_distance_matrix = unifrac_distances[np.ix_(val_indices, val_indices)]
        elif unifrac_metric_name == "faith_pd":
            if isinstance(unifrac_distances, pd.Series):
                train_distance_matrix = unifrac_distances.loc[train_ids]
                val_distance_matrix = unifrac_distances.loc[val_ids]
            elif isinstance(unifrac_distances, np.ndarray):
                train_indices = [sample_ids.index(sid) for sid in train_ids]
                val_indices = [sample_ids.index(sid) for sid in val_ids]
                train_distance_matrix = unifrac_distances[train_indices]
                val_distance_matrix = unifrac_distances[val_indices]

        logger.info("Creating datasets...")
        train_dataset = ASVDataset(
            table=train_table,
            metadata=None,
            unifrac_distances=train_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=None,
            unifrac_metric=unifrac_metric_name,
            cache_sequences=not no_sequence_cache,
        )

        val_dataset = ASVDataset(
            table=val_table,
            metadata=None,
            unifrac_distances=val_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=None,
            unifrac_metric=unifrac_metric_name,
            cache_sequences=not no_sequence_cache,
        )

        train_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=train_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
            asv_sampling=asv_sampling,
        )
        val_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=val_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
            asv_sampling=asv_sampling,
        )

        # Create dataloaders (with distributed sampler if using distributed training)
        if use_distributed:
            train_loader, train_sampler = create_distributed_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=train_collate,
            )
            val_loader, val_sampler = create_distributed_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=val_collate,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=train_collate,
                drop_last=True,
                prefetch_factor=2 if num_workers > 0 else None,
                pin_memory=device == "cuda",
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=val_collate,
                drop_last=True,
                prefetch_factor=2 if num_workers > 0 else None,
                pin_memory=device == "cuda",
            )

        logger.info("Creating model...")
        # Convert asv_chunk_size=0 to None (disabled)
        effective_asv_chunk_size = asv_chunk_size if asv_chunk_size is not None and asv_chunk_size > 0 else None
        model = SequenceEncoder(
            encoder_type=encoder_type,
            vocab_size=7,
            embedding_dim=embedding_dim,
            max_bp=max_bp,
            token_limit=token_limit,
            asv_num_layers=attention_layers,
            asv_num_heads=attention_heads,
            sample_num_layers=attention_layers,
            sample_num_heads=attention_heads,
            encoder_num_layers=attention_layers,
            encoder_num_heads=attention_heads,
            base_output_dim=base_output_dim,
            predict_nucleotides=True,
            asv_chunk_size=effective_asv_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            mask_ratio=nuc_mask_ratio,
            mask_strategy=nuc_mask_strategy,
            attn_implementation=cast(AttnImplementation, attn_implementation),
        )

        log_model_summary(model, logger)

        # Log memory after model creation (before moving to device)
        if memory_profile and torch.cuda.is_available():
            log_gpu_memory_stats(label="after_model_creation", logger=logger)

        loss_fn = MultiTaskLoss(
            penalty=penalty,
            nuc_penalty=nuc_penalty,
            count_penalty=count_penalty,
            class_weights=None,
            target_loss_type="huber",  # Default for pretraining (not used, but consistent)
        )

        # Auto batch size finding (only for single-GPU CUDA training)
        if auto_batch_size and device == "cuda" and not use_distributed and not data_parallel:
            if not torch.cuda.is_available():
                logger.warning("Auto batch size requires CUDA. Using --batch-size value directly.")
            else:
                logger.info("Finding optimal batch size...")
                # Move model to GPU temporarily for batch size finding
                model = model.to(device_obj)

                finder = BatchSizeFinder(
                    model=model,
                    loss_fn=loss_fn,
                    device=device_obj,
                    collate_fn=train_collate,
                )

                try:
                    result = finder.find_batch_size(
                        dataset=train_dataset,
                        min_batch_size=2,
                        max_batch_size=256,
                        target_effective_batch_size=target_effective_batch_size,
                        max_memory_fraction=max_memory_fraction,
                    )

                    old_batch_size = batch_size
                    old_grad_accum = gradient_accumulation_steps
                    batch_size = result.batch_size
                    gradient_accumulation_steps = result.gradient_accumulation_steps

                    logger.info(
                        f"Auto batch size: {batch_size} "
                        f"(was: {old_batch_size}, effective: {result.effective_batch_size}, "
                        f"grad_accum: {gradient_accumulation_steps}, "
                        f"memory: {result.peak_memory_mb:.0f}MB = {result.memory_fraction:.0%})"
                    )

                    # Recreate DataLoaders with optimal batch size if it changed
                    if batch_size != old_batch_size:
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=train_collate,
                            drop_last=True,
                            prefetch_factor=2 if num_workers > 0 else None,
                            pin_memory=True,
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=val_collate,
                            drop_last=True,
                            prefetch_factor=2 if num_workers > 0 else None,
                            pin_memory=True,
                        )
                        logger.info(f"DataLoaders recreated with batch_size={batch_size}")

                except RuntimeError as e:
                    logger.warning(f"Auto batch size failed: {e}. Using --batch-size={batch_size}")
                    # Move model back to CPU to avoid issues
                    model = model.cpu()
                    torch.cuda.empty_cache()

        elif auto_batch_size and (use_distributed or data_parallel):
            logger.info("Auto batch size disabled for distributed training. Using --batch-size directly.")

        # Handle distributed training setup
        if fsdp:
            # FSDP requires CUDA - validate upfront with clear message
            if not torch.cuda.is_available():
                raise click.ClickException(
                    "FSDP requires CUDA but no GPU is available. "
                    "Use --distributed for DDP (which supports CPU via gloo backend), "
                    "or ensure CUDA is properly installed (run 'nvidia-smi' to check)."
                )

            # FSDP handles device placement internally, don't move model beforehand
            # Convert BatchNorm to SyncBatchNorm if requested
            if sync_batchnorm:
                model = sync_batch_norm(model)
                if is_main_process():
                    logger.info("Converted BatchNorm to SyncBatchNorm for FSDP training")

            # Wrap model with FSDP for memory-efficient distributed training
            from aam.training.distributed import wrap_model_fsdp

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
                if memory_profile:
                    log_gpu_memory_stats(label="after_model_to_device", logger=logger)

        elif distributed:
            # Move model to device
            model = model.to(device_obj)

            # Convert BatchNorm to SyncBatchNorm if requested
            if sync_batchnorm:
                model = sync_batch_norm(model)
                if is_main_process():
                    logger.info("Converted BatchNorm to SyncBatchNorm for distributed training")

            # Wrap model with DDP
            model = wrap_model_ddp(model, device_id=get_local_rank())
            if is_main_process():
                logger.info("Model wrapped with DistributedDataParallel")
                if memory_profile:
                    log_gpu_memory_stats(label="after_model_to_device", logger=logger)

        elif data_parallel:
            # DataParallel for single-node multi-GPU pretraining
            # Unlike DDP, DP gathers outputs to GPU 0 before loss computation,
            # preserving full pairwise comparisons for UniFrac loss
            if not torch.cuda.is_available():
                raise click.ClickException("--data-parallel requires CUDA GPUs")

            num_gpus = torch.cuda.device_count()
            if num_gpus < 2:
                logger.warning(f"--data-parallel specified but only {num_gpus} GPU(s) available. Running on single GPU.")

            # Explicitly specify all available GPUs
            device_ids = list(range(num_gpus))
            logger.info(f"Using GPUs: {device_ids}")

            # Move model to primary GPU (device_ids[0])
            model = model.to(f"cuda:{device_ids[0]}")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel across {num_gpus} GPU(s)")
            logger.info("Note: GPU 0 has higher memory usage as it gathers all outputs for loss computation")

            if memory_profile:
                log_gpu_memory_stats(label="after_model_to_device", logger=logger)

        num_training_steps = len(train_loader) * epochs
        optimizer_obj = create_optimizer(model, optimizer_type=optimizer, lr=lr, weight_decay=weight_decay, freeze_base=False)
        # Build scheduler kwargs based on scheduler type and provided options
        scheduler_kwargs = {}
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

        scheduler_obj = create_scheduler(
            optimizer_obj,
            scheduler_type=scheduler,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            **scheduler_kwargs,
        )

        # Normalize mixed_precision: "none" -> None
        mixed_precision_normalized = None if mixed_precision == "none" else mixed_precision

        # Only log to TensorBoard on main process in distributed mode
        tensorboard_dir = str(output_path) if (not use_distributed or is_main_process()) else None

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer_obj,
            scheduler=scheduler_obj,
            device=device_obj,
            freeze_base=False,
            tensorboard_dir=tensorboard_dir,
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision_normalized,
            compile_model=compile_model,
            train_sampler=train_sampler,
            use_sharded_checkpoint=fsdp_sharded_checkpoint,
            gather_for_distributed=fsdp,  # Enable gathering for FSDP pretraining
        )

        if resume_from is not None:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            trainer.load_checkpoint(resume_from, load_optimizer=True, load_scheduler=True)

        logger.info("Starting pre-training...")
        checkpoint_dir = output_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Initialize memory profiler if enabled
        profiler = MemoryProfiler(enabled=memory_profile)
        if memory_profile and torch.cuda.is_available():
            log_gpu_memory_stats(label="before_training", logger=logger)
            logger.info("Memory profiling enabled - will log peak memory per epoch")

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            early_stopping_patience=patience,
            checkpoint_dir=str(checkpoint_dir),
            resume_from=resume_from,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        logger.info("Pre-training completed")
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
        logger.info(f"Best validation loss: {best_val_loss}")

        # Log final memory statistics if profiling enabled
        if memory_profile and torch.cuda.is_available():
            log_gpu_memory_stats(label="after_training", logger=logger)
            logger.info("-" * 60)
            logger.info("MEMORY PROFILING SUMMARY")
            logger.info("-" * 60)
            bytes_to_mb = 1024 * 1024
            bytes_to_gb = 1024 * 1024 * 1024
            peak_allocated = torch.cuda.max_memory_allocated() / bytes_to_mb
            peak_reserved = torch.cuda.max_memory_reserved() / bytes_to_mb
            logger.info(f"Peak memory allocated: {peak_allocated:.1f} MB ({peak_allocated / 1024:.2f} GB)")
            logger.info(f"Peak memory reserved:  {peak_reserved:.1f} MB ({peak_reserved / 1024:.2f} GB)")
            logger.info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
            logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
            logger.info(f"Attention implementation: {attn_implementation}")
            logger.info(f"Gradient checkpointing: {gradient_checkpointing}")
            # Get GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / bytes_to_gb
            utilization = (peak_allocated / 1024) / total_memory_gb * 100
            logger.info(f"GPU: {gpu_props.name} ({total_memory_gb:.1f} GB)")
            logger.info(f"Memory utilization: {utilization:.1f}%")
            logger.info("-" * 60)

        # Only save final model on main process in distributed mode
        if not use_distributed or is_main_process():
            final_model_path = output_path / "pretrained_encoder.pt"
            trainer.save_checkpoint(str(final_model_path), epoch=epochs - 1, best_val_loss=best_val_loss, metrics=history)
            logger.info(f"Pre-trained encoder saved to {final_model_path}")

        # Cleanup distributed training
        if use_distributed:
            cleanup_distributed()

    except Exception as e:
        if "logger" in locals():
            logger.error(f"Pre-training failed: {e}", exc_info=True)
        else:
            logging.error(f"Pre-training failed: {e}", exc_info=True)
        # Cleanup distributed training on error
        if "use_distributed" in locals() and use_distributed:
            cleanup_distributed()
        raise click.ClickException(str(e))
