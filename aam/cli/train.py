"""Train command for AAM CLI."""

import click
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, cast
from functools import partial
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader
from aam.data.dataset import ASVDataset, collate_fn
from aam.data.categorical import CategoricalEncoder
from aam.data.normalization import CategoryNormalizer, GlobalNormalizer, parse_target_transform
from skbio import DistanceMatrix
from aam.models.sequence_predictor import SequencePredictor
from aam.models.transformer import AttnImplementation
from aam.training.losses import MultiTaskLoss
from aam.training.trainer import Trainer, create_optimizer, create_scheduler, load_pretrained_encoder
from aam.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    create_distributed_dataloader,
    is_main_process,
    get_local_rank,
    sync_batch_norm,
    wrap_model_ddp,
)
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
@click.option("--metadata", required=True, type=click.Path(exists=True), help="Path to metadata file (.tsv)")
@click.option("--metadata-column", required=True, help="Column name for target prediction")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory for checkpoints and logs")
@click.option("--epochs", default=100, type=int, help="Number of training epochs")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--patience", default=10, type=int, help="Early stopping patience")
@click.option("--warmup-steps", default=10000, type=int, help="Learning rate warmup steps")
@click.option("--weight-decay", default=0.01, type=float, help="Weight decay for AdamW")
@click.option("--embedding-dim", default=128, type=int, help="Embedding dimension")
@click.option("--attention-heads", default=4, type=int, help="Number of attention heads")
@click.option("--attention-layers", default=4, type=int, help="Number of transformer layers")
@click.option("--max-bp", default=150, type=int, help="Maximum base pairs per sequence")
@click.option("--token-limit", default=1024, type=int, help="Maximum ASVs per sample")
@click.option("--out-dim", default=1, type=int, help="Output dimension")
@click.option("--classifier", is_flag=True, help="Use classification mode")
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
@click.option("--target-penalty", default=1.0, type=float, help="Weight for target loss (default: 1.0)")
@click.option("--count-penalty", default=1.0, type=float, help="Weight for count loss (default: 1.0)")
@click.option(
    "--count-prediction/--no-count-prediction",
    default=True,
    help="Enable/disable count prediction head (default: enabled). Use --no-count-prediction to save memory.",
)
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
@click.option("--class-weights", default=None, help="Class weights for classification (optional)")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option(
    "--num-workers",
    default=4,
    type=int,
    help="Number of DataLoader worker processes (default: 4, use 0 to disable multiprocessing)",
)
@click.option("--resume-from", default=None, type=click.Path(exists=True), help="Path to checkpoint to resume from")
@click.option("--freeze-base", is_flag=True, help="Freeze base model parameters")
@click.option(
    "--pretrained-encoder", default=None, type=click.Path(exists=True), help="Path to pretrained SequenceEncoder checkpoint"
)
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
    "--target-transform",
    default=None,
    type=click.Choice(["none", "minmax", "zscore", "zscore-category", "log-minmax", "log-zscore"]),
    help="Target normalization strategy: "
    "none (no transform), "
    "minmax (scale to [0,1]), "
    "zscore (global z-score), "
    "zscore-category (per-category z-score, requires --normalize-by), "
    "log-minmax (log(y+1) then minmax), "
    "log-zscore (log(y+1) then z-score). "
    "Default: minmax. Replaces legacy --normalize-targets, --normalize-targets-by, --log-transform-targets.",
)
@click.option(
    "--normalize-by",
    default=None,
    help="Categorical columns for per-category normalization (comma-separated). "
    "Required with --target-transform zscore-category. Example: 'location' or 'location,season'.",
)
@click.option(
    "--normalize-targets/--no-normalize-targets",
    default=True,
    help="[DEPRECATED] Use --target-transform instead. Normalize targets to [0, 1] range (default: enabled).",
)
@click.option(
    "--normalize-targets-by",
    default=None,
    help="[DEPRECATED] Use --target-transform zscore-category --normalize-by instead. "
    "Per-category z-score normalization using comma-separated column names.",
)
@click.option(
    "--log-transform-targets",
    is_flag=True,
    help="[DEPRECATED] Use --target-transform log-minmax or log-zscore instead. Apply log(y+1) transform.",
)
@click.option(
    "--loss-type",
    type=click.Choice(["mse", "mae", "huber", "quantile", "asymmetric"]),
    default="huber",
    help="Loss function for regression targets: mse, mae, huber (default), quantile, or asymmetric",
)
@click.option(
    "--quantiles",
    default=None,
    help="Comma-separated quantile levels for quantile regression, e.g., '0.1,0.5,0.9'. Required when --loss-type=quantile.",
)
@click.option(
    "--over-penalty",
    type=float,
    default=1.0,
    help="Penalty weight for over-predictions (pred > actual) when using --loss-type=asymmetric. Default: 1.0",
)
@click.option(
    "--under-penalty",
    type=float,
    default=1.0,
    help="Penalty weight for under-predictions (pred < actual) when using --loss-type=asymmetric. Default: 1.0",
)
@click.option(
    "--no-sequence-cache",
    is_flag=True,
    help="Disable sequence tokenization cache (enabled by default for faster training)",
)
@click.option(
    "--distributed",
    is_flag=True,
    help="Enable distributed training with DDP. Use with torchrun: torchrun --nproc_per_node=4 -m aam.cli train --distributed ...",
)
@click.option(
    "--data-parallel",
    is_flag=True,
    help="Enable DataParallel for multi-GPU training on a single node. Unlike DDP, DataParallel preserves full pairwise comparisons for UniFrac loss. Note: GPU 0 has higher memory usage as it gathers all outputs.",
)
@click.option(
    "--sync-batchnorm",
    is_flag=True,
    help="Convert BatchNorm to SyncBatchNorm for distributed training (recommended for small batch sizes)",
)
@click.option(
    "--fsdp",
    is_flag=True,
    help="Enable FSDP (Fully Sharded Data Parallel) for memory-efficient distributed training. Use with torchrun: torchrun --nproc_per_node=4 -m aam.cli train --fsdp ...",
)
@click.option(
    "--fsdp-sharded-checkpoint",
    is_flag=True,
    help="Save FSDP checkpoints in sharded format (each rank saves its own shard). Faster for large models but requires same world size to load. Default: save full state dict on rank 0 for checkpoint compatibility.",
)
@click.option(
    "--target-layer-norm/--no-target-layer-norm",
    default=True,
    help="Apply LayerNorm before target projection (default: enabled)",
)
@click.option(
    "--bounded-targets",
    is_flag=True,
    help="Apply sigmoid to bound regression output to [0, 1] (default: unbounded)",
)
@click.option(
    "--learnable-output-scale",
    is_flag=True,
    help="Add learnable scale and bias after target projection",
)
@click.option(
    "--categorical-columns",
    default=None,
    help="Comma-separated categorical column names from metadata for conditioning target predictions",
)
@click.option(
    "--categorical-embed-dim",
    default=16,
    type=int,
    help="Embedding dimension for categorical features (default: 16)",
)
@click.option(
    "--categorical-fusion",
    default="concat",
    type=click.Choice(["concat", "add", "gmu", "cross-attention"]),
    help="Fusion strategy for categorical embeddings: concat (concatenate + project), add (project + add), gmu (gated multimodal unit after pooling), or cross-attention (position-specific via cross-attention). Default: concat",
)
@click.option(
    "--cross-attn-heads",
    default=8,
    type=click.IntRange(1, 64),
    help="Number of attention heads for cross-attention fusion (default: 8). Only used with --categorical-fusion cross-attention.",
)
@click.option(
    "--output-activation",
    default="none",
    type=click.Choice(["none", "relu", "softplus", "exp"]),
    help="Output activation for non-negative regression: none (default), relu, softplus (recommended), exp. Cannot be used with --bounded-targets or --classifier.",
)
@click.option(
    "--regressor-hidden-dims",
    default=None,
    help="Comma-separated hidden layer dimensions for MLP regression head. E.g., '64,32' creates MLP: embedding_dim -> 64 -> 32 -> out_dim. Default: single linear layer.",
)
@click.option(
    "--regressor-dropout",
    default=0.0,
    type=click.FloatRange(0.0, 1.0, max_open=True),
    help="Dropout rate between MLP regression head layers (default: 0.0, no dropout). Must be in [0.0, 1.0).",
)
@click.option(
    "--conditional-output-scaling",
    default=None,
    help="Comma-separated categorical column names for conditional output scaling. Learns per-category scale and bias applied after base prediction. Requires --categorical-columns.",
)
@click.option(
    "--best-metric",
    default="val_loss",
    type=click.Choice(["val_loss", "r2", "mae", "accuracy", "f1"]),
    help="Metric to use for best model selection: val_loss (default), r2 (higher better), mae (lower better), accuracy (higher better), f1 (higher better).",
)
def train(
    table: str,
    unifrac_matrix: str,
    metadata: str,
    metadata_column: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    warmup_steps: int,
    weight_decay: float,
    embedding_dim: int,
    attention_heads: int,
    attention_layers: int,
    max_bp: int,
    token_limit: int,
    out_dim: int,
    classifier: bool,
    rarefy_depth: int,
    test_size: float,
    unifrac_metric: str,
    penalty: float,
    nuc_penalty: float,
    target_penalty: float,
    count_penalty: float,
    count_prediction: bool,
    nuc_mask_ratio: float,
    nuc_mask_strategy: str,
    class_weights: Optional[str],
    device: str,
    seed: Optional[int],
    num_workers: int,
    resume_from: Optional[str],
    freeze_base: bool,
    pretrained_encoder: Optional[str],
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
    asv_chunk_size: int,
    target_transform: Optional[str],
    normalize_by: Optional[str],
    normalize_targets: bool,
    normalize_targets_by: Optional[str],
    log_transform_targets: bool,
    loss_type: str,
    quantiles: Optional[str],
    over_penalty: float,
    under_penalty: float,
    no_sequence_cache: bool,
    distributed: bool,
    data_parallel: bool,
    sync_batchnorm: bool,
    fsdp: bool,
    fsdp_sharded_checkpoint: bool,
    target_layer_norm: bool,
    bounded_targets: bool,
    learnable_output_scale: bool,
    categorical_columns: Optional[str],
    categorical_embed_dim: int,
    categorical_fusion: str,
    cross_attn_heads: int,
    output_activation: str,
    regressor_hidden_dims: Optional[str],
    regressor_dropout: float,
    conditional_output_scaling: Optional[str],
    best_metric: str,
):
    """Train AAM model on microbial sequencing data."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_logging(output_path)
        logger = logging.getLogger(__name__)

        if tf32:
            torch.set_float32_matmul_precision("high")
            logger.info("TensorFloat32 enabled for faster matrix multiplication")

        logger.info("Starting AAM training")
        logger.info(f"Arguments: table={table}, unifrac_matrix={unifrac_matrix}, metadata={metadata}")

        validate_file_path(table, "BIOM table")
        validate_file_path(unifrac_matrix, "UniFrac matrix")
        validate_file_path(metadata, "Metadata file")

        validate_arguments(
            batch_size=batch_size,
            classifier=classifier,
            out_dim=out_dim,
            lr=lr,
            test_size=test_size,
            epochs=epochs,
        )

        # Parse and validate target transform configuration
        # Supports both new --target-transform and legacy flags with deprecation warnings
        resolved_transform, uses_log_transform = parse_target_transform(
            target_transform=target_transform,
            normalize_targets=normalize_targets,
            normalize_targets_by=normalize_targets_by,
            log_transform_targets=log_transform_targets,
        )

        # Handle --normalize-by (new) and --normalize-targets-by (legacy) columns
        normalize_by_columns: list[str] = []
        if normalize_by:
            normalize_by_columns = [col.strip() for col in normalize_by.split(",")]
        elif normalize_targets_by:
            normalize_by_columns = [col.strip() for col in normalize_targets_by.split(",")]

        # Validate zscore-category requires normalize-by columns
        if resolved_transform == "zscore-category" and not normalize_by_columns:
            raise click.ClickException(
                "--target-transform zscore-category requires --normalize-by to specify categorical columns. "
                "Example: --target-transform zscore-category --normalize-by location,season"
            )

        # Map resolved transform to internal flags for backward compatibility with dataset
        effective_normalize_targets = resolved_transform in ("minmax", "log-minmax")
        effective_log_transform = uses_log_transform
        use_zscore = resolved_transform in ("zscore", "log-zscore")
        use_category_zscore = resolved_transform == "zscore-category"

        logger.info(f"Target transform: {resolved_transform}")
        if normalize_by_columns:
            logger.info(f"Per-category normalization columns: {normalize_by_columns}")

        # Auto-enable bounded_targets when using log_transform with normalization
        # Without bounds, model can output values > 1 which explode after exp()
        if effective_log_transform and effective_normalize_targets and not bounded_targets and not classifier:
            bounded_targets = True
            logger.info(
                "Auto-enabling --bounded-targets for log transform with normalization "
                "(prevents exp() overflow from unbounded predictions)"
            )

        setup_expandable_segments(use_expandable_segments)

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

        # Parse and validate quantile regression configuration
        quantiles_list: Optional[List[float]] = None
        num_quantiles: Optional[int] = None
        if loss_type == "quantile":
            if quantiles is None:
                quantiles = "0.1,0.5,0.9"  # Default quantiles
                logger.info(f"Using default quantiles: {quantiles}")
            try:
                quantiles_list = [float(q.strip()) for q in quantiles.split(",")]
            except ValueError as e:
                raise click.ClickException(
                    f"Invalid --quantiles: could not parse '{quantiles}'. "
                    f"Expected comma-separated values in (0, 1) like '0.1,0.5,0.9'. Error: {e}"
                )
            for q in quantiles_list:
                if not (0 < q < 1):
                    raise click.ClickException(f"Quantile values must be in (0, 1), got {q}. Example: --quantiles 0.1,0.5,0.9")
            num_quantiles = len(quantiles_list)
            if classifier:
                raise click.ClickException("--loss-type quantile cannot be used with --classifier")
        elif loss_type == "asymmetric":
            if classifier:
                raise click.ClickException("--loss-type asymmetric cannot be used with --classifier")
            if over_penalty <= 0:
                raise click.ClickException(f"--over-penalty must be positive, got {over_penalty}")
            if under_penalty <= 0:
                raise click.ClickException(f"--under-penalty must be positive, got {under_penalty}")
        else:
            # Warn if asymmetric penalty flags are set but loss type is not asymmetric
            if over_penalty != 1.0 or under_penalty != 1.0:
                logger.warning(
                    f"--over-penalty ({over_penalty}) and --under-penalty ({under_penalty}) are ignored "
                    f"when --loss-type is not 'asymmetric'. Current loss type: {loss_type}"
                )
        if quantiles is not None and loss_type != "quantile":
            raise click.ClickException("--quantiles requires --loss-type quantile")

        if not count_prediction and count_penalty != 0.0:
            logger.warning(
                f"--count-penalty ({count_penalty}) is ignored when --no-count-prediction is set. "
                "Count prediction head is disabled."
            )

        # Setup distributed training if enabled
        train_sampler = None
        val_sampler = None
        if fsdp or distributed:
            rank, world_size, device_obj = setup_distributed(backend="nccl")
            mode = "FSDP" if fsdp else "Distributed"
            if is_main_process():
                logger.info(f"{mode} training enabled: rank {rank}/{world_size}")

            # Validate batch size for distributed training
            per_gpu_batch_size = batch_size // world_size
            if per_gpu_batch_size < 2:
                unifrac_note = " UniFrac requires at least 2 samples per GPU for pairwise distances." if distributed else ""
                raise click.ClickException(
                    f"batch_size={batch_size} is too small for {world_size} GPUs. "
                    f"Each GPU would only get {per_gpu_batch_size} sample(s).{unifrac_note} "
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

        metadata_df = pd.read_csv(metadata, sep="\t", encoding="utf-8-sig")
        metadata_df.columns = metadata_df.columns.str.strip()
        if "sample_id" not in metadata_df.columns:
            found_columns = list(metadata_df.columns)
            raise ValueError(
                f"Metadata file must have 'sample_id' column.\n"
                f"Found columns: {found_columns}\n"
                f"Expected: 'sample_id'\n"
                f"Tip: Check for whitespace or encoding issues in column names."
            )
        if metadata_column not in metadata_df.columns:
            found_columns = list(metadata_df.columns)
            raise ValueError(
                f"Metadata column '{metadata_column}' not found in metadata file.\n"
                f"Found columns: {found_columns}\n"
                f"Tip: Check for whitespace or encoding issues in column names."
            )

        # Parse and validate categorical columns
        categorical_column_list: list[str] = []
        categorical_encoder: Optional[CategoricalEncoder] = None
        categorical_cardinalities: Optional[dict[str, int]] = None

        if categorical_columns:
            categorical_column_list = [col.strip() for col in categorical_columns.split(",")]
            found_columns = list(metadata_df.columns)
            for col in categorical_column_list:
                if col not in found_columns:
                    raise ValueError(
                        f"Categorical column '{col}' not found in metadata file.\n"
                        f"Found columns: {found_columns}\n"
                        f"Tip: Check for whitespace or encoding issues in column names."
                    )
            logger.info(f"Categorical columns: {categorical_column_list}")

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

        # Get actual sample IDs from loaded matrix (may be filtered to intersection)
        if isinstance(unifrac_distances, DistanceMatrix):
            matrix_sample_ids = list(unifrac_distances.ids)
        elif isinstance(unifrac_distances, pd.Series):
            matrix_sample_ids = list(unifrac_distances.index)
        else:
            # For numpy arrays, use original sample_ids (matrix should match)
            matrix_sample_ids = sample_ids

        # Filter table to only include samples present in the matrix
        if set(matrix_sample_ids) != set(sample_ids):
            logger.info(
                f"Filtering BIOM table to match matrix samples: "
                f"{len(matrix_sample_ids)} samples (from {len(sample_ids)} original)"
            )
            # Filter table to only include samples in matrix
            table_obj = table_obj.filter(matrix_sample_ids, axis="sample", inplace=False)
            sample_ids = matrix_sample_ids

        if unifrac_metric == "unifrac":
            unifrac_metric_name = "unweighted"
            encoder_type = "unifrac"
        else:
            unifrac_metric_name = "faith_pd"
            encoder_type = "faith_pd"

        logger.info(
            f"Loaded UniFrac matrix: {type(unifrac_distances).__name__}, shape: {getattr(unifrac_distances, 'shape', 'N/A')}"
        )

        logger.info("Splitting data...")
        train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)

        # For DDP/FSDP, broadcast train/val splits from rank 0 to ensure consistency
        if distributed or fsdp:
            import torch.distributed as dist

            split_data = [train_ids, val_ids]
            dist.broadcast_object_list(split_data, src=0)
            train_ids, val_ids = split_data[0], split_data[1]

        logger.info(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

        # Save train/val sample IDs to output directory (only on main process)
        if not (distributed or fsdp) or is_main_process():
            train_samples_path = output_path / "train_samples.txt"
            val_samples_path = output_path / "val_samples.txt"
            with open(train_samples_path, "w") as f:
                for sample_id in train_ids:
                    f.write(f"{sample_id}\n")
            with open(val_samples_path, "w") as f:
                for sample_id in val_ids:
                    f.write(f"{sample_id}\n")
            logger.info(f"Saved sample lists: {train_samples_path}, {val_samples_path}")

        logger.info("Filtering tables for train/val splits...")
        train_table = table_obj.filter(train_ids, axis="sample", inplace=False)
        val_table = table_obj.filter(val_ids, axis="sample", inplace=False)
        logger.info("Table filtering complete")

        train_metadata = metadata_df[metadata_df["sample_id"].isin(train_ids)]
        val_metadata = metadata_df[metadata_df["sample_id"].isin(val_ids)]

        # Fit categorical encoder on training data only
        # Note: train_metadata is consistent across DDP processes due to broadcast above
        if categorical_column_list:
            categorical_encoder = CategoricalEncoder()
            categorical_encoder.fit(train_metadata, columns=categorical_column_list)
            categorical_cardinalities = categorical_encoder.get_cardinalities()
            logger.info(f"Categorical cardinalities: {categorical_cardinalities}")

        conditional_scaling_columns: Optional[list[str]] = None
        if conditional_output_scaling:
            if not categorical_columns:
                raise click.ClickException("--conditional-output-scaling requires --categorical-columns to be set")
            conditional_scaling_columns = [col.strip() for col in conditional_output_scaling.split(",")]
            for col in conditional_scaling_columns:
                if col not in categorical_column_list:
                    raise click.ClickException(
                        f"Conditional scaling column '{col}' not in --categorical-columns. Available: {categorical_column_list}"
                    )
            logger.info(f"Conditional output scaling columns: {conditional_scaling_columns}")

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

        # Fit normalizer based on resolved transform
        category_normalizer: Optional[CategoryNormalizer] = None
        global_normalizer: Optional[GlobalNormalizer] = None

        if use_category_zscore:
            # Per-category z-score normalization
            for col in normalize_by_columns:
                if col not in train_metadata.columns:
                    raise click.ClickException(
                        f"Column '{col}' specified in --normalize-by not found in metadata. "
                        f"Available columns: {list(train_metadata.columns)}"
                    )

            # Extract training targets for normalization
            train_targets = train_metadata.set_index("sample_id")[metadata_column].values.astype(float)

            # Apply log transform to targets before fitting normalizer (if enabled)
            if effective_log_transform:
                train_targets = np.log(train_targets + 1)

            # Fit the category normalizer
            category_normalizer = CategoryNormalizer()
            category_normalizer.fit(
                targets=train_targets,
                metadata=train_metadata,
                columns=normalize_by_columns,
                sample_ids=list(train_metadata["sample_id"]),
            )

            # Log category statistics
            logger.info(f"Fitted CategoryNormalizer with {len(category_normalizer.stats)} categories")
            logger.info(f"  Global: mean={category_normalizer.global_mean:.4f}, std={category_normalizer.global_std:.4f}")
            for cat_key, stats in list(category_normalizer.stats.items())[:5]:
                logger.info(f"  {cat_key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            if len(category_normalizer.stats) > 5:
                logger.info(f"  ... and {len(category_normalizer.stats) - 5} more categories")

        elif use_zscore:
            # Global z-score normalization
            train_targets = train_metadata.set_index("sample_id")[metadata_column].values.astype(float)

            # Apply log transform before z-score (if enabled)
            if effective_log_transform:
                train_targets = np.log(train_targets + 1)

            global_normalizer = GlobalNormalizer(method="zscore")
            global_normalizer.fit(train_targets)
            logger.info(f"Fitted GlobalNormalizer (zscore): mean={global_normalizer.mean:.4f}, std={global_normalizer.std:.4f}")

        logger.info("Creating datasets...")
        train_dataset = ASVDataset(
            table=train_table,
            metadata=train_metadata,
            unifrac_distances=train_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=metadata_column,
            unifrac_metric=unifrac_metric_name,
            normalize_targets=effective_normalize_targets,
            log_transform_targets=effective_log_transform,
            normalize_counts=effective_normalize_targets,  # Normalize counts along with targets
            cache_sequences=not no_sequence_cache,
            categorical_encoder=categorical_encoder,
            category_normalizer=category_normalizer,
            global_normalizer=global_normalizer,
        )

        # Get normalization parameters from training set
        target_normalization_params = train_dataset.get_normalization_params()
        if target_normalization_params:
            if target_normalization_params.get("log_transform"):
                logger.info("Target log transform enabled: log(y+1) applied, inverse is exp(x)-1")
            if "target_scale" in target_normalization_params:
                logger.info(
                    f"Target normalization enabled: min={target_normalization_params['target_min']:.4f}, "
                    f"max={target_normalization_params['target_max']:.4f}, "
                    f"scale={target_normalization_params['target_scale']:.4f}"
                )

        count_normalization_params = train_dataset.get_count_normalization_params()
        if count_normalization_params:
            logger.info(
                f"Count normalization enabled: min={count_normalization_params['count_min']:.4f}, "
                f"max={count_normalization_params['count_max']:.4f}, "
                f"scale={count_normalization_params['count_scale']:.4f}"
            )

        val_dataset = ASVDataset(
            table=val_table,
            metadata=val_metadata,
            unifrac_distances=val_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=metadata_column,
            unifrac_metric=unifrac_metric_name,
            normalize_targets=effective_normalize_targets,
            log_transform_targets=effective_log_transform,
            # Use same normalization params as training set for consistency
            target_min=train_dataset.target_min if effective_normalize_targets else None,
            target_max=train_dataset.target_max if effective_normalize_targets else None,
            normalize_counts=effective_normalize_targets,  # Normalize counts along with targets
            # Use same count normalization params as training set for consistency
            count_min=train_dataset.count_min if effective_normalize_targets else None,
            count_max=train_dataset.count_max if effective_normalize_targets else None,
            cache_sequences=not no_sequence_cache,
            categorical_encoder=categorical_encoder,
            category_normalizer=category_normalizer,  # Use same normalizer as training set
            global_normalizer=global_normalizer,  # Use same normalizer as training set
        )

        train_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=train_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
        )
        val_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=val_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
        )

        # Create dataloaders (with distributed sampler if distributed or fsdp)
        if distributed or fsdp:
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

        # Parse regressor hidden dims
        regressor_hidden_dims_list: Optional[list[int]] = None
        if regressor_hidden_dims:
            try:
                parts = regressor_hidden_dims.split(",")
                regressor_hidden_dims_list = []
                for i, part in enumerate(parts):
                    stripped = part.strip()
                    if not stripped:
                        raise click.ClickException(
                            f"Invalid --regressor-hidden-dims: empty value at position {i + 1}. "
                            f"Expected comma-separated positive integers like '64,32'. Got: '{regressor_hidden_dims}'"
                        )
                    dim = int(stripped)
                    if dim <= 0:
                        raise click.ClickException(
                            f"Invalid --regressor-hidden-dims: dimension must be positive, got {dim} at position {i + 1}. "
                            f"Expected comma-separated positive integers like '64,32'."
                        )
                    regressor_hidden_dims_list.append(dim)
                logger.info(f"MLP regression head: {regressor_hidden_dims_list}")
            except ValueError as e:
                raise click.ClickException(
                    f"Invalid --regressor-hidden-dims: could not parse '{regressor_hidden_dims}'. "
                    f"Expected comma-separated positive integers like '64,32'. Error: {e}"
                )

        model = SequencePredictor(
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
            count_num_layers=attention_layers,
            count_num_heads=attention_heads,
            target_num_layers=attention_layers,
            target_num_heads=attention_heads,
            out_dim=out_dim,
            is_classifier=classifier,
            freeze_base=freeze_base,
            predict_nucleotides=True,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=cast(AttnImplementation, attn_implementation),
            asv_chunk_size=effective_asv_chunk_size,
            mask_ratio=nuc_mask_ratio,
            mask_strategy=nuc_mask_strategy,
            target_layer_norm=target_layer_norm,
            bounded_targets=bounded_targets,
            learnable_output_scale=learnable_output_scale,
            categorical_cardinalities=categorical_cardinalities,
            categorical_embed_dim=categorical_embed_dim,
            categorical_fusion=categorical_fusion,
            cross_attn_heads=cross_attn_heads,
            output_activation=output_activation,
            regressor_hidden_dims=regressor_hidden_dims_list,
            regressor_dropout=regressor_dropout,
            conditional_scaling_columns=conditional_scaling_columns,
            num_quantiles=num_quantiles,
            count_prediction=count_prediction,
        )

        log_model_summary(model, logger)

        if pretrained_encoder is not None:
            load_result = load_pretrained_encoder(pretrained_encoder, model, strict=False, logger=logger)
            if load_result["loaded_keys"] == 0:
                raise click.ClickException(
                    "No keys were loaded from pretrained encoder. "
                    "Check that pretrain and train use the same model configuration."
                )

        # Auto-disable nuc_penalty when freeze_base is True (frozen encoder can't improve)
        effective_nuc_penalty = nuc_penalty
        if freeze_base and nuc_penalty > 0:
            logger.info(f"Auto-disabling nucleotide loss (--freeze-base is set). Original nuc_penalty={nuc_penalty} -> 0.0")
            effective_nuc_penalty = 0.0

        # Log parameter counts (frozen vs trainable)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        if total_params > 0:
            logger.info(
                f"Parameters: Total={total_params:,}, Trainable={trainable_params:,} "
                f"({trainable_params / total_params * 100:.1f}%), Frozen={frozen_params:,}"
            )

        class_weights_tensor = None
        if class_weights is not None and classifier:
            weights_list = [float(w) for w in class_weights.split(",")]
            class_weights_tensor = torch.tensor(weights_list)

        effective_count_penalty = count_penalty if count_prediction else 0.0

        loss_fn = MultiTaskLoss(
            penalty=penalty,
            nuc_penalty=effective_nuc_penalty,
            target_penalty=target_penalty,
            count_penalty=effective_count_penalty,
            class_weights=class_weights_tensor,
            target_loss_type=loss_type,
            quantiles=quantiles_list,
            over_penalty=over_penalty,
            under_penalty=under_penalty,
        )
        if loss_type == "quantile":
            logger.info(f"Using quantile loss with quantiles: {quantiles_list}")
        elif loss_type == "asymmetric":
            logger.info(f"Using asymmetric loss: over_penalty={over_penalty}, under_penalty={under_penalty}")
        else:
            logger.info(f"Using {loss_type} loss for regression targets")
        logger.info(
            f"Loss weights: target={target_penalty}, unifrac={penalty}, nuc={effective_nuc_penalty}, count={effective_count_penalty}"
        )
        if not count_prediction:
            logger.info("Count prediction head disabled (--no-count-prediction)")

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
            except Exception as e:
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

        elif distributed:
            # Move model to device
            model = model.to(device_obj)

            # Convert BatchNorm to SyncBatchNorm if requested
            if sync_batchnorm:
                model = sync_batch_norm(model)
                if is_main_process():
                    logger.info("Converted BatchNorm to SyncBatchNorm for distributed training")

            # Wrap model with DDP
            # find_unused_parameters needed when freeze_base=True (base model params unused)
            # or when categorical features may not always contribute to loss
            model = wrap_model_ddp(
                model,
                device_id=get_local_rank(),
                find_unused_parameters=freeze_base or bool(categorical_column_list),
            )
            if is_main_process():
                logger.info("Model wrapped with DistributedDataParallel")

        elif data_parallel:
            # DataParallel for single-node multi-GPU training
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

        effective_batches_per_epoch = len(train_loader) // gradient_accumulation_steps
        num_training_steps = effective_batches_per_epoch * epochs
        optimizer_obj = create_optimizer(
            model, optimizer_type=optimizer, lr=lr, weight_decay=weight_decay, freeze_base=freeze_base
        )
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
        tensorboard_dir = str(output_path) if (not distributed or is_main_process()) else None

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer_obj,
            scheduler=scheduler_obj,
            device=device_obj,
            freeze_base=freeze_base,
            tensorboard_dir=tensorboard_dir,
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision_normalized,
            compile_model=compile_model,
            target_normalization_params=target_normalization_params,
            count_normalization_params=count_normalization_params,
            train_sampler=train_sampler,
            use_sharded_checkpoint=fsdp_sharded_checkpoint,
            best_metric=best_metric,
        )

        if resume_from is not None:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            trainer.load_checkpoint(resume_from, load_optimizer=True, load_scheduler=True, target_lr=lr)

        logger.info("Starting training...")
        checkpoint_dir = output_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            early_stopping_patience=patience,
            checkpoint_dir=str(checkpoint_dir),
            resume_from=resume_from,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        logger.info("Training completed")
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
        logger.info(f"Best validation loss: {best_val_loss}")

        # Only save final model on main process in distributed mode
        if not (distributed or fsdp) or is_main_process():
            # Build model config for inference
            model_config = {
                "encoder_type": encoder_type,
                "embedding_dim": embedding_dim,
                "max_bp": max_bp,
                "token_limit": token_limit,
                "out_dim": out_dim,
                "is_classifier": classifier,
                "attention_layers": attention_layers,
                "attention_heads": attention_heads,
                "target_layer_norm": target_layer_norm,
                "bounded_targets": bounded_targets,
                "learnable_output_scale": learnable_output_scale,
                "categorical_embed_dim": categorical_embed_dim,
                "categorical_fusion": categorical_fusion,
                "cross_attn_heads": cross_attn_heads,
                "output_activation": output_activation,
                "log_transform_targets": log_transform_targets,
                "regressor_hidden_dims": regressor_hidden_dims_list,
                "regressor_dropout": regressor_dropout,
                "conditional_scaling_columns": conditional_scaling_columns,
            }
            # Include categorical encoder state if categoricals are used
            if categorical_encoder is not None:
                model_config["categorical_encoder"] = categorical_encoder.to_dict()

            # Include category normalizer state if per-category normalization is used
            if category_normalizer is not None:
                model_config["category_normalizer"] = category_normalizer.to_dict()

            final_model_path = output_path / "final_model.pt"
            trainer.save_checkpoint(
                str(final_model_path),
                epoch=epochs - 1,
                best_val_loss=best_val_loss,
                metrics=history,
                config=model_config,
            )
            logger.info(f"Final model saved to {final_model_path}")

        # Cleanup distributed training
        if distributed or fsdp:
            cleanup_distributed()

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # Cleanup distributed training on error
        if distributed or fsdp:
            cleanup_distributed()
        raise click.ClickException(str(e))
