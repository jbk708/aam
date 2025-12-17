"""Command-line interface for AAM training and inference."""

import click
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from functools import partial
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader
from aam.data.dataset import ASVDataset, collate_fn
from skbio import DistanceMatrix
from aam.models.sequence_predictor import SequencePredictor
from aam.models.sequence_encoder import SequenceEncoder
from aam.training.losses import MultiTaskLoss
from aam.training.trainer import Trainer, create_optimizer, create_scheduler, load_pretrained_encoder
from aam.models.model_summary import log_model_summary


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging to console and file.

    Args:
        output_dir: Directory to write log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def setup_device(device: str) -> torch.device:
    """Setup device (CPU or CUDA).

    Args:
        device: Device string ('cpu' or 'cuda')

    Returns:
        torch.device object

    Raises:
        ValueError: If device is invalid or CUDA not available
    """
    if device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Use --device cpu")
        return torch.device("cuda")
    else:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")


def setup_expandable_segments(use_expandable_segments: bool) -> None:
    """Setup PyTorch CUDA memory allocator with expandable segments.

    Args:
        use_expandable_segments: Whether to enable expandable segments
    """
    if use_expandable_segments and torch.cuda.is_available():
        import os

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup_random_seed(seed: Optional[int]):
    """Setup random seed for reproducibility.

    Args:
        seed: Random seed (None for no seed)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def validate_file_path(path: str, file_type: str = "file"):
    """Validate that a file path exists.

    Args:
        path: File path to validate
        file_type: Type of file for error message

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{file_type} not found: {path}")


def validate_arguments(**kwargs):
    """Validate CLI arguments.

    Args:
        **kwargs: Arguments to validate

    Raises:
        ValueError: If validation fails
    """
    batch_size = kwargs.get("batch_size")
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even (for UniFrac), got {batch_size}")

    classifier = kwargs.get("classifier", False)
    out_dim = kwargs.get("out_dim", 1)
    if classifier and out_dim <= 1:
        raise ValueError(f"classifier requires out_dim > 1, got {out_dim}")

    lr = kwargs.get("lr")
    if lr is not None and lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    test_size = kwargs.get("test_size")
    if test_size is not None:
        if test_size < 0 or test_size > 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    epochs = kwargs.get("epochs")
    if epochs is not None and epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")


@click.group()
def cli():
    """AAM (Attention All Microbes) - Deep learning for microbial sequencing data."""
    pass


@cli.command()
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--unifrac-matrix", required=True, type=click.Path(exists=True), help="Path to pre-computed UniFrac distance matrix (.npy, .h5, or .csv format)")
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
@click.option("--unifrac-metric", default="unifrac", type=click.Choice(["unifrac", "faith_pd"]), help="UniFrac metric type (unifrac for pairwise, faith_pd for per-sample values)")
@click.option("--penalty", default=1.0, type=float, help="Weight for base/UniFrac loss")
@click.option("--nuc-penalty", default=1.0, type=float, help="Weight for nucleotide loss")
@click.option("--class-weights", default=None, help="Class weights for classification (optional)")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--num-workers", default=4, type=int, help="Number of DataLoader worker processes (default: 4, use 0 to disable multiprocessing)")
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
@click.option("--scheduler-t0", default=None, type=int, help="Initial restart period for cosine_restarts scheduler (default: num_training_steps // 4)")
@click.option("--scheduler-t-mult", default=None, type=int, help="Restart period multiplier for cosine_restarts scheduler (default: 2)")
@click.option("--scheduler-eta-min", default=None, type=float, help="Minimum learning rate for cosine/cosine_restarts schedulers (default: 0.0)")
@click.option("--scheduler-patience", default=None, type=int, help="Patience for plateau scheduler (epochs to wait before reducing LR, default: 5)")
@click.option("--scheduler-factor", default=None, type=float, help="LR reduction factor for plateau scheduler (default: 0.3)")
@click.option("--scheduler-min-lr", default=None, type=float, help="Minimum learning rate for plateau scheduler (default: 0.0)")
@click.option(
    "--mixed-precision",
    default=None,
    type=click.Choice(["fp16", "bf16", "none"]),
    help="Mixed precision training mode (fp16, bf16, or none)",
)
@click.option("--compile-model", is_flag=True, help="Compile model with torch.compile() for optimization (PyTorch 2.0+)")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True, help="Use gradient checkpointing to reduce memory (default: enabled, use --no-gradient-checkpointing to disable)")
@click.option(
    "--attn-implementation",
    default="mem_efficient",
    type=click.Choice(["sdpa", "flash", "mem_efficient", "math"]),
    help="Attention implementation: mem_efficient (default), sdpa (auto-select), flash (Flash Attention), or math",
)
@click.option("--asv-chunk-size", default=256, type=int, help="Process ASVs in chunks to reduce memory (default: 256, use 0 to disable)")
@click.option("--normalize-targets", is_flag=True, help="Normalize target and count values to [0, 1] range during training (recommended for regression tasks)")
@click.option("--loss-type", type=click.Choice(["mse", "mae", "huber"]), default="huber", help="Loss function for regression targets: mse, mae, or huber (default)")
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
    gradient_checkpointing: bool,
    attn_implementation: str,
    asv_chunk_size: int,
    normalize_targets: bool,
    loss_type: str,
):
    """Train AAM model on microbial sequencing data."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_logging(output_path)
        logger = logging.getLogger(__name__)
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

        setup_expandable_segments(use_expandable_segments)
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
        
        logger.info(f"Loaded UniFrac matrix: {type(unifrac_distances).__name__}, shape: {getattr(unifrac_distances, 'shape', 'N/A')}")

        logger.info("Splitting data...")
        train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)
        logger.info(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

        logger.info("Filtering tables for train/val splits...")
        train_table = table_obj.filter(train_ids, axis="sample", inplace=False)
        val_table = table_obj.filter(val_ids, axis="sample", inplace=False)
        logger.info("Table filtering complete")

        train_metadata = metadata_df[metadata_df["sample_id"].isin(train_ids)]
        val_metadata = metadata_df[metadata_df["sample_id"].isin(val_ids)]

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
        
        reference_sample_ids = None

        logger.info("Creating datasets...")
        train_dataset = ASVDataset(
            table=train_table,
            metadata=train_metadata,
            unifrac_distances=train_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=metadata_column,
            unifrac_metric=unifrac_metric_name,
            lazy_unifrac=False,
            unifrac_computer=None,
            stripe_mode=False,
            reference_sample_ids=None,
            normalize_targets=normalize_targets,
            normalize_counts=normalize_targets,  # Normalize counts along with targets
        )

        # Get normalization parameters from training set
        target_normalization_params = train_dataset.get_normalization_params()
        if target_normalization_params:
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
            lazy_unifrac=False,
            unifrac_computer=None,
            stripe_mode=False,
            reference_sample_ids=None,
            normalize_targets=normalize_targets,
            # Use same normalization params as training set for consistency
            target_min=train_dataset.target_min if normalize_targets else None,
            target_max=train_dataset.target_max if normalize_targets else None,
            normalize_counts=normalize_targets,  # Normalize counts along with targets
            # Use same count normalization params as training set for consistency
            count_min=train_dataset.count_min if normalize_targets else None,
            count_max=train_dataset.count_max if normalize_targets else None,
        )

        train_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=train_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
            lazy_unifrac=False,
            stripe_mode=False,
            reference_sample_ids=None,
            all_sample_ids=None,
        )
        val_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=val_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
            lazy_unifrac=False,
            stripe_mode=False,
            reference_sample_ids=None,
            all_sample_ids=None,
        )

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
        effective_asv_chunk_size = asv_chunk_size if asv_chunk_size > 0 else None
        model = SequencePredictor(
            encoder_type=encoder_type,
            vocab_size=6,
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
            attn_implementation=attn_implementation,
            asv_chunk_size=effective_asv_chunk_size,
        )

        log_model_summary(model, logger)

        if pretrained_encoder is not None:
            logger.info(f"Loading pretrained encoder from {pretrained_encoder}")
            load_pretrained_encoder(pretrained_encoder, model, strict=False)
            logger.info("Pretrained encoder loaded successfully")

        class_weights_tensor = None
        if class_weights is not None and classifier:
            weights_list = [float(w) for w in class_weights.split(",")]
            class_weights_tensor = torch.tensor(weights_list)

        loss_fn = MultiTaskLoss(
            penalty=penalty,
            nuc_penalty=nuc_penalty,
            class_weights=class_weights_tensor,
            target_loss_type=loss_type,
        )
        logger.info(f"Using {loss_type} loss for regression targets")

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
            optimizer_obj, scheduler_type=scheduler, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps, **scheduler_kwargs
        )

        # Normalize mixed_precision: "none" -> None
        mixed_precision_normalized = None if mixed_precision == "none" else mixed_precision

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer_obj,
            scheduler=scheduler_obj,
            device=device_obj,
            freeze_base=freeze_base,
            tensorboard_dir=str(output_path),
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision_normalized,
            compile_model=compile_model,
            target_normalization_params=target_normalization_params,
            count_normalization_params=count_normalization_params,
        )

        if resume_from is not None:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            trainer.load_checkpoint(resume_from, load_optimizer=True, load_scheduler=True)

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
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        logger.info(f"Best validation loss: {best_val_loss}")

        final_model_path = output_path / "final_model.pt"
        trainer.save_checkpoint(str(final_model_path), epoch=epochs - 1, best_val_loss=best_val_loss, metrics=history)
        logger.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--unifrac-matrix", required=True, type=click.Path(exists=True), help="Path to pre-computed UniFrac distance matrix (.npy, .h5, or .csv format)")
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
@click.option("--rarefy-depth", default=5000, type=int, help="Rarefaction depth")
@click.option("--test-size", default=0.2, type=float, help="Validation split size")
@click.option("--unifrac-metric", default="unifrac", type=click.Choice(["unifrac", "faith_pd"]), help="UniFrac metric type (unifrac for pairwise, faith_pd for per-sample values)")
@click.option("--penalty", default=1.0, type=float, help="Weight for base/UniFrac loss")
@click.option("--nuc-penalty", default=1.0, type=float, help="Weight for nucleotide loss")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--num-workers", default=4, type=int, help="Number of DataLoader worker processes (default: 4, use 0 to disable multiprocessing)")
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
@click.option("--scheduler-t0", default=None, type=int, help="Initial restart period for cosine_restarts scheduler (default: num_training_steps // 4)")
@click.option("--scheduler-t-mult", default=None, type=int, help="Restart period multiplier for cosine_restarts scheduler (default: 2)")
@click.option("--scheduler-eta-min", default=None, type=float, help="Minimum learning rate for cosine/cosine_restarts schedulers (default: 0.0)")
@click.option("--scheduler-patience", default=None, type=int, help="Patience for plateau scheduler (epochs to wait before reducing LR, default: 5)")
@click.option("--scheduler-factor", default=None, type=float, help="LR reduction factor for plateau scheduler (default: 0.3)")
@click.option("--scheduler-min-lr", default=None, type=float, help="Minimum learning rate for plateau scheduler (default: 0.0)")
@click.option(
    "--mixed-precision",
    default=None,
    type=click.Choice(["fp16", "bf16", "none"]),
    help="Mixed precision training mode (fp16, bf16, or none)",
)
@click.option("--compile-model", is_flag=True, help="Compile model with torch.compile() for optimization (PyTorch 2.0+)")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True, help="Use gradient checkpointing to reduce memory (default: enabled, use --no-gradient-checkpointing to disable)")
@click.option(
    "--attn-implementation",
    default="mem_efficient",
    type=click.Choice(["sdpa", "flash", "mem_efficient", "math"]),
    help="Attention implementation: mem_efficient (default), sdpa (auto-select), flash (Flash Attention), or math",
)
@click.option(
    "--asv-chunk-size", default=256, type=int, help="Process ASVs in chunks to reduce memory (default: 256, use 0 to disable)"
)
def pretrain(
    table: str,
    unifrac_matrix: str,
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
    rarefy_depth: int,
    test_size: float,
    unifrac_metric: str,
    penalty: float,
    nuc_penalty: float,
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
    gradient_checkpointing: bool,
    attn_implementation: str,
    asv_chunk_size: Optional[int],
):
    """Pre-train SequenceEncoder on UniFrac and nucleotide prediction (self-supervised)."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_logging(output_path)
        logger = logging.getLogger(__name__)
        logger.info("Starting SequenceEncoder pre-training")
        logger.info(f"Arguments: table={table}, unifrac_matrix={unifrac_matrix}")

        validate_file_path(table, "BIOM table")
        validate_file_path(unifrac_matrix, "UniFrac matrix")

        validate_arguments(
            batch_size=batch_size,
            lr=lr,
            test_size=test_size,
            epochs=epochs,
        )

        setup_expandable_segments(use_expandable_segments)
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
        
        logger.info(f"Loaded UniFrac matrix: {type(unifrac_distances).__name__}, shape: {getattr(unifrac_distances, 'shape', 'N/A')}")

        logger.info("Splitting data...")
        train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)
        logger.info(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

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
        
        reference_sample_ids = None

        logger.info("Creating datasets...")
        train_dataset = ASVDataset(
            table=train_table,
            metadata=None,
            unifrac_distances=train_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=None,
            unifrac_metric=unifrac_metric_name,
            lazy_unifrac=False,
            unifrac_computer=None,
            stripe_mode=False,
            reference_sample_ids=None,
        )

        val_dataset = ASVDataset(
            table=val_table,
            metadata=None,
            unifrac_distances=val_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=None,
            unifrac_metric=unifrac_metric_name,
            lazy_unifrac=False,
            unifrac_computer=None,
            stripe_mode=False,
            reference_sample_ids=None,
        )

        train_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=train_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
            lazy_unifrac=False,
            stripe_mode=False,
            reference_sample_ids=None,
            all_sample_ids=None,
        )
        val_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=val_distance_matrix,
            unifrac_metric=unifrac_metric_name,
            unifrac_loader=unifrac_loader,
            lazy_unifrac=False,
            stripe_mode=False,
            reference_sample_ids=None,
            all_sample_ids=None,
        )

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
        effective_asv_chunk_size = asv_chunk_size if asv_chunk_size > 0 else None
        model = SequenceEncoder(
            encoder_type=encoder_type,
            vocab_size=6,
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
            attn_implementation=attn_implementation,
        )

        log_model_summary(model, logger)

        loss_fn = MultiTaskLoss(
            penalty=penalty,
            nuc_penalty=nuc_penalty,
            class_weights=None,
            target_loss_type="huber",  # Default for pretraining (not used, but consistent)
        )

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
            optimizer_obj, scheduler_type=scheduler, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps, **scheduler_kwargs
        )

        # Normalize mixed_precision: "none" -> None
        mixed_precision_normalized = None if mixed_precision == "none" else mixed_precision

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer_obj,
            scheduler=scheduler_obj,
            device=device_obj,
            freeze_base=False,
            tensorboard_dir=str(output_path),
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision_normalized,
            compile_model=compile_model,
        )

        if resume_from is not None:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            trainer.load_checkpoint(resume_from, load_optimizer=True, load_scheduler=True)

        logger.info("Starting pre-training...")
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

        logger.info("Pre-training completed")
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        logger.info(f"Best validation loss: {best_val_loss}")

        final_model_path = output_path / "pretrained_encoder.pt"
        trainer.save_checkpoint(str(final_model_path), epoch=epochs - 1, best_val_loss=best_val_loss, metrics=history)
        logger.info(f"Pre-trained encoder saved to {final_model_path}")

    except Exception as e:
        if "logger" in locals():
            logger.error(f"Pre-training failed: {e}", exc_info=True)
        else:
            logging.error(f"Pre-training failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to trained model checkpoint")
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--output", required=True, type=click.Path(), help="Output file for predictions")
@click.option("--batch-size", default=8, type=int, help="Batch size for inference")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
def predict(
    model: str,
    table: str,
    output: str,
    batch_size: int,
    device: str,
):
    """Run inference with trained AAM model."""
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)
        logger.info("Starting AAM inference")

        validate_file_path(model, "Model checkpoint")
        validate_file_path(table, "BIOM table")

        validate_arguments(batch_size=batch_size)

        device_obj = setup_device(device)

        logger.info("Loading model...")
        checkpoint = torch.load(model, map_location=device_obj)

        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            model_config = checkpoint.get("config", {})
        else:
            model_state = checkpoint
            model_config = {}

        logger.info("Loading data...")
        biom_loader = BIOMLoader()
        table_obj = biom_loader.load_table(table)

        dataset = ASVDataset(
            table=table_obj,
            max_bp=model_config.get("max_bp", 150),
            token_limit=model_config.get("token_limit", 1024),
        )

        inference_collate = partial(
            collate_fn,
            token_limit=model_config.get("token_limit", 1024),
            unifrac_distances=None,
            unifrac_metric="unweighted",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=inference_collate,
            pin_memory=device == "cuda",
        )

        logger.info("Creating model...")
        model_obj = SequencePredictor(
            encoder_type=model_config.get("encoder_type", "unifrac"),
            vocab_size=6,
            embedding_dim=model_config.get("embedding_dim", 128),
            max_bp=model_config.get("max_bp", 150),
            token_limit=model_config.get("token_limit", 1024),
            out_dim=model_config.get("out_dim", 1),
            is_classifier=model_config.get("is_classifier", False),
        )
        model_obj.load_state_dict(model_state)
        model_obj.to(device_obj)
        model_obj.eval()

        logger.info("Running inference...")
        predictions = []
        sample_ids_list = []

        with torch.no_grad():
            for batch in dataloader:
                tokens = batch["tokens"].to(device_obj)
                outputs = model_obj(tokens, return_nucleotides=False)

                if "target_prediction" in outputs:
                    pred = outputs["target_prediction"].cpu().numpy()
                    if pred.ndim == 1:
                        predictions.extend(pred.tolist())
                    elif pred.ndim == 2 and pred.shape[1] == 1:
                        predictions.extend(pred.squeeze(1).tolist())
                    else:
                        predictions.extend([p.tolist() for p in pred])
                    sample_ids_list.extend(batch["sample_ids"])

        logger.info(f"Writing predictions to {output}...")
        if predictions and isinstance(predictions[0], list):
            pred_cols = {f"prediction_{i}": [p[i] for p in predictions] for i in range(len(predictions[0]))}
            output_df = pd.DataFrame({"sample_id": sample_ids_list, **pred_cols})
        else:
            output_df = pd.DataFrame({"sample_id": sample_ids_list, "prediction": predictions})
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, sep="\t", index=False)

        logger.info(f"Inference completed. Predictions saved to {output}")

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
