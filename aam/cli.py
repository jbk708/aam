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
from aam.data.unifrac import UniFracComputer
from aam.data.dataset import ASVDataset, collate_fn
from skbio import DistanceMatrix
from aam.models.sequence_predictor import SequencePredictor
from aam.models.sequence_encoder import SequenceEncoder
from aam.training.losses import MultiTaskLoss
from aam.training.trainer import Trainer, create_optimizer, create_scheduler, load_pretrained_encoder


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
@click.option("--tree", required=True, type=click.Path(exists=True), help="Path to phylogenetic tree file (.nwk)")
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
@click.option("--unifrac-metric", default="unifrac", type=click.Choice(["unifrac", "faith_pd"]), help="UniFrac metric")
@click.option("--penalty", default=1.0, type=float, help="Weight for base/UniFrac loss")
@click.option("--nuc-penalty", default=1.0, type=float, help="Weight for nucleotide loss")
@click.option("--class-weights", default=None, help="Class weights for classification (optional)")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--num-workers", default=0, type=int, help="DataLoader workers")
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
    type=click.Choice(["warmup_cosine", "cosine", "plateau", "onecycle"]),
    help="Learning rate scheduler type",
)
def train(
    table: str,
    tree: str,
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
):
    """Train AAM model on microbial sequencing data."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_logging(output_path)
        logger = logging.getLogger(__name__)
        logger.info("Starting AAM training")
        logger.info(f"Arguments: table={table}, tree={tree}, metadata={metadata}")

        validate_file_path(table, "BIOM table")
        validate_file_path(tree, "Phylogenetic tree")
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

        metadata_df = pd.read_csv(metadata, sep="\t")
        if "sample_id" not in metadata_df.columns:
            raise ValueError("Metadata file must have 'sample_id' column")
        if metadata_column not in metadata_df.columns:
            raise ValueError(f"Metadata column '{metadata_column}' not found in metadata file")

        logger.info("Computing UniFrac distances...")
        unifrac_computer = UniFracComputer()
        if unifrac_metric == "unifrac":
            unifrac_distances = unifrac_computer.compute_unweighted(table_obj, tree)
            unifrac_metric_name = "unweighted"
            encoder_type = "unifrac"
        else:
            unifrac_distances = unifrac_computer.compute_faith_pd(table_obj, tree)
            unifrac_metric_name = "faith_pd"
            encoder_type = "faith_pd"

        logger.info("Splitting data...")
        sample_ids = list(table_obj.ids(axis="sample"))
        train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)

        train_table = table_obj.filter(train_ids, axis="sample", inplace=False)
        val_table = table_obj.filter(val_ids, axis="sample", inplace=False)

        train_metadata = metadata_df[metadata_df["sample_id"].isin(train_ids)]
        val_metadata = metadata_df[metadata_df["sample_id"].isin(val_ids)]

        train_distance_matrix = None
        val_distance_matrix = None
        if unifrac_metric_name == "unweighted":
            train_distance_matrix = (
                unifrac_distances.filter(train_ids) if isinstance(unifrac_distances, DistanceMatrix) else None
            )
            val_distance_matrix = unifrac_distances.filter(val_ids) if isinstance(unifrac_distances, DistanceMatrix) else None
        elif unifrac_metric_name == "faith_pd":
            train_distance_matrix = unifrac_distances.loc[train_ids] if isinstance(unifrac_distances, pd.Series) else None
            val_distance_matrix = unifrac_distances.loc[val_ids] if isinstance(unifrac_distances, pd.Series) else None

        logger.info("Creating datasets...")
        train_dataset = ASVDataset(
            table=train_table,
            metadata=train_metadata,
            unifrac_distances=train_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=metadata_column,
            unifrac_metric=unifrac_metric_name,
        )

        val_dataset = ASVDataset(
            table=val_table,
            metadata=val_metadata,
            unifrac_distances=val_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=metadata_column,
            unifrac_metric=unifrac_metric_name,
        )

        train_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=train_distance_matrix,
            unifrac_metric=unifrac_metric_name,
        )
        val_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=val_distance_matrix,
            unifrac_metric=unifrac_metric_name,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate,
            drop_last=True,
        )

        logger.info("Creating model...")
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
        )

        if pretrained_encoder is not None:
            logger.info(f"Loading pretrained encoder from {pretrained_encoder}")
            load_pretrained_encoder(pretrained_encoder, model, strict=False)
            logger.info("Pretrained encoder loaded successfully")

        class_weights_tensor = None
        if class_weights is not None and classifier:
            weights_list = [float(w) for w in class_weights.split(",")]
            class_weights_tensor = torch.tensor(weights_list)

        loss_fn = MultiTaskLoss(penalty=penalty, nuc_penalty=nuc_penalty, class_weights=class_weights_tensor)

        effective_batches_per_epoch = len(train_loader) // gradient_accumulation_steps
        num_training_steps = effective_batches_per_epoch * epochs
        optimizer_obj = create_optimizer(
            model, optimizer_type=optimizer, lr=lr, weight_decay=weight_decay, freeze_base=freeze_base
        )
        scheduler_obj = create_scheduler(
            optimizer_obj, scheduler_type=scheduler, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer_obj,
            scheduler=scheduler_obj,
            device=device_obj,
            freeze_base=freeze_base,
            tensorboard_dir=str(output_path),
            max_grad_norm=max_grad_norm,
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
        logger.info(f"Best validation loss: {min(history['val_loss']) if history['val_loss'] else 'N/A'}")

        final_model_path = output_path / "final_model.pt"
        trainer.save_checkpoint(str(final_model_path), epoch=epochs - 1, metrics=history)
        logger.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--tree", required=True, type=click.Path(exists=True), help="Path to phylogenetic tree file (.nwk)")
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
@click.option("--unifrac-metric", default="unifrac", type=click.Choice(["unifrac", "faith_pd"]), help="UniFrac metric")
@click.option("--penalty", default=1.0, type=float, help="Weight for base/UniFrac loss")
@click.option("--nuc-penalty", default=1.0, type=float, help="Weight for nucleotide loss")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--num-workers", default=0, type=int, help="DataLoader workers")
@click.option("--resume-from", default=None, type=click.Path(exists=True), help="Path to checkpoint to resume from")
@click.option("--gradient-accumulation-steps", default=1, type=int, help="Number of gradient accumulation steps")
@click.option("--use-expandable-segments", is_flag=True, help="Enable PyTorch CUDA expandable segments for memory optimization")
@click.option("--max-grad-norm", default=None, type=float, help="Maximum gradient norm for clipping (None to disable)")
@click.option("--optimizer", default="adamw", type=click.Choice(["adamw", "adam", "sgd"]), help="Optimizer type")
@click.option(
    "--scheduler",
    default="warmup_cosine",
    type=click.Choice(["warmup_cosine", "cosine", "plateau", "onecycle"]),
    help="Learning rate scheduler type",
)
@click.option(
    "--asv-chunk-size", default=None, type=int, help="Process ASVs in chunks of this size to reduce memory (None = process all)"
)
def pretrain(
    table: str,
    tree: str,
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
    asv_chunk_size: Optional[int],
):
    """Pre-train SequenceEncoder on UniFrac and nucleotide prediction (self-supervised)."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        setup_logging(output_path)
        logger = logging.getLogger(__name__)
        logger.info("Starting SequenceEncoder pre-training")
        logger.info(f"Arguments: table={table}, tree={tree}")

        validate_file_path(table, "BIOM table")
        validate_file_path(tree, "Phylogenetic tree")

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

        logger.info("Computing UniFrac distances...")
        unifrac_computer = UniFracComputer()
        if unifrac_metric == "unifrac":
            unifrac_distances = unifrac_computer.compute_unweighted(table_obj, tree)
            unifrac_metric_name = "unweighted"
            encoder_type = "unifrac"
            base_output_dim = batch_size
        else:
            unifrac_distances = unifrac_computer.compute_faith_pd(table_obj, tree)
            unifrac_metric_name = "faith_pd"
            encoder_type = "faith_pd"
            base_output_dim = 1

        logger.info("Splitting data...")
        sample_ids = list(table_obj.ids(axis="sample"))
        train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)

        train_table = table_obj.filter(train_ids, axis="sample", inplace=False)
        val_table = table_obj.filter(val_ids, axis="sample", inplace=False)

        train_distance_matrix = None
        val_distance_matrix = None
        if unifrac_metric_name == "unweighted":
            train_distance_matrix = (
                unifrac_distances.filter(train_ids) if isinstance(unifrac_distances, DistanceMatrix) else None
            )
            val_distance_matrix = unifrac_distances.filter(val_ids) if isinstance(unifrac_distances, DistanceMatrix) else None
        elif unifrac_metric_name == "faith_pd":
            train_distance_matrix = unifrac_distances.loc[train_ids] if isinstance(unifrac_distances, pd.Series) else None
            val_distance_matrix = unifrac_distances.loc[val_ids] if isinstance(unifrac_distances, pd.Series) else None

        logger.info("Creating datasets...")
        train_dataset = ASVDataset(
            table=train_table,
            metadata=None,
            unifrac_distances=train_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=None,
            unifrac_metric=unifrac_metric_name,
        )

        val_dataset = ASVDataset(
            table=val_table,
            metadata=None,
            unifrac_distances=val_distance_matrix,
            max_bp=max_bp,
            token_limit=token_limit,
            target_column=None,
            unifrac_metric=unifrac_metric_name,
        )

        train_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=train_distance_matrix,
            unifrac_metric=unifrac_metric_name,
        )
        val_collate = partial(
            collate_fn,
            token_limit=token_limit,
            unifrac_distances=val_distance_matrix,
            unifrac_metric=unifrac_metric_name,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate,
            drop_last=True,
        )

        logger.info("Creating model...")
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
            asv_chunk_size=asv_chunk_size,
        )

        loss_fn = MultiTaskLoss(penalty=penalty, nuc_penalty=nuc_penalty, class_weights=None)

        num_training_steps = len(train_loader) * epochs
        optimizer_obj = create_optimizer(model, optimizer_type=optimizer, lr=lr, weight_decay=weight_decay, freeze_base=False)
        scheduler_obj = create_scheduler(
            optimizer_obj, scheduler_type=scheduler, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer_obj,
            scheduler=scheduler_obj,
            device=device_obj,
            freeze_base=False,
            tensorboard_dir=str(output_path),
            max_grad_norm=max_grad_norm,
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
        logger.info(f"Best validation loss: {min(history['val_loss']) if history['val_loss'] else 'N/A'}")

        final_model_path = output_path / "pretrained_encoder.pt"
        trainer.save_checkpoint(str(final_model_path), epoch=epochs - 1, metrics=history)
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
@click.option("--tree", required=True, type=click.Path(exists=True), help="Path to phylogenetic tree file")
@click.option("--output", required=True, type=click.Path(), help="Output file for predictions")
@click.option("--batch-size", default=8, type=int, help="Batch size for inference")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
def predict(
    model: str,
    table: str,
    tree: str,
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
        validate_file_path(tree, "Phylogenetic tree")

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
