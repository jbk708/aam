"""Training and validation loops for AAM model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union, Tuple, List
import os
from pathlib import Path
import math
from tqdm import tqdm

from aam.training.metrics import compute_regression_metrics, compute_count_metrics, compute_classification_metrics


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup + cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
        """
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step_count = 0

    def step(self):
        """Update learning rate."""
        self._step_count += 1

        if self._step_count <= self.num_warmup_steps:
            lr_scale = self._step_count / self.num_warmup_steps
        else:
            progress = (self._step_count - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * lr_scale

    def get_last_lr(self):
        """Get current learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]


class Trainer:
    """Trainer for AAM models with support for staged training."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, WarmupCosineScheduler]] = None,
        device: Union[str, torch.device] = "cpu",
        freeze_base: bool = False,
    ):
        """Initialize Trainer.

        Args:
            model: Model to train (SequenceEncoder or SequencePredictor)
            loss_fn: Loss function (MultiTaskLoss)
            optimizer: Optimizer (if None, will be created)
            scheduler: Learning rate scheduler (if None, will be created)
            device: Device to train on
            freeze_base: Whether base model parameters are frozen
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = torch.device(device) if isinstance(device, str) else device
        self.freeze_base = freeze_base

        if optimizer is None:
            self.optimizer = create_optimizer(model, freeze_base=freeze_base)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

    def _prepare_batch(self, batch: Union[Dict, Tuple]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare batch for model forward pass.

        Args:
            batch: Batch from DataLoader (dict or tuple)

        Returns:
            Tuple of (tokens, targets_dict)
        """
        if isinstance(batch, dict):
            tokens = batch["tokens"].to(self.device)
            targets = {}

            if "counts" in batch:
                targets["counts"] = batch["counts"].to(self.device)
            if "y_target" in batch:
                targets["target"] = batch["y_target"].to(self.device)
            if "unifrac_target" in batch:
                targets["base_target"] = batch["unifrac_target"].to(self.device)
            if "tokens" in batch:
                targets["tokens"] = tokens

            return tokens, targets
        else:
            tokens = batch[0].to(self.device)
            targets = {"tokens": tokens}

            if len(batch) > 1:
                if len(batch) == 2:
                    targets["base_target"] = batch[1].to(self.device)
                elif len(batch) >= 3:
                    targets["counts"] = batch[1].to(self.device)
                    targets["target"] = batch[2].to(self.device)
                    if len(batch) >= 4:
                        targets["base_target"] = batch[3].to(self.device)

            return tokens, targets

    def _get_encoder_type(self) -> str:
        """Get encoder type from model."""
        if hasattr(self.model, "encoder_type"):
            return self.model.encoder_type
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "encoder_type"):
            return self.model.base_model.encoder_type
        return "unifrac"

    def _get_is_classifier(self) -> bool:
        """Check if model is a classifier."""
        if hasattr(self.model, "is_classifier"):
            return self.model.is_classifier
        return False

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with average losses
        """
        self.model.train()
        total_losses = {}
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            tokens, targets = self._prepare_batch(batch)

            self.optimizer.zero_grad()

            return_nucleotides = "nucleotides" in targets or self.loss_fn.nuc_penalty > 0
            outputs = self.model(tokens, return_nucleotides=return_nucleotides)

            encoder_type = self._get_encoder_type()
            is_classifier = self._get_is_classifier()
            losses = self.loss_fn(outputs, targets, is_classifier=is_classifier, encoder_type=encoder_type)

            losses["total_loss"].backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()

            num_batches += 1

        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses

    def validate_epoch(
        self,
        dataloader: DataLoader,
        compute_metrics: bool = True,
    ) -> Dict[str, float]:
        """Run one validation epoch.

        Args:
            dataloader: Validation data loader
            compute_metrics: Whether to compute metrics

        Returns:
            Dictionary with losses and metrics
        """
        self.model.eval()
        total_losses = {}
        all_predictions = {}
        all_targets = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                tokens, targets = self._prepare_batch(batch)

                return_nucleotides = "nucleotides" in targets or self.loss_fn.nuc_penalty > 0
                outputs = self.model(tokens, return_nucleotides=return_nucleotides)

                encoder_type = self._get_encoder_type()
                is_classifier = self._get_is_classifier()
                losses = self.loss_fn(outputs, targets, is_classifier=is_classifier, encoder_type=encoder_type)

                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()

                if compute_metrics:
                    if "target_prediction" in outputs and "target" in targets:
                        if "target_prediction" not in all_predictions:
                            all_predictions["target_prediction"] = []
                            all_targets["target"] = []
                        all_predictions["target_prediction"].append(outputs["target_prediction"])
                        all_targets["target"].append(targets["target"])

                    if "count_prediction" in outputs and "counts" in targets:
                        if "count_prediction" not in all_predictions:
                            all_predictions["count_prediction"] = []
                            all_targets["counts"] = []
                            all_targets["mask"] = []
                        all_predictions["count_prediction"].append(outputs["count_prediction"])
                        all_targets["counts"].append(targets["counts"])
                        mask = targets.get("mask", (tokens.sum(dim=-1) > 0).long())
                        all_targets["mask"].append(mask)

                num_batches += 1

        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        if compute_metrics and all_predictions:
            if "target_prediction" in all_predictions:
                target_pred = torch.cat(all_predictions["target_prediction"], dim=0)
                target_true = torch.cat(all_targets["target"], dim=0)

                is_classifier = self._get_is_classifier()
                if is_classifier:
                    metrics = compute_classification_metrics(target_pred, target_true)
                else:
                    metrics = compute_regression_metrics(target_pred, target_true)
                avg_losses.update(metrics)

            if "count_prediction" in all_predictions:
                count_pred = torch.cat(all_predictions["count_prediction"], dim=0)
                count_true = torch.cat(all_targets["counts"], dim=0)
                mask = torch.cat(all_targets["mask"], dim=0)
                metrics = compute_count_metrics(count_pred, count_true, mask)
                avg_losses.update({f"count_{k}": v for k, v in metrics.items()})

        return avg_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 50,
        checkpoint_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, list]:
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 0

        if resume_from is not None:
            checkpoint_info = self.load_checkpoint(resume_from)
            start_epoch = checkpoint_info["epoch"] + 1
            best_val_loss = checkpoint_info["best_val_loss"]

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, num_epochs):
            train_losses = self.train_epoch(train_loader)
            history["train_loss"].append(train_losses["total_loss"])

            if val_loader is not None:
                val_results = self.validate_epoch(val_loader, compute_metrics=True)
                val_loss = val_results["total_loss"]
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    if checkpoint_dir is not None:
                        checkpoint_path = Path(checkpoint_dir) / f"best_model_epoch_{epoch}.pt"
                        self.save_checkpoint(
                            str(checkpoint_path),
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                            metrics=val_results,
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                history["val_loss"].append(None)

        return history

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        best_val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            best_val_loss: Best validation loss so far
            metrics: Optional metrics dictionary
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "metrics": metrics or {},
        }

        if self.scheduler is not None:
            if isinstance(self.scheduler, WarmupCosineScheduler):
                checkpoint["scheduler_step"] = self.scheduler._step_count
            else:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict:
        """Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state

        Returns:
            Dictionary with checkpoint info (epoch, best_val_loss, metrics)
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_scheduler and self.scheduler is not None:
            if isinstance(self.scheduler, WarmupCosineScheduler) and "scheduler_step" in checkpoint:
                self.scheduler._step_count = checkpoint["scheduler_step"]
            elif "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
            "metrics": checkpoint.get("metrics", {}),
        }


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    freeze_base: bool = False,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer, excluding frozen parameters if needed.

    Args:
        model: Model to create optimizer for
        lr: Learning rate
        weight_decay: Weight decay
        freeze_base: Whether base model parameters are frozen

    Returns:
        AdamW optimizer
    """
    if freeze_base and hasattr(model, "base_model"):
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith("base_model."):
                trainable_params.append(param)
        if not trainable_params:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int = 10000,
    num_training_steps: int = 100000,
) -> WarmupCosineScheduler:
    """Create learning rate scheduler with warmup + cosine decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps

    Returns:
        Learning rate scheduler
    """
    return WarmupCosineScheduler(optimizer, num_warmup_steps, num_training_steps)


def load_pretrained_encoder(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = True,
) -> None:
    """Load pre-trained SequenceEncoder checkpoint into model.

    Args:
        checkpoint_path: Path to SequenceEncoder checkpoint
        model: Model to load weights into (SequencePredictor with base_model)
        strict: Whether to strictly match state dict keys
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if hasattr(model, "base_model"):
        model.base_model.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)
