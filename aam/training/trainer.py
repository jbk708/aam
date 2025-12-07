"""Training and validation loops for AAM model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Union, Tuple, List
import os
from pathlib import Path
import math
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        tensorboard_dir: Optional[str] = None,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize Trainer.

        Args:
            model: Model to train (SequenceEncoder or SequencePredictor)
            loss_fn: Loss function (MultiTaskLoss)
            optimizer: Optimizer (if None, will be created)
            scheduler: Learning rate scheduler (if None, will be created)
            device: Device to train on
            freeze_base: Whether base model parameters are frozen
            tensorboard_dir: Directory for TensorBoard logs (if None, TensorBoard disabled)
            max_grad_norm: Maximum gradient norm for clipping (None to disable)
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = torch.device(device) if isinstance(device, str) else device
        self.freeze_base = freeze_base
        self.tensorboard_dir = tensorboard_dir
        self.max_grad_norm = max_grad_norm
        self.writer: Optional[SummaryWriter] = None
        self.log_histograms: bool = True
        self.histogram_frequency: int = 50

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

    def _is_pretraining(self) -> bool:
        """Check if model is SequenceEncoder (pretraining mode)."""
        # Use string comparison to avoid circular import
        return self.model.__class__.__name__ == "SequenceEncoder"

    def _create_prediction_plot(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        r2: float,
    ) -> plt.Figure:
        """Create prediction vs actual scatter plot for regression tasks.

        Args:
            predictions: Predicted values [B, 1] or [B]
            targets: Actual values [B, 1] or [B]
            epoch: Current epoch number
            r2: R² score

        Returns:
            Matplotlib figure
        """
        pred_np = np.array(predictions.detach().cpu().tolist()).flatten()
        target_np = np.array(targets.detach().cpu().tolist()).flatten()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.scatter(target_np, pred_np, alpha=0.6, s=20)

        min_val = min(target_np.min(), pred_np.min())
        max_val = max(target_np.max(), pred_np.max())

        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Perfect Prediction", alpha=0.5)

        if len(target_np) > 1:
            z = np.polyfit(target_np, pred_np, 1)
            p = np.poly1d(z)
            ax.plot(target_np, p(target_np), "b-", linewidth=2, label=f"Linear Fit, R² = {r2:.4f}")

        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"Predicted vs Actual (Epoch {epoch}, R² = {r2:.4f})", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_unifrac_prediction_plot(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        r2: float,
    ) -> plt.Figure:
        """Create prediction vs actual scatter plot for UniFrac predictions (pretraining).

        Args:
            predictions: Predicted UniFrac distances (1D array, already flattened upper triangle)
            targets: Actual UniFrac distances (1D array, already flattened upper triangle)
            epoch: Current epoch number
            r2: R² score

        Returns:
            Matplotlib figure
        """
        # Predictions and targets are already flattened (upper triangle only, diagonal excluded)
        # Convert to numpy arrays
        pred_np = predictions.detach().cpu()
        target_np = targets.detach().cpu()

        # Convert to numpy (handle both tensor and numpy)
        try:
            pred_flat = pred_np.numpy() if hasattr(pred_np, "numpy") else np.array(pred_np)
            target_flat = target_np.numpy() if hasattr(target_np, "numpy") else np.array(target_np)
        except RuntimeError:
            # Fallback: convert to list then numpy
            pred_flat = np.array(pred_np.tolist())
            target_flat = np.array(target_np.tolist())

        # Ensure 1D arrays
        if pred_flat.ndim > 1:
            pred_flat = pred_flat.flatten()
        if target_flat.ndim > 1:
            target_flat = target_flat.flatten()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.scatter(target_flat, pred_flat, alpha=0.6, s=20)

        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())

        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Perfect Prediction", alpha=0.5)

        if len(target_flat) > 1:
            z = np.polyfit(target_flat, pred_flat, 1)
            p = np.poly1d(z)
            ax.plot(target_flat, p(target_flat), "b-", linewidth=2, label=f"Linear Fit, R² = {r2:.4f}")

        ax.set_xlabel("Actual UniFrac Distance", fontsize=12)
        ax.set_ylabel("Predicted UniFrac Distance", fontsize=12)
        ax.set_title(f"UniFrac Prediction vs Actual (Epoch {epoch}, R² = {r2:.4f})", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_confusion_matrix_plot(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
    ) -> plt.Figure:
        """Create confusion matrix plot for classification tasks.

        Args:
            predictions: Predicted class indices [B] or logits [B, num_classes]
            targets: Actual class indices [B]
            epoch: Current epoch number
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1: F1 score

        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix

        target_np = np.array(targets.detach().cpu().tolist())

        if predictions.dim() > 1 and predictions.size(-1) > 1:
            pred_np = np.array(predictions.detach().cpu().argmax(dim=-1).tolist())
        else:
            pred_np = np.array(predictions.detach().cpu().tolist()).flatten()

        num_classes = max(int(pred_np.max()), int(target_np.max())) + 1
        cm = confusion_matrix(target_np, pred_np, labels=list(range(num_classes)))

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=list(range(num_classes)),
            yticklabels=list(range(num_classes)),
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        metrics_text = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}"
        ax.text(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_title(f"Confusion Matrix (Epoch {epoch})", fontsize=14, pad=20)
        plt.tight_layout()
        return fig

    def _save_prediction_plots(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: Optional[str],
        is_unifrac: bool = False,
    ) -> None:
        """Create and save prediction plots when validation improves.

        Args:
            predictions: Predicted values
            targets: Actual values
            epoch: Current epoch number
            metrics: Metrics dictionary
            checkpoint_dir: Directory to save plots
            is_unifrac: Whether this is UniFrac prediction plot (pretraining)
        """
        if checkpoint_dir is None:
            return

        is_classifier = self._get_is_classifier()

        if is_classifier:
            if "accuracy" not in metrics or "precision" not in metrics or "recall" not in metrics or "f1" not in metrics:
                return
            fig = self._create_confusion_matrix_plot(
                predictions,
                targets,
                epoch,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )
        elif is_unifrac:
            if "r2" not in metrics:
                return
            fig = self._create_unifrac_prediction_plot(predictions, targets, epoch, metrics["r2"])
        else:
            if "r2" not in metrics:
                return
            fig = self._create_prediction_plot(predictions, targets, epoch, metrics["r2"])

        plots_dir = Path(checkpoint_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_file = plots_dir / "pred_vs_actual_best.png"
        if plot_file.exists():
            plot_file.unlink()

        fig.savefig(plot_file, dpi=100, bbox_inches="tight")

        if self.writer is not None:
            plot_tag = "validation/unifrac_prediction_plot" if is_unifrac else "validation/prediction_plot"
            self.writer.add_figure(plot_tag, fig, epoch)

        plt.close(fig)

    def _log_to_tensorboard(self, epoch: int, train_losses: Dict[str, float], val_results: Optional[Dict[str, float]] = None):
        """Log metrics to TensorBoard.

        Args:
            epoch: Current epoch number
            train_losses: Training losses dictionary
            val_results: Validation results dictionary (optional)
        """
        if self.writer is None:
            return

        for key, value in train_losses.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)

        if val_results is not None:
            for key, value in val_results.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val/{key}", value, epoch)

        if self.scheduler is not None:
            if hasattr(self.scheduler, "get_last_lr"):
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]["lr"]
        else:
            lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/learning_rate", lr, epoch)

        if self.log_histograms and epoch % self.histogram_frequency == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Only log histograms for parameters with non-zero gradients
                    grad_norm = param.grad.data.norm().item()
                    if grad_norm > 0:
                        self.writer.add_histogram(f"weights/{name}", param.data, epoch)
                        self.writer.add_histogram(f"gradients/{name}", param.grad.data, epoch)

    def train_epoch(
        self,
        dataloader: DataLoader,
        gradient_accumulation_steps: int = 1,
        epoch: int = 0,
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            dataloader: Training data loader
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
            epoch: Current epoch number
            num_epochs: Total number of epochs

        Returns:
            Dictionary with average losses
        """
        self.model.train()
        total_losses = {}
        num_batches = 0
        accumulated_steps = 0
        total_steps = len(dataloader)
        running_avg_loss = 0.0
        running_avg_unifrac_loss = 0.0
        running_avg_nuc_loss = 0.0

        self.optimizer.zero_grad()

        current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for step, batch in enumerate(pbar, 1):
            try:
                tokens, targets = self._prepare_batch(batch)

                # Check for invalid token values before model forward pass
                import sys

                # Get vocab_size from model (supports both SequenceEncoder and SequencePredictor)
                if hasattr(self.model, "vocab_size"):
                    vocab_size = self.model.vocab_size
                elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "vocab_size"):
                    vocab_size = self.model.base_model.vocab_size
                elif hasattr(self.model, "sample_encoder") and hasattr(self.model.sample_encoder, "asv_encoder"):
                    vocab_size = self.model.sample_encoder.asv_encoder.vocab_size
                else:
                    # Default to 6 if we can't find it (new default with START_TOKEN)
                    vocab_size = 6

                if torch.any(tokens < 0) or torch.any(tokens >= vocab_size):
                    invalid_mask = (tokens < 0) | (tokens >= vocab_size)
                    invalid_count = invalid_mask.sum().item()
                    print(f"ERROR: Invalid token values detected before model forward", file=sys.stderr, flush=True)
                    print(
                        f"tokens shape={tokens.shape}, min={tokens.min().item()}, max={tokens.max().item()}, invalid_count={invalid_count}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise ValueError(
                        f"Invalid token values: min={tokens.min().item()}, max={tokens.max().item()}, vocab_size={vocab_size}"
                    )
                if torch.any(torch.isnan(tokens)):
                    print(f"ERROR: NaN in tokens before model forward", file=sys.stderr, flush=True)
                    raise ValueError("NaN values found in tokens")

                return_nucleotides = "nucleotides" in targets or self.loss_fn.nuc_penalty > 0
                outputs = self.model(tokens, return_nucleotides=return_nucleotides)

                # Check for NaN in model outputs before loss computation
                import sys

                # For UniFrac, check embeddings instead of base_prediction
                encoder_type = self._get_encoder_type()
                if encoder_type == "unifrac" and "embeddings" in outputs:
                    if torch.any(torch.isnan(outputs["embeddings"])):
                        print(f"ERROR: NaN in embeddings before loss computation", file=sys.stderr, flush=True)
                        print(f"embeddings shape={outputs['embeddings'].shape}", file=sys.stderr, flush=True)
                        print(
                            f"embeddings min={outputs['embeddings'].min().item()}, max={outputs['embeddings'].max().item()}",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Check sample_embeddings too for debugging
                        if "sample_embeddings" in outputs:
                            print(f"sample_embeddings has NaN: {torch.any(torch.isnan(outputs['sample_embeddings']))}", file=sys.stderr, flush=True)
                        raise ValueError("NaN values found in embeddings before loss computation")
                elif "base_prediction" in outputs:
                    if torch.any(torch.isnan(outputs["base_prediction"])):
                        print(f"ERROR: NaN in base_prediction before loss computation", file=sys.stderr, flush=True)
                        print(f"base_prediction shape={outputs['base_prediction'].shape}", file=sys.stderr, flush=True)
                        print(
                            f"base_prediction min={outputs['base_prediction'].min().item()}, max={outputs['base_prediction'].max().item()}",
                            file=sys.stderr,
                            flush=True,
                        )

                if "base_target" in targets:
                    if torch.any(torch.isnan(targets["base_target"])):
                        print(f"ERROR: NaN in base_target before loss computation", file=sys.stderr, flush=True)
                        print(f"base_target shape={targets['base_target'].shape}", file=sys.stderr, flush=True)
                        print(
                            f"base_target min={targets['base_target'].min().item()}, max={targets['base_target'].max().item()}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if isinstance(batch, dict) and "sample_ids" in batch:
                            print(f"sample_ids in batch: {batch['sample_ids']}", file=sys.stderr, flush=True)

                if "nuc_predictions" in outputs:
                    if torch.any(torch.isnan(outputs["nuc_predictions"])):
                        print(f"ERROR: NaN in nuc_predictions before loss computation", file=sys.stderr, flush=True)
                        print(f"nuc_predictions shape={outputs['nuc_predictions'].shape}", file=sys.stderr, flush=True)
                        print(
                            f"nuc_predictions min={outputs['nuc_predictions'].min().item()}, max={outputs['nuc_predictions'].max().item()}",
                            file=sys.stderr,
                            flush=True,
                        )

                encoder_type = self._get_encoder_type()
                is_classifier = self._get_is_classifier()
                losses = self.loss_fn(outputs, targets, is_classifier=is_classifier, encoder_type=encoder_type)

                # Check for NaN in loss after computation
                if torch.any(torch.isnan(losses["total_loss"])):
                    print(f"ERROR: NaN in total_loss after computation", file=sys.stderr, flush=True)
                    print(f"losses: {losses}", file=sys.stderr, flush=True)
                    if "unifrac_loss" in losses:
                        print(f"unifrac_loss: {losses['unifrac_loss']}", file=sys.stderr, flush=True)

                current_loss_val = losses["total_loss"]
                if isinstance(current_loss_val, torch.Tensor):
                    current_loss_val = current_loss_val.detach().item()
                else:
                    current_loss_val = float(current_loss_val)

                scaled_loss = losses["total_loss"] / gradient_accumulation_steps
                scaled_loss.backward()

                del outputs, tokens, targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                accumulated_steps += 1

                if accumulated_steps % gradient_accumulation_steps == 0:
                    # Apply gradient clipping if enabled
                    if self.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        # Log gradient norm to TensorBoard if available
                        if self.writer is not None:
                            global_step = epoch * len(dataloader) + step
                            self.writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            pass
                        else:
                            self.scheduler.step()
                        current_lr = (
                            self.scheduler.get_last_lr()[0]
                            if hasattr(self.scheduler, "get_last_lr")
                            else self.optimizer.param_groups[0]["lr"]
                        )
                    else:
                        current_lr = self.optimizer.param_groups[0]["lr"]

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    if isinstance(value, torch.Tensor):
                        total_losses[key] += value.detach().item()
                    else:
                        total_losses[key] += float(value)

                if num_batches == 0:
                    running_avg_loss = current_loss_val
                    if "unifrac_loss" in losses:
                        unifrac_val = losses["unifrac_loss"]
                        if isinstance(unifrac_val, torch.Tensor):
                            running_avg_unifrac_loss = unifrac_val.detach().item()
                        else:
                            running_avg_unifrac_loss = float(unifrac_val)
                    if "nuc_loss" in losses:
                        nuc_val = losses["nuc_loss"]
                        if isinstance(nuc_val, torch.Tensor):
                            running_avg_nuc_loss = nuc_val.detach().item()
                        else:
                            running_avg_nuc_loss = float(nuc_val)
                else:
                    running_avg_loss = (running_avg_loss * num_batches + current_loss_val) / (num_batches + 1)
                    if "unifrac_loss" in losses:
                        unifrac_val = losses["unifrac_loss"]
                        if isinstance(unifrac_val, torch.Tensor):
                            unifrac_val = unifrac_val.detach().item()
                        else:
                            unifrac_val = float(unifrac_val)
                        running_avg_unifrac_loss = (running_avg_unifrac_loss * num_batches + unifrac_val) / (num_batches + 1)
                    if "nuc_loss" in losses:
                        nuc_val = losses["nuc_loss"]
                        if isinstance(nuc_val, torch.Tensor):
                            nuc_val = nuc_val.detach().item()
                        else:
                            nuc_val = float(nuc_val)
                        running_avg_nuc_loss = (running_avg_nuc_loss * num_batches + nuc_val) / (num_batches + 1)

                # Format progress bar
                postfix_dict = {
                    "TL": f"{running_avg_loss:.6f}" if running_avg_loss < 0.0001 else f"{running_avg_loss:.4f}",
                    "LR": f"{current_lr:.2e}",
                }
                if "unifrac_loss" in losses:
                    postfix_dict["UL"] = (
                        f"{running_avg_unifrac_loss:.6f}"
                        if running_avg_unifrac_loss < 0.0001
                        else f"{running_avg_unifrac_loss:.4f}"
                    )
                if "nuc_loss" in losses:
                    postfix_dict["NL"] = (
                        f"{running_avg_nuc_loss:.6f}" if running_avg_nuc_loss < 0.0001 else f"{running_avg_nuc_loss:.4f}"
                    )

                pbar.set_postfix(postfix_dict)

                del losses, scaled_loss
                num_batches += 1

            except torch.cuda.OutOfMemoryError as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error_msg = (
                    f"CUDA out of memory during training. "
                    f"Try: (1) reducing batch_size, (2) increasing gradient_accumulation_steps "
                    f"(current: {gradient_accumulation_steps}), (3) reducing model size, "
                    f"(4) setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                )
                raise RuntimeError(error_msg) from e

        if accumulated_steps % gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass
                else:
                    self.scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses

    def validate_epoch(
        self,
        dataloader: DataLoader,
        compute_metrics: bool = True,
        epoch: int = 0,
        num_epochs: int = 1,
        return_predictions: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Run one validation epoch.

        Args:
            dataloader: Validation data loader
            compute_metrics: Whether to compute metrics
            epoch: Current epoch number
            num_epochs: Total number of epochs

        Returns:
            Dictionary with losses and metrics
        """
        self.model.eval()
        total_losses = {}
        all_predictions = {}
        all_targets = {}
        num_batches = 0
        total_steps = len(dataloader)
        is_pretraining = self._is_pretraining()

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                leave=False,
            )

            for step, batch in enumerate(pbar, 1):
                try:
                    tokens, targets = self._prepare_batch(batch)

                    return_nucleotides = "nucleotides" in targets or self.loss_fn.nuc_penalty > 0
                    outputs = self.model(tokens, return_nucleotides=return_nucleotides)

                    encoder_type = self._get_encoder_type()
                    is_classifier = self._get_is_classifier()
                    losses = self.loss_fn(outputs, targets, is_classifier=is_classifier, encoder_type=encoder_type)

                    current_loss_val = losses["total_loss"]
                    if isinstance(current_loss_val, torch.Tensor):
                        current_loss_val = current_loss_val.detach().item()
                    else:
                        current_loss_val = float(current_loss_val)

                    for key, value in losses.items():
                        if key not in total_losses:
                            total_losses[key] = 0.0
                        if isinstance(value, torch.Tensor):
                            total_losses[key] += value.detach().item()
                        else:
                            total_losses[key] += float(value)

                    if num_batches == 0:
                        running_avg_loss = current_loss_val
                        if "unifrac_loss" in losses:
                            unifrac_val = losses["unifrac_loss"]
                            if isinstance(unifrac_val, torch.Tensor):
                                running_avg_unifrac_loss = unifrac_val.detach().item()
                            else:
                                running_avg_unifrac_loss = float(unifrac_val)
                        if "nuc_loss" in losses:
                            nuc_val = losses["nuc_loss"]
                            if isinstance(nuc_val, torch.Tensor):
                                running_avg_nuc_loss = nuc_val.detach().item()
                            else:
                                running_avg_nuc_loss = float(nuc_val)
                    else:
                        running_avg_loss = (running_avg_loss * num_batches + current_loss_val) / (num_batches + 1)
                        if "unifrac_loss" in losses:
                            unifrac_val = losses["unifrac_loss"]
                            if isinstance(unifrac_val, torch.Tensor):
                                unifrac_val = unifrac_val.detach().item()
                            else:
                                unifrac_val = float(unifrac_val)
                            running_avg_unifrac_loss = (running_avg_unifrac_loss * num_batches + unifrac_val) / (
                                num_batches + 1
                            )
                        if "nuc_loss" in losses:
                            nuc_val = losses["nuc_loss"]
                            if isinstance(nuc_val, torch.Tensor):
                                nuc_val = nuc_val.detach().item()
                            else:
                                nuc_val = float(nuc_val)
                            running_avg_nuc_loss = (running_avg_nuc_loss * num_batches + nuc_val) / (num_batches + 1)

                    # Format validation progress bar
                    postfix_dict = {
                        "TL": f"{running_avg_loss:.6f}" if running_avg_loss < 0.0001 else f"{running_avg_loss:.4f}",
                    }
                    if "unifrac_loss" in losses:
                        postfix_dict["UL"] = (
                            f"{running_avg_unifrac_loss:.6f}"
                            if running_avg_unifrac_loss < 0.0001
                            else f"{running_avg_unifrac_loss:.4f}"
                        )
                    if "nuc_loss" in losses:
                        postfix_dict["NL"] = (
                            f"{running_avg_nuc_loss:.6f}" if running_avg_nuc_loss < 0.0001 else f"{running_avg_nuc_loss:.4f}"
                        )

                    pbar.set_postfix(postfix_dict)

                    if compute_metrics:
                        if is_pretraining:
                            # For pretraining, collect base_prediction (UniFrac) for plotting
                            # For UniFrac, compute distances from embeddings if available
                            from aam.training.losses import compute_pairwise_distances
                            
                            if "base_target" in targets:
                                encoder_type = self._get_encoder_type()
                                if encoder_type == "unifrac" and "embeddings" in outputs:
                                    # Compute distances from embeddings
                                    embeddings = outputs["embeddings"]
                                    base_pred_batch = compute_pairwise_distances(embeddings)
                                elif "base_prediction" in outputs:
                                    base_pred_batch = outputs["base_prediction"]
                                else:
                                    base_pred_batch = None
                                
                                if base_pred_batch is not None:
                                    if "base_prediction" not in all_predictions:
                                        all_predictions["base_prediction"] = []
                                        all_targets["base_target"] = []
                                    # Extract upper triangle (excluding diagonal) from each batch matrix
                                    # to avoid including diagonal 0.0 values in validation plots
                                    base_true_batch = targets["base_target"]
                                    if base_pred_batch.dim() == 2 and base_pred_batch.shape[0] == base_pred_batch.shape[1]:
                                        batch_size = base_pred_batch.shape[0]
                                        triu_indices = torch.triu_indices(
                                            batch_size, batch_size, offset=1, device=base_pred_batch.device
                                        )
                                        base_pred_flat = base_pred_batch[triu_indices[0], triu_indices[1]]
                                        base_true_flat = base_true_batch[triu_indices[0], triu_indices[1]]
                                        all_predictions["base_prediction"].append(base_pred_flat)
                                        all_targets["base_target"].append(base_true_flat)
                                    else:
                                        # If not square, just flatten (shouldn't happen for UniFrac)
                                        all_predictions["base_prediction"].append(base_pred_batch.flatten())
                                        all_targets["base_target"].append(base_true_batch.flatten())
                        else:
                            # For regular training, collect target_prediction for plotting
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

                except torch.cuda.OutOfMemoryError as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    error_msg = (
                        f"CUDA out of memory during validation. "
                        f"Try: (1) reducing batch_size, (2) reducing model size, "
                        f"(3) setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                    )
                    raise RuntimeError(error_msg) from e

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        target_pred_tensor = None
        target_true_tensor = None
        base_pred_tensor = None
        base_true_tensor = None

        if compute_metrics and all_predictions:
            if is_pretraining and "base_prediction" in all_predictions:
                # For pretraining, compute metrics for UniFrac predictions
                # Predictions are already flattened (upper triangle only) from collection step
                base_pred_flat = torch.cat(all_predictions["base_prediction"], dim=0)
                base_true_flat = torch.cat(all_targets["base_target"], dim=0)

                metrics = compute_regression_metrics(base_pred_flat, base_true_flat)
                avg_losses.update(metrics)

                # For return_predictions, we need to return the flattened tensors
                # (they're already flattened, so we can use them directly)
                base_pred_tensor = base_pred_flat
                base_true_tensor = base_true_flat
            elif "target_prediction" in all_predictions:
                target_pred_tensor = torch.cat(all_predictions["target_prediction"], dim=0)
                target_true_tensor = torch.cat(all_targets["target"], dim=0)

                is_classifier = self._get_is_classifier()
                if is_classifier:
                    metrics = compute_classification_metrics(target_pred_tensor, target_true_tensor)
                else:
                    metrics = compute_regression_metrics(target_pred_tensor, target_true_tensor)
                avg_losses.update(metrics)

            if "count_prediction" in all_predictions:
                count_pred = torch.cat(all_predictions["count_prediction"], dim=0)
                count_true = torch.cat(all_targets["counts"], dim=0)
                mask = torch.cat(all_targets["mask"], dim=0)
                metrics = compute_count_metrics(count_pred, count_true, mask)
                avg_losses.update({f"count_{k}": v for k, v in metrics.items()})

        if return_predictions:
            if is_pretraining:
                # base_pred_tensor and base_true_tensor are already flattened (upper triangle only)
                return avg_losses, base_pred_tensor, base_true_tensor
            else:
                return avg_losses, target_pred_tensor, target_true_tensor
        return avg_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        save_plots: bool = True,
    ) -> Dict[str, list]:
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
            save_plots: Whether to save prediction plots when validation improves

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

        if self.tensorboard_dir is not None:
            tensorboard_path = Path(self.tensorboard_dir) / "tensorboard"
            tensorboard_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_path))

        try:
            for epoch in range(start_epoch, num_epochs):
                train_losses = self.train_epoch(
                    train_loader,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    epoch=epoch,
                    num_epochs=num_epochs,
                )
                history["train_loss"].append(train_losses["total_loss"])

                val_results = None
                if val_loader is not None:
                    return_predictions = save_plots
                    val_output = self.validate_epoch(
                        val_loader,
                        compute_metrics=True,
                        epoch=epoch,
                        num_epochs=num_epochs,
                        return_predictions=return_predictions,
                    )
                    if return_predictions:
                        val_results, val_predictions, val_targets = val_output
                    else:
                        val_results = val_output
                        val_predictions = None
                        val_targets = None
                    val_loss = val_results["total_loss"]
                    history["val_loss"].append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0

                        if checkpoint_dir is not None:
                            checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
                            if checkpoint_path.exists():
                                checkpoint_path.unlink()
                            self.save_checkpoint(
                                str(checkpoint_path),
                                epoch=epoch,
                                best_val_loss=best_val_loss,
                                metrics=val_results,
                            )

                        if save_plots and val_predictions is not None and val_targets is not None:
                            is_pretraining = self._is_pretraining()
                            self._save_prediction_plots(
                                val_predictions,
                                val_targets,
                                epoch,
                                val_results,
                                checkpoint_dir,
                                is_unifrac=is_pretraining,
                            )
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                else:
                    history["val_loss"].append(None)
                    train_loss = train_losses["total_loss"]
                    if train_loss < best_val_loss:
                        best_val_loss = train_loss
                        patience_counter = 0

                        if checkpoint_dir is not None:
                            checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
                            if checkpoint_path.exists():
                                checkpoint_path.unlink()
                            self.save_checkpoint(
                                str(checkpoint_path),
                                epoch=epoch,
                                best_val_loss=best_val_loss,
                                metrics=train_losses,
                            )
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

                if self.writer is not None:
                    self._log_to_tensorboard(epoch, train_losses, val_results)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loader is not None:
                            val_loss = val_results["total_loss"] if val_results else train_losses["total_loss"]
                        else:
                            val_loss = train_losses["total_loss"]
                        self.scheduler.step(val_loss)
                    elif isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        pass
                    elif not isinstance(self.scheduler, WarmupCosineScheduler):
                        self.scheduler.step()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

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
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    freeze_base: bool = False,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Create optimizer, excluding frozen parameters if needed.

    Args:
        model: Model to create optimizer for
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay (for AdamW/Adam)
        freeze_base: Whether base model parameters are frozen
        momentum: Momentum for SGD

    Returns:
        Optimizer instance
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

    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Must be one of: 'adamw', 'adam', 'sgd'")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "warmup_cosine",
    num_warmup_steps: int = 10000,
    num_training_steps: int = 100000,
    **kwargs,
) -> Union[WarmupCosineScheduler, torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('warmup_cosine', 'cosine', 'plateau', 'onecycle')
        num_warmup_steps: Number of warmup steps (for warmup_cosine)
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler-specific parameters

    Returns:
        Learning rate scheduler instance
    """
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max", num_training_steps)
        eta_min = kwargs.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == "plateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_type == "onecycle":
        max_lr = kwargs.get("max_lr", optimizer.param_groups[0]["lr"])
        pct_start = kwargs.get("pct_start", 0.3)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=num_training_steps, pct_start=pct_start
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Must be one of: 'warmup_cosine', 'cosine', 'plateau', 'onecycle'"
        )


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
