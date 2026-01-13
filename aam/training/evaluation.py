"""Validation and evaluation logic for AAM model training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Tuple, Union, cast
from pathlib import Path
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aam.training.metrics import (
    StreamingRegressionMetrics,
    StreamingClassificationMetrics,
    StreamingCountMetrics,
)


def create_prediction_plot(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    r2: float,
    mae: Optional[float] = None,
    title_prefix: str = "Target",
) -> plt.Figure:
    """Create prediction vs actual scatter plot for regression tasks.

    Args:
        predictions: Predicted values [B, 1] or [B]
        targets: Actual values [B, 1] or [B]
        epoch: Current epoch number
        r2: R² score
        mae: Mean Absolute Error (optional)
        title_prefix: Prefix for plot title (e.g., "Target", "Count", "UniFrac")

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

    metrics_str = f"R² = {r2:.4f}"
    if mae is not None:
        metrics_str += f", MAE = {mae:.4f}"

    if len(target_np) > 1:
        z = np.polyfit(target_np, pred_np, 1)
        p = np.poly1d(z)
        ax.plot(target_np, p(target_np), "b-", linewidth=2, label=f"Linear Fit, {metrics_str}")

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(f"{title_prefix} Prediction vs Actual (Epoch {epoch}, {metrics_str})", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_unifrac_prediction_plot(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    r2: float,
    mae: Optional[float] = None,
) -> plt.Figure:
    """Create prediction vs actual scatter plot for UniFrac predictions (pretraining).

    Args:
        predictions: Predicted UniFrac distances (1D array, already flattened upper triangle)
        targets: Actual UniFrac distances (1D array, already flattened upper triangle)
        epoch: Current epoch number
        r2: R² score
        mae: Mean Absolute Error (optional)

    Returns:
        Matplotlib figure
    """
    pred_np = predictions.detach().cpu()
    target_np = targets.detach().cpu()

    try:
        pred_flat = pred_np.numpy() if hasattr(pred_np, "numpy") else np.array(pred_np)
        target_flat = target_np.numpy() if hasattr(target_np, "numpy") else np.array(target_np)
    except RuntimeError:
        pred_flat = np.array(pred_np.tolist())
        target_flat = np.array(target_np.tolist())

    if pred_flat.ndim > 1:
        pred_flat = pred_flat.flatten()
    if target_flat.ndim > 1:
        target_flat = target_flat.flatten()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.scatter(target_flat, pred_flat, alpha=0.6, s=20)

    min_val = min(target_flat.min(), pred_flat.min())
    max_val = max(target_flat.max(), pred_flat.max())

    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Perfect Prediction", alpha=0.5)

    metrics_str = f"R² = {r2:.4f}"
    if mae is not None:
        metrics_str += f", MAE = {mae:.4f}"

    if len(target_flat) > 1:
        z = np.polyfit(target_flat, pred_flat, 1)
        p = np.poly1d(z)
        ax.plot(target_flat, p(target_flat), "b-", linewidth=2, label=f"Linear Fit, {metrics_str}")

    ax.set_xlabel("Actual UniFrac Distance", fontsize=12)
    ax.set_ylabel("Predicted UniFrac Distance", fontsize=12)
    ax.set_title(f"UniFrac Prediction vs Actual (Epoch {epoch}, {metrics_str})", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_confusion_matrix_plot(
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


class Evaluator:
    """Handles validation and evaluation for AAM models.

    This class extracts validation logic from the Trainer to provide a cleaner
    separation of concerns. It handles:
    - Running validation epochs
    - Computing streaming metrics
    - Creating and saving prediction plots
    - Logging figures to TensorBoard
    """

    def __init__(
        self,
        model: Any,  # nn.Module or torch.compile wrapped model
        loss_fn: nn.Module,
        device: torch.device,
        mixed_precision: Optional[str] = None,
        target_normalization_params: Optional[Dict[str, float]] = None,
        count_normalization_params: Optional[Dict[str, float]] = None,
    ):
        """Initialize Evaluator.

        Args:
            model: Model to evaluate (can be nn.Module or torch.compile wrapped)
            loss_fn: Loss function
            device: Device to run evaluation on
            mixed_precision: Mixed precision mode ('fp16', 'bf16', or None)
            target_normalization_params: Dict with 'target_min', 'target_max', 'target_scale'
            count_normalization_params: Dict with 'count_min', 'count_max', 'count_scale'
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.mixed_precision = mixed_precision
        self.target_normalization_params = target_normalization_params
        self.count_normalization_params = count_normalization_params

    def _get_encoder_type(self) -> str:
        """Get encoder type from model."""
        if hasattr(self.model, "encoder_type"):
            return str(self.model.encoder_type)
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "encoder_type"):
            return str(self.model.base_model.encoder_type)
        return "unifrac"

    def _get_is_classifier(self) -> bool:
        """Check if model is a classifier."""
        if hasattr(self.model, "is_classifier"):
            return bool(self.model.is_classifier)
        return False

    def _is_pretraining(self) -> bool:
        """Check if model is SequenceEncoder (pretraining mode)."""
        model_class_name = self.model.__class__.__name__
        if model_class_name == "SequenceEncoder":
            return True
        if hasattr(self.model, "_orig_mod"):
            return self.model._orig_mod.__class__.__name__ == "SequenceEncoder"
        return False

    def _denormalize_targets(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize/inverse-transform target values back to original scale."""
        if self.target_normalization_params is None:
            return values

        result = values

        # First, denormalize if normalization was applied
        if "target_scale" in self.target_normalization_params:
            target_min = self.target_normalization_params["target_min"]
            target_scale = self.target_normalization_params["target_scale"]
            result = result * target_scale + target_min

        # Then, inverse log transform if it was applied
        if self.target_normalization_params.get("log_transform", False):
            # Clamp to prevent exp() overflow (exp(88.7) overflows float32)
            MAX_EXP_INPUT = 88.0
            result = torch.exp(torch.clamp(result, max=MAX_EXP_INPUT)) - 1

        return result

    def _denormalize_counts(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize count values back to original scale."""
        if self.count_normalization_params is None:
            return values
        count_min = self.count_normalization_params["count_min"]
        count_scale = self.count_normalization_params["count_scale"]
        return values * count_scale + count_min

    def _prepare_batch(
        self, batch: Union[Dict, Tuple]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Prepare batch for model forward pass.

        Args:
            batch: Batch from DataLoader (dict or tuple)

        Returns:
            Tuple of (tokens, targets_dict, categorical_ids)
        """
        categorical_ids: Optional[Dict[str, torch.Tensor]] = None

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
            if "categorical_ids" in batch:
                categorical_ids = {col: ids.to(self.device) for col, ids in batch["categorical_ids"].items()}

            return tokens, targets, categorical_ids
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

            return tokens, targets, categorical_ids

    def validate_epoch(
        self,
        dataloader: DataLoader,
        compute_metrics: bool = True,
        epoch: int = 0,
        num_epochs: int = 1,
        return_predictions: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Run one validation epoch with streaming metrics computation.

        Uses O(batch) memory instead of O(dataset) by computing metrics incrementally.
        Plot data is retained via reservoir sampling (max 1000 samples by default).

        Args:
            dataloader: Validation data loader
            compute_metrics: Whether to compute metrics
            epoch: Current epoch number
            num_epochs: Total number of epochs
            return_predictions: Whether to return sampled predictions for plotting

        Returns:
            Dictionary with losses and metrics, optionally with prediction samples
        """
        self.model.eval()
        total_losses: Dict[str, float] = {}
        num_batches = 0
        is_pretraining = self._is_pretraining()
        is_classifier = self._get_is_classifier()
        encoder_type = self._get_encoder_type()

        unifrac_metrics = StreamingRegressionMetrics(max_plot_samples=1000)
        target_metrics: Union[StreamingClassificationMetrics, StreamingRegressionMetrics] = (
            StreamingClassificationMetrics(max_plot_samples=1000)
            if is_classifier
            else StreamingRegressionMetrics(max_plot_samples=1000)
        )
        count_metrics = StreamingCountMetrics(max_plot_samples=1000)

        has_unifrac = False
        has_target = False
        has_count = False

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                leave=False,
            )

            last_targets = None
            last_outputs = None

            running_avg_loss = 0.0
            running_avg_unifrac_loss = 0.0
            running_avg_nuc_loss = 0.0
            running_avg_nuc_accuracy = 0.0
            running_avg_target_loss = 0.0

            show_nuc_metrics = cast(float, self.loss_fn.nuc_penalty) > 0

            for step, batch in enumerate(pbar, 1):
                try:
                    tokens, targets, categorical_ids = self._prepare_batch(batch)
                    last_targets = targets

                    return_nucleotides = "nucleotides" in targets or cast(float, self.loss_fn.nuc_penalty) > 0

                    autocast_dtype = None
                    if self.mixed_precision == "fp16":
                        autocast_dtype = torch.float16
                    elif self.mixed_precision == "bf16":
                        autocast_dtype = torch.bfloat16

                    # Only pass categorical_ids if the model supports it (SequencePredictor has categorical_embedder)
                    supports_categorical = hasattr(self.model, "categorical_embedder")
                    forward_kwargs: Dict[str, Any] = {"return_nucleotides": return_nucleotides}
                    if supports_categorical and categorical_ids is not None:
                        forward_kwargs["categorical_ids"] = categorical_ids

                    if autocast_dtype is not None and self.device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                            outputs = self.model(tokens, **forward_kwargs)
                    else:
                        outputs = self.model(tokens, **forward_kwargs)
                    last_outputs = outputs

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

                    nuc_accuracy_val = 0.0
                    if "nuc_predictions" in outputs:
                        nuc_preds = outputs["nuc_predictions"]
                        nuc_targets = targets.get("tokens", targets.get("nucleotides"))
                        if nuc_targets is not None:
                            predicted_tokens = nuc_preds.argmax(dim=-1)
                            correct = predicted_tokens == nuc_targets

                            mask_indices = outputs.get("mask_indices")
                            if mask_indices is not None and mask_indices.any():
                                nuc_accuracy_val = correct[mask_indices].float().mean().item()
                            else:
                                valid_mask = (nuc_targets >= 1) & (nuc_targets <= 4)
                                if valid_mask.any():
                                    nuc_accuracy_val = correct[valid_mask].float().mean().item()

                        if "nuc_accuracy" not in total_losses:
                            total_losses["nuc_accuracy"] = 0.0
                        total_losses["nuc_accuracy"] += nuc_accuracy_val

                    if num_batches == 0:
                        running_avg_loss = current_loss_val
                        running_avg_nuc_accuracy = nuc_accuracy_val
                        if "unifrac_loss" in losses:
                            unifrac_val = losses["unifrac_loss"]
                            running_avg_unifrac_loss = (
                                unifrac_val.detach().item() if isinstance(unifrac_val, torch.Tensor) else float(unifrac_val)
                            )
                        if "target_loss" in losses:
                            target_val = losses["target_loss"]
                            running_avg_target_loss = (
                                target_val.detach().item() if isinstance(target_val, torch.Tensor) else float(target_val)
                            )
                        if "nuc_loss" in losses:
                            nuc_val = losses["nuc_loss"]
                            running_avg_nuc_loss = (
                                nuc_val.detach().item() if isinstance(nuc_val, torch.Tensor) else float(nuc_val)
                            )
                    else:
                        running_avg_loss = (running_avg_loss * num_batches + current_loss_val) / (num_batches + 1)
                        running_avg_nuc_accuracy = (running_avg_nuc_accuracy * num_batches + nuc_accuracy_val) / (
                            num_batches + 1
                        )
                        if "unifrac_loss" in losses:
                            unifrac_val = losses["unifrac_loss"]
                            unifrac_val = (
                                unifrac_val.detach().item() if isinstance(unifrac_val, torch.Tensor) else float(unifrac_val)
                            )
                            running_avg_unifrac_loss = (running_avg_unifrac_loss * num_batches + unifrac_val) / (
                                num_batches + 1
                            )
                        if "target_loss" in losses:
                            target_val = losses["target_loss"]
                            target_val = (
                                target_val.detach().item() if isinstance(target_val, torch.Tensor) else float(target_val)
                            )
                            running_avg_target_loss = (running_avg_target_loss * num_batches + target_val) / (num_batches + 1)
                        if "nuc_loss" in losses:
                            nuc_val = losses["nuc_loss"]
                            nuc_val = nuc_val.detach().item() if isinstance(nuc_val, torch.Tensor) else float(nuc_val)
                            running_avg_nuc_loss = (running_avg_nuc_loss * num_batches + nuc_val) / (num_batches + 1)

                    postfix_dict = {
                        "TL": f"{running_avg_loss:.6f}" if running_avg_loss < 0.0001 else f"{running_avg_loss:.4f}",
                    }
                    if not is_pretraining and "target_loss" in losses:
                        loss_label = "CL" if is_classifier else "RL"
                        postfix_dict[loss_label] = (
                            f"{running_avg_target_loss:.6f}"
                            if running_avg_target_loss < 0.0001
                            else f"{running_avg_target_loss:.4f}"
                        )
                    if "unifrac_loss" in losses:
                        postfix_dict["UL"] = (
                            f"{running_avg_unifrac_loss:.6f}"
                            if running_avg_unifrac_loss < 0.0001
                            else f"{running_avg_unifrac_loss:.4f}"
                        )
                    if show_nuc_metrics and "nuc_loss" in losses:
                        postfix_dict["NL"] = (
                            f"{running_avg_nuc_loss:.6f}" if running_avg_nuc_loss < 0.0001 else f"{running_avg_nuc_loss:.4f}"
                        )
                    if show_nuc_metrics and "nuc_predictions" in outputs:
                        postfix_dict["NA"] = f"{running_avg_nuc_accuracy:.2%}"

                    pbar.set_postfix(postfix_dict)

                    if compute_metrics:
                        if "base_target" in targets:
                            from aam.training.losses import compute_pairwise_distances

                            base_pred_batch = None

                            if encoder_type == "unifrac" and "embeddings" in outputs:
                                embeddings = outputs["embeddings"]
                                if embeddings is not None:
                                    try:
                                        base_pred_batch = compute_pairwise_distances(embeddings.detach()).detach()
                                    except Exception:
                                        base_pred_batch = None

                            if base_pred_batch is None and "base_prediction" in outputs:
                                base_pred_batch = outputs["base_prediction"]

                            if base_pred_batch is not None:
                                base_true_batch = targets["base_target"]
                                if base_pred_batch.dim() == 2 and base_pred_batch.shape[0] == base_pred_batch.shape[1]:
                                    batch_size = base_pred_batch.shape[0]
                                    triu_indices = torch.triu_indices(
                                        batch_size, batch_size, offset=1, device=base_pred_batch.device
                                    )
                                    base_pred_flat = base_pred_batch[triu_indices[0], triu_indices[1]]
                                    base_true_flat = base_true_batch[triu_indices[0], triu_indices[1]]
                                else:
                                    base_pred_flat = base_pred_batch.flatten()
                                    base_true_flat = base_true_batch.flatten()

                                unifrac_metrics.update(base_pred_flat, base_true_flat)
                                has_unifrac = True

                        if not is_pretraining and "target_prediction" in outputs and "target" in targets:
                            pred = outputs["target_prediction"]
                            true = targets["target"]
                            pred_denorm = self._denormalize_targets(pred)
                            true_denorm = self._denormalize_targets(true)
                            target_metrics.update(pred_denorm, true_denorm)
                            has_target = True

                        if "count_prediction" in outputs and "counts" in targets:
                            count_pred = outputs["count_prediction"]
                            count_true = targets["counts"]
                            mask = targets.get("mask", (tokens.sum(dim=-1) > 0).long())
                            count_pred_denorm = self._denormalize_counts(count_pred)
                            count_true_denorm = self._denormalize_counts(count_true)
                            count_metrics.update(count_pred_denorm, count_true_denorm, mask)
                            has_count = True

                    num_batches += 1

                except torch.cuda.OutOfMemoryError as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    error_msg = (
                        "CUDA out of memory during validation. "
                        "Try: (1) reducing batch_size, (2) reducing model size, "
                        "(3) setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                    )
                    raise RuntimeError(error_msg) from e

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        if compute_metrics:
            if is_pretraining and has_unifrac:
                metrics = unifrac_metrics.compute()
                avg_losses.update(metrics)
            elif has_target:
                metrics = target_metrics.compute()
                avg_losses.update(metrics)

            if has_count:
                metrics = count_metrics.compute()
                avg_losses.update({f"count_{k}": v for k, v in metrics.items()})

        if compute_metrics and not (has_unifrac or has_target or has_count):
            import logging

            logger = logging.getLogger(__name__)
            target_keys = list(last_targets.keys()) if last_targets is not None else []
            output_keys = list(last_outputs.keys()) if last_outputs is not None else []
            logger.warning(
                f"No predictions collected for metrics computation at epoch {epoch}. "
                f"Outputs keys: {output_keys}, Target keys: {target_keys}, is_pretraining: {is_pretraining}"
            )

        if return_predictions:
            all_preds: Dict[str, torch.Tensor] = {}
            all_targs: Dict[str, torch.Tensor] = {}

            if has_unifrac:
                pred_samples, targ_samples = unifrac_metrics.get_plot_data()
                all_preds["unifrac"] = pred_samples
                all_targs["unifrac"] = targ_samples

            if has_target:
                pred_samples, targ_samples = target_metrics.get_plot_data()
                all_preds["target"] = pred_samples
                all_targs["target"] = targ_samples

            if has_count:
                pred_samples, targ_samples = count_metrics.get_plot_data()
                all_preds["count"] = pred_samples
                all_targs["count"] = targ_samples

            return avg_losses, all_preds, all_targs

        return avg_losses

    def save_prediction_plots(
        self,
        predictions_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: Optional[str],
    ) -> None:
        """Save prediction plots when validation improves.

        Args:
            predictions_dict: Dictionary of predictions by type (target, unifrac, count)
            targets_dict: Dictionary of targets by type
            epoch: Current epoch number
            metrics: Metrics dictionary
            checkpoint_dir: Directory to save plots
        """
        if checkpoint_dir is None:
            return

        is_classifier = self._get_is_classifier()
        plots_dir = Path(checkpoint_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if "target" in predictions_dict:
            if is_classifier:
                if all(k in metrics for k in ["accuracy", "precision", "recall", "f1"]):
                    fig = create_confusion_matrix_plot(
                        predictions_dict["target"],
                        targets_dict["target"],
                        epoch,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                    )
                    plot_file = plots_dir / "prediction_plot_best.png"
                    if plot_file.exists():
                        plot_file.unlink()
                    fig.savefig(plot_file, dpi=100, bbox_inches="tight")
                    plt.close(fig)
            else:
                if "r2" in metrics:
                    fig = create_prediction_plot(
                        predictions_dict["target"],
                        targets_dict["target"],
                        epoch,
                        metrics["r2"],
                        mae=metrics.get("mae"),
                        title_prefix="Target",
                    )
                    plot_file = plots_dir / "prediction_plot_best.png"
                    if plot_file.exists():
                        plot_file.unlink()
                    fig.savefig(plot_file, dpi=100, bbox_inches="tight")
                    plt.close(fig)

        if "unifrac" in predictions_dict:
            if "r2" in metrics:
                fig = create_unifrac_prediction_plot(
                    predictions_dict["unifrac"],
                    targets_dict["unifrac"],
                    epoch,
                    metrics["r2"],
                    mae=metrics.get("mae"),
                )
                plot_file = plots_dir / "unifrac_plot_best.png"
                if plot_file.exists():
                    plot_file.unlink()
                fig.savefig(plot_file, dpi=100, bbox_inches="tight")
                plt.close(fig)

        if "count" in predictions_dict:
            r2 = metrics.get("count_r2", metrics.get("r2"))
            mae = metrics.get("count_mae", metrics.get("mae"))
            if r2 is not None:
                fig = create_prediction_plot(
                    predictions_dict["count"],
                    targets_dict["count"],
                    epoch,
                    r2,
                    mae=mae,
                    title_prefix="Count",
                )
                plot_file = plots_dir / "count_plot_best.png"
                if plot_file.exists():
                    plot_file.unlink()
                fig.savefig(plot_file, dpi=100, bbox_inches="tight")
                plt.close(fig)

    def log_figures_to_tensorboard(
        self,
        epoch: int,
        predictions_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        writer: SummaryWriter,
    ) -> None:
        """Log prediction figures to TensorBoard at every epoch.

        Args:
            epoch: Current epoch number
            predictions_dict: Dictionary of predictions by type (target, unifrac, count)
            targets_dict: Dictionary of targets by type
            metrics: Metrics dictionary
            writer: TensorBoard SummaryWriter
        """
        is_classifier = self._get_is_classifier()

        if "target" in predictions_dict:
            if is_classifier:
                if all(k in metrics for k in ["accuracy", "precision", "recall", "f1"]):
                    fig = create_confusion_matrix_plot(
                        predictions_dict["target"],
                        targets_dict["target"],
                        epoch,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                    )
                    writer.add_figure("validation/prediction_plot", fig, epoch)
                    plt.close(fig)
            else:
                if "r2" in metrics:
                    fig = create_prediction_plot(
                        predictions_dict["target"],
                        targets_dict["target"],
                        epoch,
                        metrics["r2"],
                        mae=metrics.get("mae"),
                        title_prefix="Target",
                    )
                    writer.add_figure("validation/prediction_plot", fig, epoch)
                    plt.close(fig)

        if "unifrac" in predictions_dict:
            r2 = metrics.get("r2") if "target" not in predictions_dict else None
            if r2 is None:
                from sklearn.metrics import r2_score

                try:
                    pred_np = predictions_dict["unifrac"].cpu().numpy().flatten()
                    true_np = targets_dict["unifrac"].cpu().numpy().flatten()
                    r2 = r2_score(true_np, pred_np)
                    mae = float(np.abs(pred_np - true_np).mean())
                except Exception:
                    r2 = None
                    mae = None
            else:
                mae = metrics.get("mae")

            if r2 is not None:
                fig = create_unifrac_prediction_plot(
                    predictions_dict["unifrac"],
                    targets_dict["unifrac"],
                    epoch,
                    r2,
                    mae=mae,
                )
                writer.add_figure("validation/unifrac_plot", fig, epoch)
                plt.close(fig)

        if "count" in predictions_dict:
            r2 = metrics.get("count_r2")
            mae = metrics.get("count_mae")
            if r2 is not None:
                fig = create_prediction_plot(
                    predictions_dict["count"],
                    targets_dict["count"],
                    epoch,
                    r2,
                    mae=mae,
                    title_prefix="Count",
                )
                writer.add_figure("validation/count_plot", fig, epoch)
                plt.close(fig)
