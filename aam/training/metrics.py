"""Metrics computation for AAM model evaluation."""

import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class StreamingRegressionMetrics:
    """Streaming computation of regression metrics (MAE, MSE, R²).

    Uses Welford's online algorithm for numerically stable computation of
    running statistics. Memory usage is O(1) regardless of dataset size.
    """

    def __init__(self, max_plot_samples: int = 1000):
        """Initialize streaming metrics.

        Args:
            max_plot_samples: Maximum samples to retain for plotting (reservoir sampling)
        """
        self.n = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.mean_true = 0.0
        self.m2_true = 0.0  # For computing variance of true values (Welford)
        self.sum_sq_total = 0.0  # Will be finalized after mean is known

        # Reservoir sampling for plot data
        self.max_plot_samples = max_plot_samples
        self.plot_predictions = []
        self.plot_targets = []
        self._reservoir_idx = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with a new batch.

        Args:
            predictions: Predicted values (any shape, will be flattened)
            targets: True values (same shape as predictions)
        """
        pred_flat = predictions.detach().cpu().float().flatten()
        true_flat = targets.detach().cpu().float().flatten()

        batch_size = pred_flat.numel()
        if batch_size == 0:
            return

        # Update error sums
        errors = pred_flat - true_flat
        self.sum_abs_error += torch.abs(errors).sum().item()
        self.sum_sq_error += (errors**2).sum().item()

        # Update mean of true values using Welford's algorithm (batch update)
        for val in true_flat.tolist():
            self.n += 1
            delta = val - self.mean_true
            self.mean_true += delta / self.n
            delta2 = val - self.mean_true
            self.m2_true += delta * delta2

        # Reservoir sampling for plot data
        for i in range(batch_size):
            if len(self.plot_predictions) < self.max_plot_samples:
                self.plot_predictions.append(pred_flat[i].item())
                self.plot_targets.append(true_flat[i].item())
            else:
                # Reservoir sampling: replace with decreasing probability
                j = np.random.randint(0, self._reservoir_idx + 1)
                if j < self.max_plot_samples:
                    self.plot_predictions[j] = pred_flat[i].item()
                    self.plot_targets[j] = true_flat[i].item()
            self._reservoir_idx += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with 'mae', 'mse', 'r2' keys
        """
        if self.n == 0:
            return {"mae": 0.0, "mse": 0.0, "r2": 0.0}

        mae = self.sum_abs_error / self.n
        mse = self.sum_sq_error / self.n

        # R² = 1 - SSres/SStot where SStot = variance * n
        variance_true = self.m2_true / self.n if self.n > 0 else 0.0
        ss_tot = variance_true * self.n

        if ss_tot > 0:
            r2 = 1.0 - (self.sum_sq_error / ss_tot)
        else:
            r2 = 0.0  # All true values are the same

        return {"mae": float(mae), "mse": float(mse), "r2": float(r2)}

    def get_plot_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sampled data for plotting.

        Returns:
            Tuple of (predictions, targets) tensors
        """
        return (
            torch.tensor(self.plot_predictions, dtype=torch.float32),
            torch.tensor(self.plot_targets, dtype=torch.float32),
        )

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self.n = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.mean_true = 0.0
        self.m2_true = 0.0
        self.plot_predictions = []
        self.plot_targets = []
        self._reservoir_idx = 0

    def _merge_from(self, other: "StreamingRegressionMetrics") -> None:
        """Merge statistics from another StreamingRegressionMetrics instance.

        Uses parallel Welford algorithm to correctly combine running variance.
        After merging, this instance contains statistics for the combined dataset.

        Args:
            other: Another StreamingRegressionMetrics instance to merge from.
        """
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.sum_abs_error = other.sum_abs_error
            self.sum_sq_error = other.sum_sq_error
            self.mean_true = other.mean_true
            self.m2_true = other.m2_true
            return

        # Merge sums (these can simply be added)
        n_combined = self.n + other.n
        self.sum_abs_error += other.sum_abs_error
        self.sum_sq_error += other.sum_sq_error

        # Parallel Welford merge for mean and m2
        # Formula: delta = mean_b - mean_a
        #          mean_combined = mean_a + delta * n_b / n_combined
        #          m2_combined = m2_a + m2_b + delta^2 * n_a * n_b / n_combined
        delta = other.mean_true - self.mean_true
        self.mean_true = self.mean_true + delta * other.n / n_combined
        self.m2_true = self.m2_true + other.m2_true + delta**2 * self.n * other.n / n_combined

        self.n = n_combined

    def sync_distributed(self) -> None:
        """Synchronize metrics across all distributed processes.

        Uses all_reduce for sums and parallel Welford merge for variance stats.
        After calling this method, all ranks will have identical metrics computed
        over the full validation set.

        This is a no-op if not running in distributed mode.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        # Gather all statistics from all ranks
        # We need to gather individual stats and merge using parallel Welford
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pack local statistics into a tensor for all_gather
        local_stats = torch.tensor(
            [self.n, self.sum_abs_error, self.sum_sq_error, self.mean_true, self.m2_true],
            dtype=torch.float64,
            device=device,
        )

        # Gather from all ranks
        gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]
        dist.all_gather(gathered, local_stats)

        # Reset and merge all gathered statistics
        self.n = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.mean_true = 0.0
        self.m2_true = 0.0

        for stats in gathered:
            other_n = int(stats[0].item())
            if other_n == 0:
                continue

            other = StreamingRegressionMetrics.__new__(StreamingRegressionMetrics)
            other.n = other_n
            other.sum_abs_error = stats[1].item()
            other.sum_sq_error = stats[2].item()
            other.mean_true = stats[3].item()
            other.m2_true = stats[4].item()
            self._merge_from(other)


class StreamingClassificationMetrics:
    """Streaming computation of classification metrics.

    Accumulates a confusion matrix incrementally. Memory usage is O(num_classes²).
    """

    def __init__(self, num_classes: Optional[int] = None, max_plot_samples: int = 1000):
        """Initialize streaming metrics.

        Args:
            num_classes: Number of classes (if None, inferred from data)
            max_plot_samples: Maximum samples to retain for plotting
        """
        self.num_classes = num_classes
        self.confusion_matrix = None
        self.n = 0

        # Store samples for confusion matrix plot
        self.max_plot_samples = max_plot_samples
        self.plot_predictions = []
        self.plot_targets = []
        self._reservoir_idx = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with a new batch.

        Args:
            predictions: Predicted logits [batch, num_classes] or class indices [batch]
            targets: True class indices [batch]
        """
        # Convert predictions to class indices if needed
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            pred_indices = predictions.detach().cpu().argmax(dim=-1)
        else:
            pred_indices = predictions.detach().cpu().flatten().long()

        true_indices = targets.detach().cpu().flatten().long()
        batch_size = pred_indices.numel()

        if batch_size == 0:
            return

        # Determine required size from data
        max_class = max(int(pred_indices.max().item()), int(true_indices.max().item()))
        required_size = max_class + 1
        if self.num_classes is None:
            self.num_classes = required_size
        elif required_size > self.num_classes:
            self.num_classes = required_size

        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        else:
            self._expand_confusion_matrix(self.num_classes)

        # Update confusion matrix
        for pred, true in zip(pred_indices.tolist(), true_indices.tolist()):
            self.confusion_matrix[true, pred] += 1
            self.n += 1

        # Reservoir sampling for plot data
        for i in range(batch_size):
            if len(self.plot_predictions) < self.max_plot_samples:
                self.plot_predictions.append(pred_indices[i].item())
                self.plot_targets.append(true_indices[i].item())
            else:
                j = np.random.randint(0, self._reservoir_idx + 1)
                if j < self.max_plot_samples:
                    self.plot_predictions[j] = pred_indices[i].item()
                    self.plot_targets[j] = true_indices[i].item()
            self._reservoir_idx += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with 'accuracy', 'precision', 'recall', 'f1' keys
        """
        if self.n == 0 or self.confusion_matrix is None:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Accuracy from confusion matrix diagonal
        accuracy = np.trace(self.confusion_matrix) / self.n

        # Compute precision, recall, F1 (weighted average)
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        total_support = 0

        assert self.num_classes is not None  # Set in update() before compute()
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            support = self.confusion_matrix[c, :].sum()

            if support == 0:
                continue

            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0.0

            precision_sum += precision_c * support
            recall_sum += recall_c * support
            f1_sum += f1_c * support
            total_support += support

        if total_support > 0:
            precision = precision_sum / total_support
            recall = recall_sum / total_support
            f1 = f1_sum / total_support
        else:
            precision = recall = f1 = 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def get_plot_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sampled data for confusion matrix plotting.

        Returns:
            Tuple of (predictions, targets) tensors
        """
        return (
            torch.tensor(self.plot_predictions, dtype=torch.long),
            torch.tensor(self.plot_targets, dtype=torch.long),
        )

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self.confusion_matrix = None
        self.n = 0
        self.plot_predictions = []
        self.plot_targets = []
        self._reservoir_idx = 0

    def _expand_confusion_matrix(self, new_size: int) -> None:
        """Expand confusion matrix to accommodate more classes."""
        if self.confusion_matrix is None or self.confusion_matrix.shape[0] >= new_size:
            return
        new_cm = np.zeros((new_size, new_size), dtype=np.int64)
        old_size = self.confusion_matrix.shape[0]
        new_cm[:old_size, :old_size] = self.confusion_matrix
        self.confusion_matrix = new_cm
        self.num_classes = new_size

    def _merge_from(self, other: "StreamingClassificationMetrics") -> None:
        """Merge statistics from another StreamingClassificationMetrics instance.

        Simply adds confusion matrices element-wise.

        Args:
            other: Another StreamingClassificationMetrics instance to merge from.
        """
        if other.n == 0 or other.confusion_matrix is None:
            return

        self.n += other.n

        if self.confusion_matrix is None:
            self.num_classes = other.num_classes
            self.confusion_matrix = other.confusion_matrix.copy()
            return

        other_size = other.confusion_matrix.shape[0]
        self._expand_confusion_matrix(other_size)
        self.confusion_matrix[:other_size, :other_size] += other.confusion_matrix

    def sync_distributed(self) -> None:
        """Synchronize metrics across all distributed processes.

        Uses all_reduce to sum confusion matrices across all ranks.
        After calling this method, all ranks will have identical metrics computed
        over the full validation set.

        This is a no-op if not running in distributed mode.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        if self.confusion_matrix is None:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert confusion matrix to tensor for all_reduce
        cm_tensor = torch.tensor(self.confusion_matrix, dtype=torch.int64, device=device)

        # All-reduce to sum confusion matrices across all ranks
        dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)

        # Update confusion matrix and count
        self.confusion_matrix = cm_tensor.cpu().numpy()
        self.n = int(self.confusion_matrix.sum())


class StreamingCountMetrics:
    """Streaming computation of count prediction metrics (masked MAE, MSE).

    Memory usage is O(1) regardless of dataset size.
    """

    def __init__(self, max_plot_samples: int = 1000):
        """Initialize streaming metrics.

        Args:
            max_plot_samples: Maximum samples to retain for plotting
        """
        self.n = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0

        # Reservoir sampling for plot data
        self.max_plot_samples = max_plot_samples
        self.plot_predictions = []
        self.plot_targets = []
        self._reservoir_idx = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        """Update metrics with a new batch.

        Args:
            predictions: Predicted counts [batch, num_asvs] or [batch, num_asvs, 1]
            targets: True counts (same shape as predictions)
            mask: Valid ASV mask [batch, num_asvs] (1=valid, 0=padding)
        """
        pred = predictions.detach().cpu().float()
        true = targets.detach().cpu().float()
        m = mask.detach().cpu().bool()

        # Handle extra dimension
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if true.dim() == 3:
            true = true.squeeze(-1)

        # Apply mask
        valid_pred = pred[m]
        valid_true = true[m]

        batch_valid = valid_pred.numel()
        if batch_valid == 0:
            return

        # Update error sums
        errors = valid_pred - valid_true
        self.sum_abs_error += torch.abs(errors).sum().item()
        self.sum_sq_error += (errors**2).sum().item()
        self.n += batch_valid

        # Reservoir sampling for plot data
        for i in range(batch_valid):
            if len(self.plot_predictions) < self.max_plot_samples:
                self.plot_predictions.append(valid_pred[i].item())
                self.plot_targets.append(valid_true[i].item())
            else:
                j = np.random.randint(0, self._reservoir_idx + 1)
                if j < self.max_plot_samples:
                    self.plot_predictions[j] = valid_pred[i].item()
                    self.plot_targets[j] = valid_true[i].item()
            self._reservoir_idx += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with 'mae', 'mse' keys
        """
        if self.n == 0:
            return {"mae": 0.0, "mse": 0.0}

        mae = self.sum_abs_error / self.n
        mse = self.sum_sq_error / self.n

        return {"mae": float(mae), "mse": float(mse)}

    def get_plot_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sampled data for plotting.

        Returns:
            Tuple of (predictions, targets) tensors
        """
        return (
            torch.tensor(self.plot_predictions, dtype=torch.float32),
            torch.tensor(self.plot_targets, dtype=torch.float32),
        )

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self.n = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.plot_predictions = []
        self.plot_targets = []
        self._reservoir_idx = 0

    def _merge_from(self, other: "StreamingCountMetrics") -> None:
        """Merge statistics from another StreamingCountMetrics instance.

        Simply adds error sums and counts.

        Args:
            other: Another StreamingCountMetrics instance to merge from.
        """
        if other.n == 0:
            return

        self.n += other.n
        self.sum_abs_error += other.sum_abs_error
        self.sum_sq_error += other.sum_sq_error

    def sync_distributed(self) -> None:
        """Synchronize metrics across all distributed processes.

        Uses all_reduce to sum error statistics across all ranks.
        After calling this method, all ranks will have identical metrics computed
        over the full validation set.

        This is a no-op if not running in distributed mode.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pack statistics into tensor for all_reduce
        local_stats = torch.tensor(
            [self.n, self.sum_abs_error, self.sum_sq_error],
            dtype=torch.float64,
            device=device,
        )

        # All-reduce to sum statistics across all ranks
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        # Update local statistics
        self.n = int(local_stats[0].item())
        self.sum_abs_error = local_stats[1].item()
        self.sum_sq_error = local_stats[2].item()


def compute_regression_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """Compute regression metrics (MAE, MSE, R2).

    Args:
        y_pred: Predicted values [batch_size, out_dim] or [batch_size]
        y_true: True values [batch_size, out_dim] or [batch_size]

    Returns:
        Dictionary with metrics
    """
    y_pred_np = np.array(y_pred.detach().cpu().tolist()).flatten()
    y_true_np = np.array(y_true.detach().cpu().tolist()).flatten()

    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2),
    }


def compute_classification_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute classification metrics (accuracy, precision, recall, F1).

    Args:
        y_pred: Predicted log probabilities [batch_size, num_classes] or predicted class indices [batch_size]
        y_true: True class indices [batch_size]
        num_classes: Number of classes (if None, inferred from y_pred/y_true)

    Returns:
        Dictionary with metrics
    """
    y_true_np = np.array(y_true.detach().cpu().tolist())

    if y_pred.dim() > 1 and y_pred.size(-1) > 1:
        y_pred_tensor = y_pred.detach().cpu()
        y_pred_np = np.array(y_pred_tensor.argmax(dim=-1).tolist())
    else:
        y_pred_np = np.array(y_pred.detach().cpu().tolist()).flatten()

    if num_classes is None:
        num_classes = max(int(y_pred_np.max()), int(y_true_np.max())) + 1

    accuracy = accuracy_score(y_true_np, y_pred_np)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average="weighted", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_count_metrics(
    count_pred: torch.Tensor,
    count_true: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute metrics for ASV count prediction (masked MAE, MSE).

    Args:
        count_pred: Predicted counts [batch_size, num_asvs, 1]
        count_true: True counts [batch_size, num_asvs, 1]
        mask: Mask for valid ASVs [batch_size, num_asvs] (1=valid, 0=padding)

    Returns:
        Dictionary with metrics
    """
    valid_mask = mask.unsqueeze(-1).bool()

    count_pred_np = np.array(count_pred.detach().cpu().tolist())
    count_true_np = np.array(count_true.detach().cpu().tolist())
    valid_mask_np = np.array(valid_mask.detach().cpu().tolist())

    valid_pred = count_pred_np[valid_mask_np]
    valid_true = count_true_np[valid_mask_np]

    if len(valid_pred) == 0:
        return {"mae": 0.0, "mse": 0.0}

    mae = np.mean(np.abs(valid_pred - valid_true))
    mse = np.mean((valid_pred - valid_true) ** 2)

    return {
        "mae": float(mae),
        "mse": float(mse),
    }
