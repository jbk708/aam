"""Metrics computation for AAM model evaluation."""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


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
    pass


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
    pass


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
    pass
