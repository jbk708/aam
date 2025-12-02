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
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="weighted", zero_division=0
    )
    
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
