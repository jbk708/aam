"""Validation and evaluation logic for AAM model training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast
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
    """Create prediction vs actual scatter plot for regression tasks."""
    raise NotImplementedError


def create_unifrac_prediction_plot(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    r2: float,
    mae: Optional[float] = None,
) -> plt.Figure:
    """Create prediction vs actual scatter plot for UniFrac predictions."""
    raise NotImplementedError


def create_confusion_matrix_plot(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> plt.Figure:
    """Create confusion matrix plot for classification tasks."""
    raise NotImplementedError


class Evaluator:
    """Handles validation and evaluation for AAM models."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        mixed_precision: Optional[str] = None,
        target_normalization_params: Optional[Dict[str, float]] = None,
        count_normalization_params: Optional[Dict[str, float]] = None,
    ):
        """Initialize Evaluator."""
        raise NotImplementedError

    def validate_epoch(
        self,
        dataloader: DataLoader,
        compute_metrics: bool = True,
        epoch: int = 0,
        num_epochs: int = 1,
        return_predictions: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Run one validation epoch with streaming metrics computation."""
        raise NotImplementedError

    def save_prediction_plots(
        self,
        predictions_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: Optional[str],
    ) -> None:
        """Save prediction plots when validation improves."""
        raise NotImplementedError

    def log_figures_to_tensorboard(
        self,
        epoch: int,
        predictions_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        writer: SummaryWriter,
    ) -> None:
        """Log prediction figures to TensorBoard."""
        raise NotImplementedError
