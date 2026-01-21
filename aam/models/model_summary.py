"""Model summary utilities for logging layer structure and parameter counts."""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import logging


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a module.

    Args:
        module: PyTorch module
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def get_model_summary(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
) -> str:
    """Generate a summary string of model architecture.

    Args:
        model: PyTorch model to summarize
        input_shape: Optional input shape for shape inference (without batch dim)
        batch_size: Batch size for shape inference

    Returns:
        Formatted summary string
    """
    lines = []
    separator = "=" * 80
    lines.append(separator)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append(separator)
    lines.append(f"{'Layer (type)':<45} {'Params':>15} {'Trainable':>10}")
    lines.append("-" * 80)

    total_params = 0
    trainable_params = 0

    # Get immediate children modules (top-level components)
    for name, module in model.named_children():
        params = count_parameters(module)
        trainable = count_parameters(module, trainable_only=True)
        total_params += params
        trainable_params += trainable

        module_type = module.__class__.__name__
        layer_name = f"{name} ({module_type})"

        trainable_str = "Yes" if trainable > 0 else "No"
        if trainable > 0 and trainable < params:
            trainable_str = f"Partial ({trainable:,})"

        lines.append(f"{layer_name:<45} {format_number(params):>15} {trainable_str:>10}")

        # Show sub-components for key modules
        if hasattr(module, "named_children"):
            for sub_name, sub_module in module.named_children():
                sub_params = count_parameters(sub_module)
                sub_trainable = count_parameters(sub_module, trainable_only=True)
                sub_type = sub_module.__class__.__name__
                sub_layer_name = f"  └─ {sub_name} ({sub_type})"

                sub_trainable_str = "Yes" if sub_trainable > 0 else "No"
                if sub_trainable > 0 and sub_trainable < sub_params:
                    sub_trainable_str = "Partial"

                lines.append(f"{sub_layer_name:<45} {format_number(sub_params):>15} {sub_trainable_str:>10}")

    lines.append(separator)
    lines.append(f"{'Total params:':<45} {format_number(total_params):>15}")
    lines.append(f"{'Trainable params:':<45} {format_number(trainable_params):>15}")
    lines.append(f"{'Non-trainable params:':<45} {format_number(total_params - trainable_params):>15}")
    lines.append(separator)

    return "\n".join(lines)


def log_model_summary(
    model: nn.Module,
    logger: Optional[logging.Logger] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> None:
    """Log model summary to logger.

    Args:
        model: PyTorch model to summarize
        logger: Logger to use (defaults to module logger)
        input_shape: Optional input shape for shape inference
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    summary = get_model_summary(model, input_shape)

    # Log each line separately for proper formatting
    for line in summary.split("\n"):
        logger.info(line)


def print_model_summary(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> None:
    """Print model summary to stdout.

    Args:
        model: PyTorch model to summarize
        input_shape: Optional input shape for shape inference
    """
    print(get_model_summary(model, input_shape))
