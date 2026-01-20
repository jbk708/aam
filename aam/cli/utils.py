"""Utility functions for AAM CLI."""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional


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
