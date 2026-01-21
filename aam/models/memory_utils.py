"""Memory optimization utilities for AAM models."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def chunk_tensor(tensor: torch.Tensor, chunk_size: int, dim: int = 0) -> Tuple[torch.Tensor, ...]:
    """Split tensor into chunks along specified dimension.

    Args:
        tensor: Input tensor
        chunk_size: Size of each chunk
        dim: Dimension to chunk along

    Returns:
        List of tensor chunks
    """
    return torch.chunk(tensor, chunks=(tensor.size(dim) + chunk_size - 1) // chunk_size, dim=dim)


def process_in_chunks(model: nn.Module, input_tensor: torch.Tensor, chunk_size: int, dim: int = 0, **kwargs) -> torch.Tensor:
    """Process input tensor in chunks to reduce memory usage.

    Args:
        model: Model to apply
        input_tensor: Input tensor to process
        chunk_size: Size of each chunk
        dim: Dimension to chunk along
        **kwargs: Additional arguments to pass to model

    Returns:
        Concatenated output tensor
    """
    chunks = chunk_tensor(input_tensor, chunk_size, dim=dim)
    outputs = []

    for chunk in chunks:
        output = model(chunk, **kwargs)
        outputs.append(output)

    return torch.cat(outputs, dim=dim)
