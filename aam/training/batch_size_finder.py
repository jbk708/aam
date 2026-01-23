"""Batch size finder utility to identify optimal batch size for available GPU memory.

Implements binary search to find the maximum batch size that fits in GPU memory
while maintaining a safety margin for training stability.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeFinderResult:
    """Result from batch size finder."""

    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    peak_memory_mb: float
    memory_fraction: float


class BatchSizeFinder:
    """Find optimal batch size that fits in GPU memory.

    Uses binary search to find the maximum batch size that fits in GPU memory
    while maintaining a configurable safety margin. Can also auto-tune gradient
    accumulation steps to maintain a target effective batch size.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        collate_fn: Callable,
    ):
        """Initialize batch size finder.

        Args:
            model: Model to test batch sizes with
            loss_fn: Loss function for forward/backward pass
            device: Device to run on (must be CUDA for memory profiling)
            collate_fn: Collate function for DataLoader
        """
        if device.type != "cuda":
            raise ValueError("BatchSizeFinder requires CUDA device for memory profiling. Use --no-auto-batch-size on CPU.")

        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.collate_fn = collate_fn

    def find_batch_size(
        self,
        dataset: Dataset,
        min_batch_size: int = 2,
        max_batch_size: int = 256,
        target_effective_batch_size: Optional[int] = None,
        max_memory_fraction: float = 0.8,
        num_iterations: int = 2,
    ) -> BatchSizeFinderResult:
        """Find optimal batch size using binary search.

        Args:
            dataset: Dataset to sample batches from
            min_batch_size: Minimum batch size to try (must be >= 2 and even)
            max_batch_size: Maximum batch size to try
            target_effective_batch_size: Target effective batch size for gradient
                accumulation tuning. If None, gradient_accumulation_steps=1.
            max_memory_fraction: Maximum fraction of GPU memory to use (0.0-1.0)
            num_iterations: Number of forward/backward passes to test stability

        Returns:
            BatchSizeFinderResult with optimal batch_size, gradient_accumulation_steps,
            effective_batch_size, peak_memory_mb, and memory_fraction

        Raises:
            RuntimeError: If even min_batch_size causes OOM
            ValueError: If parameters are invalid
        """
        if min_batch_size < 2:
            raise ValueError("min_batch_size must be at least 2 (UniFrac requires pairwise distances)")

        if max_memory_fraction <= 0 or max_memory_fraction > 1.0:
            raise ValueError(f"max_memory_fraction must be in (0.0, 1.0], got {max_memory_fraction}")

        min_batch_size = self._round_to_even(min_batch_size)
        max_batch_size = self._round_to_even(max_batch_size)

        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        max_memory_bytes = total_memory * max_memory_fraction

        logger.info(
            f"Finding optimal batch size (min={min_batch_size}, max={max_batch_size}, "
            f"memory_fraction={max_memory_fraction:.0%})"
        )

        success, peak_memory = self._try_batch_size(min_batch_size, dataset, num_iterations)
        if not success:
            raise RuntimeError(
                f"Even minimum batch size ({min_batch_size}) causes OOM. "
                f"Try reducing --token-limit, enabling --mixed-precision, "
                f"or using --no-auto-batch-size with a smaller --batch-size."
            )

        best_batch_size = min_batch_size
        best_memory = peak_memory

        low = min_batch_size
        high = max_batch_size

        while low <= high:
            mid = self._round_to_even((low + high) // 2)

            if mid == best_batch_size:
                if mid + 2 <= high:
                    mid = mid + 2
                else:
                    break

            logger.debug(f"Trying batch size {mid}...")

            success, peak_memory = self._try_batch_size(mid, dataset, num_iterations)

            if success and peak_memory * 1024 * 1024 <= max_memory_bytes:
                logger.debug(f"Batch size {mid} succeeded (peak memory: {peak_memory:.1f} MB)")
                best_batch_size = mid
                best_memory = peak_memory
                low = mid + 2
            else:
                logger.debug(f"Batch size {mid} failed or exceeded memory limit")
                high = mid - 2

        grad_accum_steps = 1
        effective_batch_size = best_batch_size

        if target_effective_batch_size is not None and target_effective_batch_size > best_batch_size:
            grad_accum_steps = max(1, target_effective_batch_size // best_batch_size)
            effective_batch_size = best_batch_size * grad_accum_steps

        memory_fraction_used = (best_memory * 1024 * 1024) / total_memory

        result = BatchSizeFinderResult(
            batch_size=best_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            effective_batch_size=effective_batch_size,
            peak_memory_mb=best_memory,
            memory_fraction=memory_fraction_used,
        )

        logger.info(
            f"Optimal batch size: {result.batch_size} "
            f"(effective: {result.effective_batch_size} with {result.gradient_accumulation_steps}x grad accum, "
            f"peak memory: {result.peak_memory_mb:.1f} MB = {result.memory_fraction:.1%} of GPU)"
        )

        return result

    def _try_batch_size(
        self,
        batch_size: int,
        dataset: Dataset,
        num_iterations: int,
    ) -> Tuple[bool, float]:
        """Test if a batch size fits in GPU memory.

        Args:
            batch_size: Batch size to test
            dataset: Dataset to sample from
            num_iterations: Number of forward/backward passes to test

        Returns:
            Tuple of (success, peak_memory_mb)
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        subset_size = min(batch_size * num_iterations, len(dataset))  # type: ignore[arg-type]
        subset = Subset(dataset, list(range(subset_size)))

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        self.model.train()

        try:
            for batch in loader:
                if isinstance(batch, dict):
                    tokens = batch["tokens"].to(self.device)
                    targets = {"tokens": tokens}
                    if "counts" in batch:
                        targets["counts"] = batch["counts"].to(self.device)
                    if "unifrac_target" in batch:
                        targets["base_target"] = batch["unifrac_target"].to(self.device)
                else:
                    tokens = batch[0].to(self.device)
                    targets = {"tokens": tokens}

                outputs = self.model(tokens)

                encoder_type = "unifrac"
                if hasattr(self.model, "encoder_type"):
                    encoder_type = self.model.encoder_type
                elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "encoder_type"):
                    encoder_type = self.model.base_model.encoder_type

                loss_dict = self.loss_fn(outputs, targets, encoder_type=encoder_type)
                loss = loss_dict["total_loss"]

                loss.backward()

                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)

            return True, peak_memory_mb

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False, 0.0

        finally:
            torch.cuda.empty_cache()
            self.model.eval()

    def _round_to_even(self, batch_size: int) -> int:
        """Round batch size down to nearest even number (UniFrac requirement)."""
        if batch_size < 2:
            return 2
        return batch_size if batch_size % 2 == 0 else batch_size - 1
