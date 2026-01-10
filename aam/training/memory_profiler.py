"""Memory profiling utilities for GPU training analysis."""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager
import time


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""

    timestamp: float
    label: str
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float

    def __str__(self) -> str:
        return (
            f"{self.label}: allocated={self.allocated_mb:.1f}MB, "
            f"reserved={self.reserved_mb:.1f}MB, "
            f"peak_allocated={self.max_allocated_mb:.1f}MB"
        )


@dataclass
class EpochMemoryStats:
    """Memory statistics for a single training epoch."""

    epoch: int
    peak_allocated_mb: float
    peak_reserved_mb: float
    avg_allocated_mb: float
    batch_peaks_mb: List[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Epoch {self.epoch}: peak={self.peak_allocated_mb:.1f}MB, "
            f"avg={self.avg_allocated_mb:.1f}MB, "
            f"reserved={self.peak_reserved_mb:.1f}MB"
        )


class MemoryProfiler:
    """GPU memory profiler for tracking memory usage during training.

    Usage:
        profiler = MemoryProfiler(device=0, enabled=True)

        with profiler.profile_epoch(epoch=0):
            for batch in dataloader:
                with profiler.profile_batch():
                    # training step
                    pass

        profiler.print_summary()
    """

    def __init__(
        self,
        device: int = 0,
        enabled: bool = True,
        log_every_n_batches: int = 0,
    ):
        """Initialize memory profiler.

        Args:
            device: GPU device index to profile
            enabled: Whether profiling is enabled
            log_every_n_batches: Log memory every N batches (0 to disable per-batch logging)
        """
        self.device = device
        self.enabled = enabled and torch.cuda.is_available()
        self.log_every_n_batches = log_every_n_batches

        self.snapshots: List[MemorySnapshot] = []
        self.epoch_stats: List[EpochMemoryStats] = []
        self._current_epoch: Optional[int] = None
        self._batch_count: int = 0
        self._epoch_allocated_samples: List[float] = []
        self._epoch_peak: float = 0.0

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_memory_mb(self) -> Dict[str, float]:
        """Get current memory statistics in MB.

        Returns:
            Dict with allocated, reserved, max_allocated, max_reserved in MB
        """
        if not self.enabled:
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
                "max_reserved_mb": 0.0,
            }

        bytes_to_mb = 1024 * 1024
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / bytes_to_mb,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / bytes_to_mb,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / bytes_to_mb,
            "max_reserved_mb": torch.cuda.max_memory_reserved(self.device) / bytes_to_mb,
        }

    def snapshot(self, label: str) -> Optional[MemorySnapshot]:
        """Take a memory snapshot with the given label.

        Args:
            label: Description of this snapshot point

        Returns:
            MemorySnapshot if enabled, None otherwise
        """
        if not self.enabled:
            return None

        stats = self.get_memory_mb()
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            allocated_mb=stats["allocated_mb"],
            reserved_mb=stats["reserved_mb"],
            max_allocated_mb=stats["max_allocated_mb"],
            max_reserved_mb=stats["max_reserved_mb"],
        )
        self.snapshots.append(snapshot)
        return snapshot

    @contextmanager
    def profile_epoch(self, epoch: int):
        """Context manager to profile memory for an epoch.

        Args:
            epoch: Epoch number
        """
        if not self.enabled:
            yield
            return

        self._current_epoch = epoch
        self._batch_count = 0
        self._epoch_allocated_samples = []
        self._epoch_peak = 0.0

        self.reset_peak_stats()
        self.snapshot(f"epoch_{epoch}_start")

        try:
            yield
        finally:
            end_stats = self.get_memory_mb()
            self.snapshot(f"epoch_{epoch}_end")

            avg_allocated = (
                sum(self._epoch_allocated_samples) / len(self._epoch_allocated_samples)
                if self._epoch_allocated_samples else 0.0
            )

            epoch_stats = EpochMemoryStats(
                epoch=epoch,
                peak_allocated_mb=end_stats["max_allocated_mb"],
                peak_reserved_mb=end_stats["max_reserved_mb"],
                avg_allocated_mb=avg_allocated,
                batch_peaks_mb=list(self._epoch_allocated_samples),
            )
            self.epoch_stats.append(epoch_stats)
            self._current_epoch = None

    @contextmanager
    def profile_batch(self):
        """Context manager to profile memory for a batch."""
        if not self.enabled:
            yield
            return

        self._batch_count += 1

        try:
            yield
        finally:
            stats = self.get_memory_mb()
            self._epoch_allocated_samples.append(stats["allocated_mb"])

            if stats["allocated_mb"] > self._epoch_peak:
                self._epoch_peak = stats["allocated_mb"]

            if (
                self.log_every_n_batches > 0
                and self._batch_count % self.log_every_n_batches == 0
            ):
                self.snapshot(f"batch_{self._batch_count}")

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all profiled epochs.

        Returns:
            Dict with overall memory statistics
        """
        if not self.epoch_stats:
            return {
                "peak_allocated_mb": 0.0,
                "peak_reserved_mb": 0.0,
                "avg_peak_mb": 0.0,
            }

        peaks = [s.peak_allocated_mb for s in self.epoch_stats]
        reserved_peaks = [s.peak_reserved_mb for s in self.epoch_stats]

        return {
            "peak_allocated_mb": max(peaks),
            "peak_reserved_mb": max(reserved_peaks),
            "avg_peak_mb": sum(peaks) / len(peaks),
            "num_epochs_profiled": len(self.epoch_stats),
        }

    def print_summary(self, logger=None) -> str:
        """Print memory profiling summary.

        Args:
            logger: Optional logger to use (uses print if None)

        Returns:
            Summary string
        """
        if not self.enabled:
            return "Memory profiling disabled (no CUDA)"

        summary = self.get_summary()

        lines = [
            "=" * 60,
            "MEMORY PROFILING SUMMARY",
            "=" * 60,
            f"Peak allocated:  {summary['peak_allocated_mb']:.1f} MB",
            f"Peak reserved:   {summary['peak_reserved_mb']:.1f} MB",
            f"Avg peak/epoch:  {summary['avg_peak_mb']:.1f} MB",
            f"Epochs profiled: {summary.get('num_epochs_profiled', 0)}",
        ]

        if self.epoch_stats:
            lines.append("-" * 60)
            lines.append("Per-epoch breakdown:")
            for stat in self.epoch_stats[-5:]:  # Last 5 epochs
                lines.append(f"  {stat.summary()}")

        lines.append("=" * 60)

        output = "\n".join(lines)

        if logger:
            for line in lines:
                logger.info(line)
        else:
            print(output)

        return output

    def get_recommendations(self, batch_size: int, gpu_memory_gb: float = 128.0) -> List[str]:
        """Get memory optimization recommendations based on profiling data.

        Args:
            batch_size: Current batch size
            gpu_memory_gb: Total GPU memory in GB

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self.epoch_stats:
            return ["No profiling data available. Run training with memory profiling enabled."]

        summary = self.get_summary()
        peak_gb = summary["peak_allocated_mb"] / 1024
        utilization = peak_gb / gpu_memory_gb

        if utilization < 0.5:
            headroom = (gpu_memory_gb - peak_gb) / peak_gb
            suggested_batch = int(batch_size * (1 + headroom * 0.7))  # Conservative increase
            recommendations.append(
                f"Memory utilization is {utilization:.0%}. "
                f"Consider increasing batch_size from {batch_size} to ~{suggested_batch}."
            )
        elif utilization > 0.9:
            recommendations.append(
                f"Memory utilization is {utilization:.0%}. "
                f"Consider reducing batch_size or enabling gradient checkpointing."
            )
        else:
            recommendations.append(
                f"Memory utilization is {utilization:.0%} - good balance."
            )

        return recommendations


def log_gpu_memory_stats(device: int = 0, label: str = "", logger=None) -> Dict[str, float]:
    """Log current GPU memory statistics.

    Args:
        device: GPU device index
        label: Label for this log entry
        logger: Optional logger (uses print if None)

    Returns:
        Dict with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {}

    bytes_to_mb = 1024 * 1024
    stats = {
        "allocated_mb": torch.cuda.memory_allocated(device) / bytes_to_mb,
        "reserved_mb": torch.cuda.memory_reserved(device) / bytes_to_mb,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / bytes_to_mb,
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / bytes_to_mb,
    }

    msg = (
        f"[{label}] GPU Memory: "
        f"allocated={stats['allocated_mb']:.1f}MB, "
        f"reserved={stats['reserved_mb']:.1f}MB, "
        f"peak={stats['max_allocated_mb']:.1f}MB"
    )

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return stats
