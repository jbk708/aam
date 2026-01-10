"""Tests for memory profiler utilities."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from aam.training.memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    EpochMemoryStats,
    log_gpu_memory_stats,
)


class TestMemorySnapshot:
    def test_snapshot_creation(self):
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            label="test_snapshot",
            allocated_mb=100.0,
            reserved_mb=200.0,
            max_allocated_mb=150.0,
            max_reserved_mb=250.0,
        )
        assert snapshot.label == "test_snapshot"
        assert snapshot.allocated_mb == 100.0
        assert "allocated=100.0MB" in str(snapshot)

    def test_snapshot_str(self):
        snapshot = MemorySnapshot(
            timestamp=0.0,
            label="test",
            allocated_mb=512.5,
            reserved_mb=1024.0,
            max_allocated_mb=600.0,
            max_reserved_mb=1200.0,
        )
        s = str(snapshot)
        assert "test:" in s
        assert "512.5MB" in s
        assert "peak_allocated=600.0MB" in s


class TestEpochMemoryStats:
    def test_epoch_stats_creation(self):
        stats = EpochMemoryStats(
            epoch=5,
            peak_allocated_mb=2048.0,
            peak_reserved_mb=4096.0,
            avg_allocated_mb=1500.0,
            batch_peaks_mb=[1400.0, 1500.0, 1600.0],
        )
        assert stats.epoch == 5
        assert stats.peak_allocated_mb == 2048.0
        assert len(stats.batch_peaks_mb) == 3

    def test_epoch_stats_summary(self):
        stats = EpochMemoryStats(
            epoch=3,
            peak_allocated_mb=1024.0,
            peak_reserved_mb=2048.0,
            avg_allocated_mb=800.0,
        )
        summary = stats.summary()
        assert "Epoch 3" in summary
        assert "peak=1024.0MB" in summary
        assert "avg=800.0MB" in summary


class TestMemoryProfiler:
    def test_profiler_disabled_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            profiler = MemoryProfiler(enabled=True)
            assert not profiler.enabled

    def test_profiler_disabled_explicitly(self):
        profiler = MemoryProfiler(enabled=False)
        assert not profiler.enabled

    def test_get_memory_mb_disabled(self):
        profiler = MemoryProfiler(enabled=False)
        stats = profiler.get_memory_mb()
        assert stats["allocated_mb"] == 0.0
        assert stats["reserved_mb"] == 0.0

    def test_snapshot_disabled(self):
        profiler = MemoryProfiler(enabled=False)
        snapshot = profiler.snapshot("test")
        assert snapshot is None

    def test_profile_epoch_disabled(self):
        profiler = MemoryProfiler(enabled=False)
        with profiler.profile_epoch(epoch=0):
            pass
        assert len(profiler.epoch_stats) == 0

    def test_profile_batch_disabled(self):
        profiler = MemoryProfiler(enabled=False)
        with profiler.profile_batch():
            pass
        assert profiler._batch_count == 0

    def test_get_summary_no_data(self):
        profiler = MemoryProfiler(enabled=False)
        summary = profiler.get_summary()
        assert summary["peak_allocated_mb"] == 0.0
        assert summary["avg_peak_mb"] == 0.0

    def test_print_summary_disabled(self):
        profiler = MemoryProfiler(enabled=False)
        result = profiler.print_summary()
        assert "disabled" in result.lower()

    def test_recommendations_no_data(self):
        profiler = MemoryProfiler(enabled=False)
        recs = profiler.get_recommendations(batch_size=8)
        assert len(recs) == 1
        assert "No profiling data" in recs[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profiler_enabled_with_cuda(self):
        profiler = MemoryProfiler(enabled=True)
        assert profiler.enabled

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_mb_with_cuda(self):
        profiler = MemoryProfiler(enabled=True)
        stats = profiler.get_memory_mb()
        assert "allocated_mb" in stats
        assert "reserved_mb" in stats
        assert "max_allocated_mb" in stats

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_snapshot_with_cuda(self):
        profiler = MemoryProfiler(enabled=True)
        snapshot = profiler.snapshot("test_label")
        assert snapshot is not None
        assert snapshot.label == "test_label"
        assert len(profiler.snapshots) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_epoch_with_cuda(self):
        profiler = MemoryProfiler(enabled=True)

        with profiler.profile_epoch(epoch=0):
            # Simulate some GPU work
            x = torch.randn(100, 100, device="cuda")
            y = x @ x.T
            del x, y

        assert len(profiler.epoch_stats) == 1
        assert profiler.epoch_stats[0].epoch == 0
        assert profiler.epoch_stats[0].peak_allocated_mb >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_batch_with_cuda(self):
        profiler = MemoryProfiler(enabled=True)

        with profiler.profile_epoch(epoch=0):
            for _ in range(3):
                with profiler.profile_batch():
                    x = torch.randn(50, 50, device="cuda")
                    del x

        assert len(profiler.epoch_stats) == 1
        stats = profiler.epoch_stats[0]
        assert len(stats.batch_peaks_mb) == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_recommendations_with_cuda(self):
        profiler = MemoryProfiler(enabled=True)

        with profiler.profile_epoch(epoch=0):
            x = torch.randn(100, 100, device="cuda")
            del x

        recs = profiler.get_recommendations(batch_size=8, gpu_memory_gb=24.0)
        assert len(recs) >= 1


class TestLogGpuMemoryStats:
    def test_log_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            stats = log_gpu_memory_stats(label="test")
            assert stats == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_log_with_cuda(self):
        stats = log_gpu_memory_stats(label="test")
        assert "allocated_mb" in stats
        assert "reserved_mb" in stats

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_log_with_logger(self):
        mock_logger = MagicMock()
        stats = log_gpu_memory_stats(label="test", logger=mock_logger)
        mock_logger.info.assert_called_once()
        assert "GPU Memory" in mock_logger.info.call_args[0][0]
