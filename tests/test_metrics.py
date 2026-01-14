"""Unit tests for metrics computation."""

import pytest
import torch
import numpy as np

from aam.training.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_count_metrics,
    StreamingRegressionMetrics,
    StreamingClassificationMetrics,
    StreamingCountMetrics,
)


class TestRegressionMetrics:
    """Test regression metrics computation."""

    def test_regression_metrics_1d(self):
        """Test regression metrics for 1D output."""
        batch_size = 10
        y_pred = torch.randn(batch_size)
        y_true = torch.randn(batch_size)

        metrics = compute_regression_metrics(y_pred, y_true)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "r2" in metrics

        y_pred_np = np.array(y_pred.detach().cpu().tolist())
        y_true_np = np.array(y_true.detach().cpu().tolist())

        expected_mae = np.mean(np.abs(y_pred_np - y_true_np))
        expected_mse = np.mean((y_pred_np - y_true_np) ** 2)
        expected_r2 = 1 - np.sum((y_true_np - y_pred_np) ** 2) / np.sum((y_true_np - np.mean(y_true_np)) ** 2)

        assert abs(metrics["mae"] - expected_mae) < 1e-5
        assert abs(metrics["mse"] - expected_mse) < 1e-5
        assert abs(metrics["r2"] - expected_r2) < 1e-5

    def test_regression_metrics_2d(self):
        """Test regression metrics for 2D output."""
        batch_size = 10
        out_dim = 3
        y_pred = torch.randn(batch_size, out_dim)
        y_true = torch.randn(batch_size, out_dim)

        metrics = compute_regression_metrics(y_pred, y_true)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "r2" in metrics

        y_pred_np = np.array(y_pred.detach().cpu().tolist()).flatten()
        y_true_np = np.array(y_true.detach().cpu().tolist()).flatten()

        expected_mae = np.mean(np.abs(y_pred_np - y_true_np))
        expected_mse = np.mean((y_pred_np - y_true_np) ** 2)
        expected_r2 = 1 - np.sum((y_true_np - y_pred_np) ** 2) / np.sum((y_true_np - np.mean(y_true_np)) ** 2)

        assert abs(metrics["mae"] - expected_mae) < 1e-5
        assert abs(metrics["mse"] - expected_mse) < 1e-5
        assert abs(metrics["r2"] - expected_r2) < 1e-5

    def test_regression_metrics_perfect_prediction(self):
        """Test regression metrics for perfect prediction."""
        batch_size = 10
        y_pred = torch.randn(batch_size)
        y_true = y_pred.clone()

        metrics = compute_regression_metrics(y_pred, y_true)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert abs(metrics["r2"] - 1.0) < 1e-5


class TestClassificationMetrics:
    """Test classification metrics computation."""

    def test_classification_metrics_from_log_probs(self):
        """Test classification metrics from log probabilities."""
        batch_size = 10
        num_classes = 3

        log_probs = torch.randn(batch_size, num_classes)
        log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
        y_pred = log_probs.argmax(dim=-1)
        y_true = torch.randint(0, num_classes, (batch_size,))

        metrics = compute_classification_metrics(log_probs, y_true, num_classes=num_classes)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        y_pred_np = np.array(y_pred.detach().cpu().tolist())
        y_true_np = np.array(y_true.detach().cpu().tolist())

        expected_accuracy = np.mean(y_pred_np == y_true_np)
        assert abs(metrics["accuracy"] - expected_accuracy) < 1e-5

    def test_classification_metrics_from_indices(self):
        """Test classification metrics from predicted class indices."""
        batch_size = 10
        num_classes = 3

        y_pred = torch.randint(0, num_classes, (batch_size,))
        y_true = torch.randint(0, num_classes, (batch_size,))

        metrics = compute_classification_metrics(y_pred, y_true, num_classes=num_classes)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        y_pred_np = np.array(y_pred.detach().cpu().tolist())
        y_true_np = np.array(y_true.detach().cpu().tolist())

        expected_accuracy = np.mean(y_pred_np == y_true_np)
        assert abs(metrics["accuracy"] - expected_accuracy) < 1e-5

    def test_classification_metrics_perfect_prediction(self):
        """Test classification metrics for perfect prediction."""
        batch_size = 10
        num_classes = 3

        y_true = torch.randint(0, num_classes, (batch_size,))
        y_pred = y_true.clone()

        metrics = compute_classification_metrics(y_pred, y_true, num_classes=num_classes)

        assert metrics["accuracy"] == 1.0

    def test_classification_metrics_infer_num_classes(self):
        """Test classification metrics with inferred number of classes."""
        batch_size = 10

        y_pred = torch.randint(0, 5, (batch_size,))
        y_true = torch.randint(0, 5, (batch_size,))

        metrics = compute_classification_metrics(y_pred, y_true, num_classes=None)

        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0.0
        assert metrics["accuracy"] <= 1.0


class TestCountMetrics:
    """Test count metrics computation."""

    def test_count_metrics_masked(self):
        """Test masked metrics for counts."""
        batch_size = 4
        num_asvs = 10

        count_pred = torch.randn(batch_size, num_asvs, 1)
        count_true = torch.randn(batch_size, num_asvs, 1)
        mask = torch.ones(batch_size, num_asvs)
        mask[:, 5:] = 0

        metrics = compute_count_metrics(count_pred, count_true, mask)

        assert "mae" in metrics
        assert "mse" in metrics

        valid_mask = mask.unsqueeze(-1)
        valid_pred = (count_pred * valid_mask).detach().cpu()
        valid_true = (count_true * valid_mask).detach().cpu()

        valid_mask_bool = valid_mask.bool().detach().cpu()
        valid_pred_np = np.array(valid_pred.tolist())
        valid_true_np = np.array(valid_true.tolist())
        valid_mask_np = np.array(valid_mask_bool.tolist())

        valid_pred_flat = valid_pred_np[valid_mask_np]
        valid_true_flat = valid_true_np[valid_mask_np]

        expected_mae = np.mean(np.abs(valid_pred_flat - valid_true_flat))
        expected_mse = np.mean((valid_pred_flat - valid_true_flat) ** 2)

        assert abs(metrics["mae"] - expected_mae) < 1e-5
        assert abs(metrics["mse"] - expected_mse) < 1e-5

    def test_count_metrics_all_valid(self):
        """Test count metrics when all ASVs are valid."""
        batch_size = 4
        num_asvs = 10

        count_pred = torch.randn(batch_size, num_asvs, 1)
        count_true = torch.randn(batch_size, num_asvs, 1)
        mask = torch.ones(batch_size, num_asvs)

        metrics = compute_count_metrics(count_pred, count_true, mask)

        assert "mae" in metrics
        assert "mse" in metrics

        count_pred_np = np.array(count_pred.detach().cpu().tolist()).flatten()
        count_true_np = np.array(count_true.detach().cpu().tolist()).flatten()

        expected_mae = np.mean(np.abs(count_pred_np - count_true_np))
        expected_mse = np.mean((count_pred_np - count_true_np) ** 2)

        assert abs(metrics["mae"] - expected_mae) < 1e-5
        assert abs(metrics["mse"] - expected_mse) < 1e-5

    def test_count_metrics_perfect_prediction(self):
        """Test count metrics for perfect prediction."""
        batch_size = 4
        num_asvs = 10

        count_pred = torch.randn(batch_size, num_asvs, 1)
        count_true = count_pred.clone()
        mask = torch.ones(batch_size, num_asvs)

        metrics = compute_count_metrics(count_pred, count_true, mask)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0


class TestDeviceHandling:
    """Test device handling for metrics."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_metrics_on_cuda(self):
        """Test that metrics work on CUDA tensors."""
        device = torch.device("cuda")

        batch_size = 10
        y_pred = torch.randn(batch_size, device=device)
        y_true = torch.randn(batch_size, device=device)

        metrics = compute_regression_metrics(y_pred, y_true)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "r2" in metrics


class TestStreamingRegressionMetrics:
    """Test streaming regression metrics computation."""

    def test_streaming_matches_batch(self):
        """Test that streaming metrics match batch computation."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate data
        n_samples = 100
        y_pred = torch.randn(n_samples)
        y_true = torch.randn(n_samples)

        # Batch computation
        batch_metrics = compute_regression_metrics(y_pred, y_true)

        # Streaming computation (process in chunks)
        streaming = StreamingRegressionMetrics(max_plot_samples=1000)
        batch_size = 10
        for i in range(0, n_samples, batch_size):
            streaming.update(y_pred[i : i + batch_size], y_true[i : i + batch_size])

        streaming_metrics = streaming.compute()

        # Should match within tolerance
        assert abs(streaming_metrics["mae"] - batch_metrics["mae"]) < 1e-4
        assert abs(streaming_metrics["mse"] - batch_metrics["mse"]) < 1e-4
        assert abs(streaming_metrics["r2"] - batch_metrics["r2"]) < 1e-4

    def test_streaming_single_update(self):
        """Test streaming metrics with a single batch update."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.1, 2.2, 2.8])

        streaming = StreamingRegressionMetrics()
        streaming.update(y_pred, y_true)
        metrics = streaming.compute()

        batch_metrics = compute_regression_metrics(y_pred, y_true)

        assert abs(metrics["mae"] - batch_metrics["mae"]) < 1e-5
        assert abs(metrics["mse"] - batch_metrics["mse"]) < 1e-5
        assert abs(metrics["r2"] - batch_metrics["r2"]) < 1e-5

    def test_streaming_empty(self):
        """Test streaming metrics with no data."""
        streaming = StreamingRegressionMetrics()
        metrics = streaming.compute()

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["r2"] == 0.0

    def test_streaming_perfect_prediction(self):
        """Test streaming metrics with perfect predictions."""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_true = y_pred.clone()

        streaming = StreamingRegressionMetrics()
        streaming.update(y_pred, y_true)
        metrics = streaming.compute()

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_streaming_plot_data(self):
        """Test that plot data is retained via reservoir sampling."""
        streaming = StreamingRegressionMetrics(max_plot_samples=50)

        # Add more samples than max_plot_samples
        for _ in range(10):
            streaming.update(torch.randn(20), torch.randn(20))

        pred_samples, targ_samples = streaming.get_plot_data()

        # Should have exactly max_plot_samples
        assert len(pred_samples) == 50
        assert len(targ_samples) == 50

    def test_streaming_reset(self):
        """Test that reset clears all state."""
        streaming = StreamingRegressionMetrics()
        streaming.update(torch.randn(10), torch.randn(10))
        streaming.reset()

        assert streaming.n == 0
        assert streaming.sum_abs_error == 0.0
        assert len(streaming.plot_predictions) == 0

    def test_streaming_2d_input(self):
        """Test streaming with 2D input (gets flattened)."""
        y_pred = torch.randn(5, 3)
        y_true = torch.randn(5, 3)

        streaming = StreamingRegressionMetrics()
        streaming.update(y_pred, y_true)
        metrics = streaming.compute()

        batch_metrics = compute_regression_metrics(y_pred, y_true)

        assert abs(metrics["mae"] - batch_metrics["mae"]) < 1e-5
        assert abs(metrics["mse"] - batch_metrics["mse"]) < 1e-5
        assert abs(metrics["r2"] - batch_metrics["r2"]) < 1e-5


class TestStreamingClassificationMetrics:
    """Test streaming classification metrics computation."""

    def test_streaming_matches_batch(self):
        """Test that streaming metrics match batch computation."""
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = 100
        num_classes = 5
        y_pred = torch.randint(0, num_classes, (n_samples,))
        y_true = torch.randint(0, num_classes, (n_samples,))

        batch_metrics = compute_classification_metrics(y_pred, y_true)

        streaming = StreamingClassificationMetrics(num_classes=num_classes)
        batch_size = 10
        for i in range(0, n_samples, batch_size):
            streaming.update(y_pred[i : i + batch_size], y_true[i : i + batch_size])

        streaming_metrics = streaming.compute()

        assert abs(streaming_metrics["accuracy"] - batch_metrics["accuracy"]) < 1e-5
        assert abs(streaming_metrics["precision"] - batch_metrics["precision"]) < 1e-4
        assert abs(streaming_metrics["recall"] - batch_metrics["recall"]) < 1e-4
        assert abs(streaming_metrics["f1"] - batch_metrics["f1"]) < 1e-4

    def test_streaming_from_logits(self):
        """Test streaming classification with logits input."""
        n_samples = 50
        num_classes = 3

        logits = torch.randn(n_samples, num_classes)
        y_true = torch.randint(0, num_classes, (n_samples,))

        streaming = StreamingClassificationMetrics()
        streaming.update(logits, y_true)
        metrics = streaming.compute()

        batch_metrics = compute_classification_metrics(logits, y_true)

        assert abs(metrics["accuracy"] - batch_metrics["accuracy"]) < 1e-5

    def test_streaming_empty(self):
        """Test streaming classification metrics with no data."""
        streaming = StreamingClassificationMetrics()
        metrics = streaming.compute()

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_streaming_perfect_prediction(self):
        """Test streaming classification with perfect predictions."""
        y_true = torch.tensor([0, 1, 2, 0, 1, 2])
        y_pred = y_true.clone()

        streaming = StreamingClassificationMetrics()
        streaming.update(y_pred, y_true)
        metrics = streaming.compute()

        assert metrics["accuracy"] == 1.0

    def test_streaming_infer_num_classes(self):
        """Test that num_classes is inferred from data."""
        streaming = StreamingClassificationMetrics()

        # First batch with classes 0-2
        streaming.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
        # Second batch introduces class 3
        streaming.update(torch.tensor([3, 3]), torch.tensor([0, 3]))

        assert streaming.num_classes == 4


class TestStreamingCountMetrics:
    """Test streaming count metrics computation."""

    def test_streaming_matches_batch(self):
        """Test that streaming metrics match batch computation."""
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size = 4
        num_asvs = 20
        count_pred = torch.randn(batch_size, num_asvs, 1)
        count_true = torch.randn(batch_size, num_asvs, 1)
        mask = torch.ones(batch_size, num_asvs)
        mask[:, 15:] = 0  # Mask out last 5 ASVs

        batch_metrics = compute_count_metrics(count_pred, count_true, mask)

        streaming = StreamingCountMetrics()
        # Process in smaller chunks
        for i in range(batch_size):
            streaming.update(count_pred[i : i + 1], count_true[i : i + 1], mask[i : i + 1])

        streaming_metrics = streaming.compute()

        assert abs(streaming_metrics["mae"] - batch_metrics["mae"]) < 1e-5
        assert abs(streaming_metrics["mse"] - batch_metrics["mse"]) < 1e-5

    def test_streaming_empty(self):
        """Test streaming count metrics with no data."""
        streaming = StreamingCountMetrics()
        metrics = streaming.compute()

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0

    def test_streaming_all_masked(self):
        """Test streaming with all values masked."""
        streaming = StreamingCountMetrics()
        streaming.update(
            torch.randn(4, 10, 1),
            torch.randn(4, 10, 1),
            torch.zeros(4, 10),  # All masked
        )
        metrics = streaming.compute()

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0

    def test_streaming_perfect_prediction(self):
        """Test streaming count metrics with perfect predictions."""
        count_pred = torch.randn(4, 10, 1)
        count_true = count_pred.clone()
        mask = torch.ones(4, 10)

        streaming = StreamingCountMetrics()
        streaming.update(count_pred, count_true, mask)
        metrics = streaming.compute()

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0

    def test_streaming_plot_data(self):
        """Test that plot data is retained for counts."""
        streaming = StreamingCountMetrics(max_plot_samples=30)

        # Add samples (4 batches * 10 valid per batch = 40 valid samples)
        for _ in range(4):
            streaming.update(torch.randn(2, 10, 1), torch.randn(2, 10, 1), torch.ones(2, 10))

        pred_samples, targ_samples = streaming.get_plot_data()

        # Should have max_plot_samples
        assert len(pred_samples) == 30
        assert len(targ_samples) == 30


class TestDistributedMetricSync:
    """Test distributed synchronization of streaming metrics."""

    def test_regression_sync_noop_when_not_distributed(self):
        """Test that sync_distributed is no-op when not in distributed mode."""
        streaming = StreamingRegressionMetrics()
        streaming.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.1, 2.0, 2.9]))

        # Should not raise and should preserve state
        original_n = streaming.n
        original_mae_sum = streaming.sum_abs_error
        streaming.sync_distributed()

        assert streaming.n == original_n
        assert streaming.sum_abs_error == original_mae_sum

    def test_classification_sync_noop_when_not_distributed(self):
        """Test that sync_distributed is no-op when not in distributed mode."""
        streaming = StreamingClassificationMetrics()
        streaming.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))

        original_n = streaming.n
        streaming.sync_distributed()

        assert streaming.n == original_n

    def test_count_sync_noop_when_not_distributed(self):
        """Test that sync_distributed is no-op when not in distributed mode."""
        streaming = StreamingCountMetrics()
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        true = torch.tensor([[1.1, 2.1], [3.1, 4.1]])
        mask = torch.ones(2, 2)
        streaming.update(pred, true, mask)

        original_n = streaming.n
        streaming.sync_distributed()

        assert streaming.n == original_n

    def test_regression_sync_combines_stats_correctly(self):
        """Test that sync correctly combines statistics from multiple ranks.

        Simulates what happens when each rank has different data and we need
        to merge the statistics to get metrics over the full dataset.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        # Simulate 4 ranks, each with 25 samples
        all_preds = torch.randn(100)
        all_targets = torch.randn(100)

        # Compute expected metrics from full dataset
        expected_metrics = compute_regression_metrics(all_preds, all_targets)

        # Now simulate distributed: each "rank" processes 25 samples
        rank_metrics = []
        for rank in range(4):
            start = rank * 25
            end = start + 25
            streaming = StreamingRegressionMetrics()
            streaming.update(all_preds[start:end], all_targets[start:end])
            rank_metrics.append(streaming)

        # Merge all rank statistics into rank 0's metrics object
        merged = rank_metrics[0]
        for other in rank_metrics[1:]:
            merged._merge_from(other)

        merged_metrics = merged.compute()

        # Should match full dataset metrics
        assert abs(merged_metrics["mae"] - expected_metrics["mae"]) < 1e-4
        assert abs(merged_metrics["mse"] - expected_metrics["mse"]) < 1e-4
        assert abs(merged_metrics["r2"] - expected_metrics["r2"]) < 1e-4

    def test_classification_sync_combines_confusion_matrices(self):
        """Test that sync correctly combines confusion matrices from multiple ranks."""
        # Simulate 2 ranks with different predictions
        targets_rank0 = torch.tensor([0, 0, 1, 1])
        preds_rank0 = torch.tensor([0, 1, 1, 1])  # 1 FP for class 1

        targets_rank1 = torch.tensor([0, 0, 1, 1])
        preds_rank1 = torch.tensor([0, 0, 0, 1])  # 1 FN for class 1

        # Full dataset
        all_targets = torch.cat([targets_rank0, targets_rank1])
        all_preds = torch.cat([preds_rank0, preds_rank1])

        expected_metrics = compute_classification_metrics(all_preds, all_targets)

        # Per-rank streaming
        streaming0 = StreamingClassificationMetrics()
        streaming0.update(preds_rank0, targets_rank0)

        streaming1 = StreamingClassificationMetrics()
        streaming1.update(preds_rank1, targets_rank1)

        # Merge
        streaming0._merge_from(streaming1)
        merged_metrics = streaming0.compute()

        assert abs(merged_metrics["accuracy"] - expected_metrics["accuracy"]) < 1e-4
        assert abs(merged_metrics["f1"] - expected_metrics["f1"]) < 1e-4

    def test_count_sync_combines_stats_correctly(self):
        """Test that sync correctly combines count statistics from multiple ranks."""
        # Rank 0 data
        pred0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        true0 = torch.tensor([[1.1, 2.2], [2.9, 4.1]])
        mask0 = torch.ones(2, 2)

        # Rank 1 data
        pred1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        true1 = torch.tensor([[5.2, 5.9], [7.1, 8.0]])
        mask1 = torch.ones(2, 2)

        # Full dataset
        all_pred = torch.cat([pred0.flatten(), pred1.flatten()])
        all_true = torch.cat([true0.flatten(), true1.flatten()])

        expected_mae = (all_pred - all_true).abs().mean().item()
        expected_mse = ((all_pred - all_true) ** 2).mean().item()

        # Per-rank streaming
        streaming0 = StreamingCountMetrics()
        streaming0.update(pred0, true0, mask0)

        streaming1 = StreamingCountMetrics()
        streaming1.update(pred1, true1, mask1)

        # Merge
        streaming0._merge_from(streaming1)
        merged_metrics = streaming0.compute()

        assert abs(merged_metrics["mae"] - expected_mae) < 1e-4
        assert abs(merged_metrics["mse"] - expected_mse) < 1e-4

    def test_regression_sync_welford_merge_accuracy(self):
        """Test parallel Welford algorithm accuracy for variance computation.

        The key challenge in distributed metrics is correctly merging the
        running variance (m2_true) used for R² computation.
        """
        np.random.seed(123)
        torch.manual_seed(123)

        # Large dataset to test numerical stability
        n_samples = 10000
        all_preds = torch.randn(n_samples)
        all_targets = torch.randn(n_samples) * 10 + 5  # Different scale

        expected_metrics = compute_regression_metrics(all_preds, all_targets)

        # Split into 4 unequal chunks (simulating uneven batches)
        chunk_sizes = [2500, 2500, 2500, 2500]
        streaming_objects = []
        offset = 0
        for size in chunk_sizes:
            streaming = StreamingRegressionMetrics()
            streaming.update(all_preds[offset : offset + size], all_targets[offset : offset + size])
            streaming_objects.append(streaming)
            offset += size

        # Merge all into first
        merged = streaming_objects[0]
        for other in streaming_objects[1:]:
            merged._merge_from(other)

        merged_metrics = merged.compute()

        # R² computation depends on accurate variance merge
        assert abs(merged_metrics["r2"] - expected_metrics["r2"]) < 1e-3, (
            f"R² mismatch: expected {expected_metrics['r2']}, got {merged_metrics['r2']}"
        )
