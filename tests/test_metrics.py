"""Unit tests for metrics computation."""

import pytest
import torch
import numpy as np

from aam.training.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_count_metrics,
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
        
        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        
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
        
        y_pred_np = y_pred.detach().cpu().numpy().flatten()
        y_true_np = y_true.detach().cpu().numpy().flatten()
        
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
        
        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        
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
        
        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        
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
        valid_pred = (count_pred * valid_mask).detach().cpu().numpy()
        valid_true = (count_true * valid_mask).detach().cpu().numpy()
        
        valid_pred_flat = valid_pred[valid_mask.bool().cpu().numpy()]
        valid_true_flat = valid_true[valid_mask.bool().cpu().numpy()]
        
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
        
        count_pred_np = count_pred.detach().cpu().numpy().flatten()
        count_true_np = count_true.detach().cpu().numpy().flatten()
        
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
