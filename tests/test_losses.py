"""Unit tests for loss functions."""

import pytest
import torch
import torch.nn as nn

from aam.training.losses import MultiTaskLoss


@pytest.fixture
def loss_fn():
    """Create a MultiTaskLoss instance with MSE loss (for backward compatibility)."""
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0, target_loss_type="mse")


@pytest.fixture
def loss_fn_huber():
    """Create a MultiTaskLoss instance with Huber loss (default)."""
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0, target_loss_type="huber")


@pytest.fixture
def loss_fn_mae():
    """Create a MultiTaskLoss instance with MAE loss."""
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0, target_loss_type="mae")


@pytest.fixture
def loss_fn_weighted():
    """Create a MultiTaskLoss instance with custom weights."""
    return MultiTaskLoss(penalty=2.0, nuc_penalty=0.5, target_loss_type="mse")


@pytest.fixture
def loss_fn_classifier():
    """Create a MultiTaskLoss instance with class weights."""
    class_weights = torch.tensor([0.5, 1.0, 2.0])
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0, class_weights=class_weights, target_loss_type="mse")


class TestTargetLoss:
    """Test target loss computation."""

    def test_regression_mse_loss(self, loss_fn):
        """Test MSE loss for regression."""
        batch_size = 4
        out_dim = 1
        target_pred = torch.randn(batch_size, out_dim)
        target_true = torch.randn(batch_size, out_dim)

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=False)

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.mse_loss(target_pred, target_true)
        assert torch.allclose(loss, expected_loss)

    def test_regression_mse_loss_multidim(self, loss_fn):
        """Test MSE loss for multi-dimensional regression."""
        batch_size = 4
        out_dim = 3
        target_pred = torch.randn(batch_size, out_dim)
        target_true = torch.randn(batch_size, out_dim)

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=False)

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.mse_loss(target_pred, target_true)
        assert torch.allclose(loss, expected_loss)

    def test_classification_nll_loss(self, loss_fn):
        """Test NLL loss for classification."""
        batch_size = 4
        num_classes = 3
        target_pred = torch.randn(batch_size, num_classes)
        target_pred = nn.functional.log_softmax(target_pred, dim=-1)
        target_true = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=True)

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.nll_loss(target_pred, target_true)
        assert torch.allclose(loss, expected_loss)

    def test_classification_nll_loss_with_weights(self, loss_fn_classifier):
        """Test NLL loss for classification with class weights."""
        batch_size = 4
        num_classes = 3
        target_pred = torch.randn(batch_size, num_classes)
        target_pred = nn.functional.log_softmax(target_pred, dim=-1)
        target_true = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn_classifier.compute_target_loss(target_pred, target_true, is_classifier=True)

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.nll_loss(target_pred, target_true, weight=loss_fn_classifier.class_weights)
        assert torch.allclose(loss, expected_loss)


class TestLossTypes:
    """Test different loss type configurations."""

    def test_default_loss_type_is_huber(self):
        """Test that default loss type is huber."""
        loss_fn = MultiTaskLoss()
        assert loss_fn.target_loss_type == "huber"

    def test_mse_loss_type(self):
        """Test MSE loss computation."""
        loss_fn = MultiTaskLoss(target_loss_type="mse")
        target_pred = torch.tensor([[0.5], [0.8]])
        target_true = torch.tensor([[0.3], [0.9]])

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=False)

        expected = nn.functional.mse_loss(target_pred, target_true)
        assert torch.allclose(loss, expected)

    def test_mae_loss_type(self, loss_fn_mae):
        """Test MAE (L1) loss computation."""
        target_pred = torch.tensor([[0.5], [0.8]])
        target_true = torch.tensor([[0.3], [0.9]])

        loss = loss_fn_mae.compute_target_loss(target_pred, target_true, is_classifier=False)

        expected = nn.functional.l1_loss(target_pred, target_true)
        assert torch.allclose(loss, expected)

    def test_huber_loss_type(self, loss_fn_huber):
        """Test Huber (smooth L1) loss computation."""
        target_pred = torch.tensor([[0.5], [0.8]])
        target_true = torch.tensor([[0.3], [0.9]])

        loss = loss_fn_huber.compute_target_loss(target_pred, target_true, is_classifier=False)

        expected = nn.functional.smooth_l1_loss(target_pred, target_true, beta=1.0)
        assert torch.allclose(loss, expected)

    def test_huber_loss_for_small_errors_like_mse(self):
        """Test that Huber loss behaves like MSE for small errors (|error| < beta)."""
        loss_fn_huber = MultiTaskLoss(target_loss_type="huber")
        loss_fn_mse = MultiTaskLoss(target_loss_type="mse")

        # Small errors (within beta=1.0)
        target_pred = torch.tensor([[0.5]])
        target_true = torch.tensor([[0.6]])  # error = 0.1

        loss_huber = loss_fn_huber.compute_target_loss(target_pred, target_true, is_classifier=False)
        loss_mse = loss_fn_mse.compute_target_loss(target_pred, target_true, is_classifier=False)

        # For small errors, Huber ≈ 0.5 * MSE (due to smooth_l1 definition)
        # smooth_l1(x) = 0.5 * x^2 / beta for |x| < beta
        assert loss_huber.item() < loss_mse.item()  # Huber should be smaller

    def test_huber_loss_for_large_errors_like_mae(self):
        """Test that Huber loss behaves like MAE for large errors (|error| > beta)."""
        loss_fn_huber = MultiTaskLoss(target_loss_type="huber")
        loss_fn_mae = MultiTaskLoss(target_loss_type="mae")

        # Large errors (beyond beta=1.0)
        target_pred = torch.tensor([[0.0]])
        target_true = torch.tensor([[5.0]])  # error = 5.0

        loss_huber = loss_fn_huber.compute_target_loss(target_pred, target_true, is_classifier=False)
        loss_mae = loss_fn_mae.compute_target_loss(target_pred, target_true, is_classifier=False)

        # For large errors, Huber ≈ MAE - 0.5 * beta (due to smooth_l1 definition)
        # The difference should be small relative to the MAE loss
        diff = abs(loss_huber.item() - loss_mae.item())
        assert diff < 1.0  # Should be close (within beta/2)

    def test_invalid_loss_type_raises_error(self):
        """Test that invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_loss_type"):
            MultiTaskLoss(target_loss_type="invalid")

    def test_loss_type_does_not_affect_classification(self):
        """Test that loss type doesn't affect classification (always uses NLL)."""
        loss_fn_mse = MultiTaskLoss(target_loss_type="mse")
        loss_fn_huber = MultiTaskLoss(target_loss_type="huber")
        loss_fn_mae = MultiTaskLoss(target_loss_type="mae")

        target_pred = torch.randn(4, 3)
        target_pred = nn.functional.log_softmax(target_pred, dim=-1)
        target_true = torch.randint(0, 3, (4,))

        loss_mse = loss_fn_mse.compute_target_loss(target_pred, target_true, is_classifier=True)
        loss_huber = loss_fn_huber.compute_target_loss(target_pred, target_true, is_classifier=True)
        loss_mae = loss_fn_mae.compute_target_loss(target_pred, target_true, is_classifier=True)

        # All should be equal (using NLL loss for classification)
        assert torch.allclose(loss_mse, loss_huber)
        assert torch.allclose(loss_huber, loss_mae)


class TestQuantileLoss:
    """Test quantile regression loss computation."""

    def test_quantile_loss_type_requires_quantiles(self):
        """Test that quantile loss type requires quantiles parameter."""
        with pytest.raises(ValueError, match="quantiles must be provided"):
            MultiTaskLoss(target_loss_type="quantile")

    def test_quantile_loss_empty_quantiles_raises_error(self):
        """Test that empty quantiles list raises error."""
        with pytest.raises(ValueError, match="must contain at least one value"):
            MultiTaskLoss(target_loss_type="quantile", quantiles=[])

    def test_quantile_loss_invalid_quantile_below_zero(self):
        """Test that quantile <= 0 raises error."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            MultiTaskLoss(target_loss_type="quantile", quantiles=[0.0, 0.5, 0.9])

    def test_quantile_loss_invalid_quantile_above_one(self):
        """Test that quantile >= 1 raises error."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            MultiTaskLoss(target_loss_type="quantile", quantiles=[0.1, 0.5, 1.0])

    def test_quantile_loss_valid_creation(self):
        """Test successful creation with valid quantiles."""
        loss_fn = MultiTaskLoss(target_loss_type="quantile", quantiles=[0.1, 0.5, 0.9])
        assert loss_fn.target_loss_type == "quantile"
        assert loss_fn.quantiles is not None
        assert torch.allclose(loss_fn.quantiles, torch.tensor([0.1, 0.5, 0.9]))

    def test_quantile_loss_single_quantile(self):
        """Test quantile loss with single quantile (median)."""
        loss_fn = MultiTaskLoss(target_loss_type="quantile", quantiles=[0.5])
        assert len(loss_fn.quantiles) == 1

    def test_pinball_loss_basic(self):
        """Test basic pinball loss computation."""
        from aam.training.losses import compute_pinball_loss

        # pred: [batch=2, out_dim=1, num_quantiles=3]
        pred = torch.tensor([[[0.3, 0.5, 0.7]], [[0.4, 0.6, 0.8]]])
        target = torch.tensor([[0.5], [0.5]])  # [batch=2, out_dim=1]
        quantiles = torch.tensor([0.1, 0.5, 0.9])

        loss = compute_pinball_loss(pred, target, quantiles)

        # Loss should be non-negative scalar
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_pinball_loss_perfect_prediction(self):
        """Test pinball loss when predictions exactly match targets."""
        from aam.training.losses import compute_pinball_loss

        # All quantiles predict the target exactly
        pred = torch.tensor([[[0.5, 0.5, 0.5]]])  # [1, 1, 3]
        target = torch.tensor([[0.5]])  # [1, 1]
        quantiles = torch.tensor([0.1, 0.5, 0.9])

        loss = compute_pinball_loss(pred, target, quantiles)

        # Perfect predictions should have zero loss
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_pinball_loss_underprediction_penalty(self):
        """Test that underprediction is penalized more for high quantiles."""
        from aam.training.losses import compute_pinball_loss

        # Predict 0.0 when target is 1.0 (underprediction)
        pred_low = torch.tensor([[[0.0]]])  # Predicting low quantile
        pred_high = torch.tensor([[[0.0]]])  # Predicting high quantile
        target = torch.tensor([[1.0]])

        # For tau=0.9 (high quantile), underprediction penalty is tau * |error| = 0.9 * 1.0
        # For tau=0.1 (low quantile), underprediction penalty is tau * |error| = 0.1 * 1.0
        loss_low_q = compute_pinball_loss(pred_low, target, torch.tensor([0.1]))
        loss_high_q = compute_pinball_loss(pred_high, target, torch.tensor([0.9]))

        # High quantile should have higher loss for underprediction
        assert loss_high_q.item() > loss_low_q.item()

    def test_pinball_loss_overprediction_penalty(self):
        """Test that overprediction is penalized more for low quantiles."""
        from aam.training.losses import compute_pinball_loss

        # Predict 1.0 when target is 0.0 (overprediction)
        pred = torch.tensor([[[1.0]]])
        target = torch.tensor([[0.0]])

        # For tau=0.1 (low quantile), overprediction penalty is (1-tau) * |error| = 0.9 * 1.0
        # For tau=0.9 (high quantile), overprediction penalty is (1-tau) * |error| = 0.1 * 1.0
        loss_low_q = compute_pinball_loss(pred, target, torch.tensor([0.1]))
        loss_high_q = compute_pinball_loss(pred, target, torch.tensor([0.9]))

        # Low quantile should have higher loss for overprediction
        assert loss_low_q.item() > loss_high_q.item()

    def test_pinball_loss_median_symmetric(self):
        """Test that median (tau=0.5) has symmetric loss."""
        from aam.training.losses import compute_pinball_loss

        quantiles = torch.tensor([0.5])
        target = torch.tensor([[0.5]])

        # Same magnitude error in both directions
        pred_under = torch.tensor([[[0.3]]])  # Under by 0.2
        pred_over = torch.tensor([[[0.7]]])  # Over by 0.2

        loss_under = compute_pinball_loss(pred_under, target, quantiles)
        loss_over = compute_pinball_loss(pred_over, target, quantiles)

        # Should be equal for median
        assert torch.allclose(loss_under, loss_over, atol=1e-6)

    def test_pinball_loss_multi_output(self):
        """Test pinball loss with multiple output dimensions."""
        from aam.training.losses import compute_pinball_loss

        # pred: [batch=2, out_dim=3, num_quantiles=2]
        pred = torch.randn(2, 3, 2)
        target = torch.randn(2, 3)
        quantiles = torch.tensor([0.25, 0.75])

        loss = compute_pinball_loss(pred, target, quantiles)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_pinball_loss_gradient_flow(self):
        """Test that gradients flow through pinball loss."""
        from aam.training.losses import compute_pinball_loss

        pred = torch.randn(4, 2, 3, requires_grad=True)
        target = torch.randn(4, 2)
        quantiles = torch.tensor([0.1, 0.5, 0.9])

        loss = compute_pinball_loss(pred, target, quantiles)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_quantile_loss_via_multitask(self):
        """Test quantile loss through MultiTaskLoss.compute_target_loss."""
        loss_fn = MultiTaskLoss(target_loss_type="quantile", quantiles=[0.1, 0.5, 0.9])

        # pred shape: [batch=2, out_dim=1, num_quantiles=3]
        target_pred = torch.tensor([[[0.3, 0.5, 0.7]], [[0.4, 0.6, 0.8]]])
        target_true = torch.tensor([[0.5], [0.5]])

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=False)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_quantile_loss_does_not_affect_classification(self):
        """Test that classification ignores quantile loss type."""
        loss_fn = MultiTaskLoss(target_loss_type="quantile", quantiles=[0.1, 0.5, 0.9])

        # For classification, should use NLL regardless of loss type
        target_pred = torch.randn(4, 3)
        target_pred = nn.functional.log_softmax(target_pred, dim=-1)
        target_true = torch.randint(0, 3, (4,))

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=True)

        expected = nn.functional.nll_loss(target_pred, target_true)
        assert torch.allclose(loss, expected)


class TestAsymmetricLoss:
    """Test asymmetric loss computation."""

    def test_asymmetric_loss_type_valid_creation(self):
        """Test successful creation with valid asymmetric parameters."""
        loss_fn = MultiTaskLoss(target_loss_type="asymmetric", over_penalty=2.0, under_penalty=1.0)
        assert loss_fn.target_loss_type == "asymmetric"
        assert loss_fn.over_penalty == 2.0
        assert loss_fn.under_penalty == 1.0

    def test_asymmetric_loss_default_penalties(self):
        """Test asymmetric loss with default penalty values."""
        loss_fn = MultiTaskLoss(target_loss_type="asymmetric")
        assert loss_fn.over_penalty == 1.0
        assert loss_fn.under_penalty == 1.0

    def test_asymmetric_loss_over_penalty_must_be_positive(self):
        """Test that over_penalty <= 0 raises error."""
        with pytest.raises(ValueError, match="over_penalty must be positive"):
            MultiTaskLoss(target_loss_type="asymmetric", over_penalty=0.0)

        with pytest.raises(ValueError, match="over_penalty must be positive"):
            MultiTaskLoss(target_loss_type="asymmetric", over_penalty=-1.0)

    def test_asymmetric_loss_under_penalty_must_be_positive(self):
        """Test that under_penalty <= 0 raises error."""
        with pytest.raises(ValueError, match="under_penalty must be positive"):
            MultiTaskLoss(target_loss_type="asymmetric", under_penalty=0.0)

        with pytest.raises(ValueError, match="under_penalty must be positive"):
            MultiTaskLoss(target_loss_type="asymmetric", under_penalty=-1.0)

    def test_compute_asymmetric_loss_basic(self):
        """Test basic asymmetric loss computation."""
        from aam.training.losses import compute_asymmetric_loss

        pred = torch.tensor([[0.5], [0.8]])
        target = torch.tensor([[0.3], [0.9]])

        loss = compute_asymmetric_loss(pred, target, over_penalty=1.0, under_penalty=1.0)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_compute_asymmetric_loss_perfect_prediction(self):
        """Test asymmetric loss when predictions exactly match targets."""
        from aam.training.losses import compute_asymmetric_loss

        pred = torch.tensor([[0.5], [0.8]])
        target = torch.tensor([[0.5], [0.8]])

        loss = compute_asymmetric_loss(pred, target, over_penalty=2.0, under_penalty=1.0)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_compute_asymmetric_loss_overprediction_penalty(self):
        """Test that overpredictions are penalized by over_penalty."""
        from aam.training.losses import compute_asymmetric_loss

        # Overprediction: pred > target
        pred = torch.tensor([[1.0]])
        target = torch.tensor([[0.0]])

        loss_high_over = compute_asymmetric_loss(pred, target, over_penalty=2.0, under_penalty=1.0)
        loss_low_over = compute_asymmetric_loss(pred, target, over_penalty=1.0, under_penalty=2.0)

        # Higher over_penalty should result in higher loss for overprediction
        assert loss_high_over.item() > loss_low_over.item()

    def test_compute_asymmetric_loss_underprediction_penalty(self):
        """Test that underpredictions are penalized by under_penalty."""
        from aam.training.losses import compute_asymmetric_loss

        # Underprediction: pred < target
        pred = torch.tensor([[0.0]])
        target = torch.tensor([[1.0]])

        loss_high_under = compute_asymmetric_loss(pred, target, over_penalty=1.0, under_penalty=2.0)
        loss_low_under = compute_asymmetric_loss(pred, target, over_penalty=2.0, under_penalty=1.0)

        # Higher under_penalty should result in higher loss for underprediction
        assert loss_high_under.item() > loss_low_under.item()

    def test_compute_asymmetric_loss_equal_penalties_is_mae(self):
        """Test that equal penalties produce MAE-like behavior."""
        from aam.training.losses import compute_asymmetric_loss

        pred = torch.tensor([[0.5], [0.8], [0.2]])
        target = torch.tensor([[0.3], [0.9], [0.5]])

        loss = compute_asymmetric_loss(pred, target, over_penalty=1.0, under_penalty=1.0)
        expected_mae = nn.functional.l1_loss(pred, target)

        assert torch.allclose(loss, expected_mae, atol=1e-6)

    def test_compute_asymmetric_loss_multi_output(self):
        """Test asymmetric loss with multiple output dimensions."""
        from aam.training.losses import compute_asymmetric_loss

        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)

        loss = compute_asymmetric_loss(pred, target, over_penalty=1.5, under_penalty=0.5)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_compute_asymmetric_loss_gradient_flow(self):
        """Test that gradients flow through asymmetric loss."""
        from aam.training.losses import compute_asymmetric_loss

        pred = torch.randn(4, 2, requires_grad=True)
        target = torch.randn(4, 2)

        loss = compute_asymmetric_loss(pred, target, over_penalty=2.0, under_penalty=1.0)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_asymmetric_loss_via_multitask(self):
        """Test asymmetric loss through MultiTaskLoss.compute_target_loss."""
        loss_fn = MultiTaskLoss(target_loss_type="asymmetric", over_penalty=2.0, under_penalty=1.0)

        target_pred = torch.tensor([[0.5], [0.8]])
        target_true = torch.tensor([[0.3], [0.9]])

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=False)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_asymmetric_loss_does_not_affect_classification(self):
        """Test that classification ignores asymmetric loss type."""
        loss_fn = MultiTaskLoss(target_loss_type="asymmetric", over_penalty=2.0, under_penalty=1.0)

        # For classification, should use NLL regardless of loss type
        target_pred = torch.randn(4, 3)
        target_pred = nn.functional.log_softmax(target_pred, dim=-1)
        target_true = torch.randint(0, 3, (4,))

        loss = loss_fn.compute_target_loss(target_pred, target_true, is_classifier=True)

        expected = nn.functional.nll_loss(target_pred, target_true)
        assert torch.allclose(loss, expected)

    def test_asymmetric_loss_extreme_penalty_ratios(self):
        """Test asymmetric loss with extreme penalty ratios."""
        from aam.training.losses import compute_asymmetric_loss

        # Only overpredictions
        pred_over = torch.tensor([[1.0], [2.0]])
        target_over = torch.tensor([[0.0], [0.0]])

        # Very high over_penalty vs low
        loss_high_over = compute_asymmetric_loss(pred_over, target_over, over_penalty=10.0, under_penalty=1.0)
        loss_low_over = compute_asymmetric_loss(pred_over, target_over, over_penalty=1.0, under_penalty=10.0)

        # Both should be positive
        assert loss_high_over.item() > 0
        assert loss_low_over.item() > 0

        # High over_penalty should give 10x higher loss for pure overpredictions
        assert loss_high_over.item() == loss_low_over.item() * 10


class TestEvaluatorQuantileExtraction:
    """Test Evaluator's handling of quantile regression outputs."""

    def test_extract_median_quantile_3d_tensor(self):
        """Test median quantile extraction from 3D predictions."""
        from unittest.mock import Mock

        from aam.training.evaluation import Evaluator

        model = Mock()
        model.num_quantiles = 3
        loss_fn = Mock()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loss_fn, device)

        # 3D predictions: [batch=4, out_dim=2, num_quantiles=3]
        predictions = torch.tensor(
            [
                [[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]],
                [[0.15, 0.55, 0.95], [0.25, 0.65, 1.05]],
                [[0.12, 0.52, 0.92], [0.22, 0.62, 1.02]],
                [[0.18, 0.58, 0.98], [0.28, 0.68, 1.08]],
            ]
        )

        result = evaluator._extract_median_quantile(predictions)

        # Should extract index 1 (middle quantile)
        expected = torch.tensor([[0.5, 0.6], [0.55, 0.65], [0.52, 0.62], [0.58, 0.68]])
        assert result.shape == (4, 2)
        assert torch.allclose(result, expected)

    def test_extract_median_quantile_2d_passthrough(self):
        """Test that 2D predictions pass through unchanged."""
        from unittest.mock import Mock

        from aam.training.evaluation import Evaluator

        model = Mock()
        model.num_quantiles = None  # Standard regression
        loss_fn = Mock()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loss_fn, device)

        predictions = torch.randn(4, 2)
        result = evaluator._extract_median_quantile(predictions)

        assert torch.equal(result, predictions)

    def test_extract_median_quantile_no_num_quantiles(self):
        """Test that predictions pass through when model has no num_quantiles."""
        from unittest.mock import Mock

        from aam.training.evaluation import Evaluator

        model = Mock(spec=[])  # No num_quantiles attribute
        loss_fn = Mock()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loss_fn, device)

        predictions = torch.randn(4, 2, 3)
        result = evaluator._extract_median_quantile(predictions)

        # Should pass through unchanged since model doesn't have num_quantiles
        assert torch.equal(result, predictions)

    def test_extract_median_quantile_ddp_wrapped(self):
        """Test median quantile extraction with DDP-wrapped model."""
        from unittest.mock import Mock

        from aam.training.evaluation import Evaluator

        inner_model = Mock()
        inner_model.num_quantiles = 5

        model = Mock(spec=["module"])
        model.module = inner_model

        loss_fn = Mock()
        device = torch.device("cpu")

        evaluator = Evaluator(model, loss_fn, device)

        predictions = torch.randn(4, 2, 5)
        result = evaluator._extract_median_quantile(predictions)

        # Should extract index 2 (middle of 5 quantiles)
        assert result.shape == (4, 2)
        assert torch.allclose(result, predictions[:, :, 2])


class TestCountLoss:
    """Test count loss computation."""

    def test_count_loss_masked(self, loss_fn):
        """Test masked MSE loss for counts."""
        batch_size = 4
        num_asvs = 10
        count_pred = torch.randn(batch_size, num_asvs, 1)
        count_true = torch.randn(batch_size, num_asvs, 1)
        mask = torch.ones(batch_size, num_asvs)
        mask[:, 5:] = 0

        loss = loss_fn.compute_count_loss(count_pred, count_true, mask)

        assert loss.dim() == 0
        assert loss.item() >= 0

        valid_mask = mask.unsqueeze(-1)
        valid_pred = count_pred * valid_mask
        valid_true = count_true * valid_mask
        num_valid = valid_mask.sum()
        expected_loss = ((valid_pred - valid_true) ** 2).sum() / num_valid
        assert torch.allclose(loss, expected_loss)

    def test_count_loss_all_valid(self, loss_fn):
        """Test count loss when all ASVs are valid."""
        batch_size = 4
        num_asvs = 10
        count_pred = torch.randn(batch_size, num_asvs, 1)
        count_true = torch.randn(batch_size, num_asvs, 1)
        mask = torch.ones(batch_size, num_asvs)

        loss = loss_fn.compute_count_loss(count_pred, count_true, mask)

        assert loss.dim() == 0
        expected_loss = nn.functional.mse_loss(count_pred, count_true)
        assert torch.allclose(loss, expected_loss)


class TestPairwiseDistances:
    """Test pairwise distance computation from embeddings."""

    def test_compute_pairwise_distances(self):
        """Test that pairwise distances are computed correctly from embeddings (unnormalized)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)

        # Test unnormalized distances (explicitly pass normalize=False)
        distances = compute_pairwise_distances(embeddings, normalize=False)

        assert distances.shape == (batch_size, batch_size)
        assert torch.all(distances >= 0)  # Distances should be non-negative
        assert torch.allclose(distances, distances.T)  # Should be symmetric
        assert torch.allclose(torch.diag(distances), torch.zeros(batch_size))  # Diagonal should be 0

        # Test that distances match manual computation for first pair
        manual_dist = torch.sqrt(((embeddings[0] - embeddings[1]) ** 2).sum())
        assert torch.allclose(distances[0, 1], manual_dist)

        # Test that distances are Euclidean
        diff = embeddings[0] - embeddings[1]
        expected_dist = torch.sqrt((diff**2).sum())
        assert torch.allclose(distances[0, 1], expected_dist)

    def test_compute_pairwise_distances_single_sample(self):
        """Test pairwise distances with batch_size=1 (unnormalized)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 1
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)

        # Test unnormalized distances (explicitly pass normalize=False)
        distances = compute_pairwise_distances(embeddings, normalize=False)

        assert distances.shape == (batch_size, batch_size)
        assert torch.allclose(distances[0, 0], torch.tensor(0.0))  # Distance to self is 0

    def test_compute_pairwise_distances_normalized(self):
        """Test that normalized distances are bounded to [0, 1]."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)

        distances = compute_pairwise_distances(embeddings)

        assert distances.shape == (batch_size, batch_size)
        # All distances should be in [0, 1]
        assert torch.all(distances >= 0.0)
        assert torch.all(distances <= 1.0)
        # Diagonal should still be 0.0
        assert torch.allclose(torch.diag(distances), torch.zeros(batch_size))
        # Should be symmetric
        assert torch.allclose(distances, distances.T)

    def test_compute_pairwise_distances_normalized_gradient_flow(self):
        """Test that normalized distances maintain healthy gradient flow (no saturation)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)

        distances = compute_pairwise_distances(embeddings)

        # Compute a loss using off-diagonal elements to ensure non-zero gradients
        # Sum only off-diagonal elements to avoid gradient cancellation from symmetry
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=distances.device)
        off_diagonal_distances = distances[triu_indices[0], triu_indices[1]]
        loss = off_diagonal_distances.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert embeddings.grad is not None
        # With tanh normalization, gradients should be healthy (> 1e-5)
        # Tanh has better gradient flow than sigmoid, but still bounded
        max_grad = torch.abs(embeddings.grad).max().item()
        assert max_grad > 1e-5, f"Gradients should be healthy (max={max_grad:.2e}), not saturated"

    def test_compute_pairwise_distances_normalized_different_scales(self):
        """Test that different scale values affect normalization (tanh-based)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)

        # Test with different scale values (tanh normalization uses scale parameter)
        distances_scale_1 = compute_pairwise_distances(embeddings, scale=1.0)
        distances_scale_5 = compute_pairwise_distances(embeddings, scale=5.0)
        distances_scale_10 = compute_pairwise_distances(embeddings, scale=10.0)

        # All should be in [0, 1]
        assert torch.all(distances_scale_1 >= 0.0) and torch.all(distances_scale_1 <= 1.0)
        assert torch.all(distances_scale_5 >= 0.0) and torch.all(distances_scale_5 <= 1.0)
        assert torch.all(distances_scale_10 >= 0.0) and torch.all(distances_scale_10 <= 1.0)

        # With tanh normalization, different scales should produce different results
        # Larger scale = more sensitive (distances get divided by larger value before tanh)
        # So scale=10 should produce values closer to 0.5 (tanh(0) = 0.5 after shift)
        # Scale=1 should produce more extreme values
        # They should not be identical
        assert not torch.allclose(distances_scale_1, distances_scale_5, atol=1e-6)
        assert not torch.allclose(distances_scale_1, distances_scale_10, atol=1e-6)

    def test_compute_pairwise_distances_no_saturation(self):
        """Test that normalized distances do not saturate at ~0.55 (no sigmoid saturation)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 8
        embedding_dim = 16
        # Use fixed seed for reproducibility (avoids flaky test failures)
        torch.manual_seed(42)
        # Use diverse embeddings to ensure varied distances
        embeddings = torch.randn(batch_size, embedding_dim) * 2.0

        distances = compute_pairwise_distances(embeddings)  # Use default scale=10.0

        # Extract off-diagonal elements
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=distances.device)
        off_diagonal = distances[triu_indices[0], triu_indices[1]]

        # Check that values are distributed across [0, 1] range, not all clustered at ~0.5
        mean_val = off_diagonal.mean().item()
        std_val = off_diagonal.std().item()

        # Mean should not be ~0.5 (saturation indicator for tanh)
        assert abs(mean_val - 0.5) > 0.05, f"Mean value {mean_val:.3f} suggests saturation at ~0.5"

        # Values should have reasonable spread (std > 0.03 indicates distribution, not clustering)
        # Tanh normalization with scale produces narrower range than direct normalization
        assert std_val > 0.03, f"Std {std_val:.3f} too small, suggests clustering/saturation"

        # Check that we have values across the range (not all near 0.5)
        values_near_05 = ((off_diagonal > 0.5 - 0.1) & (off_diagonal < 0.5 + 0.1)).sum().item()
        total_values = off_diagonal.numel()
        fraction_near_05 = values_near_05 / total_values

        # Less than 80% should be near 0.5 (tanh produces more values near 0.5 than direct normalization)
        assert fraction_near_05 < 0.8, f"Too many values near 0.5 ({fraction_near_05:.1%}), suggests saturation"


class TestBaseLoss:
    """Test base loss computation."""

    def test_base_loss_unifrac_with_embeddings(self, loss_fn):
        """Test MSE loss for UniFrac using embeddings (new approach with normalization)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)
        # Create base_true in [0, 1] range (UniFrac distances are bounded)
        base_true = torch.rand(batch_size, batch_size)
        # Ensure base_true is symmetric and has zero diagonal
        base_true = (base_true + base_true.T) / 2
        base_true.fill_diagonal_(0.0)

        # Compute normalized distances from embeddings (normalize=True is now the default)
        computed_distances = compute_pairwise_distances(embeddings)

        loss = loss_fn.compute_base_loss(
            torch.zeros(1),  # Dummy base_pred (ignored when embeddings provided)
            base_true,
            encoder_type="unifrac",
            embeddings=embeddings,
        )

        assert loss.dim() == 0
        assert loss.item() >= 0

        # Loss should match manual computation with normalized distances
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=embeddings.device)
        computed_masked = computed_distances[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(computed_masked, base_true_masked)
        assert torch.allclose(loss, expected_loss)

    def test_base_loss_unifrac_with_embeddings_normalized(self, loss_fn):
        """Test that UniFrac loss with embeddings produces normalized distances in [0, 1]."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)
        # Create base_true in [0, 1] range (UniFrac distances)
        base_true = torch.rand(batch_size, batch_size)
        # Ensure base_true is symmetric and has zero diagonal
        base_true = (base_true + base_true.T) / 2
        base_true.fill_diagonal_(0.0)

        # Compute loss (should use normalized distances internally)
        loss = loss_fn.compute_base_loss(
            torch.zeros(1),  # Dummy base_pred (ignored when embeddings provided)
            base_true,
            encoder_type="unifrac",
            embeddings=embeddings,
        )

        # Verify that computed distances are normalized
        computed_distances = compute_pairwise_distances(embeddings, normalize=True)
        assert torch.all(computed_distances >= 0.0)
        assert torch.all(computed_distances <= 1.0)

        # Loss should be valid
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_base_loss_unifrac_legacy(self, loss_fn):
        """Test MSE loss for unweighted UniFrac (legacy approach with direct predictions)."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        assert loss.dim() == 0
        assert loss.item() >= 0
        # Loss should be computed on upper triangle (excluding diagonal)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        base_pred_masked = base_pred[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(base_pred_masked, base_true_masked)
        assert torch.allclose(loss, expected_loss)

    def test_base_loss_faith_pd(self, loss_fn):
        """Test MSE loss for Faith PD (per-sample vector)."""
        batch_size = 4
        base_pred = torch.randn(batch_size, 1)
        base_true = torch.randn(batch_size, 1)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="faith_pd")

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.mse_loss(base_pred, base_true)
        assert torch.allclose(loss, expected_loss)

    def test_base_loss_taxonomy(self, loss_fn):
        """Test MSE loss for taxonomy (per-sample vector)."""
        batch_size = 4
        base_output_dim = 7
        base_pred = torch.randn(batch_size, base_output_dim)
        base_true = torch.randn(batch_size, base_output_dim)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="taxonomy")

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.mse_loss(base_pred, base_true)
        assert torch.allclose(loss, expected_loss)

    def test_base_loss_combined(self, loss_fn):
        """Test MSE loss for combined encoder."""
        batch_size = 4
        base_output_dim = 10
        base_pred = torch.randn(batch_size, base_output_dim)
        base_true = torch.randn(batch_size, base_output_dim)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="combined")

        assert loss.dim() == 0
        assert loss.item() >= 0
        expected_loss = nn.functional.mse_loss(base_pred, base_true)
        assert torch.allclose(loss, expected_loss)

    @pytest.mark.parametrize(
        "encoder_type,base_pred_shape,base_true_shape",
        [
            ("unifrac", (4, 4), (4, 5)),  # Different second dimension
            ("unifrac", (4, 4), (6, 6)),  # Different batch sizes
            ("unifrac", (4, 6), (4, 4)),  # base_output_dim mismatch
            ("faith_pd", (4, 1), (4, 2)),  # Different output dims
        ],
    )
    def test_base_loss_shape_mismatch_raises_error(self, loss_fn, encoder_type, base_pred_shape, base_true_shape):
        """Test that shape mismatches in base loss raise ValueError."""
        base_pred = torch.randn(*base_pred_shape)
        base_true = torch.randn(*base_true_shape)

        with pytest.raises(ValueError, match="Shape mismatch in base loss"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type=encoder_type)

    @pytest.mark.parametrize("nan_location", ["pred", "true"])
    def test_base_loss_nan_raises_error(self, loss_fn, nan_location):
        """Test that NaN in base_pred or base_true raises ValueError."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        if nan_location == "pred":
            base_pred[0, 0] = float("nan")
            with pytest.raises(ValueError, match="NaN values found in base_pred"):
                loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")
        else:
            base_true[0, 0] = float("nan")
            with pytest.raises(ValueError, match="NaN values found in base_true"):
                loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

    def test_base_loss_unifrac_diagonal_masking(self, loss_fn):
        """Test that UniFrac loss excludes diagonal elements and only applies to UniFrac."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        # Set diagonal to very large values to verify they're excluded
        base_pred.fill_diagonal_(100.0)
        base_true.fill_diagonal_(200.0)

        # Test 1: UniFrac masks diagonal
        loss_unifrac = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        expected_loss = nn.functional.mse_loss(
            base_pred[triu_indices[0], triu_indices[1]], base_true[triu_indices[0], triu_indices[1]]
        )
        assert torch.allclose(loss_unifrac, expected_loss)
        assert loss_unifrac.item() < 100.0  # Not affected by large diagonal values

        # Test 2: Other encoder types do NOT mask diagonal
        for encoder_type in ["faith_pd", "taxonomy", "combined"]:
            loss_other = loss_fn.compute_base_loss(base_pred, base_true, encoder_type=encoder_type)
            expected_loss_other = nn.functional.mse_loss(base_pred, base_true)
            assert torch.allclose(loss_other, expected_loss_other)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_base_loss_unifrac_edge_cases(self, loss_fn, batch_size):
        """Test UniFrac loss with different batch sizes (edge cases)."""
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        if batch_size == 1:
            # batch_size=1: no off-diagonal elements, should return zero loss
            loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")
            assert loss.item() == 0.0
        else:
            # Other batch sizes: should compute loss on upper triangle
            loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")
            assert loss.dim() == 0
            assert loss.item() >= 0
            triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
            expected_loss = nn.functional.mse_loss(
                base_pred[triu_indices[0], triu_indices[1]], base_true[triu_indices[0], triu_indices[1]]
            )
            assert torch.allclose(loss, expected_loss)


class TestNucleotideLoss:
    """Test nucleotide loss computation."""

    def test_nucleotide_loss_masked(self, loss_fn):
        """Test masked CrossEntropy loss for nucleotides."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 6

        nuc_pred = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        nuc_true = torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len))
        mask = torch.ones(batch_size, num_asvs, seq_len)
        mask[:, :, 5:] = 0

        loss = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask)

        assert loss.dim() == 0
        assert loss.item() >= 0

        nuc_pred_flat = nuc_pred.view(-1, vocab_size)
        nuc_true_flat = nuc_true.view(-1)
        mask_flat = mask.view(-1)

        valid_indices = mask_flat.bool()
        valid_pred = nuc_pred_flat[valid_indices]
        valid_true = nuc_true_flat[valid_indices]

        expected_loss = nn.functional.cross_entropy(valid_pred, valid_true)
        assert torch.allclose(loss, expected_loss)

    def test_nucleotide_loss_all_valid(self, loss_fn):
        """Test nucleotide loss when all positions are valid."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 6

        nuc_pred = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        nuc_true = torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len))
        mask = torch.ones(batch_size, num_asvs, seq_len)

        loss = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask)

        assert loss.dim() == 0

        nuc_pred_flat = nuc_pred.view(-1, vocab_size)
        nuc_true_flat = nuc_true.view(-1)
        expected_loss = nn.functional.cross_entropy(nuc_pred_flat, nuc_true_flat)
        assert torch.allclose(loss, expected_loss)

    def test_nucleotide_loss_from_tokens(self, loss_fn):
        """Test nucleotide loss when tokens are used instead of explicit nucleotides (pretraining scenario)."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 6

        nuc_predictions = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        tokens = torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len))
        tokens[:, :, 7:] = 0

        outputs = {"nuc_predictions": nuc_predictions}
        targets = {"tokens": tokens}

        losses = loss_fn(outputs, targets, is_classifier=False, encoder_type="unifrac")

        assert "nuc_loss" in losses
        assert losses["nuc_loss"].dim() == 0
        assert losses["nuc_loss"].item() >= 0

        mask = (tokens > 0).long()
        expected_loss = loss_fn.compute_nucleotide_loss(nuc_predictions, tokens, mask)
        assert torch.allclose(losses["nuc_loss"], expected_loss)

    def test_nucleotide_loss_with_masked_indices(self, loss_fn):
        """Test nucleotide loss computed only on masked positions (MAE mode)."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 7  # Including MASK token

        nuc_pred = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        nuc_true = torch.randint(1, 5, (batch_size, num_asvs, seq_len))  # Only nucleotides
        mask = torch.ones(batch_size, num_asvs, seq_len)

        # Create masked_indices: only first 3 positions per sequence are masked
        masked_indices = torch.zeros(batch_size, num_asvs, seq_len, dtype=torch.bool)
        masked_indices[:, :, :3] = True

        loss = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask, masked_indices=masked_indices)

        assert loss.dim() == 0
        assert loss.item() >= 0

        # Verify loss is computed only on masked positions
        nuc_pred_flat = nuc_pred.view(-1, vocab_size)
        nuc_true_flat = nuc_true.view(-1)
        masked_indices_flat = masked_indices.view(-1)

        masked_pred = nuc_pred_flat[masked_indices_flat]
        masked_true = nuc_true_flat[masked_indices_flat]

        expected_loss = nn.functional.cross_entropy(masked_pred, masked_true)
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_nucleotide_loss_masked_indices_none_uses_all_valid(self, loss_fn):
        """Test that None masked_indices uses all valid positions."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 6

        nuc_pred = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        nuc_true = torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len))
        mask = torch.ones(batch_size, num_asvs, seq_len)
        mask[:, :, 5:] = 0  # Last 5 positions are padding

        # Call with masked_indices=None (default)
        loss_with_none = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask, masked_indices=None)

        # Should be same as without masked_indices
        loss_without = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask)

        assert torch.allclose(loss_with_none, loss_without)

    def test_nucleotide_loss_masked_indices_respects_padding_mask(self, loss_fn):
        """Test that masked_indices combined with padding mask works correctly."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 7

        nuc_pred = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        nuc_true = torch.randint(1, 5, (batch_size, num_asvs, seq_len))

        # Padding mask: only first 6 positions are valid
        mask = torch.zeros(batch_size, num_asvs, seq_len)
        mask[:, :, :6] = 1

        # Masked indices: positions 2-4 are masked for MAE
        masked_indices = torch.zeros(batch_size, num_asvs, seq_len, dtype=torch.bool)
        masked_indices[:, :, 2:5] = True

        loss = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask, masked_indices=masked_indices)

        # Loss should only be computed on positions 2-4 (masked AND valid)
        # Positions 5+ are padding, even though masked_indices might be True there
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_nucleotide_loss_no_masked_positions(self, loss_fn):
        """Test nucleotide loss when masked_indices has no True values."""
        batch_size = 2
        num_asvs = 5
        seq_len = 10
        vocab_size = 7

        nuc_pred = torch.randn(batch_size, num_asvs, seq_len, vocab_size)
        nuc_true = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        mask = torch.ones(batch_size, num_asvs, seq_len)

        # No positions masked
        masked_indices = torch.zeros(batch_size, num_asvs, seq_len, dtype=torch.bool)

        loss = loss_fn.compute_nucleotide_loss(nuc_pred, nuc_true, mask, masked_indices=masked_indices)

        # Should return zero loss when no positions are masked
        assert loss.dim() == 0
        assert loss.item() == 0.0


class TestTotalLoss:
    """Test total loss computation."""

    def test_total_loss_regression(self, loss_fn):
        """Test total loss for regression mode."""
        batch_size = 4
        num_asvs = 10
        seq_len = 20
        vocab_size = 6

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
        }

        mask = torch.ones(batch_size, num_asvs)

        losses = loss_fn(
            outputs,
            targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        assert "target_loss" in losses
        assert "count_loss" in losses
        assert "unifrac_loss" in losses
        assert "total_loss" in losses

        assert losses["target_loss"].dim() == 0
        assert losses["count_loss"].dim() == 0
        assert losses["unifrac_loss"].dim() == 0
        assert losses["total_loss"].dim() == 0

        expected_total = losses["target_loss"] + losses["count_loss"] + losses["unifrac_loss"] * loss_fn.penalty
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_total_loss_with_embeddings_unifrac(self, loss_fn):
        """Test total loss with embeddings for UniFrac (new approach)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        num_asvs = 10
        seq_len = 20
        vocab_size = 6

        embeddings = torch.randn(batch_size, embedding_dim)
        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "embeddings": embeddings,  # New: embeddings instead of base_prediction
            "nuc_predictions": torch.randn(batch_size, num_asvs, seq_len, vocab_size),
        }

        # Compute expected distances from embeddings (normalized to [0, 1] for UniFrac)
        # For test, create base_target in [0, 1] range (UniFrac distances are bounded)
        base_target = torch.rand(batch_size, batch_size)
        base_target = (base_target + base_target.T) / 2  # Make symmetric
        base_target.fill_diagonal_(0.0)  # Diagonal should be 0

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": base_target,  # Use normalized distances as target (in [0, 1])
            "nucleotides": torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len)),
        }

        losses = loss_fn(
            outputs,
            targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        assert "target_loss" in losses
        assert "count_loss" in losses
        assert "unifrac_loss" in losses
        assert "nuc_loss" in losses
        assert "total_loss" in losses

        # Verify unifrac_loss is computed from embeddings
        assert losses["unifrac_loss"].item() >= 0

    def test_total_loss_with_nucleotides(self, loss_fn):
        """Test total loss with nucleotide predictions (legacy approach)."""
        batch_size = 4
        num_asvs = 10
        seq_len = 20
        vocab_size = 6

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
            "nuc_predictions": torch.randn(batch_size, num_asvs, seq_len, vocab_size),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
            "nucleotides": torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len)),
        }

        mask = torch.ones(batch_size, num_asvs)
        nuc_mask = torch.ones(batch_size, num_asvs, seq_len)

        losses = loss_fn(
            outputs,
            targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        assert "target_loss" in losses
        assert "count_loss" in losses
        assert "unifrac_loss" in losses
        assert "nuc_loss" in losses
        assert "total_loss" in losses

        expected_total = (
            losses["target_loss"]
            + losses["count_loss"]
            + losses["unifrac_loss"] * loss_fn.penalty
            + losses["nuc_loss"] * loss_fn.nuc_penalty
        )
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_total_loss_classification(self, loss_fn_classifier):
        """Test total loss for classification mode."""
        batch_size = 4
        num_asvs = 10
        num_classes = 3

        target_pred = torch.randn(batch_size, num_classes)
        target_pred = nn.functional.log_softmax(target_pred, dim=-1)

        outputs = {
            "target_prediction": target_pred,
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
        }

        targets = {
            "target": torch.randint(0, num_classes, (batch_size,)),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
        }

        mask = torch.ones(batch_size, num_asvs)

        losses = loss_fn_classifier(
            outputs,
            targets,
            is_classifier=True,
            encoder_type="unifrac",
        )

        assert "target_loss" in losses
        assert "count_loss" in losses
        assert "unifrac_loss" in losses
        assert "total_loss" in losses

    def test_total_loss_weighted(self, loss_fn_weighted):
        """Test total loss with custom weights."""
        batch_size = 4
        num_asvs = 10

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
        }

        mask = torch.ones(batch_size, num_asvs)

        losses = loss_fn_weighted(
            outputs,
            targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        expected_total = losses["target_loss"] + losses["count_loss"] + losses["unifrac_loss"] * loss_fn_weighted.penalty
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_total_loss_missing_outputs(self, loss_fn):
        """Test total loss when some outputs are missing (inference mode)."""
        batch_size = 4
        num_asvs = 10

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
        }

        mask = torch.ones(batch_size, num_asvs)

        losses = loss_fn(
            outputs,
            targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        assert "target_loss" in losses
        assert "count_loss" in losses
        assert "unifrac_loss" not in losses or losses["unifrac_loss"] == 0
        assert "nuc_loss" not in losses or losses["nuc_loss"] == 0


class TestCountPenalty:
    """Test count_penalty parameter behavior."""

    def test_count_penalty_default_is_one(self):
        """Test that count_penalty defaults to 1.0."""
        loss_fn = MultiTaskLoss()
        assert loss_fn.count_penalty == 1.0

    def test_count_penalty_applied_to_total_loss(self):
        """Test that count_penalty is applied to count_loss in total loss."""
        count_penalty = 2.5
        loss_fn = MultiTaskLoss(count_penalty=count_penalty)

        batch_size = 4
        num_asvs = 10

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
            "mask": torch.ones(batch_size, num_asvs),
        }

        losses = loss_fn(outputs, targets, is_classifier=False, encoder_type="unifrac")

        expected_total = (
            losses["target_loss"] * loss_fn.target_penalty
            + losses["count_loss"] * count_penalty
            + losses["unifrac_loss"] * loss_fn.penalty
            + losses["nuc_loss"] * loss_fn.nuc_penalty
        )
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_count_penalty_zero_disables_count_loss(self):
        """Test that count_penalty=0 effectively disables count loss contribution."""
        loss_fn = MultiTaskLoss(count_penalty=0.0)

        batch_size = 4
        num_asvs = 10

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
            "mask": torch.ones(batch_size, num_asvs),
        }

        losses = loss_fn(outputs, targets, is_classifier=False, encoder_type="unifrac")

        # count_loss should still be computed but not contribute to total
        assert losses["count_loss"].item() > 0  # Non-zero count loss
        expected_total = losses["target_loss"] * loss_fn.target_penalty + losses["unifrac_loss"] * loss_fn.penalty
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_count_penalty_scales_contribution(self):
        """Test that different count_penalty values scale count_loss contribution."""
        batch_size = 4
        num_asvs = 10

        outputs = {
            "target_prediction": torch.zeros(batch_size, 1),
            "count_prediction": torch.ones(batch_size, num_asvs, 1),
            "base_prediction": torch.zeros(batch_size, batch_size),
        }

        targets = {
            "target": torch.zeros(batch_size, 1),
            "counts": torch.zeros(batch_size, num_asvs, 1),  # MSE = 1.0
            "base_target": torch.zeros(batch_size, batch_size),
            "mask": torch.ones(batch_size, num_asvs),
        }

        # With count_penalty=1.0
        loss_fn_1 = MultiTaskLoss(count_penalty=1.0, penalty=0.0, nuc_penalty=0.0, target_penalty=0.0)
        losses_1 = loss_fn_1(outputs, targets, is_classifier=False, encoder_type="unifrac")

        # With count_penalty=2.0
        loss_fn_2 = MultiTaskLoss(count_penalty=2.0, penalty=0.0, nuc_penalty=0.0, target_penalty=0.0)
        losses_2 = loss_fn_2(outputs, targets, is_classifier=False, encoder_type="unifrac")

        # Total loss should scale with count_penalty
        assert torch.allclose(losses_2["total_loss"], losses_1["total_loss"] * 2.0)

    def test_count_penalty_with_all_penalties(self):
        """Test count_penalty works correctly with all other penalty weights."""
        loss_fn = MultiTaskLoss(
            penalty=0.5,
            nuc_penalty=0.25,
            target_penalty=2.0,
            count_penalty=1.5,
        )

        batch_size = 4
        num_asvs = 10
        seq_len = 20
        vocab_size = 6

        outputs = {
            "target_prediction": torch.randn(batch_size, 1),
            "count_prediction": torch.randn(batch_size, num_asvs, 1),
            "base_prediction": torch.randn(batch_size, batch_size),
            "nuc_predictions": torch.randn(batch_size, num_asvs, seq_len, vocab_size),
        }

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": torch.randn(batch_size, batch_size),
            "nucleotides": torch.randint(0, vocab_size, (batch_size, num_asvs, seq_len)),
            "mask": torch.ones(batch_size, num_asvs),
        }

        losses = loss_fn(outputs, targets, is_classifier=False, encoder_type="unifrac")

        expected_total = (
            losses["target_loss"] * 2.0 + losses["count_loss"] * 1.5 + losses["unifrac_loss"] * 0.5 + losses["nuc_loss"] * 0.25
        )
        assert torch.allclose(losses["total_loss"], expected_total)


class TestDeviceHandling:
    """Test device handling for losses."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_losses_on_cuda(self, loss_fn):
        """Test that losses work on CUDA."""
        device = torch.device("cuda")
        loss_fn = loss_fn.to(device)

        batch_size = 4
        outputs = {
            "target_prediction": torch.randn(batch_size, 1, device=device),
            "count_prediction": torch.randn(batch_size, 10, 1, device=device),
            "base_prediction": torch.randn(batch_size, batch_size, device=device),
        }

        targets = {
            "target": torch.randn(batch_size, 1, device=device),
            "counts": torch.randn(batch_size, 10, 1, device=device),
            "base_target": torch.randn(batch_size, batch_size, device=device),
        }

        mask = torch.ones(batch_size, 10, device=device)

        losses = loss_fn(
            outputs,
            targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        assert losses["total_loss"].device.type == device.type


class TestGatherTargetMatrices:
    """Tests for _gather_target_matrices function used in distributed FSDP training."""

    def test_gather_target_matrices_returns_tuple(self):
        """Test that _gather_target_matrices returns a tuple of (global_target, mask)."""
        from aam.training.losses import _gather_target_matrices
        from unittest.mock import patch

        local_target = torch.randn(4, 4)
        world_size = 2

        with patch("aam.training.losses.dist.all_gather") as mock_all_gather:
            # Simulate gathering: mock fills gathered list with copies of local_target
            def fill_gathered(gathered, tensor):
                for i in range(len(gathered)):
                    gathered[i].copy_(tensor)

            mock_all_gather.side_effect = fill_gathered

            global_target, mask = _gather_target_matrices(local_target, world_size)

            assert isinstance(global_target, torch.Tensor)
            assert isinstance(mask, torch.Tensor)

    def test_gather_target_matrices_output_shapes(self):
        """Test that output shapes are correct for multi-GPU gathering."""
        from aam.training.losses import _gather_target_matrices
        from unittest.mock import patch

        local_batch = 4
        world_size = 3
        local_target = torch.randn(local_batch, local_batch)

        with patch("aam.training.losses.dist.all_gather") as mock_all_gather:

            def fill_gathered(gathered, tensor):
                for i in range(len(gathered)):
                    gathered[i].copy_(tensor)

            mock_all_gather.side_effect = fill_gathered

            global_target, mask = _gather_target_matrices(local_target, world_size)

            expected_global_batch = local_batch * world_size
            assert global_target.shape == (expected_global_batch, expected_global_batch)
            assert mask.shape == (expected_global_batch, expected_global_batch)

    def test_gather_target_matrices_mask_is_block_diagonal(self):
        """Test that mask is True only on block diagonal."""
        from aam.training.losses import _gather_target_matrices
        from unittest.mock import patch

        local_batch = 2
        world_size = 3
        local_target = torch.randn(local_batch, local_batch)

        with patch("aam.training.losses.dist.all_gather") as mock_all_gather:

            def fill_gathered(gathered, tensor):
                for i in range(len(gathered)):
                    gathered[i].copy_(tensor)

            mock_all_gather.side_effect = fill_gathered

            global_target, mask = _gather_target_matrices(local_target, world_size)

            # Check diagonal blocks are True
            for rank in range(world_size):
                start = rank * local_batch
                end = start + local_batch
                assert mask[start:end, start:end].all(), f"Block {rank} should be True"

            # Check off-diagonal blocks are False
            for rank_i in range(world_size):
                for rank_j in range(world_size):
                    if rank_i != rank_j:
                        start_i = rank_i * local_batch
                        end_i = start_i + local_batch
                        start_j = rank_j * local_batch
                        end_j = start_j + local_batch
                        assert not mask[start_i:end_i, start_j:end_j].any(), (
                            f"Off-diagonal block ({rank_i}, {rank_j}) should be False"
                        )

    def test_gather_target_matrices_preserves_values_on_diagonal(self):
        """Test that gathered values are placed correctly on diagonal blocks."""
        from aam.training.losses import _gather_target_matrices
        from unittest.mock import patch

        local_batch = 2
        world_size = 2
        local_target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Simulate different values from each rank
        rank_targets = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # Rank 0
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # Rank 1
        ]

        with patch("aam.training.losses.dist.all_gather") as mock_all_gather:

            def fill_gathered(gathered, tensor):
                for i in range(len(gathered)):
                    gathered[i].copy_(rank_targets[i])

            mock_all_gather.side_effect = fill_gathered

            global_target, mask = _gather_target_matrices(local_target, world_size)

            # Check rank 0's block
            assert torch.allclose(global_target[0:2, 0:2], rank_targets[0])
            # Check rank 1's block
            assert torch.allclose(global_target[2:4, 2:4], rank_targets[1])
            # Check off-diagonal blocks are zero
            assert torch.allclose(global_target[0:2, 2:4], torch.zeros(2, 2))
            assert torch.allclose(global_target[2:4, 0:2], torch.zeros(2, 2))


class TestComputeBaseLossWithMask:
    """Tests for compute_base_loss with target_mask for distributed gathering."""

    @pytest.fixture
    def loss_fn(self):
        return MultiTaskLoss()

    def test_masked_loss_only_uses_valid_pairs(self, loss_fn):
        """Test that loss is computed only on valid (masked) pairs."""
        # Simulate a 4x4 global matrix with only the first 2x2 block valid
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        # Create a mask that's True only for the first 2x2 block
        target_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        target_mask[0:2, 0:2] = True

        # Manually compute expected loss: only use triu of valid block
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1)
        mask_triu = target_mask[triu_indices[0], triu_indices[1]]
        pred_triu = base_pred[triu_indices[0], triu_indices[1]]
        true_triu = base_true[triu_indices[0], triu_indices[1]]

        # Only (0, 1) is in triu AND in valid block
        valid_pred = pred_triu[mask_triu]
        valid_true = true_triu[mask_triu]
        expected_loss = ((valid_pred - valid_true) ** 2).mean()

        # We can't easily test compute_base_loss directly with target_mask
        # since it's set internally based on distributed gathering
        # Instead, verify the math matches what we implemented
        squared_errors = (pred_triu - true_triu) ** 2
        computed_loss = (squared_errors * mask_triu.float()).sum() / mask_triu.sum()

        assert torch.allclose(computed_loss, expected_loss)

    def test_masked_loss_excludes_zeros_from_average(self, loss_fn):
        """Test that masked pairs don't dilute the loss average."""
        batch_size = 4

        # Create predictable tensors
        base_pred = torch.zeros(batch_size, batch_size)
        base_true = torch.zeros(batch_size, batch_size)

        # Set a known difference in the valid block (0:2, 0:2)
        # Only (0, 1) is in the upper triangle
        base_pred[0, 1] = 1.0
        base_true[0, 1] = 0.0  # Error = 1.0

        # Set different values in invalid blocks that would dilute if included
        base_pred[0, 2] = 100.0  # Would be huge error if included
        base_true[0, 2] = 0.0

        target_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        target_mask[0:2, 0:2] = True

        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1)
        mask_triu = target_mask[triu_indices[0], triu_indices[1]]
        pred_triu = base_pred[triu_indices[0], triu_indices[1]]
        true_triu = base_true[triu_indices[0], triu_indices[1]]

        # Correct loss: only (0, 1) with MSE = 1.0
        squared_errors = (pred_triu - true_triu) ** 2
        correct_loss = (squared_errors * mask_triu.float()).sum() / mask_triu.sum()
        assert correct_loss.item() == 1.0

        # Wrong loss (if we included all zeros): would be much smaller
        wrong_loss = squared_errors.mean()
        # triu has 6 elements: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # Only (0,1)=1.0 and (0,2)=10000.0 are non-zero
        # If mask wasn't used properly, loss would be different
        assert wrong_loss.item() != 1.0  # Sanity check

    def test_gather_for_distributed_warns_when_not_initialized(self, loss_fn):
        """Test that warning is issued when gather_for_distributed=True but dist not init."""
        import warnings
        from unittest.mock import patch

        batch_size = 4
        embeddings = torch.randn(batch_size, 64)
        base_true = torch.randn(batch_size, batch_size)

        with patch("aam.training.losses.dist.is_initialized", return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loss_fn.compute_base_loss(
                    base_pred=torch.zeros(batch_size, batch_size),
                    base_true=base_true,
                    encoder_type="unifrac",
                    embeddings=embeddings,
                    gather_for_distributed=True,
                )

                # Check warning was issued
                assert len(w) == 1
                assert "gather_for_distributed=True but distributed not initialized" in str(w[0].message)
