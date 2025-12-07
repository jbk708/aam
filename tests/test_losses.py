"""Unit tests for loss functions."""

import pytest
import torch
import torch.nn as nn

from aam.training.losses import MultiTaskLoss


@pytest.fixture
def loss_fn():
    """Create a MultiTaskLoss instance."""
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)


@pytest.fixture
def loss_fn_weighted():
    """Create a MultiTaskLoss instance with custom weights."""
    return MultiTaskLoss(penalty=2.0, nuc_penalty=0.5)


@pytest.fixture
def loss_fn_classifier():
    """Create a MultiTaskLoss instance with class weights."""
    class_weights = torch.tensor([0.5, 1.0, 2.0])
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0, class_weights=class_weights)


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


class TestBaseLoss:
    """Test base loss computation."""

    def test_base_loss_unifrac(self, loss_fn):
        """Test MSE loss for unweighted UniFrac (pairwise matrix with diagonal masking)."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        assert loss.dim() == 0
        assert loss.item() >= 0
        # Loss should be computed on upper triangle (excluding diagonal) with clipped predictions
        base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        base_pred_masked = base_pred_clipped[triu_indices[0], triu_indices[1]]
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

    def test_base_loss_unifrac_shape_mismatch_raises_error(self, loss_fn):
        """Test that shape mismatch in base loss raises ValueError."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size + 1)

        with pytest.raises(ValueError, match="Shape mismatch in base loss"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

    def test_base_loss_unifrac_different_batch_sizes_raises_error(self, loss_fn):
        """Test that different batch sizes in unifrac base loss raises error."""
        batch_size_pred = 4
        batch_size_true = 6
        base_pred = torch.randn(batch_size_pred, batch_size_pred)
        base_true = torch.randn(batch_size_true, batch_size_true)

        with pytest.raises(ValueError, match="Shape mismatch in base loss"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

    def test_base_loss_unifrac_base_output_dim_mismatch(self, loss_fn):
        """Test that base_output_dim mismatch raises error (simulating variable batch size issue)."""
        batch_size = 4
        base_output_dim = 6
        base_pred = torch.randn(batch_size, base_output_dim)
        base_true = torch.randn(batch_size, batch_size)

        with pytest.raises(ValueError, match="Shape mismatch in base loss"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

    def test_base_loss_unifrac_consistent_batch_sizes(self, loss_fn):
        """Test that consistent batch sizes work correctly (simulating drop_last=True behavior)."""
        for batch_size in [2, 4, 6, 8]:
            base_pred = torch.randn(batch_size, batch_size)
            base_true = torch.randn(batch_size, batch_size)

            loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

            assert loss.dim() == 0
            assert loss.item() >= 0
            # Loss should be computed on upper triangle (excluding diagonal) with clipped predictions
            base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
            triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
            base_pred_masked = base_pred_clipped[triu_indices[0], triu_indices[1]]
            base_true_masked = base_true[triu_indices[0], triu_indices[1]]
            expected_loss = nn.functional.mse_loss(base_pred_masked, base_true_masked)
            assert torch.allclose(loss, expected_loss)

    def test_base_loss_faith_pd_shape_mismatch_raises_error(self, loss_fn):
        """Test that shape mismatch in Faith PD base loss raises ValueError."""
        batch_size = 4
        base_pred = torch.randn(batch_size, 1)
        base_true = torch.randn(batch_size, 2)

        with pytest.raises(ValueError, match="Shape mismatch in base loss"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type="faith_pd")

    def test_base_loss_nan_in_prediction_raises_error(self, loss_fn):
        """Test that NaN in base_pred raises ValueError."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_pred[0, 0] = float("nan")
        base_true = torch.randn(batch_size, batch_size)

        with pytest.raises(ValueError, match="NaN values found in base_pred"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

    def test_base_loss_nan_in_target_raises_error(self, loss_fn):
        """Test that NaN in base_true raises ValueError."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)
        base_true[0, 0] = float("nan")

        with pytest.raises(ValueError, match="NaN values found in base_true"):
            loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

    def test_base_loss_unifrac_masks_diagonal(self, loss_fn):
        """Test that UniFrac loss excludes diagonal elements (self-comparisons)."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        # Set diagonal to very large values to verify they're excluded
        base_pred.fill_diagonal_(100.0)
        base_true.fill_diagonal_(200.0)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        # Loss should be computed only on upper triangle (excluding diagonal) with clipped predictions
        base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        base_pred_masked = base_pred_clipped[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(base_pred_masked, base_true_masked)

        assert torch.allclose(loss, expected_loss)
        # Loss should be reasonable (not affected by large diagonal values, which are clipped to 1.0)
        assert loss.item() < 100.0

    def test_base_loss_unifrac_loss_higher_without_diagonal(self, loss_fn):
        """Test that UniFrac loss values are higher when diagonal is excluded."""
        batch_size = 4
        # Create matrices with non-zero off-diagonal errors to ensure meaningful loss
        base_pred = torch.ones(batch_size, batch_size) * 0.5
        base_true = torch.ones(batch_size, batch_size) * 0.3

        # Set diagonal to 0.0 (typical for distance matrices)
        base_pred.fill_diagonal_(0.0)
        base_true.fill_diagonal_(0.0)

        # Loss with diagonal masking (current implementation)
        loss_masked = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        # Loss without diagonal masking (old behavior - for comparison)
        loss_unmasked = nn.functional.mse_loss(base_pred, base_true)

        # Masked loss should be higher because:
        # - Unmasked: includes diagonal (0.0 - 0.0 = 0) + off-diagonal errors, divided by all elements
        # - Masked: only off-diagonal errors, divided by fewer elements (no diagonal)
        # Since we're dividing by fewer elements in masked case, loss should be higher
        assert loss_masked.item() > loss_unmasked.item(), (
            f"Masked loss ({loss_masked.item():.6f}) should be higher than unmasked loss ({loss_unmasked.item():.6f}) "
            f"when diagonal is 0.0 and off-diagonal has errors"
        )

    def test_base_loss_unifrac_diagonal_masking_only_for_unifrac(self, loss_fn):
        """Test that diagonal masking only applies to UniFrac, not other encoder types."""
        batch_size = 4
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        # Set diagonal to large values
        base_pred.fill_diagonal_(100.0)
        base_true.fill_diagonal_(200.0)

        # Faith PD should NOT mask diagonal (even if square matrix)
        loss_faith_pd = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="faith_pd")
        expected_loss_faith_pd = nn.functional.mse_loss(base_pred, base_true)
        assert torch.allclose(loss_faith_pd, expected_loss_faith_pd)

        # Taxonomy should NOT mask diagonal
        loss_taxonomy = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="taxonomy")
        expected_loss_taxonomy = nn.functional.mse_loss(base_pred, base_true)
        assert torch.allclose(loss_taxonomy, expected_loss_taxonomy)

        # Combined should NOT mask diagonal
        loss_combined = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="combined")
        expected_loss_combined = nn.functional.mse_loss(base_pred, base_true)
        assert torch.allclose(loss_combined, expected_loss_combined)

    def test_base_loss_unifrac_non_square_matrix_no_masking(self, loss_fn):
        """Test that non-square matrices don't use diagonal masking."""
        batch_size = 4
        base_output_dim = 6
        base_pred = torch.randn(batch_size, base_output_dim)
        base_true = torch.randn(batch_size, base_output_dim)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        # Should use standard MSE with clipped predictions (no masking for non-square matrices)
        base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
        expected_loss = nn.functional.mse_loss(base_pred_clipped, base_true)
        assert torch.allclose(loss, expected_loss)

    def test_base_loss_unifrac_edge_case_batch_size_2(self, loss_fn):
        """Test diagonal masking with batch_size=2 (smallest valid pairwise matrix)."""
        batch_size = 2
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        # Set diagonal to large values
        base_pred.fill_diagonal_(100.0)
        base_true.fill_diagonal_(200.0)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        # With batch_size=2, upper triangle (excluding diagonal) has only 1 element: (0, 1)
        # Predictions should be clipped before masking
        base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        assert triu_indices.shape[1] == 1  # Only one off-diagonal element
        base_pred_masked = base_pred_clipped[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(base_pred_masked, base_true_masked)

        assert torch.allclose(loss, expected_loss)

    def test_base_loss_unifrac_edge_case_batch_size_1(self, loss_fn):
        """Test that batch_size=1 works (no off-diagonal elements, should return zero loss)."""
        batch_size = 1
        base_pred = torch.randn(batch_size, batch_size)
        base_true = torch.randn(batch_size, batch_size)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")

        # With batch_size=1, there are no off-diagonal elements, so loss should be zero
        # (upper triangle with offset=1 is empty)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        assert triu_indices.shape[1] == 0  # No off-diagonal elements

        # When there are no elements, MSE on empty tensors returns 0.0
        assert loss.item() == 0.0

    def test_base_loss_unifrac_clips_predictions_to_01_range(self, loss_fn):
        """Test that UniFrac predictions are clipped to [0, 1] range."""
        batch_size = 4
        # Create predictions outside [0, 1] range
        base_pred = torch.tensor([[-0.5, 0.3, 0.7, 1.5], [0.2, -0.1, 1.2, 0.8], [0.9, 0.4, 0.6, 2.0], [0.1, 0.5, 0.3, -0.3]])
        base_true = torch.ones(batch_size, batch_size) * 0.5

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=True)

        # Predictions should be clipped to [0, 1] before loss computation
        # Verify clipping happened by checking loss matches clipped predictions
        base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        base_pred_clipped_masked = base_pred_clipped[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(base_pred_clipped_masked, base_true_masked)

        assert torch.allclose(loss, expected_loss)

    def test_base_loss_unifrac_clipping_only_for_unifrac(self, loss_fn):
        """Test that clipping only applies to UniFrac, not other encoder types."""
        batch_size = 4
        # Create predictions outside [0, 1] range
        base_pred = torch.tensor([[-0.5, 0.3, 0.7, 1.5], [0.2, -0.1, 1.2, 0.8], [0.9, 0.4, 0.6, 2.0], [0.1, 0.5, 0.3, -0.3]])
        base_true = torch.ones(batch_size, batch_size) * 0.5

        # Faith PD should NOT clip predictions (even with clip_predictions=True, it's not UniFrac)
        loss_faith_pd = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="faith_pd", clip_predictions=True)
        expected_loss_faith_pd = nn.functional.mse_loss(base_pred, base_true)  # No clipping
        assert torch.allclose(loss_faith_pd, expected_loss_faith_pd)

        # Taxonomy should NOT clip predictions
        loss_taxonomy = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="taxonomy", clip_predictions=True)
        expected_loss_taxonomy = nn.functional.mse_loss(base_pred, base_true)  # No clipping
        assert torch.allclose(loss_taxonomy, expected_loss_taxonomy)

    def test_base_loss_unifrac_clip_predictions_false_disables_clipping(self, loss_fn):
        """Test that clip_predictions=False disables clipping for UniFrac."""
        batch_size = 4
        # Create predictions outside [0, 1] range
        base_pred = torch.tensor([[-0.5, 0.3, 0.7, 1.5], [0.2, -0.1, 1.2, 0.8], [0.9, 0.4, 0.6, 2.0], [0.1, 0.5, 0.3, -0.3]])
        base_true = torch.ones(batch_size, batch_size) * 0.5

        # With clip_predictions=False, predictions should NOT be clipped
        loss_no_clip = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=False)
        
        # Loss should be computed on unclipped predictions
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        base_pred_masked = base_pred[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(base_pred_masked, base_true_masked)

        assert torch.allclose(loss_no_clip, expected_loss)

        # Loss with clipping should be different (lower) than without clipping
        loss_with_clip = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=True)
        assert not torch.allclose(loss_no_clip, loss_with_clip)
        # Clipped loss should be lower because predictions are closer to target (0.5)
        assert loss_with_clip.item() < loss_no_clip.item()

    def test_base_loss_unifrac_clipping_preserves_gradients(self, loss_fn):
        """Test that clipping preserves gradients for backpropagation."""
        batch_size = 4
        # Create predictions outside [0, 1] range that require gradients
        base_pred = torch.tensor([[-0.5, 0.3, 0.7, 1.5], [0.2, -0.1, 1.2, 0.8], [0.9, 0.4, 0.6, 2.0], [0.1, 0.5, 0.3, -0.3]], requires_grad=True)
        base_true = torch.ones(batch_size, batch_size) * 0.5

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=True)

        # Verify loss requires gradients
        assert loss.requires_grad

        # Backward pass should work
        loss.backward()

        # Verify gradients exist
        assert base_pred.grad is not None
        assert not torch.allclose(base_pred.grad, torch.zeros_like(base_pred.grad))

    def test_base_loss_unifrac_clipping_with_diagonal_masking(self, loss_fn):
        """Test that clipping works correctly with diagonal masking."""
        batch_size = 4
        # Create predictions outside [0, 1] range
        base_pred = torch.tensor([[-0.5, 0.3, 0.7, 1.5], [0.2, -0.1, 1.2, 0.8], [0.9, 0.4, 0.6, 2.0], [0.1, 0.5, 0.3, -0.3]])
        base_true = torch.ones(batch_size, batch_size) * 0.5

        # Set diagonal to large values (should be clipped but also masked out)
        base_pred.fill_diagonal_(5.0)
        base_true.fill_diagonal_(0.0)

        loss = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=True)

        # Clipping should happen first, then diagonal masking
        base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
        base_pred_clipped_masked = base_pred_clipped[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(base_pred_clipped_masked, base_true_masked)

        assert torch.allclose(loss, expected_loss)

    def test_base_loss_unifrac_clipping_default_true(self, loss_fn):
        """Test that clip_predictions defaults to True for UniFrac."""
        batch_size = 4
        # Create predictions outside [0, 1] range
        base_pred = torch.tensor([[-0.5, 0.3, 0.7, 1.5], [0.2, -0.1, 1.2, 0.8], [0.9, 0.4, 0.6, 2.0], [0.1, 0.5, 0.3, -0.3]])
        base_true = torch.ones(batch_size, batch_size) * 0.5

        # Default behavior (clip_predictions not specified) should clip
        loss_default = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac")
        loss_explicit_true = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=True)

        # Both should produce same result (clipping enabled)
        assert torch.allclose(loss_default, loss_explicit_true)

        # Should be different from no clipping
        loss_no_clip = loss_fn.compute_base_loss(base_pred, base_true, encoder_type="unifrac", clip_predictions=False)
        assert not torch.allclose(loss_default, loss_no_clip)


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

    def test_total_loss_with_nucleotides(self, loss_fn):
        """Test total loss with nucleotide predictions."""
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
