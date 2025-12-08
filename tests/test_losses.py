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


class TestPairwiseDistances:
    """Test pairwise distance computation from embeddings."""

    def test_compute_pairwise_distances(self):
        """Test that pairwise distances are computed correctly from embeddings."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)

        distances = compute_pairwise_distances(embeddings)

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
        """Test pairwise distances with batch_size=1."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 1
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)

        distances = compute_pairwise_distances(embeddings)

        assert distances.shape == (batch_size, batch_size)
        assert torch.allclose(distances[0, 0], torch.tensor(0.0))  # Distance to self is 0


class TestBaseLoss:
    """Test base loss computation."""

    def test_base_loss_unifrac_with_embeddings(self, loss_fn):
        """Test MSE loss for UniFrac using embeddings (new approach)."""
        from aam.training.losses import compute_pairwise_distances

        batch_size = 4
        embedding_dim = 8
        embeddings = torch.randn(batch_size, embedding_dim)
        base_true = torch.randn(batch_size, batch_size)
        # Ensure base_true is symmetric and has zero diagonal
        base_true = (base_true + base_true.T) / 2
        base_true.fill_diagonal_(0.0)

        # Compute distances from embeddings
        computed_distances = compute_pairwise_distances(embeddings)

        loss = loss_fn.compute_base_loss(
            torch.zeros(1),  # Dummy base_pred (ignored when embeddings provided)
            base_true,
            encoder_type="unifrac",
            embeddings=embeddings,
        )

        assert loss.dim() == 0
        assert loss.item() >= 0

        # Loss should match manual computation
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=embeddings.device)
        computed_masked = computed_distances[triu_indices[0], triu_indices[1]]
        base_true_masked = base_true[triu_indices[0], triu_indices[1]]
        expected_loss = nn.functional.mse_loss(computed_masked, base_true_masked)
        assert torch.allclose(loss, expected_loss)

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

        # Compute expected distances from embeddings
        expected_distances = compute_pairwise_distances(embeddings)

        targets = {
            "target": torch.randn(batch_size, 1),
            "counts": torch.randn(batch_size, num_asvs, 1),
            "base_target": expected_distances,  # Use computed distances as target
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
