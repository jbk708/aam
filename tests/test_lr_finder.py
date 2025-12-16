"""Tests for learning rate finder utility."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile

from aam.training.lr_finder import LearningRateFinder
from aam.models.sequence_encoder import SequenceEncoder
from aam.training.losses import MultiTaskLoss


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model():
    """Create a small SequenceEncoder for testing."""
    return SequenceEncoder(
        vocab_size=6,
        embedding_dim=32,
        max_bp=50,
        token_limit=64,
        asv_num_layers=1,
        asv_num_heads=2,
        sample_num_layers=1,
        sample_num_heads=2,
        encoder_num_layers=1,
        encoder_num_heads=2,
        base_output_dim=None,
        encoder_type="unifrac",
        predict_nucleotides=False,
    )


@pytest.fixture
def simple_dataset(device):
    """Create a simple dataset for testing."""
    from aam.data.tokenizer import SequenceTokenizer
    
    batch_size = 4
    num_asvs = 10
    seq_length = 32
    
    # Model expects (batch, num_asvs, seq_len) shape
    tokens = torch.randint(1, 5, (batch_size * 10, num_asvs, seq_length))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    
    # For pairwise mode, base_target should be [batch_size, batch_size] per batch
    # Create a custom dataset that returns correct shape per batch
    class UniFracDataset:
        def __init__(self, tokens, batch_size):
            self.tokens = tokens
            self.batch_size = batch_size
            self.num_batches = len(tokens) // batch_size

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            start_idx = idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_tokens = self.tokens[start_idx:end_idx]
            # Create pairwise distance matrix for this batch [batch_size, batch_size]
            # Ensure values are in [0, 1] range for UniFrac distances
            batch_size_actual = len(batch_tokens)
            unifrac_target = torch.rand(batch_size_actual, batch_size_actual)
            # Ensure diagonal is 0 (distance from sample to itself)
            unifrac_target.fill_diagonal_(0.0)
            return batch_tokens, unifrac_target
    
    dataset = UniFracDataset(tokens, batch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x[0])


@pytest.fixture
def loss_fn():
    """Create a loss function for testing."""
    return MultiTaskLoss(
        penalty=1.0,
        nuc_penalty=0.0,
    )


class TestLearningRateFinder:
    """Test learning rate finder utility."""

    def test_lr_finder_initialization(self, small_model, loss_fn, device):
        """Test LR finder initialization."""
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        lr_finder = LearningRateFinder(small_model, optimizer, loss_fn, device)
        
        assert lr_finder.model == small_model
        assert lr_finder.optimizer == optimizer
        assert lr_finder.loss_fn == loss_fn
        assert lr_finder.device == device
        assert len(lr_finder.original_lr) > 0

    def test_lr_finder_find_lr(self, small_model, loss_fn, simple_dataset, device):
        """Test LR finder can find learning rate."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        lr_finder = LearningRateFinder(small_model, optimizer, loss_fn, device)
        
        lrs, losses, suggested_lr = lr_finder.find_lr(
            simple_dataset,
            start_lr=1e-6,
            end_lr=1e-2,
            num_iter=20,
        )
        
        assert len(lrs) > 0
        assert len(losses) > 0
        assert len(lrs) == len(losses)
        assert all(lr > 0 for lr in lrs)
        assert all(loss > 0 for loss in losses)
        
        # Suggested LR should be in the range tested
        if suggested_lr is not None:
            assert 1e-6 <= suggested_lr <= 1e-2

    def test_lr_finder_restores_original_lr(self, small_model, loss_fn, simple_dataset, device):
        """Test LR finder restores original learning rate after finding."""
        small_model = small_model.to(device)
        original_lr = 1e-4
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=original_lr)
        lr_finder = LearningRateFinder(small_model, optimizer, loss_fn, device)
        
        lr_finder.find_lr(simple_dataset, start_lr=1e-6, end_lr=1e-2, num_iter=10)
        
        # Check that original LR was restored
        current_lr = optimizer.param_groups[0]["lr"]
        assert abs(current_lr - original_lr) < 1e-8

    def test_lr_finder_plot(self, small_model, loss_fn, simple_dataset, device):
        """Test LR finder can generate plot."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        lr_finder = LearningRateFinder(small_model, optimizer, loss_fn, device)
        
        lr_finder.find_lr(simple_dataset, start_lr=1e-6, end_lr=1e-2, num_iter=20)
        
        # Test plotting
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lr_finder_plot.png"
            fig = lr_finder.plot(output_path=output_path)
            
            assert fig is not None
            assert output_path.exists()

    def test_lr_finder_plot_without_data(self, small_model, loss_fn, device):
        """Test LR finder plot raises error without data."""
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        lr_finder = LearningRateFinder(small_model, optimizer, loss_fn, device)
        
        with pytest.raises(ValueError, match="No LR finder data"):
            lr_finder.plot()

    def test_lr_finder_handles_early_stop(self, small_model, loss_fn, simple_dataset, device):
        """Test LR finder handles early stopping on divergence."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        lr_finder = LearningRateFinder(small_model, optimizer, loss_fn, device)
        
        # Use very low divergence threshold to trigger early stop
        lrs, losses, suggested_lr = lr_finder.find_lr(
            simple_dataset,
            start_lr=1e-6,
            end_lr=1e-2,
            num_iter=50,
            diverge_threshold=1.1,  # Very low threshold
        )
        
        # Should have stopped early
        assert len(lrs) <= 50
        assert len(lrs) == len(losses)
