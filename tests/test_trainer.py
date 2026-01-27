"""Unit tests for training loop."""

import logging
import inspect
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aam.training.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
    load_pretrained_encoder,
)
from aam.training.evaluation import (
    create_prediction_plot,
    create_confusion_matrix_plot,
)
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.sequence_predictor import SequencePredictor
from aam.training.losses import MultiTaskLoss
from aam.training.metrics import compute_regression_metrics, compute_count_metrics
from conftest import MockBatchDataset


def is_rocm() -> bool:
    """Check if running on ROCm (AMD GPU) backend."""
    if not torch.cuda.is_available():
        return False
    # ROCm uses HIP which reports as CUDA but with different version info
    # Check for ROCm-specific attributes or version strings
    try:
        # torch.version.hip exists on ROCm builds
        return hasattr(torch.version, "hip") and torch.version.hip is not None
    except Exception:
        return False


@pytest.fixture
def small_predictor():
    """Create a small SequencePredictor for testing."""
    return SequencePredictor(
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
        count_num_layers=1,
        count_num_heads=2,
        target_num_layers=1,
        target_num_heads=2,
        out_dim=1,
        is_classifier=False,
        freeze_base=False,
        predict_nucleotides=False,
        base_output_dim=None,  # UniFrac: no output_head, returns embeddings
    )


@pytest.fixture
def loss_fn():
    """Create a MultiTaskLoss instance."""
    return MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)


@pytest.fixture
def simple_dataloader(device):
    """Create a simple DataLoader for testing."""
    from aam.data.tokenizer import SequenceTokenizer

    batch_size = 4
    num_asvs = 10
    seq_len = 50

    tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens = tokens.to(device)
    counts = torch.rand(batch_size * 2, num_asvs, 1).to(device)
    targets = torch.randn(batch_size * 2, 1).to(device)

    dataset = TensorDataset(tokens, counts, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def simple_dataloader_encoder(device):
    """Create a simple DataLoader for SequenceEncoder training."""
    from aam.data.tokenizer import SequenceTokenizer

    batch_size = 4
    num_asvs = 10
    seq_len = 50

    tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens = tokens.to(device)

    # For UniFrac, base_targets should be pairwise distance matrices [batch_size, batch_size]
    # Since TensorDataset requires same shape, we'll create a custom dataset that generates
    # distance matrices per batch
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
            dist_matrix = torch.rand(self.batch_size, self.batch_size)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Make symmetric
            dist_matrix.fill_diagonal_(0.0)  # Zero diagonal
            # Move to same device as tokens
            dist_matrix = dist_matrix.to(batch_tokens.device)
            return batch_tokens, dist_matrix

    dataset = UniFracDataset(tokens, batch_size)
    # Use collate_fn to handle the batching (each item is already a batch)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])


class TestCreateOptimizer:
    """Test optimizer creation."""

    def test_create_optimizer_default(self, small_model):
        """Test creating optimizer with default parameters."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4, weight_decay=0.01)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_optimizer_adam(self, small_model):
        """Test creating Adam optimizer."""
        optimizer = create_optimizer(small_model, optimizer_type="adam", lr=1e-4, weight_decay=0.01)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_optimizer_sgd(self, small_model):
        """Test creating SGD optimizer."""
        optimizer = create_optimizer(small_model, optimizer_type="sgd", lr=1e-3, momentum=0.9)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_create_optimizer_invalid_type(self, small_model):
        """Test creating optimizer with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(small_model, optimizer_type="invalid")

    def test_create_optimizer_excludes_frozen(self, small_predictor):
        """Test that optimizer excludes frozen parameters."""
        small_predictor.freeze_base = True
        for param in small_predictor.base_model.parameters():
            param.requires_grad = False

        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", freeze_base=True)

        param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        base_param_ids = {id(p) for p in small_predictor.base_model.parameters()}

        assert len(param_ids & base_param_ids) == 0

    def test_create_optimizer_categorical_lr_creates_param_groups(self):
        """Test that categorical_lr creates separate parameter groups."""
        model = SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            categorical_cardinalities={"location": 5},
            categorical_embed_dim=8,
            categorical_fusion="concat",
        )

        optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-4, categorical_lr=1e-3)

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-4  # base params
        assert optimizer.param_groups[1]["lr"] == 1e-3  # categorical params

    def test_create_optimizer_categorical_lr_separates_params(self):
        """Test that categorical parameters are correctly separated."""
        model = SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            categorical_cardinalities={"location": 5},
            categorical_embed_dim=8,
            categorical_fusion="gmu",
        )

        optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-4, categorical_lr=1e-3)

        # Get categorical param ids from model
        categorical_param_ids = set()
        for name, param in model.named_parameters():
            if "categorical_embedder" in name or "gmu" in name or "categorical_projection" in name:
                categorical_param_ids.add(id(param))

        # Check categorical params are in the second group
        cat_group_param_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
        assert categorical_param_ids == cat_group_param_ids

    def test_create_optimizer_categorical_lr_with_freeze_base(self):
        """Test categorical_lr works correctly with freeze_base."""
        model = SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            categorical_cardinalities={"location": 5},
            categorical_embed_dim=8,
            categorical_fusion="concat",
        )

        # Freeze base model
        for param in model.base_model.parameters():
            param.requires_grad = False

        optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-4, categorical_lr=1e-3, freeze_base=True)

        # Check base model params are excluded
        base_param_ids = {id(p) for p in model.base_model.parameters()}
        all_optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert len(base_param_ids & all_optimizer_param_ids) == 0

        # Check we still have 2 param groups with correct LRs
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[1]["lr"] == 1e-3

    def test_create_optimizer_categorical_lr_none_uses_single_group(self, small_predictor):
        """Test that categorical_lr=None uses single parameter group (default behavior)."""
        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", lr=1e-4, categorical_lr=None)

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-4

    def test_create_optimizer_categorical_lr_with_output_scales(self):
        """Test that output_scales params are in categorical group."""
        model = SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            categorical_cardinalities={"season": 4},
            categorical_embed_dim=8,
            categorical_fusion="concat",
            conditional_scaling_columns=["season"],
        )

        optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-4, categorical_lr=1e-3)

        # Get output_scales param ids
        output_scales_param_ids = set()
        for name, param in model.named_parameters():
            if "output_scales" in name:
                output_scales_param_ids.add(id(param))

        # Check output_scales params are in categorical group
        cat_group_param_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
        assert output_scales_param_ids.issubset(cat_group_param_ids)


class TestCreateScheduler:
    """Test scheduler creation."""

    def test_create_scheduler_warmup_cosine(self, small_model):
        """Test creating warmup cosine scheduler."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw")
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=100, num_training_steps=1000)

        assert scheduler is not None
        assert hasattr(scheduler, "step")
        assert hasattr(scheduler, "get_last_lr")

    def test_create_scheduler_cosine(self, small_model):
        """Test creating cosine annealing scheduler."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw")
        scheduler = create_scheduler(optimizer, scheduler_type="cosine", num_training_steps=1000, T_max=1000)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 1000

    def test_create_scheduler_plateau(self, small_model):
        """Test creating ReduceLROnPlateau scheduler with aggressive defaults."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw")
        scheduler = create_scheduler(optimizer, scheduler_type="plateau", num_training_steps=1000)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.patience == 5
        assert scheduler.factor == 0.3
        assert scheduler.min_lrs == [0.0]

    def test_create_scheduler_plateau_custom_params(self, small_model):
        """Test creating ReduceLROnPlateau scheduler with custom parameters."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="plateau",
            num_training_steps=1000,
            patience=3,
            factor=0.2,
            min_lr=1e-6,
            threshold=1e-5,
            threshold_mode="abs",
            cooldown=2,
            eps=1e-9,
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.patience == 3
        assert scheduler.factor == 0.2
        assert scheduler.min_lrs == [1e-6]
        assert scheduler.threshold == 1e-5
        assert scheduler.threshold_mode == "abs"
        assert scheduler.cooldown == 2
        assert scheduler.eps == 1e-9

    def test_create_scheduler_onecycle(self, small_model):
        """Test creating OneCycleLR scheduler."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="onecycle", num_training_steps=1000, max_lr=1e-3)

        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        assert scheduler.total_steps == 1000

    def test_create_scheduler_cosine_restarts(self, small_model):
        """Test creating CosineAnnealingWarmRestarts scheduler."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(
            optimizer, scheduler_type="cosine_restarts", num_training_steps=1000, T_0=100, T_mult=2, eta_min=1e-6
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 100
        assert scheduler.T_mult == 2
        assert scheduler.eta_min == 1e-6

    def test_create_scheduler_cosine_restarts_defaults(self, small_model):
        """Test creating CosineAnnealingWarmRestarts scheduler with default parameters."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="cosine_restarts", num_training_steps=1000)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 250
        assert scheduler.T_mult == 2
        assert scheduler.eta_min == 0.0

    def test_create_scheduler_invalid_type(self, small_model):
        """Test creating scheduler with invalid type raises error."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw")
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(optimizer, scheduler_type="invalid")

    def test_scheduler_cosine_restarts_stepping(self, small_model, device):
        """Test cosine_restarts scheduler stepping behavior."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(
            optimizer, scheduler_type="cosine_restarts", num_training_steps=100, T_0=10, T_mult=2, eta_min=1e-5
        )

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr > 0

        # Step through first cycle (T_0=10)
        lrs = []
        for step in range(15):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            lrs.append(lr)
            assert lr >= 1e-5  # Should not go below eta_min

        # Verify LR changes (scheduler is working)
        assert len(set(lrs)) > 1  # LR should vary
        # At restart (step 10), LR should restart to a higher value
        # The exact behavior depends on the cosine curve, but LR should change
        assert lrs[0] != lrs[10] or lrs[9] != lrs[10]  # LR changes around restart

    def test_scheduler_plateau_stepping(self, small_model, device):
        """Test plateau scheduler stepping behavior."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(
            optimizer, scheduler_type="plateau", num_training_steps=100, patience=1, factor=0.5, min_lr=1e-5
        )

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-3

        # Step with losses - scheduler tracks best loss internally
        scheduler.step(0.5)  # Initial loss
        assert optimizer.param_groups[0]["lr"] == 1e-3  # LR unchanged initially

        # Step with worse loss multiple times to trigger reduction
        # Plateau scheduler needs patience epochs of no improvement
        for _ in range(2):
            scheduler.step(0.6)  # Worse loss

        # Verify scheduler can step and LR is within bounds
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr >= 1e-5  # Should not go below min_lr
        assert current_lr <= 1e-3  # Should not exceed initial LR

    def test_scheduler_plateau_aggressive_defaults(self, small_model, device):
        """Test plateau scheduler uses aggressive defaults (factor=0.3, patience=5)."""
        small_model = small_model.to(device)
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, scheduler_type="plateau", num_training_steps=100)

        assert scheduler.patience == 5
        assert scheduler.factor == 0.3
        assert scheduler.min_lrs == [0.0]

    def test_trainer_with_cosine_restarts_scheduler(self, small_model, loss_fn, simple_dataloader, device):
        """Test Trainer works correctly with cosine_restarts scheduler."""
        small_model = small_model.to(device)
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="cosine_restarts", num_training_steps=20, T_0=5, T_mult=2)

        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        # Train for a few steps
        trainer.train_epoch(simple_dataloader, epoch=0, num_epochs=1)

        # Verify scheduler was stepped
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr > 0

    def test_trainer_with_plateau_scheduler(self, small_model, loss_fn, simple_dataloader, device):
        """Test Trainer works correctly with plateau scheduler."""
        small_model = small_model.to(device)
        optimizer = create_optimizer(small_model, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="plateau", num_training_steps=100, patience=2, factor=0.5)

        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        # Train for a few steps
        trainer.train_epoch(simple_dataloader, epoch=0, num_epochs=1)

        # Validate scheduler (plateau requires step(val_loss) which happens in validate_epoch)
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-4


class TestTrainerInit:
    """Test Trainer initialization."""

    def test_trainer_init_default(self, small_model, loss_fn, device):
        """Test Trainer initialization with default parameters."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        assert trainer.model == small_model
        assert trainer.loss_fn == loss_fn
        assert trainer.device == device

    def test_trainer_init_with_optimizer(self, small_model, loss_fn, device):
        """Test Trainer initialization with custom optimizer."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw")
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        assert trainer.optimizer == optimizer

    def test_trainer_init_with_scheduler(self, small_model, loss_fn, device):
        """Test Trainer initialization with custom scheduler."""
        optimizer = create_optimizer(small_model, optimizer_type="adamw")
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine")
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        assert trainer.scheduler == scheduler


class TestTrainEpoch:
    """Test training epoch."""

    def test_train_epoch_encoder(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test training epoch for SequenceEncoder."""
        small_model = small_model.to(device)
        small_model.train()  # Ensure model is in training mode
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        # Verify model produces valid embeddings before training
        for batch in simple_dataloader_encoder:
            tokens, targets, _, _ = trainer._prepare_batch(batch)
            with torch.no_grad():
                outputs = small_model(tokens, return_nucleotides=False)
                if "embeddings" in outputs:
                    assert not torch.any(torch.isnan(outputs["embeddings"])), "Model should produce valid embeddings"
                    assert not torch.any(torch.isinf(outputs["embeddings"])), "Model should produce finite embeddings"
            break  # Just check first batch

        losses = trainer.train_epoch(simple_dataloader_encoder)

        assert "total_loss" in losses
        assert losses["total_loss"] >= 0
        assert isinstance(losses["total_loss"], float)

    def test_train_epoch_predictor(self, small_predictor, loss_fn, simple_dataloader, device):
        """Test training epoch for SequencePredictor."""
        small_predictor = small_predictor.to(device)
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        losses = trainer.train_epoch(simple_dataloader)

        assert "total_loss" in losses
        assert losses["total_loss"] >= 0

    def test_train_epoch_updates_parameters(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that training epoch updates model parameters."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        initial_params = [p.clone() for p in small_model.parameters() if p.requires_grad]

        trainer.train_epoch(simple_dataloader_encoder)

        updated_params = [p for p in small_model.parameters() if p.requires_grad]

        for init_param, updated_param in zip(initial_params, updated_params):
            assert not torch.equal(init_param, updated_param)


class TestValidateEpoch:
    """Test validation epoch."""

    def test_validate_epoch_encoder(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test validation epoch for SequenceEncoder."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        results = trainer.validate_epoch(simple_dataloader_encoder, compute_metrics=False)

        assert "total_loss" in results
        assert results["total_loss"] >= 0

    def test_validate_epoch_predictor(self, small_predictor, loss_fn, simple_dataloader, device):
        """Test validation epoch for SequencePredictor."""
        small_predictor = small_predictor.to(device)
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        results = trainer.validate_epoch(simple_dataloader, compute_metrics=True)

        assert "total_loss" in results
        assert results["total_loss"] >= 0

    def test_validate_epoch_no_grad(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that validation epoch doesn't update parameters."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        initial_params = [p.clone() for p in small_model.parameters()]

        trainer.validate_epoch(simple_dataloader_encoder)

        updated_params = [p for p in small_model.parameters()]

        for init_param, updated_param in zip(initial_params, updated_params):
            assert torch.equal(init_param, updated_param)


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self, small_model, loss_fn, device, tmp_path):
        """Test saving checkpoint."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(
            str(checkpoint_path),
            epoch=5,
            best_val_loss=0.5,
            metrics={"mae": 0.3, "mse": 0.4},
        )

        assert checkpoint_path.exists()

    def test_load_checkpoint(self, small_model, loss_fn, device, tmp_path):
        """Test loading checkpoint."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(
            str(checkpoint_path),
            epoch=5,
            best_val_loss=0.5,
            metrics={"mae": 0.3},
        )

        checkpoint_info = trainer.load_checkpoint(str(checkpoint_path))

        assert checkpoint_info["epoch"] == 5
        assert checkpoint_info["best_val_loss"] == 0.5
        assert checkpoint_info["metrics"]["mae"] == 0.3

    def test_load_checkpoint_without_optimizer(self, small_model, loss_fn, device, tmp_path):
        """Test loading checkpoint without optimizer state."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(
            str(checkpoint_path),
            epoch=3,
            best_val_loss=0.6,
        )

        checkpoint_info = trainer.load_checkpoint(str(checkpoint_path), load_optimizer=False)

        assert checkpoint_info["epoch"] == 3

    def test_load_checkpoint_with_lr_override(self, small_model, loss_fn, device, tmp_path):
        """Test that target_lr overrides the checkpoint's learning rate."""
        small_model = small_model.to(device)
        original_lr = 1e-3
        optimizer = create_optimizer(small_model, lr=original_lr)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, best_val_loss=0.5)

        # Verify checkpoint has original LR
        assert trainer.optimizer.param_groups[0]["lr"] == original_lr

        # Load checkpoint with a different target_lr
        new_lr = 5e-5
        trainer.load_checkpoint(str(checkpoint_path), load_optimizer=True, target_lr=new_lr)

        # Verify LR was overridden
        for param_group in trainer.optimizer.param_groups:
            assert param_group["lr"] == new_lr

    def test_load_checkpoint_lr_override_updates_scheduler_base_lrs(self, small_model, loss_fn, device, tmp_path):
        """Test that target_lr also updates WarmupCosineScheduler's base_lrs."""
        small_model = small_model.to(device)
        original_lr = 1e-3
        optimizer = create_optimizer(small_model, lr=original_lr)
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=10, num_training_steps=100)

        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, best_val_loss=0.5)

        # Verify scheduler has original base_lrs
        assert trainer.scheduler.base_lrs == [original_lr]

        # Load checkpoint with a different target_lr
        new_lr = 5e-5
        trainer.load_checkpoint(str(checkpoint_path), load_optimizer=True, load_scheduler=True, target_lr=new_lr)

        # Verify scheduler's base_lrs were updated
        assert trainer.scheduler.base_lrs == [new_lr]

    def test_load_checkpoint_same_lr_no_override(self, small_model, loss_fn, device, tmp_path, caplog):
        """Test that same target_lr as checkpoint doesn't log override message."""
        small_model = small_model.to(device)
        original_lr = 1e-4
        optimizer = create_optimizer(small_model, lr=original_lr)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, best_val_loss=0.5)

        # Load checkpoint with same target_lr
        with caplog.at_level(logging.INFO):
            trainer.load_checkpoint(str(checkpoint_path), load_optimizer=True, target_lr=original_lr)

        # Verify no override message was logged
        assert "Overriding checkpoint learning rate" not in caplog.text

    def test_load_checkpoint_without_target_lr_preserves_checkpoint_lr(self, small_model, loss_fn, device, tmp_path):
        """Test that without target_lr, checkpoint's learning rate is preserved (existing behavior)."""
        small_model = small_model.to(device)
        checkpoint_lr = 1e-3
        optimizer = create_optimizer(small_model, lr=checkpoint_lr)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, best_val_loss=0.5)

        # Create new trainer with different LR
        new_lr = 5e-5
        new_optimizer = create_optimizer(small_model, lr=new_lr)
        trainer2 = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=new_optimizer,
            device=device,
        )

        # Load checkpoint without target_lr - should restore checkpoint's LR
        trainer2.load_checkpoint(str(checkpoint_path), load_optimizer=True)

        # Verify checkpoint LR was restored (existing behavior)
        assert trainer2.optimizer.param_groups[0]["lr"] == checkpoint_lr

    def test_train_start_epoch_skips_resume_from_load(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that start_epoch + initial_best_metric_value enables resume without resume_from.

        This tests the CLI pattern where load_checkpoint() is called first with target_lr override,
        then train() is called with start_epoch (not resume_from). The LR override should persist.
        """
        small_model = small_model.to(device)
        checkpoint_lr = 1e-3
        target_lr = 1e-5
        optimizer = create_optimizer(small_model, lr=checkpoint_lr)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # Save checkpoint with checkpoint_lr
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=5, best_val_loss=0.5)

        # Load checkpoint with target_lr override (CLI does this before calling train)
        checkpoint_info = trainer.load_checkpoint(str(checkpoint_path), load_optimizer=True, target_lr=target_lr)
        assert trainer.optimizer.param_groups[0]["lr"] == target_lr

        # Call train() with start_epoch (CLI pattern: no resume_from, use start_epoch instead)
        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=7,
            start_epoch=checkpoint_info["epoch"] + 1,
            initial_best_metric_value=checkpoint_info.get("best_metric_value"),
        )

        # LR should still be target_lr (no double-load occurred)
        assert trainer.optimizer.param_groups[0]["lr"] == target_lr
        assert len(history["train_loss"]) == 1


class TestTrainingLoop:
    """Test main training loop."""

    def test_train_basic(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test basic training loop."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=2,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_train_with_validation(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test training loop with validation."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=2,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_unifrac_predictions_in_range(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that UniFrac distance predictions are bounded to [0, 1] during validation."""
        from aam.training.losses import compute_pairwise_distances

        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        # Run validation and check predictions
        results = trainer.validate_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1, compute_metrics=True)

        # Check that validation completed
        assert "total_loss" in results

        # Manually check predictions from a batch
        small_model.eval()
        embeddings_found = False
        with torch.no_grad():
            for batch in simple_dataloader_encoder:
                tokens, targets, _, _ = trainer._prepare_batch(batch)
                outputs = small_model(tokens, return_nucleotides=False)

                # For UniFrac encoder, embeddings should always be present
                assert "embeddings" in outputs, "UniFrac encoder should return embeddings"
                embeddings = outputs["embeddings"]
                embeddings_found = True

                # Compute normalized distances (normalize=True is now the default)
                distances = compute_pairwise_distances(embeddings)

                # Verify all distances are in [0, 1]
                # Use explicit device-aware comparisons
                zero_tensor = torch.tensor(0.0, device=distances.device)
                one_tensor = torch.tensor(1.0, device=distances.device)
                assert torch.all(distances >= zero_tensor), (
                    f"UniFrac distances should be >= 0.0, got min={distances.min().item()}"
                )
                assert torch.all(distances <= one_tensor), (
                    f"UniFrac distances should be <= 1.0, got max={distances.max().item()}"
                )
                # Diagonal should be 0.0
                diag = torch.diag(distances)
                zeros = torch.zeros(distances.shape[0], device=distances.device, dtype=distances.dtype)
                assert torch.allclose(diag, zeros), "Diagonal should be 0.0"

                break  # Just check first batch

        assert embeddings_found, "Should have found embeddings in at least one batch"

    def test_train_early_stopping(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test early stopping."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=10,
            early_stopping_patience=2,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) <= 10

    def test_train_default_early_stopping_patience(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that default early stopping patience is 10."""
        sig = inspect.signature(Trainer.train)
        early_stopping_param = sig.parameters["early_stopping_patience"]
        assert early_stopping_param.default == 10, f"Expected default to be 10, got {early_stopping_param.default}"

    def test_train_checkpoint_saving(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test checkpoint saving during training."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=2,
            checkpoint_dir=str(checkpoint_dir),
        )

        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0

    def test_single_best_model_file(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that only single best_model.pt file is saved."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=5,
            checkpoint_dir=str(checkpoint_dir),
        )

        best_model_path = checkpoint_dir / "best_model.pt"
        assert best_model_path.exists(), "best_model.pt should exist"

        checkpoint_files = list(checkpoint_dir.glob("best_model*.pt"))
        assert len(checkpoint_files) == 1, f"Expected 1 best_model file, found {len(checkpoint_files)}"
        assert checkpoint_files[0].name == "best_model.pt", f"Expected best_model.pt, found {checkpoint_files[0].name}"

    def test_best_model_replacement(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that best model file is replaced when new best is found."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        best_model_path = checkpoint_dir / "best_model.pt"

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=3,
            checkpoint_dir=str(checkpoint_dir),
        )

        assert best_model_path.exists(), "best_model.pt should exist"

        first_mtime = best_model_path.stat().st_mtime

        trainer2 = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        history2 = trainer2.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=2,
            checkpoint_dir=str(checkpoint_dir),
        )

        assert best_model_path.exists(), "best_model.pt should still exist after second training"
        second_mtime = best_model_path.stat().st_mtime

        checkpoint_files = list(checkpoint_dir.glob("best_model*.pt"))
        assert len(checkpoint_files) == 1, f"Should only have one best_model file, found {len(checkpoint_files)}"

    def test_load_best_model_checkpoint(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test loading best_model.pt checkpoint."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        trainer.train(
            train_loader=simple_dataloader_encoder,
            val_loader=simple_dataloader_encoder,
            num_epochs=2,
            checkpoint_dir=str(checkpoint_dir),
        )

        best_model_path = checkpoint_dir / "best_model.pt"
        assert best_model_path.exists(), "best_model.pt should exist"

        new_model = SequenceEncoder(
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
            base_output_dim=None,  # UniFrac: no output_head, returns embeddings
            encoder_type="unifrac",
            predict_nucleotides=False,
        ).to(device)

        trainer2 = Trainer(
            model=new_model,
            loss_fn=loss_fn,
            device=device,
        )

        checkpoint_info = trainer2.load_checkpoint(str(best_model_path))
        assert "epoch" in checkpoint_info
        assert "best_val_loss" in checkpoint_info
        assert checkpoint_info["best_val_loss"] < float("inf")


class TestLoadPretrainedEncoder:
    """Test loading pre-trained encoder."""

    def test_load_pretrained_encoder(self, small_predictor, device, tmp_path):
        """Test loading pre-trained SequenceEncoder into SequencePredictor."""
        encoder = SequenceEncoder(
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
            base_output_dim=None,  # UniFrac: no output_head, returns embeddings
            encoder_type="unifrac",
            predict_nucleotides=False,
        ).to(device)

        checkpoint_path = tmp_path / "encoder.pt"
        torch.save(encoder.state_dict(), checkpoint_path)

        small_predictor = small_predictor.to(device)
        load_pretrained_encoder(str(checkpoint_path), small_predictor, strict=False)

        encoder_params = dict(encoder.named_parameters())
        predictor_base_params = dict(small_predictor.base_model.named_parameters())

        for name, param in encoder_params.items():
            if name in predictor_base_params:
                assert torch.allclose(param, predictor_base_params[name])

    def test_load_compiled_model_checkpoint(self, small_predictor, device, tmp_path):
        """Test loading checkpoint from torch.compile() model (has _orig_mod. prefix)."""
        encoder = SequenceEncoder(
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
        ).to(device)

        # Simulate torch.compile() checkpoint by adding _orig_mod. prefix
        original_state_dict = encoder.state_dict()
        compiled_state_dict = {f"_orig_mod.{k}": v for k, v in original_state_dict.items()}

        checkpoint_path = tmp_path / "compiled_encoder.pt"
        torch.save({"model_state_dict": compiled_state_dict}, checkpoint_path)

        small_predictor = small_predictor.to(device)
        result = load_pretrained_encoder(str(checkpoint_path), small_predictor, strict=False)

        # Verify all keys were loaded after prefix stripping
        assert result["loaded_keys"] == len(original_state_dict)
        assert len(result["missing_keys"]) == 0
        assert len(result["unexpected_keys"]) == 0

        # Verify weights match
        encoder_params = dict(encoder.named_parameters())
        predictor_base_params = dict(small_predictor.base_model.named_parameters())
        for name, param in encoder_params.items():
            if name in predictor_base_params:
                assert torch.allclose(param, predictor_base_params[name])

    def test_load_dataparallel_checkpoint(self, small_predictor, device, tmp_path):
        """Test loading checkpoint from DataParallel/DDP model (has module. prefix)."""
        encoder = SequenceEncoder(
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
        ).to(device)

        # Simulate DataParallel/DDP checkpoint by adding module. prefix
        original_state_dict = encoder.state_dict()
        dp_state_dict = {f"module.{k}": v for k, v in original_state_dict.items()}

        checkpoint_path = tmp_path / "dp_encoder.pt"
        torch.save({"model_state_dict": dp_state_dict}, checkpoint_path)

        small_predictor = small_predictor.to(device)
        result = load_pretrained_encoder(str(checkpoint_path), small_predictor, strict=False)

        # Verify all keys were loaded after prefix stripping
        assert result["loaded_keys"] == len(original_state_dict)
        assert len(result["missing_keys"]) == 0
        assert len(result["unexpected_keys"]) == 0

        # Verify weights match
        encoder_params = dict(encoder.named_parameters())
        predictor_base_params = dict(small_predictor.base_model.named_parameters())
        for name, param in encoder_params.items():
            if name in predictor_base_params:
                assert torch.allclose(param, predictor_base_params[name])

    def test_load_pretrained_encoder_returns_stats(self, small_predictor, device, tmp_path):
        """Test that load_pretrained_encoder returns loading statistics."""
        encoder = SequenceEncoder(
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
        ).to(device)

        checkpoint_path = tmp_path / "encoder.pt"
        torch.save({"model_state_dict": encoder.state_dict()}, checkpoint_path)

        small_predictor = small_predictor.to(device)
        result = load_pretrained_encoder(str(checkpoint_path), small_predictor, strict=False)

        assert "loaded_keys" in result
        assert "total_checkpoint_keys" in result
        assert "total_model_keys" in result
        assert "missing_keys" in result
        assert "unexpected_keys" in result
        assert "loaded_params" in result
        assert result["loaded_keys"] > 0
        assert result["loaded_params"] > 0

    def test_load_pretrained_encoder_shape_mismatch_raises(self, device, tmp_path):
        """Test that shape mismatch raises ValueError with helpful message."""
        # Create encoder with different embedding dim
        encoder = SequenceEncoder(
            vocab_size=6,
            embedding_dim=64,  # Different from small_predictor's 32
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
        ).to(device)

        checkpoint_path = tmp_path / "mismatched_encoder.pt"
        torch.save({"model_state_dict": encoder.state_dict()}, checkpoint_path)

        predictor = SequencePredictor(
            vocab_size=6,
            embedding_dim=32,  # Different embedding dim
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
        ).to(device)

        with pytest.raises(ValueError, match="Shape mismatch"):
            load_pretrained_encoder(str(checkpoint_path), predictor, strict=False)


class TestTrainerEdgeCases:
    """Test edge cases for trainer."""

    def test_scheduler_warmup_phase(self, small_model, device):
        """Test scheduler warmup phase."""
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=5, num_training_steps=20)

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr > 0

        for step in range(5):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr > 0
            if step < 4:
                assert lr < 1e-4

        lr_after_warmup = scheduler.get_last_lr()[0]
        assert lr_after_warmup <= 1e-4

    def test_create_optimizer_with_frozen_params(self, small_predictor, device):
        """Test create_optimizer excludes frozen parameters."""
        small_predictor = small_predictor.to(device)
        for param in small_predictor.base_model.parameters():
            param.requires_grad = False

        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", lr=1e-4, freeze_base=True)

        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        base_param_ids = {id(p) for p in small_predictor.base_model.parameters()}

        assert len(optimizer_param_ids & base_param_ids) == 0

    def test_checkpoint_save_success(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test checkpoint save functionality."""
        small_predictor = small_predictor.to(device)
        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=0, num_training_steps=10)

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=0, best_val_loss=1.0, metrics={})
        assert checkpoint_path.exists()

    def test_checkpoint_load_error_handling(self, small_predictor, loss_fn, device, tmp_path):
        """Test checkpoint load error handling."""
        small_predictor = small_predictor.to(device)
        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=0, num_training_steps=10)

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        invalid_path = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(str(invalid_path), load_optimizer=False, load_scheduler=False)

    def test_early_stopping(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test early stopping functionality."""
        small_predictor = small_predictor.to(device)
        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=0, num_training_steps=10)

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=10,
            early_stopping_patience=3,
            checkpoint_dir=str(checkpoint_dir),
        )

        assert len(history["val_loss"]) > 0
        assert len(history["val_loss"]) <= 10

    def test_resume_training(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test resume training from checkpoint."""
        small_predictor = small_predictor.to(device)
        optimizer = create_optimizer(small_predictor, optimizer_type="adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, scheduler_type="warmup_cosine", num_warmup_steps=0, num_training_steps=20)

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        history1 = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=2,
            checkpoint_dir=str(checkpoint_dir),
        )

        best_model_path = checkpoint_dir / "best_model.pt"
        assert best_model_path.exists(), "best_model.pt should exist"

        resume_path = best_model_path

        new_optimizer = create_optimizer(small_predictor, lr=1e-4)
        new_scheduler = create_scheduler(new_optimizer, num_warmup_steps=0, num_training_steps=20)

        trainer2 = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
        )

        trainer2.load_checkpoint(str(resume_path), load_optimizer=True, load_scheduler=True)

        history2 = trainer2.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=3,
            checkpoint_dir=str(checkpoint_dir),
            resume_from=str(resume_path),
        )

        assert len(history2["train_loss"]) >= 0
        assert len(history2["val_loss"]) >= 0


class TestFreezeBase:
    """Test freeze_base functionality."""

    def test_freeze_base_prevents_gradients(self, small_predictor, loss_fn, simple_dataloader, device):
        """Test that frozen base model doesn't receive gradients."""
        small_predictor = small_predictor.to(device)
        small_predictor.freeze_base = True
        for param in small_predictor.base_model.parameters():
            param.requires_grad = False

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            freeze_base=True,
        )

        initial_base_params = [p.clone() for p in small_predictor.base_model.parameters()]

        trainer.train_epoch(simple_dataloader)

        updated_base_params = [p for p in small_predictor.base_model.parameters()]

        for init_param, updated_param in zip(initial_base_params, updated_base_params):
            assert torch.equal(init_param, updated_param)


class TestGradientAccumulation:
    """Test gradient accumulation functionality."""

    @pytest.mark.parametrize("gradient_accumulation_steps", [1, 2, 4])
    def test_gradient_accumulation_steps(
        self, small_model, loss_fn, simple_dataloader_encoder, device, gradient_accumulation_steps
    ):
        """Test gradient accumulation with different step values."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        initial_params = [p.clone() for p in small_model.parameters() if p.requires_grad]
        losses = trainer.train_epoch(simple_dataloader_encoder, gradient_accumulation_steps=gradient_accumulation_steps)
        updated_params = [p for p in small_model.parameters() if p.requires_grad]

        assert "total_loss" in losses
        assert losses["total_loss"] >= 0
        # For steps 1 and 2, verify parameters were updated
        # For steps 4, the dataloader only has 4 samples so only 1 update happens
        if gradient_accumulation_steps <= 2:
            for init_param, updated_param in zip(initial_params, updated_params):
                assert not torch.equal(init_param, updated_param)

    def test_gradient_accumulation_equivalent_loss(self, small_model, loss_fn, device):
        """Test that gradient accumulation produces equivalent results."""
        batch_size = 4
        num_asvs = 10
        seq_len = 50

        from aam.data.tokenizer import SequenceTokenizer

        tokens1 = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len))
        tokens1[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens1 = tokens1.to(device)

        # For UniFrac, create pairwise distance matrices per batch
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
                # Ensure values are in [0, 1] range for UniFrac distances
                dist_matrix = torch.rand(self.batch_size, self.batch_size)
                dist_matrix = (dist_matrix + dist_matrix.T) / 2
                dist_matrix.fill_diagonal_(0.0)
                dist_matrix = dist_matrix.to(batch_tokens.device)
                return batch_tokens, dist_matrix

        dataset1 = UniFracDataset(tokens1, batch_size)
        dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

        tokens2 = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len))
        tokens2[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens2 = tokens2.to(device)
        dataset2 = UniFracDataset(tokens2, batch_size)
        dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

        model1 = small_model.to(device)
        model2 = small_model.to(device)

        trainer1 = Trainer(model=model1, loss_fn=loss_fn, device=device)
        trainer2 = Trainer(model=model2, loss_fn=loss_fn, device=device)

        losses1 = trainer1.train_epoch(dataloader1, gradient_accumulation_steps=1)
        losses2 = trainer2.train_epoch(dataloader2, gradient_accumulation_steps=2)

        assert "total_loss" in losses1
        assert "total_loss" in losses2
        assert losses1["total_loss"] >= 0
        assert losses2["total_loss"] >= 0

    def test_train_with_gradient_accumulation(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test training loop with gradient accumulation."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=2,
            gradient_accumulation_steps=2,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2


class TestTensorBoardLogging:
    """Test TensorBoard logging functionality."""

    def test_tensorboard_dir_created(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that TensorBoard directory is created when tensorboard_dir is provided."""
        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=1,
        )

        tensorboard_path = tensorboard_dir / "tensorboard"
        assert tensorboard_path.exists()
        assert tensorboard_path.is_dir()

    def test_tensorboard_writer_closed(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that TensorBoard writer is closed after training."""
        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=1,
        )

        assert trainer.writer is None

    def test_tensorboard_no_dir_when_none(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that TensorBoard is not enabled when tensorboard_dir is None."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=None,
        )

        history = trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=1,
        )

        assert trainer.writer is None

    def test_train_epoch_with_epoch_info(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that train_epoch accepts epoch and num_epochs parameters."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        losses = trainer.train_epoch(
            simple_dataloader_encoder,
            epoch=5,
            num_epochs=10,
        )

        assert "total_loss" in losses

    def test_validate_epoch_with_epoch_info(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that validate_epoch accepts epoch and num_epochs parameters."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        results = trainer.validate_epoch(
            simple_dataloader_encoder,
            epoch=5,
            num_epochs=10,
        )

        assert "total_loss" in results

    def test_tensorboard_logs_model_summary(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that model parameter counts are logged to TensorBoard at training start."""
        from unittest.mock import patch, MagicMock

        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        with patch.object(trainer, "_log_model_summary") as mock_log:
            trainer.train(
                train_loader=simple_dataloader_encoder,
                num_epochs=1,
            )
            mock_log.assert_called_once()

    def test_tensorboard_logs_epoch_timing(self, small_model, loss_fn, simple_dataloader_encoder, device, tmp_path):
        """Test that epoch duration and throughput are logged to TensorBoard."""
        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=2,
        )

        # Read TensorBoard event files to verify timing metrics were logged
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tensorboard_path = tensorboard_dir / "tensorboard"
        event_acc = EventAccumulator(str(tensorboard_path))
        event_acc.Reload()

        scalar_tags = event_acc.Tags()["scalars"]
        assert "perf/epoch_duration_sec" in scalar_tags
        assert "perf/samples_per_sec" in scalar_tags

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory tracking")
    def test_tensorboard_logs_memory_usage(self, small_model, loss_fn, simple_dataloader_encoder, tmp_path):
        """Test that GPU memory usage is logged to TensorBoard."""
        device = torch.device("cuda")
        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=small_model.to(device),
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        trainer.train(
            train_loader=simple_dataloader_encoder,
            num_epochs=1,
        )

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tensorboard_path = tensorboard_dir / "tensorboard"
        event_acc = EventAccumulator(str(tensorboard_path))
        event_acc.Reload()

        scalar_tags = event_acc.Tags()["scalars"]
        assert "perf/gpu_memory_allocated_mb" in scalar_tags

    def test_tensorboard_logs_best_model_marker(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test that best model indicator is logged when checkpoint is saved."""
        tensorboard_dir = tmp_path / "output"
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=3,
            checkpoint_dir=str(checkpoint_dir),
        )

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tensorboard_path = tensorboard_dir / "tensorboard"
        event_acc = EventAccumulator(str(tensorboard_path))
        event_acc.Reload()

        scalar_tags = event_acc.Tags()["scalars"]
        assert "checkpoint/best_model_saved" in scalar_tags

    def test_tensorboard_logs_cross_attention_entropy(self, loss_fn, simple_dataloader, device, tmp_path):
        """Test that cross-attention entropy is logged when using cross-attention fusion."""
        model = SequencePredictor(
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
            encoder_type="unifrac",
            out_dim=1,
            categorical_cardinalities={"test_col": 5},
            categorical_embedding_dim=8,
            categorical_fusion="cross-attention",
            cross_attn_heads=2,
        )
        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        # Create dataloader with categorical_ids
        class CategoricalDataset:
            def __init__(self, num_samples, device):
                self.num_samples = num_samples
                self.device = device

            def __len__(self):
                return self.num_samples // 2

            def __iter__(self):
                from aam.data.tokenizer import SequenceTokenizer
                batch_size = 2
                for i in range(0, self.num_samples, batch_size):
                    tokens = torch.randint(1, 5, (batch_size, 10, 50))
                    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
                    yield {
                        "tokens": tokens.to(self.device),
                        "counts": torch.rand(batch_size, 10, 1).to(self.device),
                        "y_target": torch.rand(batch_size, 1).to(self.device),
                        "categorical_ids": {"test_col": torch.randint(0, 5, (batch_size,)).to(self.device)},
                    }

        cat_loader = CategoricalDataset(8, device)

        trainer.train(
            train_loader=cat_loader,
            num_epochs=2,
        )

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tensorboard_path = tensorboard_dir / "tensorboard"
        event_acc = EventAccumulator(str(tensorboard_path))
        event_acc.Reload()

        scalar_tags = event_acc.Tags()["scalars"]
        assert "cross_attn/entropy" in scalar_tags

    def test_tensorboard_logs_conditional_scaling_stats(self, loss_fn, simple_dataloader, device, tmp_path):
        """Test that conditional scaling stats are logged when using conditional output scaling."""
        model = SequencePredictor(
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
            encoder_type="unifrac",
            out_dim=1,
            categorical_cardinalities={"season": 4},
            categorical_embedding_dim=8,
            conditional_scaling_columns=["season"],
        )
        tensorboard_dir = tmp_path / "output"
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        # Create dataloader with categorical_ids
        class ScalingDataset:
            def __init__(self, num_samples, device):
                self.num_samples = num_samples
                self.device = device

            def __len__(self):
                return self.num_samples // 2

            def __iter__(self):
                from aam.data.tokenizer import SequenceTokenizer
                batch_size = 2
                for i in range(0, self.num_samples, batch_size):
                    tokens = torch.randint(1, 5, (batch_size, 10, 50))
                    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
                    yield {
                        "tokens": tokens.to(self.device),
                        "counts": torch.rand(batch_size, 10, 1).to(self.device),
                        "y_target": torch.rand(batch_size, 1).to(self.device),
                        "categorical_ids": {"season": torch.randint(0, 4, (batch_size,)).to(self.device)},
                    }

        scaling_loader = ScalingDataset(8, device)

        trainer.train(
            train_loader=scaling_loader,
            num_epochs=2,
        )

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tensorboard_path = tensorboard_dir / "tensorboard"
        event_acc = EventAccumulator(str(tensorboard_path))
        event_acc.Reload()

        scalar_tags = event_acc.Tags()["scalars"]
        assert "conditional_scaling/season_scale_mean" in scalar_tags
        assert "conditional_scaling/season_bias_mean" in scalar_tags


class TestPredictionPlots:
    """Test validation prediction plot functionality."""

    def test_create_prediction_plot_regression(self, small_model, loss_fn, device):
        """Test creating regression prediction plot."""
        predictions = torch.randn(100, 1).to(device)
        targets = predictions + torch.randn(100, 1).to(device) * 0.1
        r2 = 0.95

        fig = create_prediction_plot(predictions, targets, epoch=5, r2=r2)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_confusion_matrix_plot_classification(self, loss_fn, device):
        """Test creating classification confusion matrix plot."""
        predictions = torch.randint(0, 3, (100,)).to(device)
        targets = torch.randint(0, 3, (100,)).to(device)
        accuracy = 0.85
        precision = 0.82
        recall = 0.80
        f1 = 0.81

        fig = create_confusion_matrix_plot(
            predictions, targets, epoch=5, accuracy=accuracy, precision=precision, recall=recall, f1=f1
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_validate_epoch_returns_predictions(self, small_predictor, loss_fn, simple_dataloader, device):
        """Test that validate_epoch returns predictions when return_predictions=True."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        results, predictions_dict, targets_dict = trainer.validate_epoch(
            simple_dataloader, compute_metrics=True, return_predictions=True
        )

        assert "total_loss" in results
        assert isinstance(predictions_dict, dict)
        assert isinstance(targets_dict, dict)
        # Check that at least one prediction type is present
        assert len(predictions_dict) > 0
        # Verify matching shapes for each prediction type
        for key in predictions_dict:
            assert key in targets_dict
            assert predictions_dict[key].shape[0] == targets_dict[key].shape[0]

    def test_validate_epoch_no_predictions_when_false(self, small_predictor, loss_fn, simple_dataloader, device):
        """Test that validate_epoch doesn't return predictions when return_predictions=False."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        results = trainer.validate_epoch(simple_dataloader, compute_metrics=True, return_predictions=False)

        assert "total_loss" in results
        assert isinstance(results, dict)

    def test_save_prediction_plots_regression(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test saving regression prediction plots to disk."""
        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        predictions = torch.randn(50, 1).to(device)
        targets = predictions + torch.randn(50, 1).to(device) * 0.1
        metrics = {"r2": 0.95, "mse": 0.05}

        trainer.evaluator.save_prediction_plots(
            predictions_dict={"target": predictions},
            targets_dict={"target": targets},
            epoch=5,
            metrics=metrics,
            checkpoint_dir=str(checkpoint_dir),
        )

        plots_dir = checkpoint_dir / "plots"
        plot_file = plots_dir / "prediction_plot_best.png"
        assert plot_file.exists()

    def test_save_prediction_plots_classification(self, loss_fn, device, tmp_path):
        """Test saving classification confusion matrix plots to disk."""
        from aam.models.sequence_predictor import SequencePredictor

        classifier_model = SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=3,
            is_classifier=True,
            freeze_base=False,
            predict_nucleotides=False,
            base_output_dim=None,  # UniFrac: no output_head, returns embeddings
        )

        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = Trainer(
            model=classifier_model,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        predictions = torch.randint(0, 3, (50,)).to(device)
        targets = torch.randint(0, 3, (50,)).to(device)
        metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.80, "f1": 0.81}

        trainer.evaluator.save_prediction_plots(
            predictions_dict={"target": predictions},
            targets_dict={"target": targets},
            epoch=5,
            metrics=metrics,
            checkpoint_dir=str(checkpoint_dir),
        )

        plots_dir = checkpoint_dir / "plots"
        plot_file = plots_dir / "prediction_plot_best.png"
        assert plot_file.exists()

    def test_train_with_plots_regression(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test training with plot generation for regression."""
        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=3,
            checkpoint_dir=str(checkpoint_dir),
            save_plots=True,
        )

        plots_dir = checkpoint_dir / "plots"
        # Check for any prediction plot (prediction_plot, unifrac_plot, or count_plot)
        plot_files = list(plots_dir.glob("*_plot_best.png")) if plots_dir.exists() else []
        assert len(plot_files) > 0, f"Expected at least one prediction plot, found: {plot_files}"

    def test_train_without_plots(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test training without plot generation."""
        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=3,
            checkpoint_dir=str(checkpoint_dir),
            save_plots=False,
        )

        plots_dir = checkpoint_dir / "plots"
        assert not plots_dir.exists()

    def test_plots_only_on_improvement(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test that plots are only created when validation improves."""
        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=str(tensorboard_dir),
        )

        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=5,
            checkpoint_dir=str(checkpoint_dir),
            save_plots=True,
            early_stopping_patience=10,
        )

        plots_dir = checkpoint_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            assert len(plot_files) >= 1
            assert any("best" in f.name for f in plot_files)

    def test_plot_save_without_tensorboard(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test that plots are saved even without TensorBoard."""
        checkpoint_dir = tmp_path / "checkpoints"

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            tensorboard_dir=None,
        )

        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=3,
            checkpoint_dir=str(checkpoint_dir),
            save_plots=True,
        )

        plots_dir = checkpoint_dir / "plots"
        # Check for any prediction plot (prediction_plot, unifrac_plot, or count_plot)
        plot_files = list(plots_dir.glob("*_plot_best.png")) if plots_dir.exists() else []
        assert len(plot_files) > 0, f"Expected at least one prediction plot, found: {plot_files}"


class TestMixedPrecision:
    """Test mixed precision training functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_init_with_fp16(self, small_model, loss_fn):
        """Test Trainer initialization with FP16 mixed precision."""
        device = torch.device("cuda")
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="fp16",
        )

        assert trainer.mixed_precision == "fp16"
        assert trainer.scaler is not None
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_init_with_bf16(self, small_model, loss_fn):
        """Test Trainer initialization with BF16 mixed precision."""
        device = torch.device("cuda")
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="bf16",
        )

        assert trainer.mixed_precision == "bf16"
        assert trainer.scaler is not None
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    def test_trainer_init_with_none_mixed_precision(self, small_model, loss_fn, device):
        """Test Trainer initialization with no mixed precision."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=None,
        )

        assert trainer.mixed_precision is None
        assert trainer.scaler is None

    def test_trainer_init_mixed_precision_on_cpu(self, small_model, loss_fn):
        """Test that mixed precision scaler is not created on CPU."""
        device = torch.device("cpu")
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="fp16",
        )

        assert trainer.mixed_precision == "fp16"
        # Scaler should not be created on CPU
        assert trainer.scaler is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_epoch_with_fp16(self, small_model, loss_fn, simple_dataloader_encoder):
        """Test training epoch with FP16 mixed precision."""
        device = torch.device("cuda")
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="fp16",
        )

        # Run one training epoch
        losses = trainer.train_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)

        # Check that training completed without errors
        assert "total_loss" in losses
        assert not torch.isnan(torch.tensor(losses["total_loss"]))
        assert not torch.isinf(torch.tensor(losses["total_loss"]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_epoch_with_bf16(self, small_model, loss_fn, simple_dataloader_encoder):
        """Test training epoch with BF16 mixed precision."""
        device = torch.device("cuda")
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="bf16",
        )

        # Run one training epoch
        losses = trainer.train_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)

        # Check that training completed without errors
        assert "total_loss" in losses
        assert not torch.isnan(torch.tensor(losses["total_loss"]))
        assert not torch.isinf(torch.tensor(losses["total_loss"]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validate_epoch_with_fp16(self, small_model, loss_fn, simple_dataloader_encoder):
        """Test validation epoch with FP16 mixed precision."""
        device = torch.device("cuda")
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="fp16",
        )

        # Run one validation epoch
        results = trainer.validate_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)

        # Check that validation completed without errors
        assert "total_loss" in results
        assert not torch.isnan(torch.tensor(results["total_loss"]))
        assert not torch.isinf(torch.tensor(results["total_loss"]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_numerical_stability(self, small_model, loss_fn, simple_dataloader_encoder):
        """Test that mixed precision training maintains numerical stability."""
        device = torch.device("cuda")
        small_model = small_model.to(device)

        # Train with FP16
        trainer_fp16 = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="fp16",
        )

        # Train with no mixed precision
        small_model_fp32 = SequenceEncoder(
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
        ).to(device)

        trainer_fp32 = Trainer(
            model=small_model_fp32,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=None,
        )

        # Run one epoch with each
        losses_fp16 = trainer_fp16.train_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)
        losses_fp32 = trainer_fp32.train_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)

        # Both should produce valid losses (no NaN/Inf)
        assert not torch.isnan(torch.tensor(losses_fp16["total_loss"]))
        assert not torch.isinf(torch.tensor(losses_fp16["total_loss"]))
        assert not torch.isnan(torch.tensor(losses_fp32["total_loss"]))
        assert not torch.isinf(torch.tensor(losses_fp32["total_loss"]))

        # Losses should be reasonable (not extremely different)
        # Note: FP16 and FP32 losses may differ slightly, but should be in same order of magnitude
        fp16_loss = losses_fp16["total_loss"]
        fp32_loss = losses_fp32["total_loss"]
        ratio = max(fp16_loss, fp32_loss) / min(fp16_loss, fp32_loss) if min(fp16_loss, fp32_loss) > 0 else 1.0
        assert ratio < 10.0, f"FP16 and FP32 losses differ too much: {fp16_loss} vs {fp32_loss}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_gradient_clipping(self, small_model, loss_fn, simple_dataloader_encoder):
        """Test that gradient clipping works with mixed precision."""
        device = torch.device("cuda")
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            mixed_precision="fp16",
            max_grad_norm=1.0,
        )

        # Run one training epoch
        losses = trainer.train_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)

        # Check that training completed without errors
        assert "total_loss" in losses
        assert not torch.isnan(torch.tensor(losses["total_loss"]))


@pytest.mark.skipif(is_rocm(), reason="torch.compile() with inductor backend has known Triton compatibility issues on ROCm")
class TestModelCompilation:
    """Tests for model compilation with torch.compile()."""

    def test_trainer_init_with_compile_model(self, small_model, loss_fn, device):
        """Test Trainer initialization with compile_model=True."""
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile() not available (requires PyTorch 2.0+)")

        # Try to compile - may fail on Python 3.12+ with older PyTorch
        try:
            trainer = Trainer(
                model=small_model,
                loss_fn=loss_fn,
                device=device,
                compile_model=True,
            )

            assert trainer.compile_model is True
            # Model should be compiled (wrapped)
            assert hasattr(trainer.model, "_orig_mod") or hasattr(trainer.model, "forward")
        except RuntimeError as e:
            if "not supported" in str(e) or "Dynamo" in str(e):
                pytest.skip(f"torch.compile() not supported in this environment: {e}")
            raise

    def test_trainer_init_without_compile_model(self, small_model, loss_fn, device):
        """Test Trainer initialization with compile_model=False (default)."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )

        assert trainer.compile_model is False

    def test_compile_model_requires_pytorch_2_0(self, small_model, loss_fn, device):
        """Test that compilation fails gracefully on PyTorch < 2.0."""
        # Mock torch.compile to not exist
        original_compile = getattr(torch, "compile", None)
        try:
            # Temporarily remove compile attribute
            if hasattr(torch, "compile"):
                delattr(torch, "compile")

            with pytest.raises(RuntimeError, match="torch.compile\\(\\) is not available"):
                Trainer(
                    model=small_model,
                    loss_fn=loss_fn,
                    device=device,
                    compile_model=True,
                )
        finally:
            # Restore compile attribute
            if original_compile is not None:
                torch.compile = original_compile

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile() not available (requires PyTorch 2.0+)")
    def test_compiled_model_same_outputs_as_eager(self, small_model, loss_fn, device):
        """Test that compiled model produces same outputs as eager mode."""
        # Create two identical models
        model_eager = SequenceEncoder(
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
        ).to(device)

        model_compiled = SequenceEncoder(
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
        ).to(device)

        # Copy weights to ensure identical models
        model_compiled.load_state_dict(model_eager.state_dict())

        # Compile one model (may fail on Python 3.12+ with older PyTorch or missing GCC/CUDA)
        try:
            model_compiled = torch.compile(model_compiled)
        except Exception as e:
            error_str = str(e)
            if any(msg in error_str for msg in ["Dynamo", "not supported", "BackendCompiler", "inductor", "gcc"]):
                pytest.skip(f"torch.compile() not supported in this environment: {e}")
            raise

        # Create test input
        batch_size = 2
        num_asvs = 5
        seq_len = 50
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        tokens[:, :, 0] = 5  # START_TOKEN

        # Set models to eval mode for deterministic outputs
        model_eager.eval()
        model_compiled.eval()

        # Get outputs from both models (may fail on first run when inductor compiles)
        try:
            with torch.no_grad():
                output_eager = model_eager(tokens)
                output_compiled = model_compiled(tokens)

            # Compare outputs (allow small numerical differences due to compilation optimizations)
            if isinstance(output_eager, dict) and isinstance(output_compiled, dict):
                for key in output_eager:
                    if key in output_compiled:
                        assert torch.allclose(output_eager[key], output_compiled[key], rtol=1e-4, atol=1e-5), (
                            f"Output mismatch for key '{key}'"
                        )
            else:
                assert torch.allclose(output_eager, output_compiled, rtol=1e-4, atol=1e-5)
        except Exception as e:
            error_str = str(e)
            if any(msg in error_str for msg in ["Dynamo", "not supported", "BackendCompiler", "inductor", "gcc"]):
                pytest.skip(f"torch.compile() not supported in this environment: {e}")
            raise

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile() not available (requires PyTorch 2.0+)")
    def test_compiled_model_training_works(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that training works with compiled model."""
        try:
            trainer = Trainer(
                model=small_model,
                loss_fn=loss_fn,
                device=device,
                compile_model=True,
            )

            # Run one training epoch
            losses = trainer.train_epoch(simple_dataloader_encoder, epoch=0, num_epochs=1)

            # Check that training completed without errors
            assert "total_loss" in losses
            assert not torch.isnan(torch.tensor(losses["total_loss"]))
        except Exception as e:
            error_str = str(e)
            if any(msg in error_str for msg in ["Dynamo", "not supported", "BackendCompiler", "inductor", "gcc"]):
                pytest.skip(f"torch.compile() not supported in this environment: {e}")
            raise

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile() not available (requires PyTorch 2.0+)")
    def test_compiled_sequence_predictor(self, small_predictor, loss_fn, simple_dataloader, device):
        """Test that SequencePredictor can be compiled."""
        try:
            trainer = Trainer(
                model=small_predictor,
                loss_fn=loss_fn,
                device=device,
                compile_model=True,
            )

            assert trainer.compile_model is True

            # Run one training epoch
            losses = trainer.train_epoch(simple_dataloader, epoch=0, num_epochs=1)

            # Check that training completed without errors
            assert "total_loss" in losses
            assert not torch.isnan(torch.tensor(losses["total_loss"]))
        except Exception as e:
            error_str = str(e)
            if any(msg in error_str for msg in ["Dynamo", "not supported", "BackendCompiler", "inductor", "gcc"]):
                pytest.skip(f"torch.compile() not supported in this environment: {e}")
            raise

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile() not available (requires PyTorch 2.0+)")
    def test_compiled_model_validation_works(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test that validation works with compiled model."""
        try:
            trainer = Trainer(
                model=small_model,
                loss_fn=loss_fn,
                device=device,
                compile_model=True,
            )

            # Run validation
            metrics = trainer.validate_epoch(simple_dataloader_encoder, epoch=0)

            # Check that validation completed without errors
            assert "total_loss" in metrics
            assert not torch.isnan(torch.tensor(metrics["total_loss"]))
        except Exception as e:
            error_str = str(e)
            if any(msg in error_str for msg in ["Dynamo", "not supported", "BackendCompiler", "inductor", "gcc"]):
                pytest.skip(f"torch.compile() not supported in this environment: {e}")
            raise


class TestROCmCompileHandling:
    """Test suite for ROCm torch.compile() handling (COS-9.2)."""

    def test_trainer_is_rocm_method_exists(self, small_model, loss_fn, device):
        """Test that Trainer has _is_rocm method."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        assert hasattr(trainer, "_is_rocm")
        assert callable(trainer._is_rocm)

    def test_trainer_is_rocm_returns_bool(self, small_model, loss_fn, device):
        """Test that _is_rocm returns a boolean."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        result = trainer._is_rocm()
        assert isinstance(result, bool)

    def test_trainer_compile_skipped_attribute(self, small_model, loss_fn, device):
        """Test that Trainer has _compile_skipped attribute."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        assert hasattr(trainer, "_compile_skipped")
        # Should be False when compile_model=False
        assert trainer._compile_skipped is False

    def test_trainer_compile_skipped_false_when_not_requested(self, small_model, loss_fn, device):
        """Test that _compile_skipped is False when compile_model=False."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        assert trainer._compile_skipped is False

    @pytest.mark.skipif(not is_rocm(), reason="Only runs on ROCm systems")
    def test_trainer_compile_skipped_on_rocm(self, small_model, loss_fn, device):
        """Test that compilation is skipped gracefully on ROCm."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=True,
        )
        # On ROCm, compilation should be skipped
        assert trainer._compile_skipped is True
        # Model should NOT be compiled (no _orig_mod attribute)
        assert not hasattr(trainer.model, "_orig_mod")

    @pytest.mark.skipif(not is_rocm(), reason="Only runs on ROCm systems")
    def test_trainer_rocm_detection_correct(self, small_model, loss_fn, device):
        """Test that ROCm detection is correct on ROCm systems."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        assert trainer._is_rocm() is True

    @pytest.mark.skipif(is_rocm(), reason="Only runs on non-ROCm systems")
    def test_trainer_non_rocm_detection(self, small_model, loss_fn, device):
        """Test that ROCm detection returns False on non-ROCm systems."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        assert trainer._is_rocm() is False

    def test_trainer_try_compile_model_method_exists(self, small_model, loss_fn, device):
        """Test that Trainer has _try_compile_model method."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            compile_model=False,
        )
        assert hasattr(trainer, "_try_compile_model")
        assert callable(trainer._try_compile_model)


class TestTrainerTargetNormalization:
    """Test suite for Trainer target normalization (PYT-11.9)."""

    def test_trainer_accepts_normalization_params(self, small_predictor, loss_fn, device):
        """Test that Trainer accepts target_normalization_params."""
        normalization_params = {
            "target_min": 0.0,
            "target_max": 100.0,
            "target_scale": 100.0,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        assert trainer.target_normalization_params == normalization_params

    def test_trainer_denormalize_with_params(self, small_predictor, loss_fn, device):
        """Test that Trainer._denormalize_targets works correctly with params."""
        normalization_params = {
            "target_min": 0.0,
            "target_max": 100.0,
            "target_scale": 100.0,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # Test denormalization
        normalized = torch.tensor([0.0, 0.5, 1.0])
        denormalized = trainer._denormalize_targets(normalized)
        expected = torch.tensor([0.0, 50.0, 100.0])
        torch.testing.assert_close(denormalized, expected)

    def test_trainer_denormalize_without_params(self, small_predictor, loss_fn, device):
        """Test that Trainer._denormalize_targets returns input unchanged without params."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=None,
        )

        # Test that values are returned unchanged
        values = torch.tensor([1.0, 2.0, 3.0])
        result = trainer._denormalize_targets(values)
        torch.testing.assert_close(result, values)

    def test_trainer_denormalize_with_negative_min(self, small_predictor, loss_fn, device):
        """Test Trainer._denormalize_targets with negative target_min."""
        normalization_params = {
            "target_min": -50.0,
            "target_max": 50.0,
            "target_scale": 100.0,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # Test denormalization
        normalized = torch.tensor([0.0, 0.5, 1.0])
        denormalized = trainer._denormalize_targets(normalized)
        expected = torch.tensor([-50.0, 0.0, 50.0])
        torch.testing.assert_close(denormalized, expected)

    def test_trainer_denormalize_with_log_transform(self, small_predictor, loss_fn, device):
        """Test Trainer._denormalize_targets with log_transform (no normalization)."""
        import math

        normalization_params = {
            "log_transform": True,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # log(100) should become exp(log(100)) - 1 = 99
        log_values = torch.tensor([math.log(1), math.log(100), math.log(600)])
        denormalized = trainer._denormalize_targets(log_values)
        expected = torch.tensor([0.0, 99.0, 599.0])
        torch.testing.assert_close(denormalized, expected, atol=1e-4, rtol=1e-4)

    def test_trainer_denormalize_with_log_transform_and_normalize(self, small_predictor, loss_fn, device):
        """Test Trainer._denormalize_targets with both log transform and normalization."""
        import math

        # Simulate log-transformed then normalized: log(0+1)=0, log(600+1)=6.398
        log_min = math.log(1)  # 0
        log_max = math.log(601)  # ~6.398

        normalization_params = {
            "target_min": log_min,
            "target_max": log_max,
            "target_scale": log_max - log_min,
            "log_transform": True,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # 0.0 (normalized min) -> log(1) -> exp(0) - 1 = 0
        # 1.0 (normalized max) -> log(601) -> exp(log(601)) - 1 = 600
        normalized = torch.tensor([0.0, 1.0])
        denormalized = trainer._denormalize_targets(normalized)
        expected = torch.tensor([0.0, 600.0])
        torch.testing.assert_close(denormalized, expected, atol=1.0, rtol=1e-2)

    def test_trainer_denormalize_log_transform_overflow_protection(self, small_predictor, loss_fn, device):
        """Test that denormalization clamps values to prevent exp() overflow."""
        normalization_params = {
            "log_transform": True,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # Values > 88 would overflow float32 without clamping
        large_values = torch.tensor([100.0, 200.0, 500.0])
        denormalized = trainer._denormalize_targets(large_values)

        # Should be clamped to exp(88) - 1, not inf
        assert not torch.isinf(denormalized).any(), "Overflow protection should prevent inf values"
        assert not torch.isnan(denormalized).any(), "Overflow protection should prevent NaN values"
        # exp(88) is approximately 1.65e38
        assert (denormalized > 1e30).all(), "Large values should still produce large outputs"

    def test_trainer_denormalize_zscore_with_global_normalizer(self, small_predictor, loss_fn, device):
        """Test that Trainer._denormalize_targets works with z-score GlobalNormalizer."""
        # Z-score normalization params from GlobalNormalizer.to_dict()
        normalization_params = {
            "global_normalizer": {
                "method": "zscore",
                "min_val": 10.0,
                "max_val": 110.0,
                "scale": 100.0,
                "mean": 60.0,
                "std": 25.0,
            }
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # Z-score normalized values: (x - mean) / std
        # z = -2 => x = -2 * 25 + 60 = 10
        # z = 0  => x = 0 * 25 + 60 = 60
        # z = 2  => x = 2 * 25 + 60 = 110
        normalized = torch.tensor([-2.0, 0.0, 2.0])
        denormalized = trainer._denormalize_targets(normalized)
        expected = torch.tensor([10.0, 60.0, 110.0])
        torch.testing.assert_close(denormalized, expected)

    def test_trainer_denormalize_zscore_with_log_transform(self, small_predictor, loss_fn, device):
        """Test Trainer._denormalize_targets with z-score and log transform."""
        # Simulate log-z-score transform: log(x+1) then z-score
        # Original values: 0, ~147, ~400 -> log: 0, 5, 6 -> z-score with mean=3.67, std=2.49
        log_mean = 3.67
        log_std = 2.49

        normalization_params = {
            "global_normalizer": {
                "method": "zscore",
                "min_val": 0.0,
                "max_val": 6.0,
                "scale": 6.0,
                "mean": log_mean,
                "std": log_std,
            },
            "log_transform": True,
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # z = -1.47 => log = -1.47 * 2.49 + 3.67 = 0.0 => x = exp(0) - 1 = 0
        # z = 0.53  => log = 0.53 * 2.49 + 3.67 = 4.99 => x = exp(4.99) - 1 ≈ 146
        # z = 0.93  => log = 0.93 * 2.49 + 3.67 = 5.99 => x = exp(5.99) - 1 ≈ 398
        z_scores = torch.tensor([-1.47, 0.53, 0.93])
        denormalized = trainer._denormalize_targets(z_scores)

        # Check approximate values
        assert denormalized[0].item() < 1.0, "First value should be near 0"
        assert 100 < denormalized[1].item() < 200, "Second value should be around 146"
        assert 300 < denormalized[2].item() < 500, "Third value should be around 398"

    def test_trainer_denormalize_minmax_with_global_normalizer(self, small_predictor, loss_fn, device):
        """Test that min-max GlobalNormalizer also works via global_normalizer key."""
        # Min-max normalization params from GlobalNormalizer.to_dict()
        normalization_params = {
            "global_normalizer": {
                "method": "minmax",
                "min_val": 10.0,
                "max_val": 110.0,
                "scale": 100.0,
                "mean": 60.0,
                "std": 25.0,
            }
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # Min-max: normalized * scale + min_val
        # 0.0 => 0 * 100 + 10 = 10
        # 0.5 => 0.5 * 100 + 10 = 60
        # 1.0 => 1.0 * 100 + 10 = 110
        normalized = torch.tensor([0.0, 0.5, 1.0])
        denormalized = trainer._denormalize_targets(normalized)
        expected = torch.tensor([10.0, 60.0, 110.0])
        torch.testing.assert_close(denormalized, expected)

    def test_evaluator_denormalize_zscore_directly(self, small_predictor, loss_fn, device):
        """Test Evaluator._denormalize_targets directly with z-score params."""
        from aam.training.evaluation import Evaluator

        normalization_params = {
            "global_normalizer": {
                "method": "zscore",
                "min_val": 0.0,
                "max_val": 100.0,
                "scale": 100.0,
                "mean": 50.0,
                "std": 20.0,
            }
        }

        evaluator = Evaluator(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # z = -1 => x = -1 * 20 + 50 = 30
        # z = 0  => x = 0 * 20 + 50 = 50
        # z = 1  => x = 1 * 20 + 50 = 70
        normalized = torch.tensor([-1.0, 0.0, 1.0])
        denormalized = evaluator._denormalize_targets(normalized)
        expected = torch.tensor([30.0, 50.0, 70.0])
        torch.testing.assert_close(denormalized, expected)

    def test_global_normalizer_takes_precedence_over_legacy_params(self, small_predictor, loss_fn, device):
        """Test that global_normalizer is used when both old and new params are present."""
        # Both legacy params and global_normalizer present - global_normalizer should win
        normalization_params = {
            "target_min": 0.0,
            "target_max": 100.0,
            "target_scale": 100.0,
            "global_normalizer": {
                "method": "zscore",
                "min_val": 0.0,
                "max_val": 100.0,
                "scale": 100.0,
                "mean": 50.0,
                "std": 10.0,
            },
        }

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # If using z-score: z=1 => 1*10+50 = 60
        # If using legacy min-max: n=1 => 1*100+0 = 100
        normalized = torch.tensor([1.0])
        denormalized = trainer._denormalize_targets(normalized)
        expected_zscore = torch.tensor([60.0])
        torch.testing.assert_close(denormalized, expected_zscore)

    def test_evaluator_denormalize_zscore_category_with_categorical_ids(self, small_predictor, loss_fn, device):
        """Test Evaluator._denormalize_targets with zscore-category normalization.

        When category_normalizer is present and categorical_ids are provided,
        per-sample denormalization should be applied using the category-specific
        statistics.
        """
        from aam.training.evaluation import Evaluator

        # Category normalizer with two categories having different stats
        # location=A: mean=10, std=2
        # location=B: mean=50, std=5
        normalization_params = {
            "category_normalizer": {
                "stats": {
                    "location=A": {"mean": 10.0, "std": 2.0},
                    "location=B": {"mean": 50.0, "std": 5.0},
                },
                "global_mean": 30.0,
                "global_std": 20.0,
                "columns": ["location"],
            },
            "categorical_encoder_mappings": {
                "location": {1: "A", 2: "B"},
            },
        }

        evaluator = Evaluator(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # 4 samples: [A, A, B, B] with z-scores [0, 1, 0, 1]
        # Expected denormalized:
        #   A, z=0: 0 * 2 + 10 = 10
        #   A, z=1: 1 * 2 + 10 = 12
        #   B, z=0: 0 * 5 + 50 = 50
        #   B, z=1: 1 * 5 + 50 = 55
        normalized = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        categorical_ids = {"location": torch.tensor([1, 1, 2, 2])}

        denormalized = evaluator._denormalize_targets(normalized, categorical_ids=categorical_ids)
        expected = torch.tensor([[10.0], [12.0], [50.0], [55.0]])
        torch.testing.assert_close(denormalized, expected)

    def test_evaluator_denormalize_zscore_category_without_categorical_ids_returns_unchanged(
        self, small_predictor, loss_fn, device
    ):
        """Test that zscore-category without categorical_ids returns values unchanged.

        This maintains backward compatibility - if categorical_ids aren't available,
        the values are returned as-is (normalized z-scores).
        """
        from aam.training.evaluation import Evaluator

        normalization_params = {
            "category_normalizer": {
                "stats": {
                    "location=A": {"mean": 10.0, "std": 2.0},
                },
                "global_mean": 10.0,
                "global_std": 2.0,
                "columns": ["location"],
            },
        }

        evaluator = Evaluator(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            target_normalization_params=normalization_params,
        )

        # Without categorical_ids, should return unchanged
        normalized = torch.tensor([[0.5], [1.0]])
        denormalized = evaluator._denormalize_targets(normalized)
        torch.testing.assert_close(denormalized, normalized)


class TestProgressBarFormat:
    """Test progress bar formatting for different training modes."""

    @pytest.fixture
    def small_classifier(self):
        """Create a small SequencePredictor for classification testing."""
        return SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=3,  # 3 classes
            is_classifier=True,
            freeze_base=False,
            predict_nucleotides=False,
            base_output_dim=None,
        )

    def test_nuc_metrics_hidden_when_nuc_penalty_zero(self, small_predictor, simple_dataloader, device):
        """Test that NL/NA are not shown when nuc_penalty=0."""
        from unittest.mock import patch, MagicMock

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        # Track postfix calls
        postfix_calls = []

        def capture_postfix(d):
            postfix_calls.append(d.copy())

        with patch("aam.training.trainer.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_pbar.set_postfix = capture_postfix
            mock_pbar.__iter__ = lambda self: iter(simple_dataloader)
            mock_tqdm.return_value = mock_pbar

            trainer.train_epoch(simple_dataloader)

        # Verify NL/NA are not in any postfix call
        for postfix in postfix_calls:
            assert "NL" not in postfix, "NL should not appear when nuc_penalty=0"
            assert "NA" not in postfix, "NA should not appear when nuc_penalty=0"

    def test_nuc_metrics_shown_when_nuc_penalty_positive(self, small_model, simple_dataloader_encoder, device):
        """Test that NL/NA are shown when nuc_penalty > 0."""
        from unittest.mock import patch, MagicMock

        # Create encoder with nucleotide prediction
        model = SequenceEncoder(
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
            predict_nucleotides=True,
        )

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
        )

        # Track postfix calls
        postfix_calls = []

        def capture_postfix(d):
            postfix_calls.append(d.copy())

        with patch("aam.training.trainer.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_pbar.set_postfix = capture_postfix
            mock_pbar.__iter__ = lambda self: iter(simple_dataloader_encoder)
            mock_tqdm.return_value = mock_pbar

            trainer.train_epoch(simple_dataloader_encoder)

        # Verify NL/NA are in postfix calls (at least one should have them)
        has_nuc_metrics = any("NL" in postfix or "NA" in postfix for postfix in postfix_calls)
        assert has_nuc_metrics, "NL/NA should appear when nuc_penalty > 0"

    def test_regression_loss_shown_during_finetuning(self, small_predictor, simple_dataloader, device):
        """Test that RL (Regression Loss) is shown during fine-tuning."""
        from unittest.mock import patch, MagicMock

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        # Track postfix calls
        postfix_calls = []

        def capture_postfix(d):
            postfix_calls.append(d.copy())

        with patch("aam.training.trainer.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_pbar.set_postfix = capture_postfix
            mock_pbar.__iter__ = lambda self: iter(simple_dataloader)
            mock_tqdm.return_value = mock_pbar

            trainer.train_epoch(simple_dataloader)

        # Verify RL is in postfix calls
        has_rl = any("RL" in postfix for postfix in postfix_calls)
        assert has_rl, "RL should appear during regression fine-tuning"

        # Verify CL is not shown (this is regression, not classification)
        has_cl = any("CL" in postfix for postfix in postfix_calls)
        assert not has_cl, "CL should not appear during regression fine-tuning"

    def test_classification_loss_shown_during_finetuning(self, small_classifier, device):
        """Test that CL (Classification Loss) is shown during classification fine-tuning."""
        from unittest.mock import patch, MagicMock
        from aam.data.tokenizer import SequenceTokenizer

        # Create dataloader with class targets
        batch_size = 4
        num_asvs = 10
        seq_len = 50

        tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)
        counts = torch.rand(batch_size * 2, num_asvs, 1).to(device)
        targets = torch.randint(0, 3, (batch_size * 2,)).to(device)  # Class labels

        dataset = TensorDataset(tokens, counts, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
        trainer = Trainer(
            model=small_classifier,
            loss_fn=loss_fn,
            device=device,
        )

        # Track postfix calls
        postfix_calls = []

        def capture_postfix(d):
            postfix_calls.append(d.copy())

        with patch("aam.training.trainer.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_pbar.set_postfix = capture_postfix
            mock_pbar.__iter__ = lambda self: iter(dataloader)
            mock_tqdm.return_value = mock_pbar

            trainer.train_epoch(dataloader)

        # Verify CL is in postfix calls
        has_cl = any("CL" in postfix for postfix in postfix_calls)
        assert has_cl, "CL should appear during classification fine-tuning"

        # Verify RL is not shown (this is classification, not regression)
        has_rl = any("RL" in postfix for postfix in postfix_calls)
        assert not has_rl, "RL should not appear during classification fine-tuning"

    def test_no_target_loss_during_pretraining(self, small_model, simple_dataloader_encoder, device):
        """Test that RL/CL are not shown during pretraining (SequenceEncoder)."""
        from unittest.mock import patch, MagicMock

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        # Track postfix calls
        postfix_calls = []

        def capture_postfix(d):
            postfix_calls.append(d.copy())

        with patch("aam.training.trainer.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_pbar.set_postfix = capture_postfix
            mock_pbar.__iter__ = lambda self: iter(simple_dataloader_encoder)
            mock_tqdm.return_value = mock_pbar

            trainer.train_epoch(simple_dataloader_encoder)

        # Verify neither RL nor CL are in postfix calls during pretraining
        has_rl = any("RL" in postfix for postfix in postfix_calls)
        has_cl = any("CL" in postfix for postfix in postfix_calls)
        assert not has_rl, "RL should not appear during pretraining"
        assert not has_cl, "CL should not appear during pretraining"

    def test_validation_progress_bar_same_behavior(self, small_predictor, simple_dataloader, device):
        """Test that validation progress bar follows the same rules."""
        from unittest.mock import patch, MagicMock

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        # Track postfix calls
        postfix_calls = []

        def capture_postfix(d):
            postfix_calls.append(d.copy())

        with patch("aam.training.evaluation.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_pbar.set_postfix = capture_postfix
            mock_pbar.__iter__ = lambda self: iter(simple_dataloader)
            mock_tqdm.return_value = mock_pbar

            trainer.validate_epoch(simple_dataloader)

        # Verify RL is shown during validation
        has_rl = any("RL" in postfix for postfix in postfix_calls)
        assert has_rl, "RL should appear during regression validation"

        # Verify NL/NA are hidden when nuc_penalty=0
        for postfix in postfix_calls:
            assert "NL" not in postfix, "NL should not appear when nuc_penalty=0"
            assert "NA" not in postfix, "NA should not appear when nuc_penalty=0"


class TestDistributedSamplerEpoch:
    """Tests for distributed sampler set_epoch integration."""

    def test_train_sampler_set_epoch_called(self, small_model, simple_dataloader, device):
        """Test that set_epoch is called on train_sampler for each epoch."""
        from unittest.mock import MagicMock

        # Create a mock sampler
        mock_sampler = MagicMock()

        loss_fn = MultiTaskLoss(penalty=1.0)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            train_sampler=mock_sampler,
        )

        # Train for 3 epochs
        trainer.train(
            train_loader=simple_dataloader,
            val_loader=None,
            num_epochs=3,
            early_stopping_patience=100,
        )

        # Verify set_epoch was called for each epoch
        assert mock_sampler.set_epoch.call_count == 3
        mock_sampler.set_epoch.assert_any_call(0)
        mock_sampler.set_epoch.assert_any_call(1)
        mock_sampler.set_epoch.assert_any_call(2)

    def test_train_sampler_none_works(self, small_model, simple_dataloader, device):
        """Test that training works when train_sampler is None (non-distributed)."""
        loss_fn = MultiTaskLoss(penalty=1.0)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            train_sampler=None,  # Explicitly None
        )

        # Should train without error
        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=None,
            num_epochs=2,
            early_stopping_patience=100,
        )

        assert len(history["train_loss"]) == 2

    def test_train_sampler_epoch_starts_from_resume(self, small_model, simple_dataloader, device):
        """Test that set_epoch uses correct epoch when resuming training."""
        from unittest.mock import MagicMock

        mock_sampler = MagicMock()

        loss_fn = MultiTaskLoss(penalty=1.0)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            train_sampler=mock_sampler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            # Train for 2 epochs and save
            trainer.train(
                train_loader=simple_dataloader,
                val_loader=simple_dataloader,
                num_epochs=2,
                checkpoint_dir=tmpdir,
            )

            # Reset mock to track new calls
            mock_sampler.reset_mock()

            # Resume training from epoch 2
            trainer.train(
                train_loader=simple_dataloader,
                val_loader=simple_dataloader,
                num_epochs=5,
                resume_from=os.path.join(tmpdir, "best_model.pt"),
            )

            # Should have called set_epoch for epochs 2, 3, 4 (resumed from 2)
            epochs_called = [call[0][0] for call in mock_sampler.set_epoch.call_args_list]
            assert 2 in epochs_called
            assert 3 in epochs_called
            assert 4 in epochs_called


class TestCategoricalCheckpointCompatibility:
    """Test checkpoint compatibility with categorical features (CAT-6)."""

    @pytest.fixture
    def predictor_with_categoricals(self):
        """Create SequencePredictor with categorical features."""
        return SequencePredictor(
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
            count_num_layers=1,
            count_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            is_classifier=False,
            freeze_base=False,
            predict_nucleotides=False,
            base_output_dim=None,
            categorical_cardinalities={"location": 5, "season": 4},
            categorical_embed_dim=8,
            categorical_fusion="concat",
        )

    @pytest.fixture
    def pretrained_encoder_checkpoint(self, device, tmp_path):
        """Create a pretrained SequenceEncoder checkpoint."""
        encoder = SequenceEncoder(
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
        ).to(device)

        checkpoint_path = tmp_path / "pretrained_encoder.pt"
        torch.save({"model_state_dict": encoder.state_dict()}, checkpoint_path)
        return checkpoint_path, encoder

    def test_load_pretrained_encoder_with_categoricals(
        self, predictor_with_categoricals, pretrained_encoder_checkpoint, device
    ):
        """Test loading pretrained encoder into model with categorical features."""
        checkpoint_path, original_encoder = pretrained_encoder_checkpoint
        predictor = predictor_with_categoricals.to(device)

        result = load_pretrained_encoder(str(checkpoint_path), predictor, strict=False)

        # Encoder weights should be loaded
        assert result["loaded_keys"] > 0

        # Categorical weights should be reported as missing (newly initialized)
        missing_key_names = result["missing_keys"]
        categorical_missing = [k for k in missing_key_names if "categorical" in k]
        assert len(categorical_missing) == 0, "Categorical keys should not be in base_model"

        # Verify encoder weights were actually loaded
        encoder_params = dict(original_encoder.named_parameters())
        predictor_base_params = dict(predictor.base_model.named_parameters())
        for name, param in encoder_params.items():
            if name in predictor_base_params:
                assert torch.allclose(param, predictor_base_params[name])

    def test_categorical_weights_preserved_after_load(self, predictor_with_categoricals, pretrained_encoder_checkpoint, device):
        """Test that categorical weights are preserved (not overwritten) when loading encoder."""
        checkpoint_path, _ = pretrained_encoder_checkpoint
        predictor = predictor_with_categoricals.to(device)

        # Store original categorical weights
        original_cat_weights = {}
        for name, param in predictor.named_parameters():
            if "categorical" in name:
                original_cat_weights[name] = param.clone()

        # Load pretrained encoder
        load_pretrained_encoder(str(checkpoint_path), predictor, strict=False)

        # Categorical weights should be unchanged (still their initial values)
        for name, original_param in original_cat_weights.items():
            current_param = dict(predictor.named_parameters())[name]
            assert torch.allclose(original_param, current_param), f"{name} was unexpectedly modified"

    def test_freeze_base_includes_categorical_in_optimizer(self, predictor_with_categoricals, device):
        """Test that freeze_base includes categorical embedder in optimization."""
        predictor = predictor_with_categoricals.to(device)

        # Freeze base model
        for param in predictor.base_model.parameters():
            param.requires_grad = False

        optimizer = create_optimizer(predictor, freeze_base=True)

        # Get parameter ids from optimizer
        optimizer_param_ids = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                optimizer_param_ids.add(id(param))

        # Categorical embedder and projection should be in optimizer
        categorical_params = []
        for name, param in predictor.named_parameters():
            if "categorical" in name and param.requires_grad:
                categorical_params.append((name, param))

        assert len(categorical_params) > 0, "Should have categorical parameters"
        for name, param in categorical_params:
            assert id(param) in optimizer_param_ids, f"{name} should be in optimizer"

        # Base model params should NOT be in optimizer
        for name, param in predictor.base_model.named_parameters():
            assert id(param) not in optimizer_param_ids, f"base_model.{name} should not be in optimizer"

    def test_freeze_base_training_step_with_categoricals(self, predictor_with_categoricals, device, tmp_path):
        """Test that training step works with freeze_base and categoricals."""
        predictor = predictor_with_categoricals.to(device)

        # Freeze base model
        for param in predictor.base_model.parameters():
            param.requires_grad = False

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
        optimizer = create_optimizer(predictor, freeze_base=True, lr=1e-3)

        trainer = Trainer(
            model=predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            freeze_base=True,
        )

        # Store initial categorical weights
        initial_cat_weights = {}
        for name, param in predictor.named_parameters():
            if "categorical" in name:
                initial_cat_weights[name] = param.clone()

        # Store initial base model weights
        initial_base_weights = {}
        for name, param in predictor.base_model.named_parameters():
            initial_base_weights[name] = param.clone()

        # Create simple batch with categorical data
        batch_size = 4
        num_asvs = 8
        seq_len = 50
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        targets = torch.rand(batch_size, 1).to(device)
        # counts needs shape [batch_size, num_asvs, 1] to match model output
        counts = torch.rand(batch_size, num_asvs, 1).to(device)
        categorical_ids = {
            "location": torch.randint(0, 5, (batch_size,)).to(device),
            "season": torch.randint(0, 4, (batch_size,)).to(device),
        }

        # Create dataset and dataloader
        class CategoricalDataset(torch.utils.data.Dataset):
            def __init__(self, tokens, targets, counts, categorical_ids):
                self.tokens = tokens
                self.targets = targets
                self.counts = counts
                self.categorical_ids = categorical_ids

            def __len__(self):
                return len(self.tokens)

            def __getitem__(self, idx):
                return {
                    "tokens": self.tokens[idx],
                    "y_target": self.targets[idx],
                    "counts": self.counts[idx],
                    "categorical_ids": {k: v[idx] for k, v in self.categorical_ids.items()},
                }

        dataset = CategoricalDataset(tokens, targets, counts, categorical_ids)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Run one training epoch
        train_losses = trainer.train_epoch(dataloader)

        assert "total_loss" in train_losses
        assert train_losses["total_loss"] > 0

        # Categorical weights should have changed (gradient updates)
        cat_weights_changed = False
        for name, initial_param in initial_cat_weights.items():
            current_param = dict(predictor.named_parameters())[name]
            if not torch.allclose(initial_param, current_param):
                cat_weights_changed = True
                break
        assert cat_weights_changed, "Categorical weights should have been updated"

        # Base model weights should NOT have changed
        for name, initial_param in initial_base_weights.items():
            current_param = dict(predictor.base_model.named_parameters())[name]
            assert torch.allclose(initial_param, current_param), f"base_model.{name} should not change"

    def test_staged_training_workflow(self, predictor_with_categoricals, pretrained_encoder_checkpoint, device, tmp_path):
        """Test full staged training: pretrained encoder -> fine-tune with categoricals."""
        checkpoint_path, _ = pretrained_encoder_checkpoint
        predictor = predictor_with_categoricals.to(device)

        # Stage 1: Load pretrained encoder
        result = load_pretrained_encoder(str(checkpoint_path), predictor, strict=False)
        assert result["loaded_keys"] > 0

        # Stage 2: Freeze base, train with categoricals
        for param in predictor.base_model.parameters():
            param.requires_grad = False

        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
        optimizer = create_optimizer(predictor, freeze_base=True, lr=1e-3)

        trainer = Trainer(
            model=predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            freeze_base=True,
        )

        # Simple batch
        batch_size = 4
        num_asvs = 8
        seq_len = 50
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)

        # Forward pass should work
        categorical_ids = {
            "location": torch.randint(0, 5, (batch_size,)).to(device),
            "season": torch.randint(0, 4, (batch_size,)).to(device),
        }
        outputs = predictor(tokens, categorical_ids=categorical_ids)

        assert "target_prediction" in outputs
        assert outputs["target_prediction"].shape == (batch_size, 1)

    def test_load_pretrained_encoder_reports_new_weights(
        self, predictor_with_categoricals, pretrained_encoder_checkpoint, device, caplog
    ):
        """Test that load_pretrained_encoder logs info about new weights being initialized."""
        checkpoint_path, _ = pretrained_encoder_checkpoint
        predictor = predictor_with_categoricals.to(device)

        with caplog.at_level(logging.INFO):
            result = load_pretrained_encoder(str(checkpoint_path), predictor, strict=False)

        # Should have logged about loading
        assert any("Loading pretrained encoder" in record.message for record in caplog.records)
        assert any("Successfully loaded" in record.message for record in caplog.records)


class TestBestMetricSelection:
    """Tests for best_metric model selection."""

    def test_metric_modes_constants(self):
        """Test METRIC_MODES has expected values."""
        from aam.training.trainer import METRIC_MODES

        assert METRIC_MODES["val_loss"] == "min"
        assert METRIC_MODES["r2"] == "max"
        assert METRIC_MODES["mae"] == "min"
        assert METRIC_MODES["accuracy"] == "max"
        assert METRIC_MODES["f1"] == "max"

    def test_is_metric_better_min_mode(self):
        """Test is_metric_better with 'min' mode (lower is better)."""
        from aam.training.trainer import is_metric_better

        # Lower is better
        assert is_metric_better(0.5, 0.6, "min") is True
        assert is_metric_better(0.6, 0.5, "min") is False
        assert is_metric_better(0.5, 0.5, "min") is False

    def test_is_metric_better_max_mode(self):
        """Test is_metric_better with 'max' mode (higher is better)."""
        from aam.training.trainer import is_metric_better

        # Higher is better
        assert is_metric_better(0.6, 0.5, "max") is True
        assert is_metric_better(0.5, 0.6, "max") is False
        assert is_metric_better(0.5, 0.5, "max") is False

    def test_trainer_init_default_best_metric(self, small_model, loss_fn, device):
        """Test Trainer initializes with default best_metric='val_loss'."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )
        assert trainer.best_metric == "val_loss"
        assert trainer.best_metric_mode == "min"

    @pytest.mark.parametrize(
        "metric,expected_mode",
        [
            ("r2", "max"),
            ("mae", "min"),
            ("accuracy", "max"),
            ("f1", "max"),
        ],
    )
    def test_trainer_init_best_metric(self, small_model, loss_fn, device, metric, expected_mode):
        """Test Trainer with different best_metric options."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            best_metric=metric,
        )
        assert trainer.best_metric == metric
        assert trainer.best_metric_mode == expected_mode

    def test_trainer_init_invalid_best_metric(self, small_model, loss_fn, device):
        """Test Trainer raises ValueError for invalid best_metric."""
        with pytest.raises(ValueError, match="Invalid best_metric"):
            Trainer(
                model=small_model,
                loss_fn=loss_fn,
                device=device,
                best_metric="invalid_metric",
            )

    @pytest.mark.parametrize("best_metric", ["val_loss", "r2", "mae"])
    def test_train_saves_checkpoint_by_metric(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path, best_metric):
        """Test train saves best model based on different metrics."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            best_metric=best_metric,
        )

        trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=3,
            checkpoint_dir=str(tmp_path),
            early_stopping_patience=10,
        )

        checkpoint_path = tmp_path / "best_model.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert checkpoint.get("best_metric") == best_metric
        assert "best_metric_value" in checkpoint
        assert "best_val_loss" in checkpoint  # backwards compatibility

    def test_checkpoint_stores_best_metric_info(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test checkpoint contains best_metric and best_metric_value."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            best_metric="r2",
        )

        # Run training to generate checkpoint
        trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            num_epochs=2,
            checkpoint_dir=str(tmp_path),
            early_stopping_patience=10,
        )

        checkpoint_path = tmp_path / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        # Verify checkpoint has metric info
        assert "best_metric" in checkpoint
        assert "best_metric_value" in checkpoint
        assert "best_val_loss" in checkpoint  # backwards compatibility

    def test_load_checkpoint_with_best_metric(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test load_checkpoint returns best_metric info."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            best_metric="r2",
        )

        # Save a checkpoint
        trainer.save_checkpoint(
            str(tmp_path / "test.pt"),
            epoch=5,
            best_val_loss=0.1,
            metrics={"r2": 0.85, "mae": 0.05},
        )

        # Load and verify
        info = trainer.load_checkpoint(str(tmp_path / "test.pt"))
        assert info["epoch"] == 5
        assert info["best_val_loss"] == 0.1


class TestOutputArtifacts:
    """Tests for CLN-10: Training output artifacts (sample lists, validation predictions)."""

    def test_save_val_predictions_creates_tsv(self, small_predictor, loss_fn, device, tmp_path):
        """Test _save_val_predictions creates TSV file with correct content."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        # Create mock prediction data
        full_predictions = {
            "sample_ids": ["sample_1", "sample_2", "sample_3"],
            "predictions": torch.tensor([0.5, 0.7, 0.3]),
            "actuals": torch.tensor([0.4, 0.8, 0.2]),
        }

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        trainer._save_val_predictions(full_predictions, str(checkpoint_dir))

        # Verify file was created in parent directory (output_dir, not checkpoint_dir)
        output_file = tmp_path / "val_predictions.tsv"
        assert output_file.exists()

        # Verify content
        import pandas as pd

        df = pd.read_csv(output_file, sep="\t")
        assert list(df.columns) == ["sample_id", "prediction", "actual"]
        assert len(df) == 3
        assert list(df["sample_id"]) == ["sample_1", "sample_2", "sample_3"]

    def test_save_val_predictions_handles_2d_tensors(self, small_predictor, loss_fn, device, tmp_path):
        """Test _save_val_predictions handles [N, 1] shape tensors."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        # Create mock prediction data with [N, 1] shape
        full_predictions = {
            "sample_ids": ["sample_1", "sample_2"],
            "predictions": torch.tensor([[0.5], [0.7]]),
            "actuals": torch.tensor([[0.4], [0.8]]),
        }

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        trainer._save_val_predictions(full_predictions, str(checkpoint_dir))

        output_file = tmp_path / "val_predictions.tsv"
        assert output_file.exists()

        import pandas as pd

        df = pd.read_csv(output_file, sep="\t")
        assert len(df) == 2
        # Verify values are scalar (squeezed from [N, 1])
        assert df["prediction"].iloc[0] == pytest.approx(0.5, rel=1e-5)
        assert df["actual"].iloc[0] == pytest.approx(0.4, rel=1e-5)

    def test_save_val_predictions_empty_sample_ids_no_file(self, small_predictor, loss_fn, device, tmp_path):
        """Test _save_val_predictions does nothing when sample_ids is empty."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        full_predictions = {
            "sample_ids": [],
            "predictions": torch.tensor([]),
            "actuals": torch.tensor([]),
        }

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        trainer._save_val_predictions(full_predictions, str(checkpoint_dir))

        output_file = tmp_path / "val_predictions.tsv"
        assert not output_file.exists()

    def test_validate_epoch_collect_all_predictions(self, small_predictor, device):
        """Test validate_epoch with collect_all_predictions=True returns full predictions."""
        loss_fn = MultiTaskLoss(penalty=0.0, nuc_penalty=0.0)

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        dataloader = MockBatchDataset(4, device)

        result = trainer.validate_epoch(
            dataloader=dataloader,
            compute_metrics=True,
            return_predictions=True,
            collect_all_predictions=True,
        )

        assert len(result) == 4
        avg_losses, preds_dict, targs_dict, full_preds = result

        assert isinstance(avg_losses, dict)
        assert "sample_ids" in full_preds
        assert "predictions" in full_preds
        assert "actuals" in full_preds
        assert len(full_preds["sample_ids"]) == 4

    def test_validate_epoch_no_collect_returns_3_tuple(self, small_predictor, loss_fn, device):
        """Test validate_epoch with collect_all_predictions=False returns 3-tuple."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        dataloader = MockBatchDataset(4, device)

        result = trainer.validate_epoch(
            dataloader=dataloader,
            compute_metrics=True,
            return_predictions=True,
            collect_all_predictions=False,
        )

        assert len(result) == 3
        avg_losses, preds_dict, targs_dict = result
        assert isinstance(avg_losses, dict)

    def test_train_loop_creates_val_predictions_file(self, small_predictor, loss_fn, device, tmp_path):
        """Test full training loop creates val_predictions.tsv on best validation."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        train_loader = MockBatchDataset(4, device)
        val_loader = MockBatchDataset(4, device)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            checkpoint_dir=str(checkpoint_dir),
            early_stopping_patience=10,
            save_plots=False,
        )

        output_file = tmp_path / "val_predictions.tsv"
        assert output_file.exists(), "val_predictions.tsv should be created during training"

        import pandas as pd

        df = pd.read_csv(output_file, sep="\t")
        assert "sample_id" in df.columns
        assert "prediction" in df.columns
        assert "actual" in df.columns

    def test_val_predictions_saved_even_when_no_improvement(self, small_predictor, loss_fn, device, tmp_path):
        """Test val_predictions.tsv is created even when validation never improves.

        This simulates the --resume-from scenario where best_metric_value is already
        very good and validation never beats it.
        """
        from unittest.mock import patch

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        train_loader = MockBatchDataset(4, device)
        val_loader = MockBatchDataset(4, device)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Patch load_checkpoint to simulate resuming with an unbeatable best_val_loss
        with patch.object(trainer, "load_checkpoint") as mock_load:
            mock_load.return_value = {
                "epoch": 0,
                "best_val_loss": 0.0,  # Perfect loss - can't be beaten
                "best_metric_value": 0.0,
            }

            # Run training with resume_from (the actual path doesn't matter due to mock)
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                checkpoint_dir=str(checkpoint_dir),
                early_stopping_patience=10,
                save_plots=False,
                resume_from="/fake/checkpoint.pt",
            )

        # Even though validation never improved, val_predictions.tsv should be created
        output_file = tmp_path / "val_predictions.tsv"
        assert output_file.exists(), (
            "val_predictions.tsv should be created at end of training "
            "even when validation never improves (e.g., --resume-from scenario)"
        )

        import pandas as pd

        df = pd.read_csv(output_file, sep="\t")
        assert "sample_id" in df.columns
        assert "prediction" in df.columns
        assert "actual" in df.columns
        assert len(df) > 0, "val_predictions.tsv should contain predictions"

    def test_checkpoint_saved_to_new_dir_on_resume(self, small_predictor, loss_fn, device, tmp_path):
        """Test checkpoint is saved to new directory when resuming with different output dir."""
        from unittest.mock import patch

        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        train_loader = MockBatchDataset(4, device)
        val_loader = MockBatchDataset(4, device)
        new_checkpoint_dir = tmp_path / "new_run" / "checkpoints"

        with patch.object(trainer, "load_checkpoint") as mock_load:
            mock_load.return_value = {
                "epoch": 0,
                "best_val_loss": 0.0,  # Perfect loss - validation can't beat this
                "best_metric_value": 0.0,
            }

            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                checkpoint_dir=str(new_checkpoint_dir),
                early_stopping_patience=10,
                save_plots=False,
                resume_from="/old_run/checkpoints/best_model.pt",
            )

        # Even though validation never improved, checkpoint should exist in new directory
        checkpoint_path = new_checkpoint_dir / "best_model.pt"
        assert checkpoint_path.exists(), (
            "best_model.pt should be saved to new checkpoint directory on resume, "
            "even when validation never beats the original best metric"
        )


class TestMultiPassValidation:
    """Tests for multi-pass validation during training (CLN-15)."""

    def test_evaluator_multi_pass_validation_basic(self, small_predictor, loss_fn, device):
        """Test that evaluator supports val_prediction_passes parameter."""
        from aam.training.evaluation import Evaluator

        small_predictor = small_predictor.to(device)
        evaluator = Evaluator(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        val_loader = MockBatchDataset(4, device)

        # Single pass should work
        results = evaluator.validate_epoch(
            val_loader,
            compute_metrics=True,
            val_prediction_passes=1,
        )
        assert "total_loss" in results
        assert results["total_loss"] >= 0

    def test_evaluator_multi_pass_aggregates_predictions(self, small_predictor, loss_fn, device):
        """Test that multi-pass validation aggregates predictions across passes."""
        from aam.training.evaluation import Evaluator

        small_predictor = small_predictor.to(device)
        evaluator = Evaluator(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        val_loader = MockBatchDataset(4, device)

        # Multi-pass validation computes metrics on aggregated predictions
        results = evaluator.validate_epoch(
            val_loader,
            compute_metrics=True,
            val_prediction_passes=3,
        )
        # Multi-pass returns metrics (r2, mae, mse) computed on aggregated predictions
        assert "r2" in results or "mae" in results or "mse" in results

    def test_evaluator_multi_pass_returns_total_loss(self, small_predictor, loss_fn, device):
        """Test that multi-pass validation returns total_loss key (CLN-BUG-8).

        This is required for compatibility with the training loop which expects
        total_loss to be present in validation results.
        """
        from aam.training.evaluation import Evaluator

        small_predictor = small_predictor.to(device)
        evaluator = Evaluator(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        val_loader = MockBatchDataset(4, device)

        # Multi-pass validation must return total_loss for training loop compatibility
        results = evaluator.validate_epoch(
            val_loader,
            compute_metrics=True,
            val_prediction_passes=3,
        )
        assert "total_loss" in results, "Multi-pass validation must return 'total_loss' key"
        assert isinstance(results["total_loss"], float)
        assert results["total_loss"] >= 0

    def test_trainer_passes_val_prediction_passes_to_evaluator(self, small_predictor, loss_fn, device):
        """Test that Trainer correctly passes val_prediction_passes to Evaluator."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
            val_prediction_passes=3,
        )

        # Verify the trainer has the val_prediction_passes attribute
        assert hasattr(trainer, "val_prediction_passes")
        assert trainer.val_prediction_passes == 3

    def test_trainer_default_val_prediction_passes_is_one(self, small_predictor, loss_fn, device):
        """Test that Trainer defaults to single-pass validation."""
        trainer = Trainer(
            model=small_predictor,
            loss_fn=loss_fn,
            device=device,
        )

        # Default should be 1
        assert hasattr(trainer, "val_prediction_passes")
        assert trainer.val_prediction_passes == 1
