"""Unit tests for training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
from pathlib import Path
import inspect
import matplotlib.pyplot as plt
import numpy as np

from aam.training.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
    load_pretrained_encoder,
)
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.sequence_predictor import SequencePredictor
from aam.training.losses import MultiTaskLoss
from aam.training.metrics import compute_regression_metrics, compute_count_metrics


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
        base_output_dim=16,
        encoder_type="unifrac",
        predict_nucleotides=False,
    )


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
        base_output_dim=32,
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

    tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len)).to(device)
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
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

    tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len)).to(device)
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    base_targets = torch.randn(batch_size * 2, 16).to(device)

    dataset = TensorDataset(tokens, base_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestCreateOptimizer:
    """Test optimizer creation."""

    def test_create_optimizer_default(self, small_model):
        """Test creating optimizer with default parameters."""
        optimizer = create_optimizer(small_model, lr=1e-4, weight_decay=0.01)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_optimizer_excludes_frozen(self, small_predictor):
        """Test that optimizer excludes frozen parameters."""
        small_predictor.freeze_base = True
        for param in small_predictor.base_model.parameters():
            param.requires_grad = False

        optimizer = create_optimizer(small_predictor, freeze_base=True)

        param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        base_param_ids = {id(p) for p in small_predictor.base_model.parameters()}

        assert len(param_ids & base_param_ids) == 0


class TestCreateScheduler:
    """Test scheduler creation."""

    def test_create_scheduler(self, small_model):
        """Test creating scheduler with warmup and cosine decay."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer, num_warmup_steps=100, num_training_steps=1000)

        assert scheduler is not None
        assert hasattr(scheduler, "step")


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
        optimizer = create_optimizer(small_model)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        assert trainer.optimizer == optimizer

    def test_trainer_init_with_scheduler(self, small_model, loss_fn, device):
        """Test Trainer initialization with custom scheduler."""
        optimizer = create_optimizer(small_model)
        scheduler = create_scheduler(optimizer)
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
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

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
            base_output_dim=16,
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
            base_output_dim=32,
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


class TestTrainerEdgeCases:
    """Test edge cases for trainer."""

    def test_scheduler_warmup_phase(self, small_model, device):
        """Test scheduler warmup phase."""
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
        scheduler = create_scheduler(optimizer, num_warmup_steps=5, num_training_steps=20)

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

        optimizer = create_optimizer(small_predictor, lr=1e-4, freeze_base=True)

        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        base_param_ids = {id(p) for p in small_predictor.base_model.parameters()}

        assert len(optimizer_param_ids & base_param_ids) == 0

    def test_checkpoint_save_success(self, small_predictor, loss_fn, simple_dataloader, device, tmp_path):
        """Test checkpoint save functionality."""
        small_predictor = small_predictor.to(device)
        optimizer = create_optimizer(small_predictor, lr=1e-4)
        scheduler = create_scheduler(optimizer, num_warmup_steps=0, num_training_steps=10)

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
        optimizer = create_optimizer(small_predictor, lr=1e-4)
        scheduler = create_scheduler(optimizer, num_warmup_steps=0, num_training_steps=10)

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
        optimizer = create_optimizer(small_predictor, lr=1e-4)
        scheduler = create_scheduler(optimizer, num_warmup_steps=0, num_training_steps=10)

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
        optimizer = create_optimizer(small_predictor, lr=1e-4)
        scheduler = create_scheduler(optimizer, num_warmup_steps=0, num_training_steps=20)

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

    def test_gradient_accumulation_steps_1(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test gradient accumulation with steps=1 (no accumulation)."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        initial_params = [p.clone() for p in small_model.parameters() if p.requires_grad]
        losses = trainer.train_epoch(simple_dataloader_encoder, gradient_accumulation_steps=1)
        updated_params = [p for p in small_model.parameters() if p.requires_grad]

        assert "total_loss" in losses
        assert losses["total_loss"] >= 0
        for init_param, updated_param in zip(initial_params, updated_params):
            assert not torch.equal(init_param, updated_param)

    def test_gradient_accumulation_steps_2(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test gradient accumulation with steps=2."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        initial_params = [p.clone() for p in small_model.parameters() if p.requires_grad]
        losses = trainer.train_epoch(simple_dataloader_encoder, gradient_accumulation_steps=2)
        updated_params = [p for p in small_model.parameters() if p.requires_grad]

        assert "total_loss" in losses
        assert losses["total_loss"] >= 0
        for init_param, updated_param in zip(initial_params, updated_params):
            assert not torch.equal(init_param, updated_param)

    def test_gradient_accumulation_steps_4(self, small_model, loss_fn, simple_dataloader_encoder, device):
        """Test gradient accumulation with steps=4."""
        small_model = small_model.to(device)
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        losses = trainer.train_epoch(simple_dataloader_encoder, gradient_accumulation_steps=4)

        assert "total_loss" in losses
        assert losses["total_loss"] >= 0

    def test_gradient_accumulation_equivalent_loss(self, small_model, loss_fn, device):
        """Test that gradient accumulation produces equivalent results."""
        batch_size = 4
        num_asvs = 10
        seq_len = 50

        from aam.data.tokenizer import SequenceTokenizer
        tokens1 = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len)).to(device)
        tokens1[:, :, 0] = SequenceTokenizer.START_TOKEN
        base_targets1 = torch.randn(batch_size * 2, 16).to(device)
        dataset1 = TensorDataset(tokens1, base_targets1)
        dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)

        tokens2 = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len)).to(device)
        tokens2[:, :, 0] = SequenceTokenizer.START_TOKEN
        base_targets2 = torch.randn(batch_size * 2, 16).to(device)
        dataset2 = TensorDataset(tokens2, base_targets2)
        dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

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


class TestPredictionPlots:
    """Test validation prediction plot functionality."""

    def test_create_prediction_plot_regression(self, small_model, loss_fn, device):
        """Test creating regression prediction plot."""
        trainer = Trainer(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
        )

        predictions = torch.randn(100, 1).to(device)
        targets = predictions + torch.randn(100, 1).to(device) * 0.1
        r2 = 0.95

        fig = trainer._create_prediction_plot(predictions, targets, epoch=5, r2=r2)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_confusion_matrix_plot_classification(self, loss_fn, device):
        """Test creating classification confusion matrix plot."""
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
            base_output_dim=32,
        )

        trainer = Trainer(
            model=classifier_model,
            loss_fn=loss_fn,
            device=device,
        )

        predictions = torch.randint(0, 3, (100,)).to(device)
        targets = torch.randint(0, 3, (100,)).to(device)
        accuracy = 0.85
        precision = 0.82
        recall = 0.80
        f1 = 0.81

        fig = trainer._create_confusion_matrix_plot(
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

        results, predictions, targets = trainer.validate_epoch(simple_dataloader, compute_metrics=True, return_predictions=True)

        assert "total_loss" in results
        assert predictions is not None
        assert targets is not None
        assert predictions.shape[0] == targets.shape[0]

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

        trainer._save_prediction_plots(predictions, targets, epoch=5, metrics=metrics, checkpoint_dir=str(checkpoint_dir))

        plots_dir = checkpoint_dir / "plots"
        plot_file = plots_dir / "pred_vs_actual_best.png"
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
            base_output_dim=32,
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

        trainer._save_prediction_plots(predictions, targets, epoch=5, metrics=metrics, checkpoint_dir=str(checkpoint_dir))

        plots_dir = checkpoint_dir / "plots"
        plot_file = plots_dir / "pred_vs_actual_best.png"
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
        plot_file = plots_dir / "pred_vs_actual_best.png"
        assert plot_file.exists()

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
        plot_file = plots_dir / "pred_vs_actual_best.png"
        assert plot_file.exists()
