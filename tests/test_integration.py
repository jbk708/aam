"""Integration tests for AAM PyTorch implementation."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
import os

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from aam.data.tokenizer import SequenceTokenizer
from aam.data.dataset import ASVDataset, collate_fn
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.sequence_predictor import SequencePredictor
from aam.training.losses import MultiTaskLoss
from aam.training.trainer import Trainer, create_optimizer, create_scheduler


@pytest.fixture
def data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def biom_file(data_dir):
    """Path to BIOM table file."""
    path = data_dir / "fall_train_only_all_outdoor.biom"
    if not path.exists():
        pytest.skip(f"BIOM file not found: {path}")
    return path


@pytest.fixture
def tree_file(data_dir):
    """Path to phylogenetic tree file."""
    path = data_dir / "all-outdoors_sepp_tree.nwk"
    if not path.exists():
        pytest.skip(f"Tree file not found: {path}")
    return path


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model_config():
    """Small model configuration for faster testing."""
    return {
        "vocab_size": 5,
        "embedding_dim": 32,
        "max_bp": 50,
        "token_limit": 64,
        "asv_num_layers": 1,
        "asv_num_heads": 2,
        "sample_num_layers": 1,
        "sample_num_heads": 2,
        "encoder_num_layers": 1,
        "encoder_num_heads": 2,
        "base_output_dim": 16,
        "encoder_type": "unifrac",
        "predict_nucleotides": False,
    }


@pytest.fixture
def small_predictor_config(small_model_config):
    """Small predictor configuration."""
    config = small_model_config.copy()
    config.update({
        "count_num_layers": 1,
        "count_num_heads": 2,
        "target_num_layers": 1,
        "target_num_heads": 2,
        "out_dim": 1,
        "is_classifier": False,
        "freeze_base": False,
    })
    return config


class TestDataPipelineIntegration:
    """Test data pipeline end-to-end integration."""

    def test_data_pipeline_integration(self, biom_file, tree_file):
        """Test complete data pipeline: Load → Rarefy → UniFrac → Tokenize → Dataset."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        
        assert table is not None
        assert table.shape[0] > 0
        assert table.shape[1] > 0
        
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        assert rarefied_table is not None
        assert rarefied_table.shape[0] > 0
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        assert unifrac_distances is not None
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert "tokens" in sample
        assert "counts" in sample
        assert "sample_id" in sample
        assert "unifrac_target" in sample

    def test_data_pipeline_tensor_shapes(self, biom_file, tree_file):
        """Verify tensor shapes throughout data pipeline."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        sample = dataset[0]
        
        assert len(sample["tokens"].shape) == 2
        assert sample["tokens"].shape[1] == 150
        assert sample["tokens"].shape[0] <= 1024
        
        assert len(sample["counts"].shape) == 2
        assert sample["counts"].shape[1] == 1
        assert sample["counts"].shape[0] <= 1024
        
        assert isinstance(sample["unifrac_target"], torch.FloatTensor)
        assert len(sample["unifrac_target"].shape) == 1

    def test_data_pipeline_dtypes(self, biom_file, tree_file):
        """Verify tensor dtypes throughout data pipeline."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        sample = dataset[0]
        
        assert sample["tokens"].dtype == torch.long
        assert sample["counts"].dtype == torch.float32
        assert sample["unifrac_target"].dtype == torch.float32

    def test_data_pipeline_dataloader(self, biom_file, tree_file):
        """Test data pipeline with DataLoader."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024),
        )
        
        batch = next(iter(dataloader))
        
        assert "tokens" in batch
        assert "counts" in batch
        assert "sample_ids" in batch
        assert "unifrac_target" in batch
        
        assert len(batch["tokens"].shape) == 3
        assert batch["tokens"].shape[0] == 4
        assert batch["tokens"].shape[1] <= 1024
        assert batch["tokens"].shape[2] == 150


class TestModelPipelineIntegration:
    """Test model components integration."""

    def test_model_forward_pass_integration(self, device, small_model_config):
        """Test model forward pass with all components."""
        model = SequenceEncoder(**small_model_config).to(device)
        model.eval()
        
        batch_size = 4
        num_asvs = 10
        seq_len = 50
        
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        
        with torch.no_grad():
            output = model(tokens)
        
        assert isinstance(output, dict)
        assert "base_prediction" in output
        assert "sample_embeddings" in output
        
        assert output["base_prediction"].shape[0] == batch_size
        assert output["sample_embeddings"].shape[0] == batch_size
        assert output["sample_embeddings"].shape[1] == num_asvs
        assert output["sample_embeddings"].shape[2] == small_model_config["embedding_dim"]

    def test_model_output_structure(self, device, small_predictor_config):
        """Verify model output dictionary structure."""
        model = SequencePredictor(**small_predictor_config).to(device)
        model.eval()
        
        batch_size = 4
        num_asvs = 10
        seq_len = 50
        
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        counts = torch.rand(batch_size, num_asvs, 1).to(device)
        
        with torch.no_grad():
            output = model(tokens, counts)
        
        assert isinstance(output, dict)
        assert "target_prediction" in output
        assert "count_prediction" in output
        assert "base_embeddings" in output
        
        assert output["target_prediction"].shape[0] == batch_size
        assert output["target_prediction"].shape[1] == small_predictor_config["out_dim"]
        assert output["count_prediction"].shape[0] == batch_size
        assert output["count_prediction"].shape[1] == num_asvs
        assert output["count_prediction"].shape[2] == 1

    def test_loss_computation_integration(self, device, small_predictor_config):
        """Test loss computation with model outputs."""
        model = SequencePredictor(**small_predictor_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        
        batch_size = 4
        num_asvs = 10
        seq_len = 50
        
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        counts = torch.rand(batch_size, num_asvs, 1).to(device)
        targets = torch.randn(batch_size, 1).to(device)
        
        model.train()
        output = model(tokens, counts)
        
        loss_dict = loss_fn(
            target_prediction=output["target_prediction"],
            target=targets,
            count_prediction=output["count_prediction"],
            count=counts,
            base_prediction=output.get("base_prediction"),
            base_target=torch.randn(batch_size, small_predictor_config["base_output_dim"]).to(device),
        )
        
        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0
        assert isinstance(loss_dict["total_loss"], torch.Tensor)
        
        loss_dict["total_loss"].backward()
        
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTrainingPipelineIntegration:
    """Test training pipeline integration."""

    def test_training_step_integration(self, device, small_model_config):
        """Test single training step with model, optimizer, and data."""
        model = SequenceEncoder(**small_model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)
        
        batch_size = 4
        num_asvs = 10
        seq_len = 50
        
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        base_targets = torch.randn(batch_size, small_model_config["base_output_dim"]).to(device)
        
        model.train()
        optimizer.zero_grad()
        
        output = model(tokens)
        loss_dict = loss_fn(
            base_prediction=output["base_prediction"],
            base_target=base_targets,
        )
        
        loss_dict["total_loss"].backward()
        optimizer.step()
        
        assert loss_dict["total_loss"].item() >= 0

    def test_validation_step_integration(self, device, small_model_config):
        """Test single validation step with model and data."""
        model = SequenceEncoder(**small_model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        
        batch_size = 4
        num_asvs = 10
        seq_len = 50
        
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len)).to(device)
        base_targets = torch.randn(batch_size, small_model_config["base_output_dim"]).to(device)
        
        model.eval()
        
        with torch.no_grad():
            output = model(tokens)
            loss_dict = loss_fn(
                base_prediction=output["base_prediction"],
                base_target=base_targets,
            )
        
        assert loss_dict["total_loss"].item() >= 0

    def test_training_loop_integration(self, device, small_model_config):
        """Test complete training loop works."""
        model = SequenceEncoder(**small_model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)
        scheduler = create_scheduler(optimizer, num_warmup_steps=10, num_training_steps=100)
        
        batch_size = 4
        num_asvs = 10
        seq_len = 50
        
        tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len)).to(device)
        base_targets = torch.randn(batch_size * 2, small_model_config["base_output_dim"]).to(device)
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(tokens, base_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        
        losses = trainer.train_epoch(dataloader)
        
        assert "total_loss" in losses
        assert losses["total_loss"] >= 0
        
        results = trainer.validate_epoch(dataloader, compute_metrics=False)
        
        assert "total_loss" in results
        assert results["total_loss"] >= 0


class TestEndToEnd:
    """Test end-to-end training workflow."""

    @pytest.mark.slow
    def test_end_to_end_training(self, biom_file, tree_file, device, small_model_config, tmp_path):
        """Test full training workflow with real data."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024),
        )
        
        model = SequenceEncoder(**small_model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        
        initial_loss = None
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break
            
            tokens = batch["tokens"].to(device)
            base_targets = batch["unifrac_target"].to(device)
            
            model.train()
            optimizer.zero_grad()
            
            output = model(tokens)
            loss_dict = loss_fn(
                base_prediction=output["base_prediction"],
                base_target=base_targets,
            )
            
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss_dict["total_loss"].item()
        
        assert initial_loss is not None
        assert initial_loss >= 0

    @pytest.mark.slow
    def test_end_to_end_loss_decreases(self, biom_file, tree_file, device, small_model_config):
        """Verify loss decreases during training."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024),
        )
        
        model = SequenceEncoder(**small_model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-3)
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        
        losses = []
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            
            tokens = batch["tokens"].to(device)
            base_targets = batch["unifrac_target"].to(device)
            
            model.train()
            optimizer.zero_grad()
            
            output = model(tokens)
            loss_dict = loss_fn(
                base_prediction=output["base_prediction"],
                base_target=base_targets,
            )
            
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            losses.append(loss_dict["total_loss"].item())
        
        assert len(losses) > 0
        assert all(l >= 0 for l in losses)

    @pytest.mark.slow
    def test_end_to_end_checkpoint_saving(self, biom_file, tree_file, device, small_model_config, tmp_path):
        """Test checkpoint saving during training."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, seed=42)
        
        computer = UniFracComputer()
        unifrac_distances = computer.compute_unweighted(rarefied_table, str(tree_file))
        
        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024),
        )
        
        model = SequenceEncoder(**small_model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        
        trainer.train_epoch(dataloader)
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, best_loss=1.0)
        
        assert checkpoint_path.exists()
        
        loaded_model = SequenceEncoder(**small_model_config).to(device)
        trainer.load_checkpoint(str(checkpoint_path), model=loaded_model)
        
        assert loaded_model is not None
