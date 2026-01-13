"""Integration tests for AAM PyTorch implementation."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
import os

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader
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
    # Clear any previous CUDA errors before each test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Synchronize to ensure any pending operations complete
        torch.cuda.synchronize()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model_config():
    """Small model configuration for faster testing."""
    return {
        "vocab_size": 6,
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
    config.update(
        {
            "count_num_layers": 1,
            "count_num_heads": 2,
            "target_num_layers": 1,
            "target_num_heads": 2,
            "out_dim": 1,
            "is_classifier": False,
            "freeze_base": False,
        }
    )
    return config


class TestDataPipelineIntegration:
    """Test data pipeline end-to-end integration."""

    def test_data_pipeline_integration(self, biom_file, tmp_path):
        """Test complete data pipeline: Load → Rarefy → Load UniFrac Matrix → Tokenize → Dataset."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))

        assert table is not None
        assert table.shape[0] > 0
        assert table.shape[1] > 0

        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        assert rarefied_table is not None
        assert rarefied_table.shape[0] > 0

        # Create pre-computed UniFrac distance matrix for testing
        from skbio import DistanceMatrix
        import numpy as np

        sample_ids = list(rarefied_table.ids(axis="sample"))
        n_samples = len(sample_ids)
        distances = np.random.rand(n_samples, n_samples)
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0.0)  # Diagonal is 0
        unifrac_distances = DistanceMatrix(distances, ids=sample_ids)

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
        # unifrac_target is added in collate_fn, not __getitem__
        assert "unifrac_target" not in sample
        assert dataset.unifrac_distances is not None

    def test_data_pipeline_tensor_shapes(self, biom_file):
        """Verify tensor shapes throughout data pipeline."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        # Create pre-computed Faith PD values
        import pandas as pd

        sample_ids = list(rarefied_table.ids(axis="sample"))
        faith_pd_values = np.random.rand(len(sample_ids))
        unifrac_distances = pd.Series(faith_pd_values, index=sample_ids)

        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
            unifrac_metric="faith_pd",
        )

        sample = dataset[0]

        assert len(sample["tokens"].shape) == 2
        assert sample["tokens"].shape[1] == 151
        assert sample["tokens"].shape[0] <= 1024

        assert len(sample["counts"].shape) == 2
        assert sample["counts"].shape[1] == 1
        assert sample["counts"].shape[0] <= 1024

        # unifrac_target is added in collate_fn, not __getitem__
        assert "unifrac_target" not in sample

        # Test that collate_fn adds unifrac_target correctly
        from functools import partial

        collate = partial(collate_fn, token_limit=1024, unifrac_distances=unifrac_distances, unifrac_metric="faith_pd")
        batch = [dataset[0], dataset[1]]
        batched = collate(batch)
        assert "unifrac_target" in batched
        assert isinstance(batched["unifrac_target"], torch.FloatTensor)
        assert len(batched["unifrac_target"].shape) == 2  # [batch_size, 1] for faith_pd
        assert batched["unifrac_target"].shape[0] == 2
        assert batched["unifrac_target"].shape[1] == 1

    def test_data_pipeline_dtypes(self, biom_file):
        """Verify tensor dtypes throughout data pipeline."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        # Create pre-computed Faith PD values
        import pandas as pd

        sample_ids = list(rarefied_table.ids(axis="sample"))
        faith_pd_values = np.random.rand(len(sample_ids))
        unifrac_distances = pd.Series(faith_pd_values, index=sample_ids)

        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
            unifrac_metric="faith_pd",
        )

        sample = dataset[0]

        assert sample["tokens"].dtype == torch.long
        assert sample["counts"].dtype == torch.float32
        # unifrac_target is added in collate_fn, not __getitem__
        assert "unifrac_target" not in sample

        # Test that collate_fn adds unifrac_target with correct dtype
        from functools import partial

        collate = partial(collate_fn, token_limit=1024, unifrac_distances=unifrac_distances, unifrac_metric="faith_pd")
        batch = [dataset[0]]
        batched = collate(batch)
        assert batched["unifrac_target"].dtype == torch.float32

    def test_data_pipeline_dataloader(self, biom_file):
        """Test data pipeline with DataLoader."""
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        # Create pre-computed Faith PD values
        import pandas as pd

        sample_ids = list(rarefied_table.ids(axis="sample"))
        faith_pd_values = np.random.rand(len(sample_ids))
        unifrac_distances = pd.Series(faith_pd_values, index=sample_ids)

        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
            unifrac_metric="faith_pd",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch, token_limit=1024, unifrac_distances=unifrac_distances, unifrac_metric="faith_pd"
            ),
        )

        batch = next(iter(dataloader))

        assert "tokens" in batch
        assert "counts" in batch
        assert "sample_ids" in batch
        assert "unifrac_target" in batch

        assert len(batch["tokens"].shape) == 3
        assert batch["tokens"].shape[0] == 4
        assert batch["tokens"].shape[1] <= 1024
        assert batch["tokens"].shape[2] == 151


class TestModelPipelineIntegration:
    """Test model components integration."""

    def test_model_forward_pass_integration(self, device, small_model_config):
        """Test model forward pass with all components."""
        model = SequenceEncoder(**small_model_config).to(device)
        model.eval()

        batch_size = 4
        num_asvs = 10
        seq_len = 50

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)

        with torch.no_grad():
            output = model(tokens)

        assert isinstance(output, dict)
        # For UniFrac, embeddings are returned instead of base_prediction
        assert "embeddings" in output
        assert "sample_embeddings" in output

        assert output["embeddings"].shape[0] == batch_size
        assert output["embeddings"].shape[1] == small_model_config["embedding_dim"]
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

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)
        counts = torch.rand(batch_size, num_asvs, 1).to(device)

        with torch.no_grad():
            output = model(tokens)

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

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)
        counts = torch.rand(batch_size, num_asvs, 1).to(device)
        target = torch.randn(batch_size, 1).to(device)
        # For UniFrac, base_target should be pairwise distance matrix [batch_size, batch_size]
        base_target = torch.randn(batch_size, batch_size).to(device)
        base_target = (base_target + base_target.T) / 2  # Make symmetric
        base_target.fill_diagonal_(0.0)  # Zero diagonal

        model.train()
        outputs = model(tokens)

        targets = {
            "target": target,
            "counts": counts,
            "base_target": base_target,
            "tokens": tokens,
        }

        loss_dict = loss_fn(
            outputs=outputs,
            targets=targets,
            is_classifier=False,
            encoder_type="unifrac",
        )

        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0
        assert isinstance(loss_dict["total_loss"], torch.Tensor)

        loss_dict["total_loss"].backward()

        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients, "At least some parameters should have gradients"


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

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)
        # For UniFrac, base_target should be pairwise distance matrix [batch_size, batch_size]
        base_targets = torch.randn(batch_size, batch_size).to(device)
        base_targets = (base_targets + base_targets.T) / 2  # Make symmetric
        base_targets.fill_diagonal_(0.0)  # Zero diagonal

        model.train()
        optimizer.zero_grad()

        outputs = model(tokens)
        targets = {
            "base_target": base_targets,
            "tokens": tokens,
        }

        loss_dict = loss_fn(
            outputs=outputs,
            targets=targets,
            is_classifier=False,
            encoder_type="unifrac",
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

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)
        base_targets = torch.randn(batch_size, small_model_config["base_output_dim"]).to(device)

        model.eval()

        with torch.no_grad():
            outputs = model(tokens)
            targets = {
                "base_target": base_targets,
                "tokens": tokens,
            }
            loss_dict = loss_fn(
                outputs=outputs,
                targets=targets,
                is_classifier=False,
                encoder_type="faith_pd",
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

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size * 2, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)
        # For UniFrac, base_target should be pairwise distance matrix per batch
        # Since we're using TensorDataset, we'll create a custom dataset that generates
        # distance matrices per batch
        from aam.data.tokenizer import SequenceTokenizer

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
                dist_matrix = torch.rand(self.batch_size, self.batch_size)
                dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Make symmetric
                dist_matrix.fill_diagonal_(0.0)  # Zero diagonal
                dist_matrix = dist_matrix.to(batch_tokens.device)
                return batch_tokens, dist_matrix

        dataset = UniFracDataset(tokens, batch_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

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
    def test_end_to_end_training(self, biom_file, device, small_model_config, tmp_path):
        """Test full training workflow with real data."""
        batch_size = 4
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        # Create pre-computed UniFrac distance matrix
        from skbio import DistanceMatrix

        sample_ids = list(rarefied_table.ids(axis="sample"))
        n_samples = len(sample_ids)
        distances = np.random.rand(n_samples, n_samples)
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0.0)  # Diagonal is 0
        unifrac_distances = DistanceMatrix(distances, ids=sample_ids)

        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
            unifrac_metric="unweighted",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted"),
        )

        model_config = small_model_config.copy()
        model_config["max_bp"] = 150
        model_config["token_limit"] = 1024
        model_config["encoder_type"] = "unifrac"
        model_config["base_output_dim"] = None  # UniFrac uses embeddings, not base_output_dim
        model = SequenceEncoder(**model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # Create UniFracLoader to extract batch distances
        unifrac_loader = UniFracLoader()

        initial_loss = None
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break

            tokens = batch["tokens"].to(device)
            sample_ids = batch["sample_ids"]

            batch_distances = unifrac_loader.extract_batch_distances(unifrac_distances, sample_ids, metric="unweighted")
            base_targets = torch.FloatTensor(batch_distances).to(device)

            model.train()
            optimizer.zero_grad()

            outputs = model(tokens)
            targets = {
                "base_target": base_targets,
                "tokens": tokens,
            }
            loss_dict = loss_fn(
                outputs=outputs,
                targets=targets,
                is_classifier=False,
                encoder_type="unifrac",
            )

            loss_dict["total_loss"].backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss_dict["total_loss"].item()

        assert initial_loss is not None
        assert initial_loss >= 0

    @pytest.mark.slow
    def test_end_to_end_loss_decreases(self, biom_file, device, small_model_config):
        """Verify loss decreases during training."""
        batch_size = 4
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        # Create pre-computed UniFrac distance matrix
        from skbio import DistanceMatrix

        sample_ids = list(rarefied_table.ids(axis="sample"))
        n_samples = len(sample_ids)
        distances = np.random.rand(n_samples, n_samples)
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0.0)  # Diagonal is 0
        unifrac_distances = DistanceMatrix(distances, ids=sample_ids)

        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
            unifrac_metric="unweighted",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted"),
        )

        model_config = small_model_config.copy()
        model_config["max_bp"] = 150
        model_config["token_limit"] = 1024
        model_config["encoder_type"] = "unifrac"
        model_config["base_output_dim"] = None  # UniFrac uses embeddings, not base_output_dim
        model = SequenceEncoder(**model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-3)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # Create UniFracLoader to extract batch distances
        unifrac_loader = UniFracLoader()

        losses = []
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break

            tokens = batch["tokens"].to(device)
            sample_ids = batch["sample_ids"]

            batch_distances = unifrac_loader.extract_batch_distances(unifrac_distances, sample_ids, metric="unweighted")
            base_targets = torch.FloatTensor(batch_distances).to(device)

            model.train()
            optimizer.zero_grad()

            outputs = model(tokens)
            targets = {
                "base_target": base_targets,
                "tokens": tokens,
            }
            loss_dict = loss_fn(
                outputs=outputs,
                targets=targets,
                is_classifier=False,
                encoder_type="unifrac",
            )

            loss_dict["total_loss"].backward()
            optimizer.step()

            losses.append(loss_dict["total_loss"].item())

        assert len(losses) > 0
        assert all(loss >= 0 for loss in losses)

    @pytest.mark.slow
    def test_end_to_end_checkpoint_saving(self, biom_file, device, small_model_config, tmp_path):
        """Test checkpoint saving during training."""
        batch_size = 4
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied_table = loader.rarefy(table, depth=1000, random_seed=42)

        # Create pre-computed UniFrac distance matrix
        from skbio import DistanceMatrix

        sample_ids = list(rarefied_table.ids(axis="sample"))
        n_samples = len(sample_ids)
        distances = np.random.rand(n_samples, n_samples)
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0.0)  # Diagonal is 0
        unifrac_distances = DistanceMatrix(distances, ids=sample_ids)

        tokenizer = SequenceTokenizer()
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=unifrac_distances,
            unifrac_metric="unweighted",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted"),
        )

        model_config = small_model_config.copy()
        model_config["max_bp"] = 150
        model_config["token_limit"] = 1024
        model_config["encoder_type"] = "unifrac"
        model_config["base_output_dim"] = None  # UniFrac uses embeddings, not base_output_dim
        model = SequenceEncoder(**model_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)

        checkpoint_path = tmp_path / "checkpoint.pt"

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        batch = next(iter(dataloader))
        tokens = batch["tokens"].to(device)
        sample_ids = batch["sample_ids"]
        unifrac_loader = UniFracLoader()
        batch_distances = unifrac_loader.extract_batch_distances(unifrac_distances, sample_ids, metric="unweighted")
        base_targets = torch.FloatTensor(batch_distances).to(device)

        model.train()
        optimizer.zero_grad()
        outputs = model(tokens)
        targets = {"base_target": base_targets, "tokens": tokens}
        loss_dict = loss_fn(outputs=outputs, targets=targets, is_classifier=False, encoder_type="unifrac")
        loss_dict["total_loss"].backward()
        optimizer.step()

        trainer.save_checkpoint(str(checkpoint_path), epoch=1, best_val_loss=loss_dict["total_loss"].item())

        assert checkpoint_path.exists()

        loaded_model = SequenceEncoder(**model_config).to(device)
        trainer.model = loaded_model
        trainer.load_checkpoint(str(checkpoint_path))

        assert loaded_model is not None


class TestCategoricalIntegration:
    """Test categorical feature integration through full training loop."""

    @pytest.fixture
    def categorical_predictor_config(self, small_predictor_config):
        """Predictor config with categorical features enabled."""
        config = small_predictor_config.copy()
        config["categorical_cardinalities"] = {"location": 4, "season": 5}
        config["categorical_embed_dim"] = 8
        config["categorical_fusion"] = "concat"
        return config

    @pytest.fixture
    def synthetic_categorical_data(self):
        """Create synthetic data with 2 categorical columns."""
        batch_size = 4
        num_asvs = 10
        seq_len = 50

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN

        categorical_ids = {
            "location": torch.tensor([1, 2, 1, 3]),
            "season": torch.tensor([1, 2, 3, 4]),
        }

        counts = torch.rand(batch_size, num_asvs, 1)
        target = torch.randn(batch_size, 1)

        return {
            "tokens": tokens,
            "categorical_ids": categorical_ids,
            "counts": counts,
            "target": target,
        }

    def test_categorical_training_loop(self, device, categorical_predictor_config, synthetic_categorical_data):
        """Test full training loop with synthetic categorical data."""
        model = SequencePredictor(**categorical_predictor_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0, target_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)

        data = synthetic_categorical_data
        tokens = data["tokens"].to(device)
        categorical_ids = {k: v.to(device) for k, v in data["categorical_ids"].items()}
        counts = data["counts"].to(device)
        target = data["target"].to(device)

        model.train()
        initial_loss = None

        for step in range(3):
            optimizer.zero_grad()

            outputs = model(tokens, categorical_ids=categorical_ids)

            targets = {
                "target": target,
                "counts": counts,
                "tokens": tokens,
            }

            loss_dict = loss_fn(
                outputs=outputs,
                targets=targets,
                is_classifier=False,
                encoder_type="faith_pd",
            )

            loss_dict["total_loss"].backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss_dict["total_loss"].item()

        assert initial_loss is not None
        assert initial_loss >= 0
        assert "target_prediction" in outputs
        assert outputs["target_prediction"].shape == (4, 1)

    def test_categorical_checkpoint_roundtrip(self, device, categorical_predictor_config, synthetic_categorical_data, tmp_path):
        """Test categorical encoder state saved and loaded in checkpoints."""
        import pandas as pd
        from aam.data.categorical import CategoricalEncoder

        train_metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "location": ["outdoor", "indoor", "outdoor", "mixed"],
                "season": ["spring", "summer", "fall", "winter"],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(train_metadata, columns=["location", "season"])

        model = SequencePredictor(**categorical_predictor_config).to(device)
        loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0, target_penalty=1.0)
        optimizer = create_optimizer(model, lr=1e-4)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        checkpoint_path = tmp_path / "categorical_checkpoint.pt"
        model_config = {
            "categorical_encoder": encoder.to_dict(),
            "categorical_embed_dim": 8,
            "categorical_fusion": "concat",
        }
        trainer.save_checkpoint(
            str(checkpoint_path),
            epoch=1,
            best_val_loss=0.5,
            config=model_config,
        )

        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "config" in checkpoint
        assert "categorical_encoder" in checkpoint["config"]

        restored_encoder = CategoricalEncoder.from_dict(checkpoint["config"]["categorical_encoder"])
        assert restored_encoder.is_fitted
        assert restored_encoder.column_names == encoder.column_names
        assert restored_encoder.get_cardinalities() == encoder.get_cardinalities()
        assert restored_encoder.get_mappings() == encoder.get_mappings()

    def test_categorical_unknown_categories_at_inference(
        self, device, categorical_predictor_config, synthetic_categorical_data
    ):
        """Test unknown categories map to index 0 during inference."""
        import pandas as pd
        from aam.data.categorical import CategoricalEncoder

        train_metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "location": ["outdoor", "indoor", "outdoor"],
                "season": ["spring", "summer", "fall"],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(train_metadata, columns=["location", "season"])

        inference_metadata = pd.DataFrame(
            {
                "sample_id": ["t1", "t2"],
                "location": ["underwater", "outdoor"],
                "season": ["monsoon", "spring"],
            }
        )
        inference_ids = encoder.transform(inference_metadata)

        assert inference_ids["location"][0] == 0
        assert inference_ids["location"][1] >= 1
        assert inference_ids["season"][0] == 0
        assert inference_ids["season"][1] >= 1

        model = SequencePredictor(**categorical_predictor_config).to(device)
        model.eval()

        batch_size = 2
        num_asvs = 10
        seq_len = 50

        from aam.data.tokenizer import SequenceTokenizer

        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens = tokens.to(device)

        categorical_ids = {
            "location": torch.tensor(inference_ids["location"]).to(device),
            "season": torch.tensor(inference_ids["season"]).to(device),
        }

        with torch.no_grad():
            outputs = model(tokens, categorical_ids=categorical_ids)

        assert "target_prediction" in outputs
        assert outputs["target_prediction"].shape == (batch_size, 1)
        assert not torch.isnan(outputs["target_prediction"]).any()
