"""Unit tests for ASVDataset class."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
import biom
from biom import Table
import pandas as pd
from skbio import DistanceMatrix

from aam.data.dataset import ASVDataset, collate_fn
from aam.data.tokenizer import SequenceTokenizer
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer


def generate_150bp_sequence(seed=None):
    """Generate a random 150bp DNA sequence."""
    import random

    if seed is not None:
        random.seed(seed)
    bases = "ACGT"
    return "".join(random.choice(bases) for _ in range(150))


def create_simple_tree_file(tmp_path, observation_ids):
    """Create a simple Newick tree file for testing."""
    tree_path = tmp_path / "test_tree.nwk"

    if len(observation_ids) == 1:
        tree_str = f"{observation_ids[0]}:0.1;"
    elif len(observation_ids) == 2:
        tree_str = f"({observation_ids[0]}:0.1,{observation_ids[1]}:0.1);"
    else:
        tips = [f"{obs_id}:0.1" for obs_id in observation_ids]
        tree_str = "(" + ",".join(tips) + ");"

    tree_path.write_text(tree_str)
    return str(tree_path)


@pytest.fixture
def simple_table():
    """Create a simple BIOM table for testing."""
    data = np.array([[10, 20, 5], [15, 10, 25], [5, 30, 10]])
    observation_ids = [
        generate_150bp_sequence(seed=1),
        generate_150bp_sequence(seed=2),
        generate_150bp_sequence(seed=3),
    ]
    sample_ids = ["sample1", "sample2", "sample3"]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def rarefied_table(simple_table):
    """Create a rarefied BIOM table for testing."""
    loader = BIOMLoader()
    return loader.rarefy(simple_table, depth=20, random_seed=42)


@pytest.fixture
def simple_metadata():
    """Create simple metadata DataFrame."""
    return pd.DataFrame(
        {
            "sample_id": ["sample1", "sample2", "sample3"],
            "target": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def simple_unifrac_distances(rarefied_table, tmp_path):
    """Create simple UniFrac distance matrix."""
    computer = UniFracComputer()
    observation_ids = list(rarefied_table.ids(axis="observation"))
    tree_file = create_simple_tree_file(tmp_path, observation_ids)
    return computer.compute_unweighted(rarefied_table, tree_file)


@pytest.fixture
def tokenizer():
    """Create a SequenceTokenizer instance."""
    return SequenceTokenizer()


class TestASVDataset:
    """Test suite for ASVDataset class."""

    def test_init_basic(self, rarefied_table, tokenizer):
        """Test basic ASVDataset initialization."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        assert dataset is not None
        assert isinstance(dataset, ASVDataset)

    def test_init_with_default_tokenizer(self, rarefied_table):
        """Test initialization with default tokenizer."""
        dataset = ASVDataset(
            table=rarefied_table,
            max_bp=150,
            token_limit=1024,
        )

        assert dataset is not None
        assert dataset.tokenizer is not None

    def test_len(self, rarefied_table, tokenizer):
        """Test __len__ method."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        assert len(dataset) == len(rarefied_table.ids(axis="sample"))

    def test_getitem_basic(self, rarefied_table, tokenizer):
        """Test __getitem__ returns correct structure."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "tokens" in sample
        assert "counts" in sample
        assert "sample_id" in sample
        assert isinstance(sample["tokens"], torch.LongTensor)
        assert isinstance(sample["counts"], torch.FloatTensor)

    def test_getitem_tokens_shape(self, rarefied_table, tokenizer):
        """Test that tokens have correct shape [num_asvs, max_bp]."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        sample = dataset[0]
        tokens = sample["tokens"]

        assert len(tokens.shape) == 2
        assert tokens.shape[1] == 150

    def test_getitem_counts_shape(self, rarefied_table, tokenizer):
        """Test that counts have correct shape [num_asvs, 1]."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        sample = dataset[0]
        counts = sample["counts"]

        assert len(counts.shape) == 2
        assert counts.shape[1] == 1

    def test_getitem_with_metadata(self, rarefied_table, tokenizer, simple_metadata):
        """Test __getitem__ with metadata and target column."""
        dataset = ASVDataset(
            table=rarefied_table,
            metadata=simple_metadata,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            target_column="target",
        )

        sample = dataset[0]

        assert "y_target" in sample
        assert isinstance(sample["y_target"], torch.FloatTensor)

    def test_getitem_with_unifrac(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test __getitem__ with UniFrac distances."""
        dataset = ASVDataset(
            table=rarefied_table,
            unifrac_distances=simple_unifrac_distances,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_metric="unweighted",
        )

        sample = dataset[0]

        assert "unifrac_target" in sample
        assert isinstance(sample["unifrac_target"], torch.FloatTensor)

    def test_getitem_sample_id(self, rarefied_table, tokenizer):
        """Test that sample_id is correct."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        sample_ids = rarefied_table.ids(axis="sample")
        sample = dataset[0]

        assert sample["sample_id"] == sample_ids[0]

    def test_getitem_all_samples(self, rarefied_table, tokenizer):
        """Test accessing all samples."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert "tokens" in sample
            assert "counts" in sample
            assert "sample_id" in sample

    def test_getitem_index_error(self, rarefied_table, tokenizer):
        """Test that accessing invalid index raises error."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        with pytest.raises((IndexError, KeyError)):
            _ = dataset[len(dataset)]

    def test_token_limit_padding(self, rarefied_table, tokenizer):
        """Test that samples are padded to token_limit."""
        token_limit = 10
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=token_limit,
        )

        sample = dataset[0]
        tokens = sample["tokens"]
        counts = sample["counts"]

        assert tokens.shape[0] <= token_limit
        assert counts.shape[0] <= token_limit


class TestCollateFn:
    """Test suite for collate_fn function."""

    def test_collate_fn_basic(self, tokenizer):
        """Test basic collate_fn functionality."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2, 3], [4, 1, 2]]),
                "counts": torch.FloatTensor([[10.0], [20.0]]),
                "sample_id": "sample1",
            },
            {
                "tokens": torch.LongTensor([[2, 3, 4], [1, 2, 3], [4, 1, 2]]),
                "counts": torch.FloatTensor([[15.0], [25.0], [30.0]]),
                "sample_id": "sample2",
            },
        ]

        token_limit = 5
        result = collate_fn(batch, token_limit)

        assert isinstance(result, dict)
        assert "tokens" in result
        assert "counts" in result
        assert "sample_ids" in result
        assert isinstance(result["tokens"], torch.LongTensor)
        assert isinstance(result["counts"], torch.FloatTensor)

    def test_collate_fn_batch_shape(self, tokenizer):
        """Test that collate_fn produces correct batch shapes."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2], [3, 4]]),
                "counts": torch.FloatTensor([[10.0], [20.0]]),
                "sample_id": "sample1",
            },
            {
                "tokens": torch.LongTensor([[2, 3], [4, 1], [2, 3]]),
                "counts": torch.FloatTensor([[15.0], [25.0], [30.0]]),
                "sample_id": "sample2",
            },
        ]

        token_limit = 5
        result = collate_fn(batch, token_limit)

        assert result["tokens"].shape[0] == 2
        assert result["tokens"].shape[1] <= token_limit
        assert result["counts"].shape[0] == 2
        assert result["counts"].shape[1] <= token_limit
        assert result["counts"].shape[2] == 1

    def test_collate_fn_padding(self, tokenizer):
        """Test that collate_fn pads to token_limit."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2], [3, 4]]),
                "counts": torch.FloatTensor([[10.0], [20.0]]),
                "sample_id": "sample1",
            },
        ]

        token_limit = 5
        result = collate_fn(batch, token_limit)

        assert result["tokens"].shape[1] == token_limit
        assert result["counts"].shape[1] == token_limit
        assert result["counts"].shape[2] == 1
        assert result["counts"].shape[0] == 1


class TestASVDatasetIntegration:
    """Integration tests for ASVDataset."""

    def test_dataloader_iteration(self, rarefied_table, tokenizer):
        """Test DataLoader iteration with ASVDataset."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        def custom_collate(batch):
            return collate_fn(batch, token_limit=1024)

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate, shuffle=False)

        for batch in dataloader:
            assert "tokens" in batch
            assert "counts" in batch
            assert "sample_ids" in batch
            assert batch["tokens"].shape[0] == 2
            break

    def test_dataloader_with_metadata(self, rarefied_table, tokenizer, simple_metadata):
        """Test DataLoader with metadata."""
        dataset = ASVDataset(
            table=rarefied_table,
            metadata=simple_metadata,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            target_column="target",
        )

        def custom_collate(batch):
            return collate_fn(batch, token_limit=1024)

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate, shuffle=False)

        for batch in dataloader:
            assert "y_target" in batch
            assert batch["y_target"].shape[0] == 2
            break

    def test_dataloader_with_unifrac(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test DataLoader with UniFrac distances."""
        dataset = ASVDataset(
            table=rarefied_table,
            unifrac_distances=simple_unifrac_distances,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_metric="unweighted",
        )

        def custom_collate(batch):
            return collate_fn(batch, token_limit=1024)

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate, shuffle=False)

        for batch in dataloader:
            assert "unifrac_target" in batch
            break
