"""Unit tests for ASVDataset class."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
import biom
from biom import Table
import pandas as pd
from skbio import DistanceMatrix

from aam.data.dataset import ASVDataset, collate_fn
from aam.data.tokenizer import SequenceTokenizer
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader


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
    """Create simple UniFrac distance matrix (pre-computed for testing)."""
    import numpy as np

    # Create a simple pre-computed distance matrix
    sample_ids = list(rarefied_table.ids(axis="sample"))
    n_samples = len(sample_ids)
    # Create symmetric distance matrix with zeros on diagonal
    distances = np.random.rand(n_samples, n_samples)
    distances = (distances + distances.T) / 2  # Make symmetric
    np.fill_diagonal(distances, 0.0)  # Diagonal is 0
    return DistanceMatrix(distances, ids=sample_ids)


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
        assert tokens.shape[1] == 151

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
        """Test __getitem__ with UniFrac distances (unifrac_target is added in collate_fn, not __getitem__)."""
        dataset = ASVDataset(
            table=rarefied_table,
            unifrac_distances=simple_unifrac_distances,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        sample = dataset[0]

        # unifrac_target is NOT in __getitem__ output - it's added in collate_fn
        assert "unifrac_target" not in sample
        # But the dataset should store unifrac_distances for use in collate_fn
        assert dataset.unifrac_distances is not None

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
        result = collate_fn(batch, token_limit, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False)

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
        result = collate_fn(batch, token_limit, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False)

        assert result["tokens"].shape[1] == token_limit
        assert result["counts"].shape[1] == token_limit
        assert result["counts"].shape[2] == 1
        assert result["counts"].shape[0] == 1


class TestDatasetEdgeCases:
    """Test edge cases for dataset and collate_fn."""

    def test_collate_fn_token_truncation(self, tokenizer):
        """Test collate_fn truncates when num_asvs > token_limit."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2, 3] for _ in range(15)]),
                "counts": torch.FloatTensor([[10.0] for _ in range(15)]),
                "sample_id": "sample1",
            },
        ]
        token_limit = 10
        result = collate_fn(batch, token_limit, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False)
        assert result["tokens"].shape[1] == token_limit
        assert result["counts"].shape[1] == token_limit
        assert result["tokens"].shape[0] == 1

    def test_dataset_empty_sample(self, tokenizer, tmp_path):
        """Test dataset with empty sample (no ASVs)."""
        from biom import Table
        import numpy as np

        data = np.array([[10, 20], [0, 0]])
        observation_ids = ["ACGT" * 37 + "A", "ACGT" * 37 + "C"]
        sample_ids = ["sample1", "empty_sample"]
        table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

        dataset = ASVDataset(
            table=table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        sample = dataset[1]
        assert "tokens" in sample
        assert "counts" in sample
        assert sample["tokens"].shape[0] >= 1
        assert sample["counts"].shape[0] >= 1

    def test_dataset_string_target_value(self, tokenizer, rarefied_table, tmp_path):
        """Test dataset with string target value in metadata."""
        import pandas as pd

        metadata_df = pd.DataFrame(
            {
                "sample_id": list(rarefied_table.ids(axis="sample")),
                "target": ["1.5", "2.3", "3.7"],
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            metadata=metadata_df,
            target_column="target",
        )

        sample = dataset[0]
        assert "y_target" in sample
        assert isinstance(sample["y_target"], torch.Tensor)
        assert sample["y_target"].dtype == torch.float32

    def test_dataset_faith_pd_extraction(self, tokenizer, rarefied_table):
        """Test Faith PD distance extraction (unifrac_target is added in collate_fn, not __getitem__)."""
        import pandas as pd
        import numpy as np

        sample_ids = list(rarefied_table.ids(axis="sample"))
        faith_pd_values = pd.Series([2.5, 3.1, 2.8], index=sample_ids)

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=faith_pd_values,
            unifrac_metric="faith_pd",
            stripe_mode=False,
        )

        sample = dataset[0]
        # unifrac_target is NOT in __getitem__ output - it's added in collate_fn
        assert "unifrac_target" not in sample
        # But the dataset should store unifrac_distances for use in collate_fn
        assert dataset.unifrac_distances is not None

    def test_dataset_faith_pd_missing_sample(self, tokenizer, rarefied_table):
        """Test Faith PD extraction with missing sample ID (unifrac_target is added in collate_fn, not __getitem__)."""
        import pandas as pd

        sample_ids = list(rarefied_table.ids(axis="sample"))
        faith_pd_values = pd.Series([2.5, 3.1], index=sample_ids[:2])

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_distances=faith_pd_values,
            unifrac_metric="faith_pd",
            stripe_mode=False,
        )

        # unifrac_target is NOT in __getitem__ output - it's added in collate_fn
        # The missing sample ID will cause an error in collate_fn when trying to extract distances
        sample_with_value = dataset[0]
        assert "unifrac_target" not in sample_with_value

        sample_without_value = dataset[2]
        assert "unifrac_target" not in sample_without_value

        # Test that collate_fn will raise an error for missing sample ID
        from aam.data.dataset import collate_fn
        from functools import partial

        batch = [dataset[0], dataset[2]]  # sample 2 is missing from faith_pd_values
        collate = partial(
            collate_fn, token_limit=1024, unifrac_distances=faith_pd_values, unifrac_metric="faith_pd", stripe_mode=False
        )
        # This should raise ValueError because sample_ids[2] is not in faith_pd_values
        with pytest.raises(ValueError, match="not found|reference_sample_ids"):
            collate(batch)


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
            return collate_fn(batch, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False)

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
            return collate_fn(batch, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False)

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
            stripe_mode=False,
        )

        def custom_collate(batch):
            return collate_fn(
                batch,
                token_limit=1024,
                unifrac_distances=simple_unifrac_distances,
                unifrac_metric="unweighted",
                stripe_mode=False,
            )

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate, shuffle=False)

        for batch in dataloader:
            assert "unifrac_target" in batch
            assert batch["unifrac_target"].shape == (2, 2)
            break


class TestShuffledBatchDistances:
    """Test shuffled batch distance extraction for PYT-8.5."""

    def test_collate_fn_extracts_batch_distances_unweighted(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test collate_fn extracts batch-specific distances for unweighted UniFrac."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2, 3], [4, 1, 2]]),
                "counts": torch.FloatTensor([[10.0], [20.0]]),
                "sample_id": "sample1",
            },
            {
                "tokens": torch.LongTensor([[2, 3, 4], [1, 2, 3]]),
                "counts": torch.FloatTensor([[15.0], [25.0]]),
                "sample_id": "sample2",
            },
        ]

        token_limit = 5
        result = collate_fn(
            batch,
            token_limit=token_limit,
            unifrac_distances=simple_unifrac_distances,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        assert "unifrac_target" in result
        assert result["unifrac_target"].shape == (2, 2)
        assert isinstance(result["unifrac_target"], torch.FloatTensor)

        sample_ids = result["sample_ids"]
        loader = UniFracLoader()
        expected_distances = loader.extract_batch_distances(simple_unifrac_distances, sample_ids, metric="unweighted")
        np.testing.assert_array_almost_equal(result["unifrac_target"].numpy(), expected_distances)

    def test_collate_fn_extracts_batch_distances_shuffled_order(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test collate_fn extracts distances in shuffled batch order."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2, 3], [4, 1, 2]]),
                "counts": torch.FloatTensor([[10.0], [20.0]]),
                "sample_id": "sample3",
            },
            {
                "tokens": torch.LongTensor([[2, 3, 4], [1, 2, 3]]),
                "counts": torch.FloatTensor([[15.0], [25.0]]),
                "sample_id": "sample1",
            },
        ]

        token_limit = 5
        result = collate_fn(
            batch,
            token_limit=token_limit,
            unifrac_distances=simple_unifrac_distances,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        assert "unifrac_target" in result
        assert result["unifrac_target"].shape == (2, 2)
        sample_ids = result["sample_ids"]
        assert sample_ids == ["sample3", "sample1"]

        loader = UniFracLoader()
        expected_distances = loader.extract_batch_distances(simple_unifrac_distances, sample_ids, metric="unweighted")
        np.testing.assert_array_almost_equal(result["unifrac_target"].numpy(), expected_distances)

    def test_dataloader_shuffled_batches(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test DataLoader with shuffled batches extracts correct distances."""
        dataset = ASVDataset(
            table=rarefied_table,
            unifrac_distances=simple_unifrac_distances,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        def custom_collate(batch):
            return collate_fn(
                batch,
                token_limit=1024,
                unifrac_distances=simple_unifrac_distances,
                unifrac_metric="unweighted",
                stripe_mode=False,
            )

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate, shuffle=True)

        loader = UniFracLoader()
        for batch in dataloader:
            assert "unifrac_target" in batch
            assert batch["unifrac_target"].shape == (2, 2)
            sample_ids = batch["sample_ids"]
            expected_distances = loader.extract_batch_distances(simple_unifrac_distances, sample_ids, metric="unweighted")
            np.testing.assert_array_almost_equal(batch["unifrac_target"].numpy(), expected_distances)
            break

    def test_dataloader_non_shuffled_batches(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test DataLoader with non-shuffled batches extracts correct distances."""
        dataset = ASVDataset(
            table=rarefied_table,
            unifrac_distances=simple_unifrac_distances,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        def custom_collate(batch):
            return collate_fn(
                batch,
                token_limit=1024,
                unifrac_distances=simple_unifrac_distances,
                unifrac_metric="unweighted",
                stripe_mode=False,
            )

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate, shuffle=False)

        loader = UniFracLoader()
        for batch in dataloader:
            assert "unifrac_target" in batch
            assert batch["unifrac_target"].shape == (2, 2)
            sample_ids = batch["sample_ids"]
            expected_distances = loader.extract_batch_distances(simple_unifrac_distances, sample_ids, metric="unweighted")
            np.testing.assert_array_almost_equal(batch["unifrac_target"].numpy(), expected_distances)
            break

    def test_collate_fn_no_unifrac_distances(self, tokenizer):
        """Test collate_fn works without UniFrac distances."""
        batch = [
            {
                "tokens": torch.LongTensor([[1, 2, 3], [4, 1, 2]]),
                "counts": torch.FloatTensor([[10.0], [20.0]]),
                "sample_id": "sample1",
            },
            {
                "tokens": torch.LongTensor([[2, 3, 4], [1, 2, 3]]),
                "counts": torch.FloatTensor([[15.0], [25.0]]),
                "sample_id": "sample2",
            },
        ]

        token_limit = 5
        result = collate_fn(batch, token_limit=token_limit, unifrac_distances=None, unifrac_metric="unweighted")

        assert "unifrac_target" not in result
        assert "tokens" in result
        assert "counts" in result
        assert "sample_ids" in result


class TestDataLoaderOptimizations:
    """Test suite for DataLoader optimizations (PYT-10.3)."""

    def test_dataloader_multi_worker(self, rarefied_table, tokenizer):
        """Test DataLoader with multiple workers produces correct batches."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        custom_collate = partial(
            collate_fn, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False
        )

        # Test with 2 workers
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
        )

        all_sample_ids = []
        for batch in dataloader:
            assert "tokens" in batch
            assert "counts" in batch
            assert "sample_ids" in batch
            all_sample_ids.extend(batch["sample_ids"])

        # Verify all samples were loaded (no duplicates, no missing)
        expected_sample_ids = list(rarefied_table.ids(axis="sample"))
        assert len(all_sample_ids) == len(expected_sample_ids)
        assert set(all_sample_ids) == set(expected_sample_ids)

    def test_dataloader_multi_worker_no_corruption(self, rarefied_table, tokenizer):
        """Test that multi-worker DataLoader doesn't corrupt data."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        custom_collate = partial(
            collate_fn, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False
        )

        # Load data with single worker
        dataloader_single = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=0,
        )

        # Load data with multiple workers
        dataloader_multi = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
        )

        # Collect all batches
        batches_single = list(dataloader_single)
        batches_multi = list(dataloader_multi)

        # Verify same number of batches
        assert len(batches_single) == len(batches_multi)

        # Verify data integrity: same sample IDs (order may differ)
        sample_ids_single = [batch["sample_ids"][0] for batch in batches_single]
        sample_ids_multi = [batch["sample_ids"][0] for batch in batches_multi]

        assert set(sample_ids_single) == set(sample_ids_multi)

        # Verify tensor shapes are consistent
        for batch_single, batch_multi in zip(batches_single, batches_multi):
            assert batch_single["tokens"].shape == batch_multi["tokens"].shape
            assert batch_single["counts"].shape == batch_multi["counts"].shape

    def test_dataloader_prefetch_factor(self, rarefied_table, tokenizer):
        """Test that prefetch_factor parameter works correctly."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        custom_collate = partial(
            collate_fn, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False
        )

        # Test with prefetch_factor
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
        )

        # Should iterate without errors
        batch_count = 0
        for batch in dataloader:
            assert "tokens" in batch
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break

        assert batch_count > 0

    def test_dataloader_pin_memory_cpu(self, rarefied_table, tokenizer):
        """Test that pin_memory works on CPU (should not error)."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        custom_collate = partial(
            collate_fn, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False
        )

        # pin_memory=True should work on CPU (though not as effective)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Should iterate without errors
        for batch in dataloader:
            assert "tokens" in batch
            break

    def test_dataloader_multi_worker_with_unifrac(self, rarefied_table, tokenizer, simple_unifrac_distances):
        """Test multi-worker DataLoader with UniFrac distances."""
        dataset = ASVDataset(
            table=rarefied_table,
            unifrac_distances=simple_unifrac_distances,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        custom_collate = partial(
            collate_fn,
            token_limit=1024,
            unifrac_distances=simple_unifrac_distances,
            unifrac_metric="unweighted",
            stripe_mode=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
        )

        # Verify batches have correct structure
        for batch in dataloader:
            assert "unifrac_target" in batch
            assert batch["unifrac_target"].shape[0] == batch["tokens"].shape[0]
            break

    def test_dataloader_multi_worker_shuffled(self, rarefied_table, tokenizer):
        """Test multi-worker DataLoader with shuffling."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        custom_collate = partial(
            collate_fn, token_limit=1024, unifrac_distances=None, unifrac_metric="unweighted", stripe_mode=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
        )

        # Should iterate without errors
        all_sample_ids = []
        for batch in dataloader:
            all_sample_ids.extend(batch["sample_ids"])

        # Verify all samples were loaded
        expected_sample_ids = list(rarefied_table.ids(axis="sample"))
        assert len(all_sample_ids) == len(expected_sample_ids)
        assert set(all_sample_ids) == set(expected_sample_ids)


class TestDatasetMetadataColumnHandling:
    """Tests for metadata column name handling in ASVDataset."""

    def test_dataset_with_whitespace_in_sample_id_column(self, rarefied_table, tokenizer):
        """Test that dataset handles whitespace in sample_id column name."""
        metadata = pd.DataFrame(
            {
                " sample_id ": ["sample1", "sample2", "sample3"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
        )
        assert len(dataset) > 0

    def test_dataset_with_missing_sample_id_column(self, rarefied_table, tokenizer):
        """Test that dataset raises helpful error when sample_id column missing."""
        metadata = pd.DataFrame(
            {
                "id": ["sample1", "sample2", "sample3"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="sample_id"):
            ASVDataset(
                table=rarefied_table,
                tokenizer=tokenizer,
                metadata=metadata,
                target_column="target",
            )

    def test_dataset_with_trailing_whitespace_in_sample_id_column(self, rarefied_table, tokenizer):
        """Test that dataset handles trailing whitespace in sample_id column name."""
        metadata = pd.DataFrame(
            {
                "sample_id ": ["sample1", "sample2", "sample3"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
        )
        assert len(dataset) > 0

    def test_dataset_with_leading_whitespace_in_sample_id_column(self, rarefied_table, tokenizer):
        """Test that dataset handles leading whitespace in sample_id column name."""
        metadata = pd.DataFrame(
            {
                " sample_id": ["sample1", "sample2", "sample3"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
        )
        assert len(dataset) > 0

    def test_dataset_with_normal_sample_id_column(self, rarefied_table, tokenizer, simple_metadata):
        """Test that dataset works with normal sample_id column (regression test)."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=simple_metadata,
            target_column="target",
        )
        assert len(dataset) > 0


class TestTargetNormalization:
    """Test suite for target normalization (PYT-11.9)."""

    def test_normalize_targets_disabled_by_default(self, rarefied_table, tokenizer, simple_metadata):
        """Test that target normalization is disabled by default."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=simple_metadata,
            target_column="target",
        )

        assert dataset.normalize_targets is False
        assert dataset.get_normalization_params() is None

        # Targets should be raw values
        sample = dataset[0]
        assert "y_target" in sample
        # Raw target is 1.0 for sample1
        assert sample["y_target"].item() == 1.0

    def test_normalize_targets_enabled(self, rarefied_table, tokenizer):
        """Test that target normalization normalizes values to [0, 1]."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [0.0, 50.0, 100.0],  # Simple range for easy verification
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
        )

        assert dataset.normalize_targets is True
        params = dataset.get_normalization_params()
        assert params is not None
        assert params["target_min"] == 0.0
        assert params["target_max"] == 100.0
        assert params["target_scale"] == 100.0

        # Check normalized values
        sample_ids = list(rarefied_table.ids(axis="sample"))
        for idx, sid in enumerate(sample_ids):
            sample = dataset[idx]
            expected_raw = metadata[metadata["sample_id"] == sid]["target"].values[0]
            expected_normalized = (expected_raw - 0.0) / 100.0
            np.testing.assert_almost_equal(sample["y_target"].item(), expected_normalized)

    def test_normalize_targets_with_negative_values(self, rarefied_table, tokenizer):
        """Test target normalization with negative values."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [-50.0, 0.0, 50.0],
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
        )

        params = dataset.get_normalization_params()
        assert params["target_min"] == -50.0
        assert params["target_max"] == 50.0
        assert params["target_scale"] == 100.0

        # Check first sample (target=-50 should normalize to 0)
        sample_ids = list(rarefied_table.ids(axis="sample"))
        sample1_idx = sample_ids.index("sample1")
        sample = dataset[sample1_idx]
        np.testing.assert_almost_equal(sample["y_target"].item(), 0.0)

    def test_normalize_targets_with_external_params(self, rarefied_table, tokenizer):
        """Test target normalization with externally provided min/max."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [25.0, 50.0, 75.0],
            }
        )

        # Provide external min/max (e.g., from training set)
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
            target_min=0.0,  # External min
            target_max=100.0,  # External max
        )

        params = dataset.get_normalization_params()
        assert params["target_min"] == 0.0
        assert params["target_max"] == 100.0

        # Check that values are normalized using external params
        sample_ids = list(rarefied_table.ids(axis="sample"))
        sample1_idx = sample_ids.index("sample1")
        sample = dataset[sample1_idx]
        np.testing.assert_almost_equal(sample["y_target"].item(), 0.25)  # 25/100

    def test_denormalize_targets_tensor(self, rarefied_table, tokenizer):
        """Test denormalize_targets method with tensor input."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [0.0, 50.0, 100.0],
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
        )

        normalized = torch.tensor([0.0, 0.5, 1.0])
        denormalized = dataset.denormalize_targets(normalized)
        expected = torch.tensor([0.0, 50.0, 100.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_targets_numpy(self, rarefied_table, tokenizer):
        """Test denormalize_targets method with numpy array input."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [0.0, 50.0, 100.0],
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
        )

        normalized = np.array([0.0, 0.5, 1.0])
        denormalized = dataset.denormalize_targets(normalized)
        expected = np.array([0.0, 50.0, 100.0])
        np.testing.assert_array_almost_equal(denormalized, expected)

    def test_denormalize_targets_float(self, rarefied_table, tokenizer):
        """Test denormalize_targets method with float input."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [0.0, 50.0, 100.0],
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
        )

        denormalized = dataset.denormalize_targets(0.5)
        assert denormalized == 50.0

    def test_denormalize_targets_no_normalization(self, rarefied_table, tokenizer, simple_metadata):
        """Test denormalize_targets returns input unchanged when normalization disabled."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=simple_metadata,
            target_column="target",
            normalize_targets=False,
        )

        values = torch.tensor([1.0, 2.0, 3.0])
        result = dataset.denormalize_targets(values)
        torch.testing.assert_close(result, values)

    def test_normalize_targets_identical_values(self, rarefied_table, tokenizer):
        """Test target normalization with identical target values (edge case)."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "target": [50.0, 50.0, 50.0],  # All same value
            }
        )

        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            metadata=metadata,
            target_column="target",
            normalize_targets=True,
        )

        # Should not crash, scale should be 1.0 to avoid division by zero
        params = dataset.get_normalization_params()
        assert params["target_scale"] == 1.0

        # All normalized values should be 0 (since target - min = 0)
        sample = dataset[0]
        assert sample["y_target"].item() == 0.0


class TestSequenceCache:
    """Test suite for sequence tokenization caching (PYT-12.3)."""

    def test_cache_enabled_by_default(self, rarefied_table, tokenizer):
        """Test that sequence caching is enabled by default."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        assert dataset.cache_sequences is True
        assert dataset._sequence_cache is not None
        assert len(dataset._sequence_cache) > 0

    def test_cache_disabled(self, rarefied_table, tokenizer):
        """Test that sequence caching can be disabled."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            cache_sequences=False,
        )

        assert dataset.cache_sequences is False
        assert dataset._sequence_cache is None

    def test_cache_contains_all_sequences(self, rarefied_table, tokenizer):
        """Test that cache contains all sequences from the table."""
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
        )

        # Cache should have entry for each sequence
        assert len(dataset._sequence_cache) == len(dataset.sequences)
        for seq in dataset.sequences:
            assert seq in dataset._sequence_cache

    def test_cache_produces_same_output(self, rarefied_table, tokenizer):
        """Test that cached and non-cached datasets produce identical output."""
        dataset_cached = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            cache_sequences=True,
        )

        dataset_uncached = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=150,
            token_limit=1024,
            cache_sequences=False,
        )

        # Compare output for each sample
        for idx in range(len(dataset_cached)):
            sample_cached = dataset_cached[idx]
            sample_uncached = dataset_uncached[idx]

            assert sample_cached["sample_id"] == sample_uncached["sample_id"]
            torch.testing.assert_close(sample_cached["tokens"], sample_uncached["tokens"])
            torch.testing.assert_close(sample_cached["counts"], sample_uncached["counts"])

    def test_cache_tensor_shape(self, rarefied_table, tokenizer):
        """Test that cached tensors have correct shape."""
        max_bp = 150
        dataset = ASVDataset(
            table=rarefied_table,
            tokenizer=tokenizer,
            max_bp=max_bp,
            token_limit=1024,
        )

        # Each cached tensor should have shape [max_bp + 1] (including START token)
        for seq, cached_tensor in dataset._sequence_cache.items():
            assert cached_tensor.shape == (max_bp + 1,)
            assert cached_tensor.dtype == torch.long
