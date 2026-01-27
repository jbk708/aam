"""Tests for batch size finder utility."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from unittest.mock import patch, MagicMock
from typing import Dict, List, Union

from aam.training.batch_size_finder import BatchSizeFinder, BatchSizeFinderResult
from aam.data.dataset import collate_fn
from aam.data.tokenizer import SequenceTokenizer


class SimpleASVDataset(Dataset):
    """Simple dataset that mimics ASVDataset output format."""

    def __init__(self, num_samples: int, num_asvs: int, seq_length: int):
        self.num_samples = num_samples
        self.num_asvs = num_asvs
        self.seq_length = seq_length
        self.sample_ids = [f"sample_{i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str]]:
        tokens = torch.randint(1, 5, (self.num_asvs, self.seq_length))
        tokens[:, 0] = SequenceTokenizer.START_TOKEN
        counts = torch.rand(self.num_asvs, 1)
        return {
            "tokens": tokens,
            "counts": counts,
            "sample_id": self.sample_ids[idx],
        }


def simple_collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]],
    token_limit: int = 64,
) -> Dict[str, torch.Tensor]:
    """Simple collate function for testing."""
    batch_size = len(batch)
    max_asvs = max(b["tokens"].shape[0] for b in batch)
    seq_length = batch[0]["tokens"].shape[1]

    padded_tokens = torch.zeros(batch_size, min(max_asvs, token_limit), seq_length, dtype=torch.long)
    padded_counts = torch.zeros(batch_size, min(max_asvs, token_limit), 1)

    for i, sample in enumerate(batch):
        num_asvs = min(sample["tokens"].shape[0], token_limit)
        padded_tokens[i, :num_asvs] = sample["tokens"][:num_asvs]
        padded_counts[i, :num_asvs] = sample["counts"][:num_asvs]

    unifrac_target = torch.rand(batch_size, batch_size)
    unifrac_target.fill_diagonal_(0.0)
    unifrac_target = (unifrac_target + unifrac_target.T) / 2

    return {
        "tokens": padded_tokens,
        "counts": padded_counts,
        "unifrac_target": unifrac_target,
    }


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    return SimpleASVDataset(num_samples=100, num_asvs=32, seq_length=50)


class TestBatchSizeFinderResult:
    """Test BatchSizeFinderResult dataclass."""

    def test_result_creation(self):
        """Test creating a BatchSizeFinderResult."""
        result = BatchSizeFinderResult(
            batch_size=16,
            gradient_accumulation_steps=2,
            effective_batch_size=32,
            peak_memory_mb=1024.0,
            memory_fraction=0.75,
        )
        assert result.batch_size == 16
        assert result.gradient_accumulation_steps == 2
        assert result.effective_batch_size == 32
        assert result.peak_memory_mb == 1024.0
        assert result.memory_fraction == 0.75


class TestBatchSizeFinder:
    """Test BatchSizeFinder utility."""

    def test_initialization(self, small_model, loss_fn, device):
        """Test BatchSizeFinder initialization."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        assert finder.model is small_model
        assert finder.loss_fn is loss_fn
        assert finder.device == device

    def test_initialization_cpu_raises_error(self, small_model, loss_fn):
        """Test that BatchSizeFinder raises error for CPU device."""
        cpu_device = torch.device("cpu")
        with pytest.raises(ValueError, match="CUDA"):
            BatchSizeFinder(
                model=small_model,
                loss_fn=loss_fn,
                device=cpu_device,
                collate_fn=simple_collate_fn,
            )

    def test_round_to_even(self, small_model, loss_fn, device):
        """Test _round_to_even method."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        assert finder._round_to_even(1) == 2
        assert finder._round_to_even(2) == 2
        assert finder._round_to_even(3) == 2
        assert finder._round_to_even(4) == 4
        assert finder._round_to_even(5) == 4
        assert finder._round_to_even(17) == 16

    def test_find_batch_size_returns_even(self, small_model, loss_fn, simple_dataset, device):
        """Test that find_batch_size always returns an even batch size."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        result = finder.find_batch_size(
            dataset=simple_dataset,
            min_batch_size=2,
            max_batch_size=32,
            max_memory_fraction=0.8,
        )

        assert result.batch_size % 2 == 0
        assert result.batch_size >= 2

    def test_find_batch_size_respects_min_max(self, small_model, loss_fn, simple_dataset, device):
        """Test that find_batch_size respects min/max constraints."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        result = finder.find_batch_size(
            dataset=simple_dataset,
            min_batch_size=4,
            max_batch_size=16,
            max_memory_fraction=0.8,
        )

        assert result.batch_size >= 4
        assert result.batch_size <= 16

    def test_find_batch_size_with_gradient_accumulation(self, small_model, loss_fn, simple_dataset, device):
        """Test that gradient accumulation is computed correctly."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        target_effective = 32
        result = finder.find_batch_size(
            dataset=simple_dataset,
            min_batch_size=2,
            max_batch_size=64,
            target_effective_batch_size=target_effective,
            max_memory_fraction=0.8,
        )

        assert result.effective_batch_size == result.batch_size * result.gradient_accumulation_steps
        if result.batch_size < target_effective:
            assert result.gradient_accumulation_steps > 1

    def test_find_batch_size_result_has_memory_info(self, small_model, loss_fn, simple_dataset, device):
        """Test that result includes memory information."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        result = finder.find_batch_size(
            dataset=simple_dataset,
            min_batch_size=2,
            max_batch_size=32,
            max_memory_fraction=0.8,
        )

        assert result.peak_memory_mb > 0
        assert 0 < result.memory_fraction <= 1.0

    def test_try_batch_size_clears_cache(self, small_model, loss_fn, simple_dataset, device):
        """Test that _try_batch_size clears CUDA cache."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        initial_memory = torch.cuda.memory_allocated()
        success, peak_memory = finder._try_batch_size(4, simple_dataset, num_iterations=1)
        final_memory = torch.cuda.memory_allocated()

        assert success is True
        assert abs(final_memory - initial_memory) < 10 * 1024 * 1024  # Within 10MB

    def test_invalid_min_batch_size(self, small_model, loss_fn, simple_dataset, device):
        """Test that invalid min_batch_size raises error."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        with pytest.raises(ValueError, match="min_batch_size must be at least 2"):
            finder.find_batch_size(
                dataset=simple_dataset,
                min_batch_size=1,
                max_batch_size=32,
            )

    def test_invalid_memory_fraction(self, small_model, loss_fn, simple_dataset, device):
        """Test that invalid max_memory_fraction raises error."""
        if device.type != "cuda":
            pytest.skip("BatchSizeFinder requires CUDA device")

        small_model = small_model.to(device)
        finder = BatchSizeFinder(
            model=small_model,
            loss_fn=loss_fn,
            device=device,
            collate_fn=simple_collate_fn,
        )

        with pytest.raises(ValueError, match="max_memory_fraction"):
            finder.find_batch_size(
                dataset=simple_dataset,
                min_batch_size=2,
                max_batch_size=32,
                max_memory_fraction=1.5,
            )


class TestBatchSizeFinderMocked:
    """Test BatchSizeFinder with mocked CUDA operations."""

    def test_binary_search_finds_optimal(self):
        """Test that binary search finds optimal batch size."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.train = MagicMock()
        model.eval = MagicMock()
        loss_fn = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_properties") as mock_props:
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB

                with patch.object(BatchSizeFinder, "__init__", return_value=None):
                    finder = BatchSizeFinder.__new__(BatchSizeFinder)
                    finder.model = model
                    finder.loss_fn = loss_fn
                    finder.device = torch.device("cuda")
                    finder.collate_fn = simple_collate_fn

                    memory_per_sample = 100 * 1024**2  # 100MB per sample
                    max_memory = 8 * 1024**3 * 0.8  # 80% of 8GB

                    def mock_try_batch_size(batch_size, dataset, num_iterations):
                        memory_used = batch_size * memory_per_sample
                        success = memory_used <= max_memory
                        return success, memory_used / (1024**2)

                    finder._try_batch_size = mock_try_batch_size
                    finder._round_to_even = lambda x: x if x % 2 == 0 else x - 1

                    with patch("torch.cuda.get_device_properties") as mock_props2:
                        mock_props2.return_value.total_memory = 8 * 1024**3

                        result = finder.find_batch_size(
                            dataset=MagicMock(__len__=MagicMock(return_value=100)),
                            min_batch_size=2,
                            max_batch_size=128,
                            max_memory_fraction=0.8,
                        )

                    expected_max = int(max_memory / memory_per_sample)
                    expected_even = expected_max if expected_max % 2 == 0 else expected_max - 1

                    assert result.batch_size <= expected_even
                    assert result.batch_size >= 2
                    assert result.batch_size % 2 == 0
