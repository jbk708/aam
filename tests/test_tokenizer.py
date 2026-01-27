"""Unit tests for SequenceTokenizer class."""

import pytest
import torch

from aam.data.tokenizer import SequenceTokenizer
from conftest import generate_150bp_sequence


@pytest.fixture
def tokenizer():
    """Create a SequenceTokenizer instance."""
    return SequenceTokenizer()


class TestSequenceTokenizer:
    """Test suite for SequenceTokenizer class."""

    def test_init(self, tokenizer):
        """Test SequenceTokenizer initialization."""
        assert tokenizer is not None
        assert isinstance(tokenizer, SequenceTokenizer)

    def test_tokenize_basic(self, tokenizer):
        """Test basic tokenization of a simple sequence."""
        sequence = "ACGT"
        result = tokenizer.tokenize(sequence)

        assert isinstance(result, torch.LongTensor)
        assert len(result) == 5

    def test_tokenize_mapping(self, tokenizer):
        """Test that start token is prepended and nucleotide mapping is correct."""
        sequence = "ACGT"
        result = tokenizer.tokenize(sequence)

        assert result[0].item() == SequenceTokenizer.START_TOKEN
        assert result[1].item() == 1
        assert result[2].item() == 2
        assert result[3].item() == 3
        assert result[4].item() == 4

    def test_tokenize_150bp(self, tokenizer):
        """Test tokenization of 150bp sequence."""
        sequence = generate_150bp_sequence(seed=42)
        result = tokenizer.tokenize(sequence)

        assert isinstance(result, torch.LongTensor)
        assert len(result) == 151
        assert result[0].item() == SequenceTokenizer.START_TOKEN
        assert all(1 <= val.item() <= 4 for val in result[1:])

    def test_tokenize_batch_basic(self, tokenizer):
        """Test batch tokenization."""
        sequences = ["ACGT", "TGCA", "AAAA"]
        result = tokenizer.tokenize_batch(sequences)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(t, torch.LongTensor) for t in result)
        assert len(result[0]) == 5
        assert len(result[1]) == 5
        assert len(result[2]) == 5
        assert all(t[0].item() == SequenceTokenizer.START_TOKEN for t in result)

    def test_tokenize_batch_different_lengths(self, tokenizer):
        """Test batch tokenization with different length sequences."""
        sequences = ["ACGT", "ACGTACGT", "A"]
        result = tokenizer.tokenize_batch(sequences)

        assert len(result) == 3
        assert len(result[0]) == 5
        assert len(result[1]) == 9
        assert len(result[2]) == 2
        assert all(t[0].item() == SequenceTokenizer.START_TOKEN for t in result)

    def test_pad_sequences_basic(self, tokenizer):
        """Test basic sequence padding."""
        sequences = [
            torch.LongTensor([1, 2, 3]),
            torch.LongTensor([4, 1]),
            torch.LongTensor([2, 3, 4, 1, 2]),
        ]
        max_length = 5
        result = tokenizer.pad_sequences(sequences, max_length)

        assert isinstance(result, torch.LongTensor)
        assert result.shape == (3, 5)
        assert torch.all(result[0, :3] == torch.LongTensor([1, 2, 3]))
        assert torch.all(result[0, 3:] == 0)
        assert torch.all(result[1, :2] == torch.LongTensor([4, 1]))
        assert torch.all(result[1, 2:] == 0)

    def test_pad_sequences_exact_length(self, tokenizer):
        """Test padding when sequences are exactly max_length."""
        sequences = [
            torch.LongTensor([1, 2, 3, 4]),
            torch.LongTensor([4, 3, 2, 1]),
        ]
        max_length = 4
        result = tokenizer.pad_sequences(sequences, max_length)

        assert result.shape == (2, 4)
        assert torch.all(result[0] == torch.LongTensor([1, 2, 3, 4]))
        assert torch.all(result[1] == torch.LongTensor([4, 3, 2, 1]))

    def test_pad_sequences_longer_than_max(self, tokenizer):
        """Test padding when sequences are longer than max_length."""
        sequences = [
            torch.LongTensor([1, 2, 3, 4, 5]),
        ]
        max_length = 3
        result = tokenizer.pad_sequences(sequences, max_length)

        assert result.shape == (1, 3)
        assert len(result[0]) == 3

    def test_pad_sequences_empty(self, tokenizer):
        """Test padding with empty sequence list."""
        sequences = []
        max_length = 5
        result = tokenizer.pad_sequences(sequences, max_length)

        assert isinstance(result, torch.LongTensor)
        assert result.shape == (0, 5)

    def test_tokenize_empty_sequence(self, tokenizer):
        """Test tokenization of empty sequence (should still have start token)."""
        sequence = ""
        result = tokenizer.tokenize(sequence)

        assert isinstance(result, torch.LongTensor)
        assert len(result) == 1
        assert result[0].item() == SequenceTokenizer.START_TOKEN

    def test_tokenize_invalid_characters(self, tokenizer):
        """Test tokenization with invalid characters (should handle gracefully)."""
        sequence = "ACGTN"
        result = tokenizer.tokenize(sequence)

        assert isinstance(result, torch.LongTensor)
        assert len(result) == 6
        assert result[0].item() == SequenceTokenizer.START_TOKEN

    def test_start_token_constant(self, tokenizer):
        """Test that START_TOKEN constant is defined correctly."""
        assert hasattr(SequenceTokenizer, "START_TOKEN")
        assert SequenceTokenizer.START_TOKEN == 5

    def test_start_token_never_masked(self, tokenizer):
        """Test that start token (ID 5) is never masked (mask checks for > 0)."""
        sequence = "ACGT"
        result = tokenizer.tokenize(sequence)
        mask = (result > 0).long()

        assert mask[0].item() == 1
        assert all(mask[i].item() == 1 for i in range(len(result)))

    def test_start_token_with_padding(self, tokenizer):
        """Test that start token is preserved when padding sequences."""
        sequence = "ACGT"
        tokenized = tokenizer.tokenize(sequence)
        padded = tokenizer.pad_sequences([tokenized], max_length=10)[0]

        assert padded[0].item() == SequenceTokenizer.START_TOKEN
        assert padded[1].item() == 1
        assert padded[2].item() == 2
        assert padded[3].item() == 3
        assert padded[4].item() == 4
        assert all(padded[i].item() == 0 for i in range(5, 10))
