"""Sequence tokenization for microbial sequencing data."""

from typing import List, Union
import torch


class SequenceTokenizer:
    """Tokenize nucleotide sequences for model input."""

    START_TOKEN = 5
    MASK_TOKEN = 6

    def __init__(self):
        """Initialize SequenceTokenizer."""
        self.nucleotide_to_token = {"A": 1, "C": 2, "G": 3, "T": 4}

    def tokenize(self, sequence: str) -> torch.LongTensor:
        """Convert nucleotide sequence string to tokens.

        Args:
            sequence: Nucleotide sequence string (A, C, G, T)

        Returns:
            torch.LongTensor of token IDs with start token prepended
        """
        tokens = [self.START_TOKEN]
        for char in sequence.upper():
            token = self.nucleotide_to_token.get(char, 0)
            tokens.append(token)
        return torch.LongTensor(tokens)

    def tokenize_batch(self, sequences: List[str]) -> List[torch.LongTensor]:
        """Tokenize a batch of sequences.

        Args:
            sequences: List of nucleotide sequence strings

        Returns:
            List of torch.LongTensor token sequences
        """
        return [self.tokenize(seq) for seq in sequences]

    def pad_sequences(self, sequences: List[torch.LongTensor], max_length: int) -> torch.LongTensor:
        """Pad sequences to max_length.

        Args:
            sequences: List of token sequences
            max_length: Maximum sequence length

        Returns:
            torch.LongTensor of shape [batch_size, max_length]
        """
        if not sequences:
            return torch.LongTensor([]).reshape(0, max_length)

        batch_size = len(sequences)
        padded = torch.zeros(batch_size, max_length, dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_length)
            padded[i, :seq_len] = seq[:seq_len]

        return padded
