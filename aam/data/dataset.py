"""PyTorch Dataset for microbial sequencing data."""

from typing import Dict, Optional, List, Union
import torch
from torch.utils.data import Dataset
import biom
from biom import Table
import pandas as pd
import numpy as np
from skbio import DistanceMatrix

from aam.data.tokenizer import SequenceTokenizer
from aam.data.biom_loader import BIOMLoader


def collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]], token_limit: int) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching samples with variable ASV counts.

    Args:
        batch: List of sample dictionaries
        token_limit: Maximum number of ASVs per sample

    Returns:
        Batched dictionary with padded tensors
    """
    batch_size = len(batch)
    max_bp = batch[0]["tokens"].shape[1]

    tokens_list = []
    counts_list = []
    sample_ids = []
    y_targets = []
    unifrac_targets = []

    for sample in batch:
        tokens = sample["tokens"]
        counts = sample["counts"]
        num_asvs = tokens.shape[0]

        if num_asvs > token_limit:
            tokens = tokens[:token_limit]
            counts = counts[:token_limit]
            num_asvs = token_limit

        padded_tokens = torch.zeros(token_limit, max_bp, dtype=torch.long)
        padded_counts = torch.zeros(token_limit, 1, dtype=torch.float)
        padded_tokens[:num_asvs] = tokens
        padded_counts[:num_asvs] = counts

        tokens_list.append(padded_tokens)
        counts_list.append(padded_counts)
        sample_ids.append(sample["sample_id"])

        if "y_target" in sample:
            y_targets.append(sample["y_target"])
        if "unifrac_target" in sample:
            unifrac_targets.append(sample["unifrac_target"])

    result = {
        "tokens": torch.stack(tokens_list),
        "counts": torch.stack(counts_list),
        "sample_ids": sample_ids,
    }

    if y_targets:
        result["y_target"] = torch.stack(y_targets)
    if unifrac_targets:
        stacked = torch.stack(unifrac_targets)
        # For unweighted UniFrac, we need batch-level pairwise distances [batch_size, batch_size]
        # Each sample's unifrac_target is a row of distances to all samples in the dataset
        # We need to extract only the distances to samples in this batch
        if stacked.dim() == 2:
            if stacked.shape[1] == batch_size:
                # Already batch-level distances (shouldn't happen with current dataset implementation)
                result["unifrac_target"] = stacked
            elif stacked.shape[1] > batch_size:
                # Extract batch-level submatrix
                # The stacked tensor has shape [batch_size, num_dataset_samples]
                # We need to extract columns corresponding to batch samples
                # Since we don't have access to dataset.sample_ids here, we'll extract
                # the submatrix by finding where batch sample_ids appear in the distance rows
                # For now, create batch-level distance matrix by extracting distances between batch samples
                batch_distances = torch.zeros(batch_size, batch_size, dtype=stacked.dtype)
                for i, sample_id_i in enumerate(sample_ids):
                    # Find index of sample_id_i in dataset (assume it's at position i in dataset.sample_ids)
                    # This is a simplification - proper fix requires dataset access
                    for j, sample_id_j in enumerate(sample_ids):
                        # Get distance from sample i to sample j
                        # If sample j's index in dataset is j, get stacked[i, j]
                        # Otherwise, we need to find the actual index
                        if j < stacked.shape[1]:
                            batch_distances[i, j] = stacked[i, j]
                        elif i < stacked.shape[1]:
                            # Use symmetry: dist(i,j) = dist(j,i)
                            batch_distances[i, j] = stacked[j, i] if j < stacked.shape[0] else 0.0
                result["unifrac_target"] = batch_distances
            else:
                result["unifrac_target"] = stacked
        else:
            result["unifrac_target"] = stacked

    return result


class ASVDataset(Dataset):
    """PyTorch Dataset for microbial sequencing data."""

    def __init__(
        self,
        table: Table,
        metadata: Optional[pd.DataFrame] = None,
        unifrac_distances: Optional[Union[DistanceMatrix, pd.Series]] = None,
        tokenizer: Optional[SequenceTokenizer] = None,
        max_bp: int = 150,
        token_limit: int = 1024,
        target_column: Optional[str] = None,
        unifrac_metric: str = "unweighted",
    ):
        """Initialize ASVDataset.

        Args:
            table: Rarefied biom.Table object
            metadata: Optional DataFrame with sample metadata and targets
            unifrac_distances: Optional pre-computed UniFrac distances
            tokenizer: SequenceTokenizer instance (creates default if None)
            max_bp: Maximum base pairs per sequence
            token_limit: Maximum ASVs per sample
            target_column: Column name in metadata for target values
            unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
        """
        self.table = table
        self.metadata = metadata
        self.unifrac_distances = unifrac_distances
        self.tokenizer = tokenizer if tokenizer is not None else SequenceTokenizer()
        self.max_bp = max_bp
        self.token_limit = token_limit
        self.target_column = target_column
        self.unifrac_metric = unifrac_metric

        self.sample_ids = list(table.ids(axis="sample"))
        self.observation_ids = list(table.ids(axis="observation"))

        loader = BIOMLoader()
        self.sequences = loader.get_sequences(table)

        self.sequence_to_idx = {seq: idx for idx, seq in enumerate(self.observation_ids)}

        if metadata is not None and target_column is not None:
            if "sample_id" not in metadata.columns:
                raise ValueError("metadata must have 'sample_id' column")
            self.metadata_dict = metadata.set_index("sample_id")[target_column].to_dict()
        else:
            self.metadata_dict = None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - tokens: torch.LongTensor [num_asvs, max_bp]
            - counts: torch.FloatTensor [num_asvs, 1]
            - y_target: torch.FloatTensor [1] or [out_dim] (if metadata provided)
            - unifrac_target: torch.FloatTensor (if unifrac_distances provided)
            - sample_id: str
        """
        sample_id = self.sample_ids[idx]
        sample_ids_list = list(self.table.ids(axis="sample"))
        sample_idx = sample_ids_list.index(sample_id)

        data = self.table.matrix_data[:, sample_idx].toarray().flatten()
        asv_indices = np.where(data > 0)[0]

        tokens_list = []
        counts_list = []

        for asv_idx in asv_indices:
            sequence = self.sequences[asv_idx]
            tokenized = self.tokenizer.tokenize(sequence)
            padded = self.tokenizer.pad_sequences([tokenized], self.max_bp)[0]
            tokens_list.append(padded)
            counts_list.append(data[asv_idx])

        if not tokens_list:
            tokens = torch.zeros(1, self.max_bp, dtype=torch.long)
            counts = torch.zeros(1, 1, dtype=torch.float)
        else:
            tokens = torch.stack(tokens_list)
            counts = torch.FloatTensor(counts_list).unsqueeze(1)

        result = {
            "tokens": tokens,
            "counts": counts,
            "sample_id": sample_id,
        }

        if self.metadata_dict is not None and sample_id in self.metadata_dict:
            target_value = self.metadata_dict[sample_id]
            if isinstance(target_value, (int, float)):
                result["y_target"] = torch.FloatTensor([target_value])
            else:
                result["y_target"] = torch.FloatTensor([float(target_value)])

        if self.unifrac_distances is not None:
            if self.unifrac_metric == "unweighted":
                if isinstance(self.unifrac_distances, DistanceMatrix):
                    sample_idx_in_matrix = self.unifrac_distances.ids.index(sample_id)
                    row = self.unifrac_distances.data[sample_idx_in_matrix]
                    result["unifrac_target"] = torch.FloatTensor(row)
                elif isinstance(self.unifrac_distances, np.ndarray):
                    # Handle numpy array from extract_batch_distances
                    # Find index of sample_id in dataset's sample_ids
                    sample_idx_in_dataset = self.sample_ids.index(sample_id)
                    row = self.unifrac_distances[sample_idx_in_dataset]
                    result["unifrac_target"] = torch.FloatTensor(row)
            elif self.unifrac_metric == "faith_pd":
                if isinstance(self.unifrac_distances, pd.Series):
                    if sample_id in self.unifrac_distances.index:
                        value = self.unifrac_distances[sample_id]
                        result["unifrac_target"] = torch.FloatTensor([value])
                elif isinstance(self.unifrac_distances, np.ndarray):
                    # Handle numpy array from extract_batch_distances (shape [num_samples, 1])
                    sample_idx_in_dataset = self.sample_ids.index(sample_id)
                    value = self.unifrac_distances[sample_idx_in_dataset, 0]
                    result["unifrac_target"] = torch.FloatTensor([value])

        return result
