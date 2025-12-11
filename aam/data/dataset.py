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
from aam.data.unifrac import UniFracComputer


def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]],
    token_limit: int,
    unifrac_distances: Optional[Union[DistanceMatrix, pd.Series, np.ndarray]] = None,
    unifrac_metric: str = "unweighted",
    unifrac_computer: Optional["UniFracComputer"] = None,
    lazy_unifrac: bool = True,
    stripe_mode: bool = True,
    reference_sample_ids: Optional[List[str]] = None,
    all_sample_ids: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching samples with variable ASV counts.

    Args:
        batch: List of sample dictionaries
        token_limit: Maximum number of ASVs per sample
        unifrac_distances: Optional full distance matrix/series/stripe matrix for batch extraction
        unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
        unifrac_computer: UniFracComputer instance for lazy computation
        lazy_unifrac: If True, compute distances on-the-fly (default: True)
        stripe_mode: If True, use stripe-based distances instead of pairwise (default: True)
        reference_sample_ids: List of reference sample IDs for stripe mode
        all_sample_ids: List of all sample IDs for stripe extraction (required if stripe_mode=True and not lazy)

    Returns:
        Batched dictionary with padded tensors
    """
    batch_size = len(batch)
    max_bp = batch[0]["tokens"].shape[1]

    tokens_list = []
    counts_list = []
    sample_ids = []
    y_targets = []

    for sample in batch:
        tokens = sample["tokens"]
        counts = sample["counts"]
        num_asvs = tokens.shape[0]

        if num_asvs > token_limit:
            tokens = tokens[:token_limit]
            counts = counts[:token_limit]
            num_asvs = token_limit

        # Verify sample has at least one ASV with count > 0
        # This prevents all-padding samples that cause NaN in attention pooling
        if num_asvs == 0 or (counts.sum() == 0).all():
            import sys
            error_msg = (
                f"Sample {sample['sample_id']} has no ASVs with count > 0 "
                f"(num_asvs={num_asvs}, counts_sum={counts.sum().item()})"
            )
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)

        padded_tokens = torch.zeros(token_limit, max_bp, dtype=torch.long)
        padded_counts = torch.zeros(token_limit, 1, dtype=torch.float)
        padded_tokens[:num_asvs] = tokens
        padded_counts[:num_asvs] = counts

        tokens_list.append(padded_tokens)
        counts_list.append(padded_counts)
        sample_ids.append(sample["sample_id"])

        if "y_target" in sample:
            y_targets.append(sample["y_target"])

    result = {
        "tokens": torch.stack(tokens_list),
        "counts": torch.stack(counts_list),
        "sample_ids": sample_ids,
    }

    if y_targets:
        result["y_target"] = torch.stack(y_targets)

    if unifrac_distances is not None or (lazy_unifrac and unifrac_computer is not None):
        if stripe_mode:
            # Stripe-based computation
            if lazy_unifrac and unifrac_computer is not None:
                # Lazy stripe computation: compute distances on-the-fly for this batch
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Computing stripe UniFrac distances for batch: {sample_ids[:3]}... (batch size: {len(sample_ids)})")
                if reference_sample_ids is None:
                    raise ValueError("reference_sample_ids required for stripe mode")
                if unifrac_metric == "unweighted":
                    batch_distances = unifrac_computer.compute_batch_unweighted_stripe(
                        sample_ids, reference_sample_ids=reference_sample_ids
                    )
                else:
                    # Faith PD doesn't support stripe mode, fall back to pairwise
                    logger.warning("Faith PD doesn't support stripe mode, falling back to pairwise computation")
                    batch_distances = unifrac_computer.compute_batch_faith_pd(sample_ids)
                logger.debug(f"Stripe batch distance computation complete")
            else:
                # Pre-computed stripe distances: extract from stripe matrix
                if unifrac_computer is None:
                    unifrac_computer = UniFracComputer()
                if reference_sample_ids is None:
                    raise ValueError("reference_sample_ids required for stripe mode")
                if all_sample_ids is None:
                    raise ValueError("all_sample_ids required for stripe extraction when not using lazy mode")
                batch_distances = unifrac_computer.extract_batch_stripe_distances(
                    unifrac_distances, sample_ids, 
                    reference_sample_ids=reference_sample_ids,
                    all_sample_ids=all_sample_ids
                )
        else:
            # Pairwise computation (backward compatibility)
            if lazy_unifrac and unifrac_computer is not None:
                # Lazy computation: compute distances on-the-fly for this batch
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Computing UniFrac distances for batch: {sample_ids[:3]}... (batch size: {len(sample_ids)})")
                if unifrac_metric == "unweighted":
                    batch_distances = unifrac_computer.compute_batch_unweighted(sample_ids)
                else:
                    batch_distances = unifrac_computer.compute_batch_faith_pd(sample_ids)
                logger.debug(f"Batch distance computation complete")
            else:
                # Pre-computed distances: extract from distance matrix
                if unifrac_computer is None:
                    unifrac_computer = UniFracComputer()
                batch_distances = unifrac_computer.extract_batch_distances(unifrac_distances, sample_ids, metric=unifrac_metric)
        
        # Validate extracted distances
        if np.any(np.isnan(batch_distances)):
            import sys
            error_msg = f"NaN values found in extracted batch distances for sample_ids: {sample_ids}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            print(f"batch_distances shape={batch_distances.shape}, min={np.nanmin(batch_distances)}, max={np.nanmax(batch_distances)}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if np.any(np.isinf(batch_distances)):
            import sys
            error_msg = f"Inf values found in extracted batch distances for sample_ids: {sample_ids}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
        result["unifrac_target"] = torch.FloatTensor(batch_distances)

    return result


class ASVDataset(Dataset):
    """PyTorch Dataset for microbial sequencing data."""

    def __init__(
        self,
        table: Table,
        metadata: Optional[pd.DataFrame] = None,
        unifrac_distances: Optional[Union[DistanceMatrix, pd.Series, np.ndarray]] = None,
        tokenizer: Optional[SequenceTokenizer] = None,
        max_bp: int = 150,
        token_limit: int = 1024,
        target_column: Optional[str] = None,
        unifrac_metric: str = "unweighted",
        lazy_unifrac: bool = True,
        unifrac_computer: Optional[UniFracComputer] = None,
        stripe_mode: bool = True,
        reference_sample_ids: Optional[List[str]] = None,
    ):
        """Initialize ASVDataset.

        Args:
            table: Rarefied biom.Table object
            metadata: Optional DataFrame with sample metadata and targets
            unifrac_distances: Optional pre-computed UniFrac distances/stripe matrix (ignored if lazy_unifrac=True)
            tokenizer: SequenceTokenizer instance (creates default if None)
            max_bp: Maximum base pairs per sequence
            token_limit: Maximum ASVs per sample
            target_column: Column name in metadata for target values
            unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
            lazy_unifrac: If True, compute distances on-the-fly instead of using pre-computed distances (default: True)
            unifrac_computer: UniFracComputer instance for lazy computation (required if lazy_unifrac=True)
            stripe_mode: If True, use stripe-based distances instead of pairwise (default: True)
            reference_sample_ids: List of reference sample IDs for stripe mode (required if stripe_mode=True)
        """
        self.table = table
        self.metadata = metadata
        self.unifrac_distances = unifrac_distances
        self.tokenizer = tokenizer if tokenizer is not None else SequenceTokenizer()
        self.max_bp = max_bp
        self.token_limit = token_limit
        self.target_column = target_column
        self.unifrac_metric = unifrac_metric
        self.lazy_unifrac = lazy_unifrac
        self.unifrac_computer = unifrac_computer
        self.stripe_mode = stripe_mode
        self.reference_sample_ids = reference_sample_ids

        self.sample_ids = list(table.ids(axis="sample"))
        self.observation_ids = list(table.ids(axis="observation"))
        
        # Validate stripe mode parameters
        if stripe_mode and reference_sample_ids is None:
            # Auto-select reference samples if not provided (use first 100 or all if < 100)
            if len(self.sample_ids) <= 100:
                self.reference_sample_ids = self.sample_ids.copy()
            else:
                self.reference_sample_ids = self.sample_ids[:100]
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-selected {len(self.reference_sample_ids)} reference samples for stripe mode")
        
        if stripe_mode and reference_sample_ids is not None:
            # Validate reference samples exist in table
            table_sample_ids = set(self.sample_ids)
            missing_ref = set(reference_sample_ids) - table_sample_ids
            if missing_ref:
                raise ValueError(f"Reference sample IDs not found in table: {sorted(missing_ref)}")

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
            padded = self.tokenizer.pad_sequences([tokenized], self.max_bp + 1)[0]
            tokens_list.append(padded)
            counts_list.append(data[asv_idx])

        # Ensure sample has at least one ASV with count > 0
        # If not, this is a data quality issue that should be caught
        if not tokens_list:
            import sys
            error_msg = (
                f"Sample {sample_id} has no ASVs with count > 0. "
                f"This sample should be filtered out before creating the dataset."
            )
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
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

        return result
