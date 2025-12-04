# Dataset and Tokenizer

**Status:** ✅ Completed

## Overview
PyTorch Dataset class and sequence tokenization for microbial sequencing data. Implemented in `aam/data/tokenizer.py` and `aam/data/dataset.py`.

## Key Features
- **Tokenization**: A/C/G/T → 1/2/3/4, padding → 0
- **Dataset**: `ASVDataset` class with epoch regeneration support
- **Batch Collation**: Custom `collate_fn` handles variable ASV counts, pads to `token_limit`
- **UniFrac Integration**: Extracts batch-level UniFrac distances
  - Supports both `DistanceMatrix` objects and numpy arrays (from `extract_batch_distances`)
  - Extracts batch-level pairwise distances `[batch_size, batch_size]` for unweighted UniFrac
  - Note: Current implementation assumes sample order matches dataset order (see PYT-8.5 for shuffled batch support)

## Implementation
- **Tokenizer**: `SequenceTokenizer` in `aam/data/tokenizer.py`
- **Dataset**: `ASVDataset` in `aam/data/dataset.py`
- **Testing**: Comprehensive unit tests (12 tokenizer tests + 18 dataset tests + 5 edge case tests passing)
