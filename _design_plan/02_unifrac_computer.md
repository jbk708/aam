# UniFrac Distance Computer

**Status:** ✅ Completed

## Overview
Computes phylogenetic distances (UniFrac) from BIOM table and reference phylogenetic tree. Implemented in `aam/data/unifrac.py`.

## Key Features
- **Unweighted UniFrac**: Pairwise distances between samples → `skbio.DistanceMatrix`
- **Faith PD**: Per-sample phylogenetic diversity → `pandas.Series`
- Batch-level distance extraction for training
- Supports epoch regeneration (recompute when rarefaction changes)
- Batch size validation (must be even for unweighted UniFrac)

## Implementation
- **Class**: `UniFracComputer` in `aam/data/unifrac.py`
- **Library**: Uses `unifrac` package (accepts `biom.Table` and `skbio.TreeNode` objects directly)
- **Methods**: `compute_unweighted()`, `compute_faith_pd()`, `extract_batch_distances()`
- **Testing**: Comprehensive unit tests (19 tests + 12 error handling tests passing)
