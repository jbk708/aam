# UniFrac Underfitting Analysis

**Status:** Completed
**Tickets:** PYT-8.12 through PYT-8.16b (all complete)

## Summary

This analysis investigated and resolved severe UniFrac underfitting (R² = 0.0455). The root cause was an architectural mismatch with TensorFlow.

## Key Findings

1. **Diagonal masking** was missing in loss computation
2. **Zero-distance samples** are extremely rare (0.00%) - not a concern
3. **Architecture mismatch**: PyTorch predicted distances directly; TensorFlow computes them from embeddings

## Solution Implemented

**PYT-8.16b**: Refactored to match TensorFlow approach:
- Removed direct distance prediction head
- `SequenceEncoder` returns embeddings directly
- `compute_pairwise_distances()` computes Euclidean distances from embeddings
- Diagonal masking applied (upper triangle only)
- Eliminated sigmoid saturation and boundary clustering

## Files Modified
- `aam/models/sequence_encoder.py`
- `aam/training/losses.py`
- `aam/cli.py`
