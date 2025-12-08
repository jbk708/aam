# ASVEncoder

**Status:** ✅ Completed

## Overview
Encoder that processes nucleotide sequences at the ASV (sequence) level. Implemented in `aam/models/asv_encoder.py`.

## Architecture
- **Input**: `[B, S, L]` tokens → **Output**: `[B, S, D]` ASV embeddings
- Processes all ASVs in parallel by reshaping to `[B*S, L]`
- Embedding → Position Embedding → Transformer → Attention Pooling
- Optional nucleotide prediction head for self-supervised learning
- **All-Padding Handling (PYT-8.16b)**: Handles all-padding sequences (mask sum = 0) by skipping transformer and setting embeddings to zero, preventing NaN

## Implementation
- **Class**: `ASVEncoder` in `aam/models/asv_encoder.py`
- **Features**: 
  - Handles masking, training vs inference modes, nucleotide predictions
  - **All-Padding Fix**: Identifies all-padding sequences before transformer, processes only valid sequences through transformer, sets all-padding embeddings to zero
  - Supports chunked processing for large batches
- **Testing**: Comprehensive unit tests (28 tests passing)

## All-Padding Sequence Handling (PYT-8.16b)
- **Problem**: Transformer produces NaN for all-padding sequences (mask sum = 0)
- **Solution**: 
  1. Identify all-padding sequences before transformer
  2. Process only valid sequences through transformer
  3. Set embeddings to zero for all-padding sequences (skip transformer)
  4. Combine results
- **Applied to**: Both chunked and non-chunked processing paths
- **Benefits**: Prevents NaN propagation, ensures stable training
