# ASVEncoder

**Status:** ✅ Completed

## Overview
Encoder that processes nucleotide sequences at the ASV (sequence) level. Implemented in `aam/models/asv_encoder.py`.

## Architecture
- **Input**: `[B, S, L]` tokens → **Output**: `[B, S, D]` ASV embeddings
- Processes all ASVs in parallel by reshaping to `[B*S, L]`
- Embedding → Position Embedding → Transformer → Attention Pooling
- Optional nucleotide prediction head for self-supervised learning

## Implementation
- **Class**: `ASVEncoder` in `aam/models/asv_encoder.py`
- **Features**: Handles masking, training vs inference modes, nucleotide predictions
- **Testing**: Comprehensive unit tests (28 tests passing)
