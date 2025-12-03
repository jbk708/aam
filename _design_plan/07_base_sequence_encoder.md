# SampleSequenceEncoder

**Status:** ✅ Completed

## Overview
Encoder that processes ASV embeddings at the sample level. Implemented in `aam/models/sample_sequence_encoder.py`.

## Architecture
- **Input**: `[B, S, L]` tokens → **Output**: `[B, S, D]` base embeddings
- Composes ASVEncoder + Sample-level Position Embedding + Sample-level Transformer
- Passes through nucleotide predictions from ASVEncoder for self-supervised learning

## Implementation
- **Class**: `SampleSequenceEncoder` in `aam/models/sample_sequence_encoder.py`
- **Features**: Handles ASV masking, training vs inference modes
- **Testing**: Comprehensive unit tests (30 tests passing)
