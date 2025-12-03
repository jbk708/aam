# SequencePredictor

**Status:** ✅ Completed

## Overview
Main model for predicting sample-level targets and ASV counts. Composes SequenceEncoder as base model. Implemented in `aam/models/sequence_predictor.py`.

## Architecture
- **Composition Pattern**: Uses SequenceEncoder as `base_model` (not inheritance) - enables transfer learning
- **Multi-Task Learning**: All tasks (nucleotide, UniFrac, count, target) share base embeddings but compute in parallel
- **Key Design**: Base predictions (UniFrac) are side outputs for loss only, NOT used as input to target/count encoders
- **Input**: `[B, S, L]` tokens, `[B, S, 1]` counts → **Output**: Target predictions `[B, out_dim]` + Count predictions `[B, S, 1]`

## Implementation
- **Class**: `SequencePredictor` in `aam/models/sequence_predictor.py`
- **Features**: Supports base model freezing, multiple encoder types, classification/regression modes
- **Testing**: Comprehensive unit tests (27 tests passing)
