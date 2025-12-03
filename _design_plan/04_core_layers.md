# Core Layers

**Status:** ✅ Completed

## Overview
Fundamental neural network layers: AttentionPooling and PositionEmbedding. Implemented in `aam/models/attention_pooling.py` and `aam/models/position_embedding.py`.

## Components

### AttentionPooling
- Pools sequence-level embeddings to single vector using learned attention weights
- Handles variable-length sequences via masking
- Scales attention scores by `sqrt(hidden_dim)` for stability

### PositionEmbedding
- Adds learned position information to embeddings
- Learned embeddings (not fixed sinusoidal)
- Handles variable sequence lengths

## Implementation
- **AttentionPooling**: `aam/models/attention_pooling.py` (includes masking utilities)
- **PositionEmbedding**: `aam/models/position_embedding.py`
- **Testing**: Comprehensive unit tests (17 attention pooling tests + 8 position embedding tests passing)
