# SequenceEncoder

**Status:** ✅ Completed

## Overview
Encoder that adds prediction head for UniFrac distance prediction. Serves as the base model for SequencePredictor. Implemented in `aam/models/sequence_encoder.py`.

## Architecture
- **Input**: `[B, S, L]` tokens → **Output**: `[B, base_output_dim]` base prediction + `[B, S, D]` base embeddings
- Composes SampleSequenceEncoder + Encoder Transformer + Attention Pooling + Dense head
- **Key Design**: Nucleotide and UniFrac predictions are parallel tasks sharing base embeddings (not sequential)
- Supports multiple encoder types: UniFrac, Faith PD, Taxonomy, Combined

## Implementation
- **Class**: `SequenceEncoder` in `aam/models/sequence_encoder.py`
- **Features**: Supports multiple encoder types (UniFrac, Faith PD, Taxonomy, Combined), returns base embeddings for SequencePredictor
- **Testing**: Comprehensive unit tests (25 tests passing)
