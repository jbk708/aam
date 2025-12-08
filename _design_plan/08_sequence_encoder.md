# SequenceEncoder

**Status:** ✅ Completed

## Overview
Encoder that processes sequences and returns embeddings or predictions depending on encoder type. Serves as the base model for SequencePredictor. Implemented in `aam/models/sequence_encoder.py`.

## Architecture
- **Input**: `[B, S, L]` tokens → **Output**: Varies by encoder type
  - **UniFrac**: Returns pooled embeddings `[B, D]` (distances computed from embeddings via pairwise Euclidean distance)
  - **Other types**: Returns base predictions `[B, base_output_dim]` + `[B, S, D]` base embeddings
- Composes SampleSequenceEncoder + Encoder Transformer + Attention Pooling + Optional Dense head
- **Key Design**: 
  - **UniFrac (PYT-8.16b)**: Embedding-based approach - returns embeddings, distances computed via pairwise Euclidean distance (matches TensorFlow implementation)
  - **Other types**: Direct prediction via output head
  - Nucleotide and base predictions are parallel tasks sharing base embeddings (not sequential)
- Supports multiple encoder types: UniFrac, Faith PD, Taxonomy, Combined

## Implementation
- **Class**: `SequenceEncoder` in `aam/models/sequence_encoder.py`
- **Features**: 
  - Supports multiple encoder types (UniFrac, Faith PD, Taxonomy, Combined)
  - **UniFrac**: Returns embeddings directly (no output head), distances computed in loss function
  - **Other types**: Returns base predictions via output head
  - Returns base embeddings for SequencePredictor
- **Testing**: Comprehensive unit tests (25 tests passing)

## Architectural Changes (PYT-8.16b)
- **UniFrac encoder type**: Removed output head, returns embeddings directly
- **Distance computation**: Pairwise Euclidean distances computed from embeddings in loss function
- **Benefits**: Eliminates sigmoid saturation, mode collapse, and boundary clustering issues
- **Matches TensorFlow implementation**: Computes distances from embeddings rather than predicting directly
