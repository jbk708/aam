# CLI Interface

**Status:** ✅ Completed

## Overview
Command-line interface for training and inference. Implemented in `aam/cli.py`.

## Commands
- **Training**: Full training with metadata targets
- **Pre-training**: Stage 1 self-supervised training (UniFrac + nucleotide prediction)
- **Inference**: Generate predictions from trained model

## Features
- Comprehensive argument validation and error handling
- Support for transfer learning (pretrained encoder, freeze base)
- Memory optimization options (gradient accumulation, chunked processing)
- TensorBoard logging integration
- Reproducibility support (random seed)

## Implementation
- **CLI**: `aam/cli.py` using `click` framework
- **Testing**: Comprehensive unit tests (28 tests + 10 integration tests passing)
