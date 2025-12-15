# CLI Interface

**Status:** ✅ Completed (Updated in PYT-11.4)

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
- **Pre-computed UniFrac matrices**: Uses `--unifrac-matrix` to load pre-generated matrices (PYT-11.4)

## Implementation
- **CLI**: `aam/cli.py` using `click` framework
- **UniFrac Loading**: Uses `UniFracLoader` to load pre-computed matrices from disk
- **Testing**: Comprehensive unit tests (updated for pre-computed matrices)

## Changes in PYT-11.4
- **Removed**: `--tree`, `--lazy-unifrac`, `--stripe-mode`, `--unifrac-threads`, `--prune-tree` flags
- **Added**: `--unifrac-matrix` parameter (required for training/pretrain)
- **Deprecated**: All UniFrac computation logic (users should generate matrices using unifrac-binaries)
