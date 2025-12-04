# Training Loop

**Status:** ✅ Completed

## Overview
Training and validation loops with staged training support. Implemented in `aam/training/trainer.py`.

## Features
- **Staged Training**: Pre-train SequenceEncoder (self-supervised) → Train SequencePredictor (with optional freezing)
- **Memory Optimizations**: Gradient accumulation, memory clearing, chunked ASV processing, expandable segments
- **Progress & Logging**: Enhanced progress bars with compact labels (S=Step, L=Loss, UF=UniFrac, N=Nuc), TensorBoard logging
  - In pretrain mode: Displays UniFrac loss and nucleotide loss in progress bar
  - Uses high precision (6 decimal places) for small loss values to prevent rounding to 0.0000
- **Early Stopping**: Monitor validation loss with configurable patience (default: 10 epochs)
- **Checkpointing**: Save/load model state, optimizer state, resume training

## Implementation
- **Class**: `Trainer` in `aam/training/trainer.py`
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: Warmup (10k steps) + cosine decay
- **Testing**: Comprehensive unit tests (21 tests + 6 edge case tests passing)
