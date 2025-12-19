# PyTorch Implementation Plan

**Status:** Implementation Complete

## Overview

Design documents for AAM (Attention All Microbes) PyTorch implementation. All core components are complete.

## Document Structure

- **00-14**: Core implementation (all complete)
- **FUTURE_WORK.md**: Outstanding enhancements
- **archive/**: Historical analysis documents

See [INDEX.md](INDEX.md) for navigation.

## Training Strategy

**Stage 1 (Pretraining):** Train `SequenceEncoder` on UniFrac + nucleotide prediction

**Stage 2 (Fine-tuning):** Train `SequencePredictor` with pretrained encoder
- `--freeze-base`: Freeze encoder weights
- Without flag: Joint fine-tuning

## Quick Start

```bash
pip install -e ".[dev,training]"
pytest tests/ -v
python -m aam.cli train --help
```

## Outstanding Work

See `.agents/CLEANUP_TICKETS.md` for code cleanup (1 remaining ticket).
