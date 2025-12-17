# PyTorch Implementation Plan

**Status:** Implementation Complete

## Overview

Design documents for AAM (Attention All Microbes) PyTorch implementation. All core components are complete.

## Document Structure

- **00-13**: Core implementation (all complete)
- **14**: Phase 8 training features (complete)
- **19-22**: Analysis and fixes (complete)
- **FUTURE_WORK.md**: Outstanding enhancements

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

See `.agents/PYTORCH_PORTING_TICKETS.md` (17 tickets, ~80-115 hours)
