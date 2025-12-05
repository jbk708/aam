# Debug Scripts

This directory contains debugging and investigation scripts used during development, particularly for debugging NaN issues in nucleotide predictions.

## Scripts

### `debug_nan_token_limit.py`
Main debug script for tracing NaN issues in nucleotide predictions when using `token_limit` with gradient accumulation.

**Usage:**
```bash
python debug/debug_nan_token_limit.py --batch-size 6 --token-limit 256
```

**Features:**
- Detailed layer-by-layer tensor checks
- NaN/Inf detection at each step
- Synthetic data fallback option
- Full forward pass debugging

See `DEBUG_NAN_TOKEN_LIMIT.md` for detailed usage instructions.

### `investigate_batching_logic.py`
Investigates batching logic and UniFrac distance extraction to verify order consistency.

**Usage:**
```bash
python debug/investigate_batching_logic.py
```

**Tests:**
1. Verify sample_ids order matches tokens/counts order
2. Verify shuffled batch order
3. DataLoader shuffling with token_limit

See `BATCHING_INVESTIGATION.md` for findings.

### `investigate_all_padding.py`
Analyzes why many sequences have all padding and traces sequences through the pipeline.

**Usage:**
```bash
python debug/investigate_all_padding.py --batch-size 6 --token-limit 256
```

**Features:**
- Batch analysis showing valid vs padding ASVs
- Sequence statistics (START_TOKEN, all zeros, all padding)
- Mask analysis for attention pooling
- Traces specific sequences through pipeline

See `investigate_all_padding_sequences.md` for investigation plan.

### `investigate_nucleotide_nan.py`
Traces where NaN appears in nucleotide predictions.

**Usage:**
```bash
python debug/investigate_nucleotide_nan.py --batch-size 6 --token-limit 256
```

**Features:**
- Step-by-step tracing through model
- Identifies which ASVs (valid vs padding) have NaN
- Checks transformer, pooling, and nucleotide head outputs
- Shows NaN before and after fixes

## Documentation

- `DEBUG_NAN_TOKEN_LIMIT.md` - Usage guide for debug_nan_token_limit.py
- `BATCHING_INVESTIGATION.md` - Findings from batching logic investigation
- `investigate_all_padding_sequences.md` - Investigation plan for all-padding sequences

## Related Issues

These scripts were created to investigate and fix:
- **PYT-8.9**: Fix NaN in Nucleotide Predictions During Pretraining with Token Limit

## Notes

- All scripts support `--help` for usage information
- Scripts are designed to run on GPU nodes (use `--device cuda`)
- Most scripts support synthetic data fallback if real data is unavailable
