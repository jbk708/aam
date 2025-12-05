# Debug Script for NaN in Nucleotide Predictions with Token Limit

## Overview

This debug script (`debug_nan_token_limit.py`) is designed to identify where NaN values first appear in nucleotide predictions during pretraining when using `--token-limit` with gradient accumulation.

## Usage

### Basic Usage

```bash
python debug_nan_token_limit.py
```

### With Custom Parameters

```bash
python debug_nan_token_limit.py \
    --token-limit 512 \
    --batch-size 6 \
    --gradient-accumulation-steps 2 \
    --max-steps 10 \
    --device cuda \
    --data-dir ./data
```

### Arguments

- `--token-limit`: Token limit for truncation (default: 512)
- `--batch-size`: Batch size (default: 6)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 1)
- `--data-dir`: Data directory containing BIOM and tree files (default: ./data)
- `--device`: Device to use: cuda or cpu (default: cuda)
- `--max-steps`: Maximum training steps to debug (default: 10)

## What It Does

The script performs detailed tracing through the forward pass to identify where NaN first appears:

1. **Data Loading**: Loads BIOM table and computes UniFrac distances (or creates synthetic data if files not found)

2. **Collate Function Debugging**: 
   - Checks START_TOKEN preservation after truncation
   - Validates token values (should be 0-5 for vocab_size=6)
   - Checks for all-padding sequences
   - Verifies sequence structure after truncation

3. **Model Forward Pass Tracing**:
   - Token embedding layer
   - Position embedding layer
   - Transformer encoder layers
   - Attention pooling
   - Nucleotide prediction head
   - Sample-level transformer
   - Final outputs

4. **Gradient Checking**: Monitors gradients for NaN/Inf after backward pass

5. **Detailed Diagnostics**: Prints tensor statistics, NaN counts, and identifies failure points

## Output

The script prints detailed diagnostics at each step:

- ✅ Success indicators when checks pass
- ❌ Error indicators when NaN/Inf detected
- ⚠️  Warnings for potential issues (invalid tokens, all-padding sequences)
- Tensor statistics (shape, min, max, mean, std)
- NaN/Inf counts and percentages

## Example Output

```
================================================================================
STEP 1 (Epoch 0, Batch 0)
================================================================================

--- Checking collate_fn output ---
[step_1] Batch tokens:
  Shape: torch.Size([6, 512, 151])
  Dtype: torch.int64
  Min: 0.000000, Max: 5.000000, Mean: 1.234567
  Has NaN: False
  Has Inf: False

--- Debugging model forward pass (detailed trace) ---
[step_1] Input tokens:
  Shape: torch.Size([6, 512, 151])
  ...

❌ NaN/Inf detected: nucleotide logits at step_1
================================================================================
Shape: torch.Size([6, 512, 151, 6])
Has NaN: True
NaN count: 123456 / 2799360 (4.41%)
```

## Running on Remote GPU Node

1. Copy the script to the remote node:
   ```bash
   scp debug_nan_token_limit.py user@remote-node:/path/to/aam/
   ```

2. SSH to the remote node:
   ```bash
   ssh user@remote-node
   ```

3. Activate the environment and run:
   ```bash
   cd /path/to/aam
   conda activate aam  # or your environment
   python debug_nan_token_limit.py --device cuda --max-steps 5
   ```

4. Redirect output to a file for analysis:
   ```bash
   python debug_nan_token_limit.py --device cuda --max-steps 5 2>&1 | tee debug_output.log
   ```

## Troubleshooting

### Data Files Not Found

If BIOM/tree files are not found, the script will create synthetic data automatically. This is useful for testing but may not reproduce the exact issue.

### CUDA Out of Memory

Reduce batch size or token_limit:
```bash
python debug_nan_token_limit.py --batch-size 2 --token-limit 256
```

### Script Hangs

The script may hang if there's an infinite loop or deadlock. Use `--max-steps` to limit iterations.

## Next Steps

After identifying where NaN first appears:

1. **If NaN appears in collate_fn**: Check truncation logic in `aam/data/dataset.py`
2. **If NaN appears in embeddings**: Check token/position embedding initialization
3. **If NaN appears in transformer**: Check attention mechanism numerical stability
4. **If NaN appears in nucleotide head**: Check linear layer initialization/weights
5. **If NaN appears after backward pass**: Check gradient clipping and learning rate

## Related Files

- `aam/data/dataset.py` - Collate function with truncation logic
- `aam/models/asv_encoder.py` - ASV-level encoder with nucleotide prediction
- `aam/models/sample_sequence_encoder.py` - Sample-level encoder
- `aam/models/sequence_encoder.py` - Full sequence encoder
- `.agents/PYTORCH_PORTING_TICKETS.md` - Ticket PYT-8.9
