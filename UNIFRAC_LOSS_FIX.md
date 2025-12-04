# Fix: UniFrac Loss Not Properly Reported

## Issue Summary

The loss function was not properly reporting the loss from UniFrac predictions because `base_prediction` was only included in the model output when `return_nucleotides=True`. This caused the base loss (UniFrac loss) to be set to 0 even when `base_target` existed in the training data.

## Root Cause

In `SequencePredictor.forward()` (line 209-210), `base_prediction` was conditionally included:

```python
if return_nucleotides and base_prediction is not None:
    result["base_prediction"] = base_prediction
```

However:
1. `base_prediction` is needed for computing the base loss (UniFrac loss), which is independent of nucleotide predictions
2. The trainer sets `return_nucleotides` based on whether nucleotide targets exist or `nuc_penalty > 0` (line 372 in trainer.py)
3. If `nuc_penalty = 0` and there are no nucleotide targets, then `return_nucleotides=False`
4. This causes `base_prediction` to be omitted from outputs
5. Without `base_prediction` in outputs, the loss function sets `base_loss = 0` (line 188-191 in losses.py), even though `base_target` exists

## Fix

Modified `SequencePredictor.forward()` to always include `base_prediction` if it exists, regardless of the `return_nucleotides` flag. The `return_nucleotides` flag now only controls whether `nuc_predictions` are returned.

### Changes Made

1. **`aam/models/sequence_predictor.py`**: 
   - Changed `base_prediction` to always be included if it exists
   - Changed combined encoder outputs (`unifrac_pred`, `faith_pred`, `tax_pred`) to always be included if they exist
   - `return_nucleotides` now only controls `nuc_predictions`

2. **`tests/test_sequence_predictor.py`**: 
   - Updated tests to expect `base_prediction` to always be present
   - Updated test comments to clarify that `base_prediction` is needed for loss computation

## Expected Behavior

- **`base_prediction`**: Always included in output (needed for base loss computation)
- **`nuc_predictions`**: Only included when `return_nucleotides=True`
- **Combined encoder outputs**: Always included if encoder type is "combined"

## Verification

The fix ensures that:
1. When `base_target` exists in training data, `base_prediction` will be in outputs
2. The loss function will properly compute `base_loss` using `base_prediction` and `base_target`
3. The total loss will include the weighted base loss: `total_loss = target_loss + count_loss + base_loss * penalty + nuc_loss * nuc_penalty`

## Related Files

- `aam/models/sequence_predictor.py` - Fixed forward method
- `aam/models/sequence_encoder.py` - Already correctly returns `base_prediction` always
- `aam/training/losses.py` - Loss computation (no changes needed)
- `aam/training/trainer.py` - Trainer logic (no changes needed)
- `tests/test_sequence_predictor.py` - Updated tests
