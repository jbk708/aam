# Save Best Model Only

**Status:** ⏳ Pending

## Objective
Modify checkpoint saving to keep only the single best model file, replacing previous best model instead of saving multiple epoch-specific files.

## Current State
- Currently saves `best_model_epoch_{epoch}.pt` for each epoch with improvement
- Multiple checkpoint files accumulate over training
- Final model also saved separately

## Requirements

### Single Best Model File
- Save best model as `best_model.pt` (single file, no epoch number)
- Replace previous best model file when new best is found
- Keep optimizer and scheduler state in checkpoint
- Save when validation loss improves (or when training without validation, use train loss)

### Checkpoint Contents
- Model state dict
- Optimizer state dict
- Scheduler state (if applicable)
- Epoch number
- Best validation loss
- Metrics dictionary

### File Management
- Delete or overwrite previous `best_model.pt` when new best is found
- Keep final model saving separate (optional, configurable)

## Implementation Requirements

### Changes to `trainer.py`
- Modify checkpoint saving logic in `train()` method
- Use fixed filename `best_model.pt` instead of `best_model_epoch_{epoch}.pt`
- Remove old best model file before saving new one (or overwrite directly)

### CLI Integration
- No CLI changes needed (uses existing `--checkpoint-dir` option)
- Best model saved to `{checkpoint_dir}/best_model.pt`

## Implementation Checklist

- [ ] Modify `train()` method to save single `best_model.pt` file
- [ ] Remove epoch number from best model filename
- [ ] Ensure old best model is replaced (not accumulated)
- [ ] Test checkpoint saving/loading with best model
- [ ] Update documentation to reflect single best model file
- [ ] Verify resume from checkpoint still works

## Key Considerations

- **File replacement**: Use `Path.unlink()` to remove old file before saving, or just overwrite
- **Resume training**: Ensure `load_checkpoint()` can load `best_model.pt`
- **Final model**: Decide if we still want separate final model save (probably yes, for comparison)

## Testing Requirements

- Test that only one `best_model.pt` file exists after training
- Test that best model is replaced when new best is found
- Test loading best model checkpoint
- Test resume from best model checkpoint

## Notes

- This reduces disk usage and simplifies model selection
- Users can easily identify the best model without checking multiple files
- Final model save can remain separate for comparison purposes
