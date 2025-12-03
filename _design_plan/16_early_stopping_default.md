# Early Stopping Default to 10 Epochs

**Status:** ⏳ Pending

## Objective
Change the default early stopping patience from 50 epochs to 10 epochs.

## Current State
- Default `early_stopping_patience` is 50 epochs in `trainer.py`
- CLI default `--patience` is 10 for train command, 50 for pretrain command
- Inconsistent defaults between CLI and trainer

## Requirements

### Default Value Change
- Change `trainer.py` default `early_stopping_patience` from 50 to 10
- Update CLI defaults to be consistent (both commands should use 10)
- Keep parameter configurable (users can override with `--patience`)

### Consistency
- Ensure trainer default matches CLI default
- Both `train` and `pretrain` commands should use same default

## Implementation Requirements

### Changes to `trainer.py`
- Update `train()` method signature: `early_stopping_patience: int = 10`

### Changes to `cli.py`
- Update `pretrain` command: `--patience` default from 50 to 10
- Verify `train` command already uses 10 (should be consistent)

## Implementation Checklist

- [ ] Change `trainer.py` default `early_stopping_patience` to 10
- [ ] Update `cli.py` pretrain command `--patience` default to 10
- [ ] Verify train command default is 10 (should already be)
- [ ] Test early stopping with new default
- [ ] Update documentation if needed

## Key Considerations

- **Backward compatibility**: Users can still override with `--patience` flag
- **Consistency**: All defaults should be 10 epochs
- **Documentation**: Update any docs that mention default patience value

## Testing Requirements

- Test early stopping triggers after 10 epochs without improvement
- Test that `--patience` flag still works to override default
- Verify both train and pretrain commands use same default

## Notes

- 10 epochs is more reasonable default for faster iteration
- Users can increase patience if needed for longer training
- Consistent defaults improve user experience
