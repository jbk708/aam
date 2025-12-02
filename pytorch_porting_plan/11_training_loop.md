# Training Loop

## Objective
Implement training and validation loops with staged training support.

## Training Strategy

### Stage 1: Pre-train SequenceEncoder
- Train SequenceEncoder on UniFrac + nucleotide prediction (self-supervised)
- No target labels required
- Save checkpoint for Stage 2

### Stage 2: Train SequencePredictor
- Load pre-trained SequenceEncoder
- Option A: Freeze base (`freeze_base=True`) - faster
- Option B: Fine-tune jointly (`freeze_base=False`) - better performance

## Implementation Requirements

### Training Epoch
- Set `model.train()`
- Iterate DataLoader
- Forward pass → compute loss → backward → optimizer step
- Return average loss

### Validation Epoch
- Set `model.eval()` + `torch.no_grad()`
- Iterate DataLoader
- Forward pass → compute loss and metrics
- Return losses and metrics

### Main Training Function
- Initialize model, optimizer, scheduler
- Epoch loop: train → validate → update LR → check early stopping
- Save checkpoints (best model)
- Support loading pre-trained SequenceEncoder

### Optimizer & Scheduler
- Optimizer: `AdamW` (lr=1e-4, weight_decay=0.01)
- Scheduler: Warmup (10k steps) + cosine decay
- Exclude frozen parameters from optimizer if `freeze_base=True`

### Early Stopping
- Monitor validation loss
- Patience: 50 epochs (configurable)
- Save best checkpoint

### Checkpointing
- Save: model state, optimizer state, epoch, best loss, metrics
- Load: Restore model state, optionally optimizer (for resume)

## Implementation Checklist

- [ ] Implement training epoch function
- [ ] Implement validation epoch function
- [ ] Implement main training function
- [ ] Support loading pre-trained SequenceEncoder
- [ ] Handle `freeze_base` parameter
- [ ] Implement early stopping
- [ ] Implement checkpoint saving/loading
- [ ] Setup optimizer (exclude frozen params if needed)
- [ ] Setup scheduler (warmup + decay)
- [ ] Test training loop
- [ ] Test validation loop
- [ ] Test early stopping
- [ ] Test checkpointing

## Key Considerations

- Device handling: Move model and batches to device
- Gradient management: Zero gradients, optional accumulation/clipping
- Memory: Clear cache if needed, gradient checkpointing for large models
- Progress: Use `tqdm` for progress bars
- Logging: Track losses and metrics per epoch

## Testing Requirements

- Test with small dataset (10-20 samples) for unit tests
- Test with real data for integration tests:
  - Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
- Verify loss decreases over epochs
- Verify early stopping works
- Verify checkpoint saving/loading
- Test resume from checkpoint
- Test with frozen base model

## Test Data

- Unit tests: Generate synthetic small datasets
- Integration tests: Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
