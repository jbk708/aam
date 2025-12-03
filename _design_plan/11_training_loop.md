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

- [x] Implement training epoch function
- [x] Implement validation epoch function
- [x] Implement main training function
- [x] Support loading pre-trained SequenceEncoder
- [x] Handle `freeze_base` parameter
- [x] Implement early stopping
- [x] Implement checkpoint saving/loading
- [x] Setup optimizer (exclude frozen params if needed)
- [x] Setup scheduler (warmup + decay)
- [x] Test training loop
- [x] Test validation loop
- [x] Test early stopping
- [x] Test checkpointing

## Key Considerations

- Device handling: Move model and batches to device
- Gradient management: Zero gradients, gradient accumulation support (configurable steps)
- Memory: Clear cache after batches/chunks, gradient accumulation, chunked processing for large models
- Progress: Use `tqdm` for progress bars with enhanced display (epoch, step, loss, learning rate)
- Logging: Track losses and metrics per epoch, TensorBoard logging enabled by default
- OOM handling: Catch CUDA OOM errors with helpful suggestions

## Memory Optimizations (PYT-7.2)

### Gradient Accumulation
- Accumulate gradients over N steps before optimizer.step()
- Scales loss by 1/N for correct averaging
- Reduces effective batch size without reducing memory per forward pass
- CLI option: `--gradient-accumulation-steps N` (default: 1)

### Memory Clearing
- Call `torch.cuda.empty_cache()` after each batch, chunk, and epoch
- Explicit `del` statements for intermediate tensors
- Helps reduce memory fragmentation

### Chunked ASV Processing
- Process ASVs in chunks instead of all at once
- Reduces peak memory for ASV-level attention matrices
- CLI option: `--asv-chunk-size N` (optional, None = process all)

### Expandable Segments
- Enable PyTorch's expandable segments allocator
- Reduces memory fragmentation
- CLI option: `--use-expandable-segments` (flag)

### Token Limit Reduction
- Most critical optimization: reduce `--token-limit` from default 1024
- Sample-level attention is O(token_limit^2)
- Reducing to 256 reduces sample attention by 16x

## Progress Display and Logging (PYT-4.4)

### Enhanced Progress Bars
- Display epoch number (e.g., "Epoch 5/100")
- Display step/batch number (e.g., "Step 42/500")
- Display running average loss (updated each batch)
- Display current learning rate
- Format: `"Epoch {epoch+1}/{num_epochs}"` with postfix showing Step, Loss, LR

### TensorBoard Logging
- Always enabled when `tensorboard_dir` is provided
- Logs saved to `{output_dir}/tensorboard/`
- Per-epoch logging:
  - All losses (total_loss, target_loss, count_loss, base_loss, nuc_loss)
  - All metrics (regression, classification, count metrics)
  - Learning rate
  - Weight and gradient histograms (every 10 epochs)
- Writer properly closed after training completes

### Known Issues
- Loss display may show 0.0000 in some cases - see PYT-7.3 for investigation

## Testing Requirements

- Test with small dataset (10-20 samples) for unit tests
- Test with real data for integration tests:
  - Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
- Verify loss decreases over epochs
- Verify early stopping works
- Verify checkpoint saving/loading
- Test resume from checkpoint
- Test with frozen base model
- Verify TensorBoard logs are created correctly
- Verify progress bars display correct information

## Test Data

- Unit tests: Generate synthetic small datasets
- Integration tests: Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
