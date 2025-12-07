# Attention All Microbes (AAM)

Deep learning model for microbial sequencing data analysis using transformer-based attention mechanisms. Processes nucleotide sequences at multiple levels (nucleotide, ASV, sample) with self-supervised learning from phylogenetic information.

## Installation

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/) (Mamba is recommended for faster package resolution)

### Setup

```bash
# Create and activate environment
mamba env create -f environment.yml
mamba activate aam

# Install package
pip install -e .

# For development with all dependencies
pip install -e ".[dev,docs,training]"
```

## Usage

### Pre-training (Stage 1: Self-supervised)

Pre-train SequenceEncoder on UniFrac + nucleotide prediction:

```bash
python -m aam.cli pretrain \
  --table <biom_file> \
  --tree <tree_file> \
  --output-dir <output_dir> \
  --batch-size 8 \
  --epochs 100
```

### Training (Stage 2: Fine-tuning)

Train SequencePredictor with optional pre-trained encoder:

```bash
python -m aam.cli train \
  --table <biom_file> \
  --tree <tree_file> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --output-dir <output_dir> \
  --pretrained-encoder <encoder_checkpoint.pt> \
  --freeze-base \
  --batch-size 8 \
  --epochs 100
```

### Inference

Run predictions on new data:

```bash
python -m aam.cli predict \
  --model <model_checkpoint.pt> \
  --table <biom_file> \
  --tree <tree_file> \
  --output <predictions.tsv>
```

### Key Options

**Training:**
- `--pretrained-encoder`: Load pre-trained SequenceEncoder checkpoint
- `--freeze-base`: Freeze base model parameters (faster training)
- `--classifier`: Use classification mode (requires `--out-dim > 1`)
- `--batch-size`: Must be even for unweighted UniFrac
- `--optimizer`: Optimizer type - 'adamw' (default), 'adam', or 'sgd'
- `--scheduler`: Learning rate scheduler - 'warmup_cosine' (default), 'cosine', 'plateau', or 'onecycle'
- `--lr`: Learning rate (default: 1e-4)
- `--warmup-steps`: Warmup steps for warmup_cosine scheduler (default: 10000)
- `--weight-decay`: Weight decay for optimizers (default: 0.01)

**Model:**
- `--embedding-dim`: Embedding dimension (default: 128)
- `--attention-heads`: Number of attention heads (default: 4)
- `--attention-layers`: Number of transformer layers (default: 4)
- `--max-bp`: Maximum base pairs per sequence (default: 150)
- `--token-limit`: Maximum ASVs per sample (default: 1024, **critical for memory**)

**Memory Optimization:**
- `--gradient-accumulation-steps`: Accumulate gradients over N steps (default: 1)
- `--use-expandable-segments`: Enable PyTorch CUDA expandable segments (reduces fragmentation)
- `--asv-chunk-size`: Process ASVs in chunks to reduce memory (optional)
- **Important**: Reduce `--token-limit` from 1024 to 256-512 for 24GB GPUs (reduces memory by 4-16x)

**Data:**
- `--unifrac-metric`: 'unifrac' or 'faith_pd' (default: 'unifrac')
- `--rarefy-depth`: Rarefaction depth (default: 5000)
- `--test-size`: Validation split size (default: 0.2)

See `python -m aam.cli <command> --help` for full options.

### Optimizer and Scheduler Options

**Optimizers:**
- `adamw` (default): AdamW optimizer with weight decay - recommended for transformer models
- `adam`: Standard Adam optimizer without weight decay
- `sgd`: SGD optimizer with momentum - traditional optimizer, may require different learning rates

**Schedulers:**
- `warmup_cosine` (default): Custom warmup + cosine decay scheduler - good for pretraining
- `cosine`: PyTorch CosineAnnealingLR - cosine annealing without warmup
- `plateau`: ReduceLROnPlateau - reduces LR when validation loss plateaus (requires validation set)
- `onecycle`: OneCycleLR - one cycle learning rate policy - can improve convergence speed

**Recommendations:**
- **Pretraining**: Use `adamw` + `warmup_cosine` (default) for stable training
- **Fine-tuning**: Try `adamw` + `plateau` for adaptive learning rate reduction
- **Fast experimentation**: Try `adamw` + `onecycle` for faster convergence
- **Memory-constrained**: `sgd` may use less memory than Adam variants

### Memory Optimization

For GPUs with limited memory (e.g., 24GB), use these optimizations:

```bash
python -m aam.cli pretrain \
  --table <biom_file> \
  --tree <tree_file> \
  --output-dir <output_dir> \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --token-limit 256 \
  --use-expandable-segments
```

**Key optimizations:**
- `--token-limit 256`: Reduces sample-level attention by 16x (most critical)
- `--gradient-accumulation-steps 16`: Maintains effective batch size while reducing memory
- `--use-expandable-segments`: Reduces memory fragmentation

See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for detailed memory optimization strategies.

## Monitoring Training with TensorBoard

TensorBoard is automatically enabled during training and logs are saved to `{output_dir}/tensorboard/`. Use it to monitor training progress, losses, metrics, and model weights.

### Starting TensorBoard

```bash
# Start TensorBoard server (run in separate terminal)
tensorboard --logdir <output_dir>/tensorboard

# Or specify port explicitly
tensorboard --logdir <output_dir>/tensorboard --port 6006
```

Then open your browser to `http://localhost:6006` (or the port you specified).

### What's Logged

**Losses (per epoch):**
- `train/total_loss`: Total training loss
- `train/target_loss`: Target prediction loss (MSE/NLL)
- `train/count_loss`: ASV count prediction loss (masked MSE)
- `train/unifrac_loss`: UniFrac prediction loss (MSE)
- `train/nuc_loss`: Nucleotide prediction loss (CrossEntropy)
- `val/total_loss`: Total validation loss
- `val/target_loss`, `val/count_loss`, `val/unifrac_loss`, `val/nuc_loss`: Validation component losses

**Progress Bar Abbreviations:**
- **TL**: Total Loss - Combined loss from all tasks
- **UL**: UniFrac Loss - UniFrac distance prediction loss (only shown in pretrain mode)
- **NL**: Nucleotide Loss - Nucleotide sequence prediction loss (only shown when nucleotide prediction is enabled)
- **LR**: Learning Rate - Current learning rate (only shown during training, not validation)

**Metrics (per epoch, validation only):**
- **Regression metrics**: `val/mae`, `val/mse`, `val/r2`
- **Classification metrics**: `val/accuracy`, `val/precision`, `val/recall`, `val/f1`
- **Count metrics**: `val/count_mae`, `val/count_mse`

**Training Info:**
- `train/learning_rate`: Learning rate schedule

**Model Weights & Gradients (every 10 epochs):**
- `weights/{layer_name}`: Weight histograms
- `gradients/{layer_name}`: Gradient histograms

### Interpreting Loss Values

**For Pretraining (UniFrac + Nucleotides):**
- **Epoch 1**: Total loss ~1.5-2.0 (near random baseline)
  - `nuc_loss` ~1.5-1.7 (random baseline: log(5) ≈ 1.609)
  - `unifrac_loss` ~0.1-0.5 (depends on distance scale)
- **Well-trained**: Total loss ~0.1-0.5
  - `nuc_loss` ~0.1-0.5
  - `unifrac_loss` ~0.01-0.1

**For Fine-tuning (with Target Prediction):**
- Monitor `target_loss` decreasing over epochs
- `unifrac_loss` and `nuc_loss` should remain stable if `freeze_base=True`
- Total loss should decrease steadily

### Tips

1. **Monitor loss trends**: Look for steady decreases, not just absolute values
2. **Check for overfitting**: If `val_loss` increases while `train_loss` decreases, model is overfitting
3. **Learning rate**: Watch `learning_rate` schedule - should warmup then decay
4. **Gradients**: Check gradient histograms for vanishing/exploding gradients (should be well-distributed)
5. **Compare runs**: Use different `--output-dir` for different experiments to compare in TensorBoard

### Example: Monitoring Pretraining

```bash
# Terminal 1: Start training
python -m aam.cli pretrain \
  --table data.biom \
  --tree tree.nwk \
  --output-dir runs/pretrain_exp1 \
  --epochs 1000

# Terminal 2: Start TensorBoard
tensorboard --logdir runs/pretrain_exp1/tensorboard
```

Then monitor:
- `train/total_loss` decreasing from ~1.7 → ~0.3 over 100-200 epochs
- `train/unifrac_loss` and `train/nuc_loss` both decreasing
- `val/total_loss` tracking `train/total_loss` (no overfitting)

## Testing

Run the full test suite:

```bash
pytest tests/ -v --tb=short --no-header -rA
```

**Test Coverage:**
- 359+ tests covering all components
- Unit tests for data pipeline, models, training
- Integration tests for end-to-end workflows
- CUDA GPU support verified

**Quick Test Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_biom_loader.py -v

# Run with coverage
pytest tests/ --cov=aam --cov-report=html

# Run only CUDA tests (if CUDA available)
pytest tests/ -v -k "cuda or device"
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

Implementation details and design decisions are documented in `_design_plan/`.





