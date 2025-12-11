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

**UniFrac Computation:**
- `--stripe-mode/--no-stripe-mode`: Use stripe-based UniFrac (default: enabled, more memory-efficient)
- `--lazy-unifrac/--no-lazy-unifrac`: Compute distances on-the-fly (default: enabled, faster startup)
- `--reference-samples`: Reference samples for stripe mode - number (e.g., '100') or file path (auto-selects if not specified)
- `--unifrac-metric`: 'unifrac' or 'faith_pd' (default: 'unifrac')
- `--prune-tree`: Pre-prune tree to only ASVs in table (speeds up large trees)
- **Note**: Stripe mode removes batch size restrictions (no longer requires even batch sizes)

**Memory Optimization:**
- `--gradient-accumulation-steps`: Accumulate gradients over N steps (default: 1)
- `--use-expandable-segments`: Enable PyTorch CUDA expandable segments (reduces fragmentation)
- `--asv-chunk-size`: Process ASVs in chunks to reduce memory (optional)
- **Important**: Reduce `--token-limit` from 1024 to 256-512 for 24GB GPUs (reduces memory by 4-16x)

**Data:**
- `--rarefy-depth`: Rarefaction depth (default: 5000)
- `--test-size`: Validation split size (default: 0.2)

See `python -m aam.cli <command> --help` for full options.

### Optimizer and Scheduler

**Optimizers:** `adamw` (default, recommended), `adam`, `sgd`  
**Schedulers:** `warmup_cosine` (default), `cosine`, `plateau`, `onecycle`

**Recommendations:**
- **Pretraining**: `adamw` + `warmup_cosine` (default)
- **Fine-tuning**: `adamw` + `plateau` for adaptive LR reduction
- **Fast experimentation**: `adamw` + `onecycle`

### Memory Optimization

For limited GPU memory (e.g., 24GB):

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
- `--token-limit 256`: Reduces memory by 16x (most critical)
- `--gradient-accumulation-steps 16`: Maintains effective batch size
- `--use-expandable-segments`: Reduces memory fragmentation
- Stripe mode (default) is more memory-efficient than pairwise UniFrac

See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for detailed strategies.

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

**Losses:** `train/total_loss`, `train/target_loss`, `train/unifrac_loss`, `train/nuc_loss`, `train/count_loss` (and validation equivalents)

**Progress Bar:** TL (Total Loss), UL (UniFrac Loss), NL (Nucleotide Loss), LR (Learning Rate)

**Metrics (validation):** `val/mae`, `val/mse`, `val/r2` (regression), `val/accuracy`, `val/precision`, `val/recall`, `val/f1` (classification)

**Model:** Weight and gradient histograms (every 10 epochs), learning rate schedule

### Interpreting Loss Values

**Pretraining:** Epoch 1 total loss ~1.5-2.0 (random baseline), well-trained ~0.1-0.5  
**Fine-tuning:** Monitor `target_loss` decreasing; `unifrac_loss`/`nuc_loss` stable if `freeze_base=True`

**Tips:** Monitor trends (not absolute values), watch for overfitting (val_loss increasing), check learning rate schedule, verify gradients are well-distributed

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=aam --cov-report=html
```

359+ tests covering data pipeline, models, training, and end-to-end workflows.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

### Recent Improvements

**Stripe-Based UniFrac (Default):**
- Memory-efficient stripe-based computation (O(N×R) vs O(N²) for pairwise)
- No batch size restrictions (pairwise required even batch sizes)
- Auto-selects reference samples (first 100 or all if < 100)
- Lazy computation enabled by default (faster startup)

**UniFrac Distance Prediction:**
- Computes distances from embeddings (Euclidean) instead of direct prediction
- Sigmoid normalization ensures [0, 1] range
- Eliminates boundary clustering and mode collapse issues

## Documentation

Implementation details and design decisions are documented in `_design_plan/`.





