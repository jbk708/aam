# CLI Interface

## Objective
Implement command-line interface for training and inference.

## Command Structure

### Training Command

**Required Arguments**:
- `--table`: Path to BIOM table file
- `--tree`: Path to phylogenetic tree file (.nwk)
- `--metadata`: Path to metadata file (.tsv)
- `--metadata-column`: Column name for target prediction
- `--output-dir`: Output directory for checkpoints and logs

**Training Parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--patience`: Early stopping patience (default: 10)
- `--warmup-steps`: Learning rate warmup steps (default: 10000)
- `--weight-decay`: Weight decay for AdamW (default: 0.01)

**Model Parameters**:
- `--embedding-dim`: Embedding dimension (default: 128)
- `--attention-heads`: Number of attention heads (default: 4)
- `--attention-layers`: Number of transformer layers (default: 4)
- `--max-bp`: Maximum base pairs per sequence (default: 150)
- `--token-limit`: Maximum ASVs per sample (default: 1024)
- `--out-dim`: Output dimension (default: 1)
- `--classifier`: Use classification mode (flag)

**Data Parameters**:
- `--rarefy-depth`: Rarefaction depth (default: 5000)
- `--test-size`: Validation split size (default: 0.2)
- `--unifrac-metric`: UniFrac metric ('unifrac' or 'faith_pd', default: 'unifrac')

**Loss Parameters**:
- `--penalty`: Weight for base/UniFrac loss (default: 1.0)
- `--nuc-penalty`: Weight for nucleotide loss (default: 1.0)
- `--class-weights`: Class weights for classification (optional)

**Other**:
- `--device`: Device ('cuda' or 'cpu', default: 'cuda')
- `--seed`: Random seed for reproducibility
- `--num-workers`: DataLoader workers (default: 0)

### Pre-training Command (Stage 1)

**Required Arguments**:
- `--table`: Path to BIOM table file
- `--tree`: Path to phylogenetic tree file (.nwk)
- `--output-dir`: Output directory for checkpoints and logs

**Training Parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 8, must be even)
- `--lr`: Learning rate (default: 1e-4)
- `--patience`: Early stopping patience (default: 50)
- `--warmup-steps`: Learning rate warmup steps (default: 10000)
- `--weight-decay`: Weight decay for AdamW (default: 0.01)

**Model Parameters**:
- `--embedding-dim`: Embedding dimension (default: 128)
- `--attention-heads`: Number of attention heads (default: 4)
- `--attention-layers`: Number of transformer layers (default: 4)
- `--max-bp`: Maximum base pairs per sequence (default: 150)
- `--token-limit`: Maximum ASVs per sample (default: 1024)

**Data Parameters**:
- `--rarefy-depth`: Rarefaction depth (default: 5000)
- `--test-size`: Validation split size (default: 0.2)
- `--unifrac-metric`: UniFrac metric ('unifrac' or 'faith_pd', default: 'unifrac')

**Loss Parameters**:
- `--penalty`: Weight for base/UniFrac loss (default: 1.0)
- `--nuc-penalty`: Weight for nucleotide loss (default: 1.0)

**Other**:
- `--device`: Device ('cuda' or 'cpu', default: 'cuda')
- `--seed`: Random seed for reproducibility
- `--num-workers`: DataLoader workers (default: 0)
- `--resume-from`: Path to checkpoint to resume from

**Note**: No metadata file required - pre-training is self-supervised (UniFrac + nucleotide prediction only).

### Inference Command

**Required Arguments**:
- `--model`: Path to trained model checkpoint
- `--table`: Path to BIOM table file
- `--tree`: Path to phylogenetic tree file
- `--output`: Output file for predictions

**Optional**:
- `--batch-size`: Batch size for inference
- `--device`: Device to use

## Implementation Requirements

**Argument Parsing**:
- Use `click` or `argparse`
- Validate file paths exist
- Validate parameter ranges
- Provide helpful error messages

**Setup Functions**:
- Setup device (check CUDA availability)
- Setup random seed
- Create output directory
- Setup logging

**Data Loading**:
- Load BIOM table
- Load phylogenetic tree
- Load metadata
- Create data loaders

**Model Initialization**:
- Create model with specified parameters
- Load checkpoint if resuming
- Move to device

**Training Execution**:
- Create optimizer and scheduler
- Create loss and metrics functions
- Call training function
- Save final model

## Implementation Checklist

- [x] Create CLI with argument parsing
- [x] Implement training command
- [x] Implement pre-training command (Stage 1)
- [x] Implement inference command
- [x] Validate arguments
- [x] Setup device and random seed
- [x] Integrate with data loading
- [x] Integrate with model creation
- [x] Integrate with training loop
- [x] Add logging
- [x] Test with sample data
- [x] Add help text and documentation

## Key Considerations

### Argument Validation
- Check file paths exist
- Validate parameter ranges (e.g., batch_size > 0)
- Check parameter combinations (e.g., classifier requires out_dim > 1)

### Error Handling
- Catch and report errors clearly
- Provide helpful error messages
- Handle missing files gracefully

### Logging
- Log to console and file
- Include timestamps
- Log model parameters
- Log training progress

### Reproducibility
- Set random seeds for Python, NumPy, PyTorch
- Ensure deterministic behavior
- Document seed in output

## Testing Requirements

- Test with minimal arguments
- Test with all arguments
- Test argument validation
- Test error handling
- Test with real data:
  - Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
  - Example: `python -m aam.train --table ./data/fall_train_only_all_outdoor.biom --tree ./data/all-outdoors_sepp_tree.nwk --metadata-column <column> --output-dir ./output`
- Verify output files created

## Test Data

- Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk` for CLI testing
- Ensure metadata file is available if required by the CLI

## Notes

- CLI is user-facing interface
- Clear error messages are important
- Logging helps debug issues
- Reproducibility is critical for science
