# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AAM (Attention All Microbes) is a deep learning model for microbial sequencing data analysis using transformer-based attention mechanisms. It processes nucleotide sequences at multiple levels (nucleotide, ASV, sample) with self-supervised learning from phylogenetic information.

## Common Commands

```bash
# Install for development
pip install -e ".[dev,docs,training]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_trainer.py -v

# Run tests with coverage
pytest tests/ --cov=aam --cov-report=html

# Lint with ruff
ruff check aam/

# Type check
uvx ty check aam/

# Pre-training (Stage 1: self-supervised)
python -m aam.cli pretrain --table <biom_file> --unifrac-matrix <matrix.npy> --output-dir <dir>

# Training (Stage 2: fine-tuning)
python -m aam.cli train --table <biom_file> --unifrac-matrix <matrix.npy> --metadata <meta.tsv> --metadata-column <col> --output-dir <dir>

# Inference
python -m aam.cli predict --model <checkpoint.pt> --table <biom_file> --output <predictions.tsv>
```

## Architecture

### Hierarchical Processing Flow
```
Nucleotide sequences → ASVEncoder → ASV embeddings
ASV embeddings → SampleSequenceEncoder → Sample embeddings
Sample embeddings → SequenceEncoder → Base predictions (UniFrac/Taxonomy)
Sample embeddings → SequencePredictor → Target predictions
```

### Key Model Components (aam/models/)
- **ASVEncoder**: Nucleotide → ASV embeddings with optional nucleotide prediction head
- **SampleSequenceEncoder**: Composes ASVEncoder, adds sample-level transformer
- **SequenceEncoder**: Base model with UniFrac prediction head (used in pretraining)
- **SequencePredictor**: Main model, composes SequenceEncoder, adds target prediction head

### Design Pattern
SequencePredictor **composes** SequenceEncoder (not inheritance). This enables:
- Loading pre-trained encoder checkpoints
- Freezing base model with `--freeze-base`
- Flexible base model swapping

### Multi-Task Learning
All tasks share base embeddings but compute predictions in parallel:
- **Primary**: Target prediction (regression/classification)
- **Auxiliary**: Count prediction, UniFrac prediction (self-supervised), Nucleotide prediction (self-supervised)

### Data Pipeline (aam/data/)
- **BIOMLoader**: Loads/rarefies BIOM tables
- **UniFracLoader**: Loads pre-computed UniFrac distance matrices (.npy, .h5, .csv)
- **SequenceTokenizer**: A/C/G/T → 1/2/3/4
- **ASVDataset**: PyTorch Dataset with custom collate for variable ASV counts

### Training (aam/training/)
- **losses.py**: MultiTaskLoss combining target, UniFrac, nucleotide, count losses
- **trainer.py**: Training loop, checkpointing, TensorBoard logging
- **metrics.py**: Regression/classification evaluation metrics

## Training Strategy

**Stage 1 (Pretraining)**: Train SequenceEncoder on UniFrac + nucleotide prediction (self-supervised, no labels)

**Stage 2 (Fine-tuning)**: Train SequencePredictor with pre-trained encoder
- `--freeze-base`: Faster training, freeze encoder weights
- Without flag: Fine-tune jointly for better performance

## Key Constraints

- Batch size must be even (UniFrac pairwise distances)
- `--token-limit` (default 1024) is critical for memory - reduce to 256-512 for 24GB GPUs
- UniFrac matrices must be pre-computed externally (use `ssu` from unifrac-binaries)

## Test Data

Integration test data in `./data/` (781 samples):
- `fall_train_only_all_outdoor.biom` - BIOM table
- `fall_train_only_all_outdoor.h5` - Pre-computed UniFrac matrix
- `fall_train_only_all_outdoor.tsv` - Metadata with regression targets
- `all-outdoors_sepp_tree.nwk` - Phylogenetic tree

See README Quick Start for example commands using this data.

## Development Workflow

See `.agents/` for ticket-based development workflow:
- `.agents/WORKFLOW.md` - Standard ticket workflow (branch naming, commit structure, test-driven development)
- `.agents/PYTORCH_PORTING_TICKETS.md` - Detailed ticket descriptions with acceptance criteria
- `.agents/TICKET_OVERVIEW.md` - Outstanding work summary and priorities

### Ticket Workflow Summary
1. Review `.agents/PYTORCH_PORTING_TICKETS.md` for next ticket
2. Create branch: `git checkout -b pyt-{ticket-number}-{short-name}`
3. Stub out work first → commit
4. Write tests → commit
5. Implement functionality → commit
6. After merge: Update ticket status and design plan docs

### Key Principles
- Test-driven: Write tests before implementation
- Minimize inline comments: Code should be self-documenting
- See `docs/` for user guides and `ARCHITECTURE.md` for design decisions

### Commit Messages
- Do NOT include Claude attribution or "Generated with Claude Code" in commit messages
- Use conventional commit style: `PYT-{ticket}: Brief description`
- Include a detailed body explaining the changes
