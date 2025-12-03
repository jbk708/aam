# AAM Architecture

Deep learning model for microbial sequencing data analysis using transformer-based attention mechanisms.

## Overview

AAM processes nucleotide sequences at multiple levels (nucleotide, ASV, sample) and uses phylogenetic information from reference trees for self-supervised learning.

## Architecture

**Key Design Points:**
- **SequencePredictor composes SequenceEncoder**: Composition pattern enables flexible base model swapping
- **Hierarchical processing**: Nucleotide → ASV → Sample level embeddings
- **Parallel multi-task learning**: All tasks share base embeddings but compute predictions in parallel
- **Self-supervised learning**: UniFrac and nucleotide predictions used for loss only, not as input

### Architecture Flow Diagram

```mermaid
graph TB
    subgraph "Input Data"
        BIOM[BIOM Table<br/>.biom file]
        PHYLO[Phylogenetic Tree<br/>.nwk file]
        META[Metadata<br/>.tsv file]
    end
    
    subgraph "Data Processing Pipeline"
        BIOM --> LOAD[Load & Rarefy Table]
        PHYLO --> UNIFRAC[Compute UniFrac Distances<br/>unweighted / Faith PD]
        LOAD --> UNIFRAC
        LOAD --> TOKENIZE[Tokenize Sequences<br/>A/C/G/T → 1/2/3/4]
        LOAD --> RAREFY[Rarefy to Depth]
        TOKENIZE --> PREPARE[Prepare Batches<br/>tokens, counts, targets, unifrac_targets]
        UNIFRAC --> PREPARE
        META --> PREPARE
    end
    
    subgraph "Model Architecture - Hierarchical Processing"
        PREPARE --> TOKENS[tokens<br/>B x S x L]
        PREPARE --> COUNTS[counts<br/>B x S x 1]
        
        TOKENS --> SAMPLE[SampleSequenceEncoder]
        SAMPLE --> ASV_ENC[ASVEncoder<br/>Nucleotide Level]
        ASV_ENC --> ASV_EMB[ASV Embeddings<br/>B x S x D]
        ASV_EMB --> SAMPLE_TRANS[Sample Transformer]
        SAMPLE_TRANS --> SAMPLE_EMB[Sample Embeddings<br/>B x S x D]
        
        SAMPLE_EMB --> SEQ_ENC[SequenceEncoder<br/>Base Model]
        SEQ_ENC --> ENC_TRANS[Encoder Transformer]
        ENC_TRANS --> ENC_POOL[Attention Pooling]
        ENC_POOL --> BASE_PRED[Base Prediction<br/>UniFrac/Taxonomy]
        
        SEQ_ENC --> SEQ_PRED[SequencePredictor<br/>Composes SequenceEncoder]
        SEQ_PRED --> SAMPLE_EMB_REG[Sample Embeddings<br/>from SequenceEncoder<br/>B x S x D]
        SAMPLE_EMB_REG --> COUNT_ENC[Count Encoder]
        SAMPLE_EMB_REG --> TARGET_ENC[Target Encoder]
        COUNT_ENC --> COUNT_PRED[Count Prediction]
        TARGET_ENC --> TARGET_POOL[Attention Pooling]
        TARGET_POOL --> TARGET_PRED[Target Prediction]
    end
    
    subgraph "Multi-Task Learning (Parallel Tasks)"
        SEQ_ENC --> BASE_PRED
        SEQ_PRED --> COUNT_PRED
        SEQ_PRED --> TARGET_PRED
        ASV_ENC --> NUC_PRED[Nucleotide Predictions<br/>Self-Supervised<br/>Side Output]
        
        BASE_PRED --> LOSS1[Base Loss<br/>UniFrac/Taxonomy<br/>Self-Supervised]
        COUNT_PRED --> LOSS2[Count Loss<br/>MSE<br/>Auxiliary]
        TARGET_PRED --> LOSS3[Target Loss<br/>MSE/CrossEntropy<br/>Primary]
        NUC_PRED --> LOSS4[Nucleotide Loss<br/>Self-Supervised]
        
        LOSS1 --> TOTAL[Total Loss<br/>weighted sum]
        LOSS2 --> TOTAL
        LOSS3 --> TOTAL
        LOSS4 --> TOTAL
        
        style BASE_PRED fill:#fff9c4
        style NUC_PRED fill:#fff9c4
        style COUNT_PRED fill:#e3f2fd
        style TARGET_PRED fill:#c8e6c9
    end
    
    style BIOM fill:#e1f5ff
    style PHYLO fill:#e1f5ff
    style META fill:#e1f5ff
    style UNIFRAC fill:#fff4e1
    style SEQ_ENC fill:#e8f5e9
    style SEQ_PRED fill:#ffebee
    style TOTAL fill:#ffcdd2
```

## Model Components

### Data Pipeline
- **BIOMLoader**: Loads and rarefies BIOM tables
- **UniFracComputer**: Computes phylogenetic distances (unweighted UniFrac, Faith PD)
- **SequenceTokenizer**: Converts nucleotide sequences (A/C/G/T) to tokens (1/2/3/4)
- **ASVDataset**: PyTorch Dataset with custom collate function for variable ASV counts

### Model Architecture

**ASVEncoder**: Processes nucleotide sequences → ASV embeddings `[B, S, D]`
- Token embeddings + Position embeddings
- Transformer encoder
- Attention pooling
- Optional nucleotide prediction head

**SampleSequenceEncoder**: Processes ASV embeddings at sample level
- Composes ASVEncoder
- Sample-level position embeddings
- Sample-level transformer
- Output: Sample embeddings `[B, S, D]`

**SequenceEncoder**: Base model with UniFrac prediction head
- Composes SampleSequenceEncoder
- Encoder transformer + Attention pooling
- Output: Base predictions (UniFrac/Taxonomy) + Sample embeddings
- Types: `unifrac`, `taxonomy`, `faith_pd`, `combined`

**SequencePredictor**: Main prediction model
- Composes SequenceEncoder as `base_model`
- Count encoder: Predicts ASV counts `[B, S, 1]`
- Target encoder: Predicts sample targets `[B, out_dim]`
- Supports `freeze_base` for transfer learning

## Multi-Task Learning

**Primary Task**: Target prediction (regression/classification)

**Auxiliary Tasks** (parallel, share base embeddings):
- **Count prediction**: Predicts ASV abundances
- **UniFrac prediction**: Self-supervised, phylogenetic signal (loss only)
- **Nucleotide prediction**: Self-supervised, sequence patterns (loss only)

**Important**: Base predictions and nucleotide predictions are side outputs used for loss computation only. They do NOT feed into target prediction.

## Training Strategy

### Staged Training (Recommended)

**Stage 1: Pre-train SequenceEncoder**
- Self-supervised: UniFrac + nucleotide prediction
- No target labels required
- Saves checkpoint for Stage 2

**Stage 2: Train SequencePredictor**
- Option A: Freeze base (`--freeze-base`) - faster
- Option B: Fine-tune jointly - better performance

### Training Configuration

- **Optimizer**: AdamW (weight decay: 0.01)
- **Learning Rate**: 1e-4 with warmup (10k steps) + cosine decay
- **Early Stopping**: Patience 50 epochs
- **Batch Size**: Must be even for UniFrac pairwise distances

## Dimension Reference

- **B**: Batch size (must be even)
- **S**: Maximum ASVs per sample (token_limit, default: 1024)
- **L**: Maximum base pairs per sequence (max_bp, default: 150)
- **D**: Embedding dimension (default: 128)
- **H**: Number of attention heads (default: 4)

## File Structure

```
aam/
├── data/
│   ├── __init__.py
│   ├── biom_loader.py        # BIOM table loading and rarefaction
│   ├── unifrac.py            # UniFrac distance computation
│   ├── tokenizer.py          # Sequence tokenization
│   └── dataset.py            # PyTorch Dataset and collate function
├── models/
│   ├── __init__.py
│   ├── attention_pooling.py  # Attention pooling layer
│   ├── position_embedding.py # Position embedding layer
│   ├── transformer.py        # Transformer encoder
│   ├── asv_encoder.py        # ASV-level sequence encoder
│   ├── sample_sequence_encoder.py  # Sample-level encoder
│   ├── sequence_encoder.py   # Base model with UniFrac prediction
│   └── sequence_predictor.py # Main prediction model
├── training/
│   ├── __init__.py
│   ├── losses.py             # Multi-task loss functions
│   ├── metrics.py            # Evaluation metrics
│   └── trainer.py            # Training and validation loops
└── cli.py                    # Command-line interface
```

See `_design_plan/` for detailed implementation documentation.
