# How AAM Works

This guide explains both the scientific concepts behind AAM and its implementation details.

## Table of Contents

- **Part 1: Concepts** - The science and ML behind AAM
- **Part 2: Implementation** - Code architecture and data flow

---

# Part 1: Concepts

Understanding what AAM does and why.

## Microbiome Data Primer

### What is a BIOM Table?

A BIOM (Biological Observation Matrix) table is a sparse matrix where:
- **Rows** = Microbial features (ASVs or OTUs)
- **Columns** = Samples
- **Values** = Abundance counts

Example:
```
           Sample1  Sample2  Sample3
ASV_001       150       0       45
ASV_002         0      89       12
ASV_003        67     234        0
```

### What are ASVs?

**Amplicon Sequence Variants (ASVs)** are exact DNA sequences from 16S rRNA gene sequencing. Unlike OTUs (clustered at 97% similarity), ASVs preserve single-nucleotide resolution.

Each ASV is typically 150-250 base pairs of A, C, G, T nucleotides.

### What is UniFrac Distance?

**UniFrac** measures the phylogenetic distance between microbial communities by comparing how much of the evolutionary tree is shared vs. unique to each sample.

- **Unweighted UniFrac**: Presence/absence only (are the same species present?)
- **Weighted UniFrac**: Accounts for abundance (are they present in similar amounts?)

AAM uses pre-computed UniFrac distances as a self-supervised learning signal.

### Why Phylogenetic Information?

Related microbes often have similar functions. By learning to predict phylogenetic relationships (UniFrac), the model learns biologically meaningful representations even without labeled data.

## Transformer Architecture Basics

### Attention Mechanism

Attention allows the model to learn which parts of the input are relevant to each other. For a sequence of ASVs, attention learns:
- Which ASVs co-occur meaningfully
- Which nucleotide positions are important within a sequence
- How different ASVs relate to sample-level outcomes

### Why Transformers for Sequence Data?

Transformers excel at:
1. **Variable-length inputs** - Samples have different numbers of ASVs
2. **Long-range dependencies** - ASVs far apart in the list may be related
3. **Parallel processing** - More efficient than RNNs

### Position Embeddings

Since attention is permutation-invariant, position embeddings tell the model where each element appears in the sequence.

## Hierarchical Processing

AAM processes data at three levels:

```
Nucleotide Level (150 bp per ASV)
        ↓
   ASV Level (1000s of ASVs per sample)
        ↓
   Sample Level (predictions)
```

### Why Hierarchical?

Think of it like understanding a document:
- **Nucleotides** = Characters
- **ASVs** = Words
- **Samples** = Documents

Each level captures different patterns:
- Nucleotide: Sequence motifs, taxonomic signatures
- ASV: Community composition patterns
- Sample: Ecological/health relationships

## Self-Supervised Learning

### What is Self-Supervised Learning?

Learning representations from data structure itself, without human-provided labels. AAM uses two self-supervised tasks:

### 1. Masked Nucleotide Prediction

Like BERT's masked language modeling:
1. Randomly mask 15% of nucleotides
2. Predict the masked nucleotides from context
3. Model learns sequence patterns

### 2. UniFrac Prediction

1. Compute sample embeddings
2. Predict pairwise UniFrac distances between samples in batch
3. Model learns phylogenetically-informed representations

### Why Pre-training Helps

Pre-training on self-supervised tasks:
- Learns useful representations before seeing any labels
- Reduces need for labeled training data
- Improves generalization to new datasets

## Multi-Task Learning

AAM predicts multiple outputs simultaneously:

| Task | Type | Purpose |
|------|------|---------|
| Target prediction | Primary | Your outcome of interest |
| UniFrac prediction | Auxiliary | Phylogenetic regularization |
| Nucleotide prediction | Auxiliary | Sequence pattern learning |
| Count prediction | Auxiliary | Abundance modeling |

### Why Multi-Task?

- Auxiliary tasks act as regularizers
- Shared representations learn more general features
- Prevents overfitting to limited labeled data

## Categorical Conditioning

### Why Condition on Metadata?

Samples from different environments (locations, seasons) may have systematically different relationships between microbiome and outcomes.

### Fusion Strategies

| Strategy | When to Use |
|----------|-------------|
| **Concat** | Simple conditioning, baseline |
| **GMU** | When balance of sequence vs. metadata varies |
| **Cross-attention** | When different ASVs should respond differently to metadata |

---

# Part 2: Implementation

Understanding the code architecture.

## Data Flow Overview

```
BIOM File → BIOMLoader → Rarefied Table
                              ↓
                    SequenceTokenizer → Tokens [B, S, L]
                              ↓
UniFrac File → UniFracLoader → Distances [N, N]
                              ↓
Metadata → Load targets → Labels [B]
                              ↓
            ASVDataset + DataLoader → Batches
```

## Model Forward Pass

```
Tokens [B, S, L]
      ↓
ASVEncoder
      ↓
ASV Embeddings [B, S, D]
      ↓
SampleSequenceEncoder
      ↓
Sample Embeddings [B, S, D]
      ↓
SequenceEncoder
      ↓
Pooled Embedding [B, D] + UniFrac Pred
      ↓
SequencePredictor
      ↓
Target Prediction [B, out_dim]
```

### Dimension Key

- **B** = Batch size (must be even for UniFrac)
- **S** = Max ASVs per sample (token_limit, default 1024)
- **L** = Max nucleotides per ASV (max_bp, default 150)
- **D** = Embedding dimension (default 128)

## Key Components

### ASVEncoder (`aam/models/asv_encoder.py`)

Processes individual ASV sequences:
1. Token embedding: nucleotides → vectors
2. Position embedding: add positional info
3. Transformer: learn sequence patterns
4. Attention pooling: variable length → fixed vector

### SampleSequenceEncoder (`aam/models/sample_sequence_encoder.py`)

Processes ASVs within a sample:
1. Compose ASVEncoder for each ASV
2. Add sample-level position embeddings
3. Sample-level transformer

### SequenceEncoder (`aam/models/sequence_encoder.py`)

Base model with UniFrac prediction:
1. Compose SampleSequenceEncoder
2. Encoder transformer
3. Attention pooling → sample embedding
4. UniFrac prediction head

### SequencePredictor (`aam/models/sequence_predictor.py`)

Main model (composes SequenceEncoder):
1. Get sample embeddings from SequenceEncoder
2. Optional categorical fusion
3. Target prediction head
4. Count prediction head

## Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch)

        # Multi-task loss
        loss = (
            target_penalty * target_loss +
            penalty * unifrac_loss +
            nuc_penalty * nucleotide_loss +
            count_penalty * count_loss
        )

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Validation
    val_metrics = evaluate(model, val_loader)

    # Early stopping
    if val_metrics improved:
        save_checkpoint()
    elif patience exceeded:
        break
```

## Extension Guide

### Adding a New Loss Function

1. Add to `aam/training/losses.py`
2. Register in `MultiTaskLoss`
3. Add CLI option in `aam/cli/train.py`

### Adding a New Fusion Strategy

1. Add class to `aam/models/fusion.py`
2. Register in `CategoricalFusion.STRATEGIES`
3. Add tests in `tests/test_fusion.py`

### Adding a New CLI Option

1. Add `@click.option()` in relevant command
2. Pass through to model/trainer
3. Document in user-guide.md

---

## Further Reading

- **Architecture** - See `ARCHITECTURE.md` in repo root for design decisions
- **Roadmap** - See `docs/roadmap.md` for future enhancements
- **API Reference** - Build with `cd docs && make html`
