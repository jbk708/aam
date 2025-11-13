# AAM Model Architecture Diagram

This document provides visual representations of the AAM (Attention All Microbes) model architecture.

## High-Level Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        BIOM[BIOM Table]
        METADATA[Metadata TSV]
        TREE[Phylogenetic Tree]
        TAX[Taxonomy File]
    end
    
    subgraph "Data Processing"
        GEN[Data Generators]
        GEN --> TAX_GEN[TaxonomyGenerator]
        GEN --> UNI_GEN[UniFracGenerator]
        GEN --> COMB_GEN[CombinedGenerator]
        TAX_GEN --> TOKENS1[Taxonomy Tokens]
        UNI_GEN --> DIST[UniFrac Distances]
        COMB_GEN --> COMB[Combined Targets]
    end
    
    subgraph "Model Architecture"
        INPUT[(nuc_tokens, counts)]
        INPUT --> BASE[BaseSequenceEncoder]
        BASE --> ASV_ENC[ASVEncoder<br/>Nucleotide-level Attention]
        ASV_ENC --> POS_EMB[Position Embeddings]
        POS_EMB --> SAMPLE_TRANS[Sample-level Transformer]
        SAMPLE_TRANS --> BASE_EMB[Base Embeddings]
        
        BASE_EMB --> SEQ_ENC[SequenceEncoder]
        SEQ_ENC --> ENC_TRANS[Encoder Transformer]
        ENC_TRANS --> ATT_POOL1[AttentionPooling]
        ATT_POOL1 --> BASE_PRED[Base Prediction]
        
        BASE_EMB --> SEQ_REG[SequenceRegressor]
        SEQ_REG --> COUNT_ENC[Count Encoder]
        SEQ_REG --> TARGET_ENC[Target Encoder]
        COUNT_ENC --> COUNT_PRED[Count Prediction]
        TARGET_ENC --> TARGET_EMB[Target Embeddings]
        TARGET_EMB --> ATT_POOL2[AttentionPooling]
        ATT_POOL2 --> FINAL_PRED[Final Prediction]
    end
    
    subgraph "Outputs"
        BASE_PRED --> ENC_OUT[Encoder Output]
        COUNT_PRED --> COUNT_OUT[Count Output]
        FINAL_PRED --> TARGET_OUT[Target Output]
        FINAL_PRED --> NUC_PRED[Nucleotide Prediction]
    end
    
    BIOM --> GEN
    METADATA --> GEN
    TREE --> UNI_GEN
    TAX --> TAX_GEN
    
    TOKENS1 --> INPUT
    DIST --> INPUT
    COMB --> INPUT
```

## SequenceRegressor Detailed Architecture

```mermaid
graph LR
    subgraph "Input"
        TOKENS[nuc_tokens<br/>B x S x L]
        COUNTS[counts<br/>B x S x 1]
    end
    
    subgraph "BaseModel: SequenceEncoder"
        BASE_BASE[BaseSequenceEncoder]
        BASE_BASE --> ASV[ASVEncoder]
        ASV --> NUCL[Nucleotide<br/>Attention<br/>L layers]
        NUCL --> ASV_EMB[ASV Embeddings<br/>B x S x D]
        ASV_EMB --> POS[Position<br/>Embeddings]
        POS --> SAMPLE[Sample-level<br/>Transformer<br/>L layers]
        SAMPLE --> BASE_EMB[Base Embeddings<br/>B x S x D]
        
        BASE_EMB --> ENC_TRANS[Encoder<br/>Transformer<br/>L layers]
        ENC_TRANS --> ENC_POOL[Attention<br/>Pooling]
        ENC_POOL --> BASE_FF[Dense<br/>base_output_dim]
        BASE_FF --> BASE_PRED[Base Prediction]
    end
    
    subgraph "SequenceRegressor Layers"
        BASE_EMB --> COUNT_TRANS[Count Encoder<br/>Transformer<br/>L layers]
        BASE_EMB --> TARGET_TRANS[Target Encoder<br/>Transformer<br/>L layers]
        
        COUNT_TRANS --> COUNT_FF[Dense 1]
        COUNT_FF --> COUNT_PRED[Count Prediction<br/>B x S x 1]
        
        TARGET_TRANS --> TARGET_EMB[Target Embeddings<br/>B x S x D]
        TARGET_EMB --> TARGET_POOL[Attention<br/>Pooling]
        TARGET_POOL --> TARGET_FF[Dense<br/>out_dim]
        TARGET_FF --> TARGET_PRED[Target Prediction<br/>B x out_dim]
    end
    
    subgraph "Loss Computation"
        BASE_PRED --> ENC_LOSS[Encoder Loss]
        COUNT_PRED --> COUNT_LOSS[Count Loss<br/>MSE]
        TARGET_PRED --> TARGET_LOSS[Target Loss<br/>MSE/CrossEntropy]
        NUCL --> NUC_LOSS[Nucleotide Loss<br/>CrossEntropy]
    end
    
    TOKENS --> BASE_BASE
    COUNTS --> BASE_BASE
```

## BaseSequenceEncoder Architecture

```mermaid
graph TB
    INPUT1[(tokens: B x S x L<br/>counts: B x S x 1)]
    
    subgraph "Nucleotide-Level Processing"
        INPUT1 --> ASV_ENC[ASVEncoder]
        ASV_ENC --> EMB_LAYER[Embedding Layer<br/>vocab_size x D]
        EMB_LAYER --> POS_EMB1[Position Embeddings<br/>max_bp + 1]
        POS_EMB1 --> NUCL_TRANS[Nucleotide Transformer<br/>nuc_layers x<br/>nuc_heads]
        NUCL_TRANS --> NUCL_POOL[Attention Pooling]
        NUCL_POOL --> ASV_OUT[ASV Embeddings<br/>B x S x D]
    end
    
    subgraph "Sample-Level Processing"
        ASV_OUT --> POS_EMB2[Position Embeddings<br/>token_limit + 5]
        POS_EMB2 --> SAMPLE_TRANS[Sample Transformer<br/>sample_layers x<br/>sample_heads]
        SAMPLE_TRANS --> BASE_OUT[Base Embeddings<br/>B x S x D]
    end
    
    BASE_OUT --> OUTPUT[(embeddings, ...)]
```

## Data Flow Through Generators

```mermaid
sequenceDiagram
    participant CLI as CLI Command
    participant GEN as GeneratorDataset
    participant TABLE as BIOM Table
    participant META as Metadata
    participant DS as TensorFlow Dataset
    
    CLI->>GEN: Initialize with table, metadata
    GEN->>TABLE: Load BIOM table
    GEN->>META: Load and normalize metadata
    GEN->>TABLE: Rarefy to depth
    GEN->>GEN: Tokenize sequences (A/C/G/T ? 1/2/3/4)
    GEN->>GEN: Create encoder targets
    
    CLI->>GEN: get_data()
    GEN->>GEN: _create_epoch_generator()
    loop Each Epoch
        GEN->>TABLE: New rarefied table
        loop Each Batch
            GEN->>GEN: Sample batch indices
            GEN->>GEN: Extract batch data
            GEN->>GEN: Sort ASVs by count
            GEN->>GEN: Limit to max_token_per_sample
            GEN->>GEN: Pad sequences
            GEN->>DS: Yield (tokens, counts, y_target, encoder_target)
        end
    end
    GEN->>CLI: Return dataset dict
```

## Training Pipeline Architecture

```mermaid
graph TB
    subgraph "Data Preparation"
        DATA[Data Generators]
        DATA --> TRAIN_DATA[Training Data]
        DATA --> VAL_DATA[Validation Data]
    end
    
    subgraph "Cross-Validation"
        TRAIN_DATA --> CV[CV Splitter<br/>KFold/StratifiedKFold]
        CV --> FOLD1[Fold 1]
        CV --> FOLD2[Fold 2]
        CV --> FOLDN[Fold N]
    end
    
    subgraph "Model Training"
        FOLD1 --> MODEL1[Create Model]
        FOLD2 --> MODEL2[Create Model]
        FOLDN --> MODELN[Create Model]
        
        MODEL1 --> TRAIN1[Train Fold 1]
        MODEL2 --> TRAIN2[Train Fold 2]
        MODELN --> TRAINN[Train Fold N]
        
        TRAIN1 --> CV_MODEL1[CVModel 1]
        TRAIN2 --> CV_MODEL2[CVModel 2]
        TRAINN --> CV_MODELN[CVModel N]
    end
    
    subgraph "Ensemble"
        CV_MODEL1 --> ENSEMBLE[EnsembleModel]
        CV_MODEL2 --> ENSEMBLE
        CV_MODELN --> ENSEMBLE
        ENSEMBLE --> BEST[Best Model]
        ENSEMBLE --> PRED[Ensemble Predictions]
    end
    
    subgraph "Callbacks"
        TRAIN1 --> CALLBACKS[Callbacks]
        CALLBACKS --> SAVE[SaveModel]
        CALLBACKS --> METRICS[MeanAbsoluteError]
        CALLBACKS --> CONF[ConfusionMatrix]
        CALLBACKS --> TB[TensorBoard]
    end
```

## Encoder Types Architecture

```mermaid
graph TB
    subgraph "SequenceEncoder Types"
        BASE[BaseSequenceEncoder<br/>Common Base]
        
        BASE --> TAX_ENC[Taxonomy Encoder<br/>encoder_type='taxonomy']
        BASE --> UNI_ENC[UniFrac Encoder<br/>encoder_type='unifrac']
        BASE --> FAITH_ENC[Faith PD Encoder<br/>encoder_type='faith_pd']
        BASE --> COMB_ENC[Combined Encoder<br/>encoder_type='combined']
        
        TAX_ENC --> TAX_HEAD[Taxonomy Head<br/>Dense Layer]
        UNI_ENC --> UNI_HEAD[UniFrac Head<br/>Dense Layer]
        FAITH_ENC --> FAITH_HEAD[Faith PD Head<br/>Dense Layer]
        COMB_ENC --> COMB_HEAD1[UniFrac Head]
        COMB_ENC --> COMB_HEAD2[Faith PD Head]
        COMB_ENC --> COMB_HEAD3[Taxonomy Head]
        
        TAX_HEAD --> TAX_OUT[Taxonomy Prediction]
        UNI_HEAD --> UNI_OUT[UniFrac Prediction]
        FAITH_HEAD --> FAITH_OUT[Faith PD Prediction]
        COMB_HEAD1 --> COMB_OUT1[UniFrac Output]
        COMB_HEAD2 --> COMB_OUT2[Faith PD Output]
        COMB_HEAD3 --> COMB_OUT3[Taxonomy Output]
    end
```

## Loss Computation Flow

```mermaid
graph LR
    subgraph "Model Outputs"
        TARGET_PRED[Target Prediction]
        COUNT_PRED[Count Prediction]
        BASE_PRED[Base Prediction]
        NUC_PRED[Nucleotide Prediction]
    end
    
    subgraph "Ground Truth"
        Y_TRUE[Target Values]
        COUNT_TRUE[Count Values]
        BASE_TRUE[Base Targets]
        NUC_TRUE[Nucleotide Targets]
    end
    
    subgraph "Loss Functions"
        TARGET_PRED --> TARGET_LOSS[Target Loss<br/>MSE or<br/>CrossEntropy]
        Y_TRUE --> TARGET_LOSS
        
        COUNT_PRED --> COUNT_LOSS[Count Loss<br/>MSE]
        COUNT_TRUE --> COUNT_LOSS
        
        BASE_PRED --> BASE_LOSS[Encoder Loss<br/>PairwiseLoss or<br/>MSE]
        BASE_TRUE --> BASE_LOSS
        
        NUC_PRED --> NUC_LOSS[Nucleotide Loss<br/>SparseCategorical<br/>CrossEntropy]
        NUC_TRUE --> NUC_LOSS
    end
    
    subgraph "Total Loss"
        TARGET_LOSS --> TOTAL[Total Loss<br/>weighted sum]
        COUNT_LOSS --> TOTAL
        BASE_LOSS --> TOTAL
        NUC_LOSS --> TOTAL
    end
    
    TOTAL --> BACKPROP[Backpropagation]
```

## Attention Mechanism Details

```mermaid
graph TB
    subgraph "AttentionPooling Layer"
        INPUT_EMB[Input Embeddings<br/>B x T x D]
        INPUT_EMB --> QUERY[Dense Query Layer<br/>D ? 1]
        QUERY --> SCORES[Attention Scores<br/>B x T]
        SCORES --> SCALE[Scale by ?D]
        SCALE --> MASK[Apply Mask<br/>if provided]
        MASK --> SOFTMAX[Softmax<br/>B x T]
        SOFTMAX --> WEIGHTS[Attention Weights<br/>B x T]
        WEIGHTS --> WEIGHTED_SUM[Weighted Sum<br/>B x D]
        WEIGHTED_SUM --> NORM[Layer Normalization]
        NORM --> OUTPUT[Pooled Output<br/>B x D]
    end
```

## Model Component Hierarchy

```mermaid
classDiagram
    class tf_keras_Model {
        <<abstract>>
    }
    
    class SequenceRegressor {
        +base_model: SequenceEncoder
        +count_encoder: TransformerEncoder
        +target_encoder: TransformerEncoder
        +attention_pooling: AttentionPooling
        +train_step()
        +test_step()
        +_compute_loss()
    }
    
    class SequenceEncoder {
        +base_encoder: BaseSequenceEncoder
        +encoder: TransformerEncoder
        +attention_pooling: AttentionPooling
        +encoder_ff: Dense
    }
    
    class BaseSequenceEncoder {
        +asv_encoder: ASVEncoder
        +asv_pos: PositionEmbedding
        +sample_transformer: TransformerEncoder
    }
    
    class TransformerEncoder {
        +num_layers: int
        +num_attention_heads: int
        +intermediate_size: int
    }
    
    class ASVEncoder {
        +emb_layer: Embedding
        +pos_emb: PositionEmbedding
        +asv_attention: TransformerEncoder
        +attention_pool: AttentionPooling
    }
    
    class AttentionPooling {
        +query: Dense
        +norm: LayerNormalization
    }
    
    tf_keras_Model <|-- SequenceRegressor
    tf_keras_Model <|-- SequenceEncoder
    tf_keras_layers_Layer <|-- BaseSequenceEncoder
    tf_keras_layers_Layer <|-- TransformerEncoder
    tf_keras_layers_Layer <|-- ASVEncoder
    tf_keras_layers_Layer <|-- AttentionPooling
    
    SequenceRegressor --> SequenceEncoder : uses
    SequenceEncoder --> BaseSequenceEncoder : uses
    BaseSequenceEncoder --> ASVEncoder : uses
    BaseSequenceEncoder --> TransformerEncoder : uses
    SequenceEncoder --> TransformerEncoder : uses
    SequenceEncoder --> AttentionPooling : uses
    SequenceRegressor --> TransformerEncoder : uses
    SequenceRegressor --> AttentionPooling : uses
    ASVEncoder --> TransformerEncoder : uses
    ASVEncoder --> AttentionPooling : uses
```

## Dimension Flow

```mermaid
graph LR
    subgraph "Input Dimensions"
        I1[nuc_tokens:<br/>B x S x L]
        I2[counts:<br/>B x S x 1]
    end
    
    subgraph "ASVEncoder"
        I1 --> E1[Embedding:<br/>B x S x L x D]
        E1 --> P1[+ Position:<br/>B x S x L x D]
        P1 --> T1[Transformer:<br/>B x S x L x D]
        T1 --> A1[Pooling:<br/>B x S x D]
    end
    
    subgraph "Sample-Level"
        A1 --> P2[+ Position:<br/>B x S x D]
        P2 --> T2[Transformer:<br/>B x S x D]
    end
    
    subgraph "SequenceEncoder"
        T2 --> T3[Transformer:<br/>B x S x D]
        T3 --> A2[Pooling:<br/>B x D]
        A2 --> D1[Dense:<br/>B x base_dim]
    end
    
    subgraph "SequenceRegressor"
        T2 --> T4[Count Encoder:<br/>B x S x D]
        T2 --> T5[Target Encoder:<br/>B x S x D]
        T4 --> D2[Dense 1:<br/>B x S x 1]
        T5 --> A3[Pooling:<br/>B x D]
        A3 --> D3[Dense:<br/>B x out_dim]
    end
    
    Legend:
    B = Batch size
    S = Sequence length (max ASVs per sample)
    L = Sequence length (max base pairs)
    D = Embedding dimension
```

## Notes

- **B**: Batch size
- **S**: Maximum number of ASVs per sample (token_limit)
- **L**: Maximum number of base pairs per sequence (max_bp)
- **D**: Embedding dimension (typically 128)
- **base_dim**: Base encoder output dimension
- **out_dim**: Final output dimension (1 for regression, num_classes for classification)

## Key Components

1. **ASVEncoder**: Processes individual sequences at nucleotide level
2. **BaseSequenceEncoder**: Combines ASV embeddings with sample-level processing
3. **SequenceEncoder**: Adds encoder-specific prediction head
4. **SequenceRegressor**: Adds count and target prediction heads
5. **AttentionPooling**: Pools sequence-level embeddings to sample-level
6. **TransformerEncoder**: Standard transformer encoder blocks

## Data Flow Summary

1. **Input**: Tokenized sequences (A/C/G/T ? 1/2/3/4) and counts
2. **Nucleotide-level**: ASVEncoder processes each sequence
3. **Sample-level**: Transformer processes ASV embeddings
4. **Encoder**: Predicts encoder-specific targets (taxonomy, UniFrac, etc.)
5. **Regressor**: Predicts final targets (sample metadata) and counts
6. **Output**: Predictions at multiple levels (nucleotide, ASV, sample)
