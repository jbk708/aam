# Hyperparameter Survey Plan: add_0c Regression

**Target:** MAE < 50
**Dataset:** all_outdoor_samples_mito_filtered (1984 samples)
**Epochs per run:** 50
**GPUs:** 4 (distributed DDP)
**Total runs:** 14

---

## Survey Design

The survey uses a **sequential elimination** strategy:
1. Run baseline + loss variants in parallel
2. Run fusion variants in parallel
3. Run regressor variants in parallel
4. Run learning rate variants in parallel
5. Run 3 combination runs with best settings from each phase

### Phase Structure

| Phase | Parameter | Variants | Runs | Status |
|-------|-----------|----------|------|--------|
| 1 | Loss function | huber, mse, mae | 3 | ✅ Complete |
| 2 | Fusion strategy | concat, gmu | 2 | ✅ Complete |
| 3 | Regressor arch | shallow, deep, residual | 3 | ✅ Complete |
| 4 | Learning rate | 5e-5, 2e-4, cosine_restarts | 3 | ✅ Complete |
| 5 | Combinations | best_combo, perceiver, cat_weights, cat_lr, onecycle | 5 | Pending |
| 6 | Unfreeze base | lr_1e5, lr_5e6, lr_1e5_100ep | 3 | Pending |
| **Total** | | | **19** |

---

## Base Configuration (matching pretrained encoder)

```bash
# Common args for all runs
TABLE=/sdsc/scc/ddp478/jkirkland/all_outdoor/all_outdoor_samples_mito_filtered.biom
UNIFRAC=/sdsc/scc/ddp478/jkirkland/all_outdoor/all_outdoor_samples_mito_filtered_dm.h5
METADATA=/sdsc/scc/ddp478/jkirkland/all_outdoor/metadata/metadata_all_outdoor.tsv
TARGET=add_0c
CATEGORICALS="season,facility"
PRETRAINED=/cosmos/nfs/home/jkirkland/repos/aam/data/pretrain-all-outdoor-small-5000/pretrained_encoder.pt
OUTPUT_BASE=/cosmos/nfs/home/jkirkland/repos/aam/data/survey_add_0c

# Fixed params
EPOCHS=50
TOKEN_LIMIT=5000
SEED=42

# Model architecture (must match pretrained encoder)
EMBEDDING_DIM=768
ATTENTION_HEADS=8
ATTENTION_LAYERS=4
```

---

## Phase 1: Loss Function Comparison

Compare loss functions with current best settings.

### Run 1.1: Huber (baseline)
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase1_loss/huber
```

### Run 1.2: MSE
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type mse \
    --regressor-hidden-dims "256,64" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase1_loss/mse
```

### Run 1.3: MAE
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase1_loss/mae
```

---

## Phase 2: Fusion Strategy Comparison

### Run 2.1: Concat fusion
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion concat \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase2_fusion/concat
```

### Run 2.2: Cross-attention fusion (same as 1.1, can skip)
*(Use results from Run 1.1)*

### Run 2.3: GMU fusion
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase2_fusion/gmu
```

---

## Phase 3: Regressor Architecture

### Run 3.1: Shallow MLP [128]
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "128" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase3_regressor/shallow_128
```

### Run 3.2: Deep MLP [512, 256, 64]
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "512,256,64" \
    --regressor-dropout 0.1 \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase3_regressor/deep_512_256_64
```

### Run 3.3: Residual head [256, 64]
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --residual-regression-head \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase3_regressor/residual_256_64
```

---

## Phase 4: Learning Rate & Scheduler

### Run 4.1: Lower LR (5e-5)
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr 5e-5 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase4_lr/lr_5e5
```

### Run 4.2: Higher LR (2e-4)
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr 2e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase4_lr/lr_2e4
```

### Run 4.3: Cosine restarts scheduler
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion cross-attention \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr 1e-4 \
    --scheduler cosine_restarts \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase4_lr/cosine_restarts
```

---

## Phase 5: Combination Runs (Based on Survey Results)

Best settings from phases 1-4:
- **Loss:** mae (MAE: 71.48)
- **Fusion:** gmu (MAE: 71.68, R²: 0.4413)
- **Regressor:** residual [256,64] (MAE: 71.47)
- **LR:** 5e-5 (MAE: 70.10)

### Run 5.1: Best Combo
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --pretrained-encoder $PRETRAINED --freeze-base \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 5e-5 \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/best_combo
```

### Run 5.2: Perceiver Fusion (New Feature)
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion perceiver \
    --perceiver-num-latents 64 --perceiver-num-layers 2 \
    --pretrained-encoder $PRETRAINED --freeze-base \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 5e-5 \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/perceiver
```

### Run 5.3: Category Loss Weighting
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --categorical-loss-weights auto --categorical-loss-weight-column season \
    --pretrained-encoder $PRETRAINED --freeze-base \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 5e-5 \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/category_weights
```

### Run 5.4: Separate Categorical Learning Rate
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --categorical-lr 5e-4 \
    --pretrained-encoder $PRETRAINED --freeze-base \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 5e-5 \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/categorical_lr
```

### Run 5.5: OneCycle Scheduler
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --pretrained-encoder $PRETRAINED --freeze-base \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 5e-5 --scheduler onecycle \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/onecycle
```

---

## Phase 6: Unfreeze Base Model

Higher risk, higher reward: fine-tune the pretrained encoder instead of freezing it.
Use lower LR to avoid catastrophic forgetting.

### Run 6.1: Unfreeze with lr=1e-5
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --pretrained-encoder $PRETRAINED \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 1e-5 \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase6_unfreeze/lr_1e5
```

### Run 6.2: Unfreeze with lr=5e-6
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --pretrained-encoder $PRETRAINED \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 5e-6 \
    --epochs $EPOCHS --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase6_unfreeze/lr_5e6
```

### Run 6.3: Unfreeze with 100 Epochs
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE --unifrac-matrix $UNIFRAC --metadata $METADATA \
    --metadata-column $TARGET --categorical-columns "$CATEGORICALS" \
    --categorical-fusion gmu \
    --pretrained-encoder $PRETRAINED \
    --target-transform zscore \
    --loss-type mae \
    --regressor-hidden-dims "256,64" --residual-regression-head \
    --lr 1e-5 \
    --epochs 100 --token-limit $TOKEN_LIMIT --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase6_unfreeze/lr_1e5_100ep
```

---

## Results Tracking

### Phases 1-4 (Initial Survey - Complete)

| Run | Phase | Variant | Val MAE | R² | Best Epoch | Notes |
|-----|-------|---------|---------|-----|------------|-------|
| 1.1 | Loss | huber | 71.60 | 0.4023 | 17 | baseline |
| 1.2 | Loss | mse | 73.40 | 0.3841 | 16 | worst loss |
| 1.3 | Loss | mae | 71.48 | 0.4216 | 20 | **best loss** |
| 2.1 | Fusion | concat | 101.22 | -0.1111 | 3 | broken |
| 2.3 | Fusion | gmu | 71.68 | 0.4413 | 17 | **best R²** |
| 3.1 | Regressor | shallow [128] | 75.26 | 0.3960 | 21 | |
| 3.2 | Regressor | deep [512,256,64] | 72.59 | 0.4085 | 30 | |
| 3.3 | Regressor | residual [256,64] | 71.47 | 0.4231 | 23 | **best regressor** |
| 4.1 | LR | 5e-5 | 70.10 | 0.4221 | 28 | **best overall** |
| 4.2 | LR | 2e-4 | 73.21 | 0.4198 | 14 | |
| 4.3 | LR | cosine_restarts | 74.26 | 0.3964 | 7 | hurt performance |

**Best from each phase:**
- Loss: `mae` (MAE: 71.48)
- Fusion: `gmu` (MAE: 71.68, best R²: 0.4413)
- Regressor: `residual_256_64` (MAE: 71.47)
- LR: `5e-5` (MAE: 70.10)

### Phases 5-6 (Optimization Runs)

| Run | Phase | Variant | Val MAE | R² | Best Epoch | Notes |
|-----|-------|---------|---------|-----|------------|-------|
| 5.1 | Combo | best_combo | | | | mae+gmu+residual+5e-5 |
| 5.2 | Combo | perceiver | | | | perceiver fusion (new) |
| 5.3 | Combo | category_weights | | | | auto loss weighting |
| 5.4 | Combo | categorical_lr | | | | 5e-4 for cat params |
| 5.5 | Combo | onecycle | | | | onecycle scheduler |
| 6.1 | Unfreeze | lr_1e5 | | | | unfreeze base, lr=1e-5 |
| 6.2 | Unfreeze | lr_5e6 | | | | unfreeze base, lr=5e-6 |
| 6.3 | Unfreeze | lr_1e5_100ep | | | | unfreeze, 100 epochs |

---

## Analysis Script

After runs complete, extract results:

```bash
#!/bin/bash
# analyze_survey.sh

OUTPUT_BASE=/cosmos/nfs/home/jkirkland/repos/aam/data/survey_add_0c

echo "Run,Phase,Variant,Best_MAE,Best_Epoch"

for dir in $OUTPUT_BASE/phase*/*/; do
    run_name=$(basename $dir)
    phase=$(basename $(dirname $dir))

    # Extract best MAE from training log
    if [ -f "$dir/training.log" ]; then
        best_mae=$(grep -oP "mae: \K[0-9.]+" "$dir/training.log" | sort -n | head -1)
        best_epoch=$(grep "Best model saved" "$dir/training.log" | tail -1 | grep -oP "epoch \K[0-9]+")
        echo "$run_name,$phase,$run_name,$best_mae,$best_epoch"
    fi
done
```

---

## SLURM Job Template

```bash
#!/bin/bash
#SBATCH --job-name=aam_survey
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=survey_%A_%a.out
#SBATCH --error=survey_%A_%a.err

# Load environment
source ~/.bashrc
conda activate aam

# Run the specific phase/variant based on array task ID
# See run_survey.sh for the mapping
```
