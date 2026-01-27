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

| Phase | Parameter | Variants | Runs |
|-------|-----------|----------|------|
| 1 | Loss function | huber, mse, mae | 3 |
| 2 | Fusion strategy | concat, cross-attn, gmu | 3 |
| 3 | Regressor arch | shallow, deep, residual | 3 |
| 4 | Learning rate | 5e-5, 1e-4, 2e-4 | 3 |
| 5 | Combinations | best from each phase | 2 |
| **Total** | | | **14** |

---

## Base Configuration (from your previous run)

```bash
# Common args for all runs
TABLE=/sdsc/scc/ddp478/jkirkland/all_outdoor/all_outdoor_samples_mito_filtered.biom
UNIFRAC=/sdsc/scc/ddp478/jkirkland/all_outdoor/all_outdoor_samples_mito_filtered_dm.h5
METADATA=/sdsc/scc/ddp478/jkirkland/all_outdoor/metadata/metadata_all_outdoor.tsv
TARGET=add_0c
CATEGORICALS="season,facility"
PRETRAINED=/cosmos/nfs/home/jkirkland/repos/aam/data/all-outdoor-small-model-zscore-5000token/checkpoints/best_model.pt
OUTPUT_BASE=/cosmos/nfs/home/jkirkland/repos/aam/data/survey_add_0c

# Fixed params
EPOCHS=50
TOKEN_LIMIT=5000
SEED=42
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

## Phase 5: Combination Runs

After phases 1-4 complete, select best from each and combine.

### Run 5.1: Best loss + Best regressor
*(Template - fill in after analyzing phase 1-4 results)*
```bash
# Replace BEST_LOSS and BEST_REGRESSOR_DIMS based on results
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
    --loss-type ${BEST_LOSS} \
    --regressor-hidden-dims "${BEST_REGRESSOR_DIMS}" \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/best_loss_regressor
```

### Run 5.2: Best fusion + Best LR
*(Template - fill in after analyzing phase 1-4 results)*
```bash
# Replace BEST_FUSION and BEST_LR based on results
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --table $TABLE \
    --unifrac-matrix $UNIFRAC \
    --metadata $METADATA \
    --metadata-column $TARGET \
    --categorical-columns "$CATEGORICALS" \
    --categorical-fusion ${BEST_FUSION} \
    --pretrained-encoder $PRETRAINED \
    --freeze-base \
    --target-transform zscore \
    --loss-type huber \
    --regressor-hidden-dims "256,64" \
    --lr ${BEST_LR} \
    --epochs $EPOCHS \
    --token-limit $TOKEN_LIMIT \
    --seed $SEED \
    --best-metric mae \
    --output-dir $OUTPUT_BASE/phase5_combo/best_fusion_lr
```

---

## Results Tracking

| Run | Phase | Variant | Val MAE | Best Epoch | Notes |
|-----|-------|---------|---------|------------|-------|
| 1.1 | Loss | huber | | | baseline |
| 1.2 | Loss | mse | | | |
| 1.3 | Loss | mae | | | |
| 2.1 | Fusion | concat | | | |
| 2.3 | Fusion | gmu | | | |
| 3.1 | Regressor | shallow [128] | | | |
| 3.2 | Regressor | deep [512,256,64] | | | |
| 3.3 | Regressor | residual [256,64] | | | |
| 4.1 | LR | 5e-5 | | | |
| 4.2 | LR | 2e-4 | | | |
| 4.3 | LR | cosine_restarts | | | |
| 5.1 | Combo | loss+regressor | | | |
| 5.2 | Combo | fusion+lr | | | |

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
