#!/bin/bash
# Hyperparameter survey for add_0c regression
# Target: MAE < 50
# Usage: ./run_survey.sh [phase] [run_id]
# Example: ./run_survey.sh 1 1   # Run phase 1, variant 1 (huber)
# Example: ./run_survey.sh all   # Run all phases sequentially

set -e

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
TABLE=/sdsc/scc/ddp478/jkirkland/all_outdoor/all_outdoor_samples_mito_filtered.biom
UNIFRAC=/sdsc/scc/ddp478/jkirkland/all_outdoor/all_outdoor_samples_mito_filtered_dm.h5
METADATA=/sdsc/scc/ddp478/jkirkland/all_outdoor/metadata/metadata_all_outdoor.tsv
TARGET=add_0c
CATEGORICALS="season,facility"
PRETRAINED=/cosmos/nfs/home/jkirkland/repos/aam/data/pretrain-all-outdoor-small-5000/pretrained_encoder.pt
OUTPUT_BASE=/cosmos/nfs/home/jkirkland/repos/aam/data/survey_add_0c

# Fixed hyperparameters (must match pretrained model)
EPOCHS=50
TOKEN_LIMIT=5000
SEED=42
NUM_GPUS=4

# Model architecture (must match pretrained encoder)
EMBEDDING_DIM=768
ATTENTION_HEADS=8
ATTENTION_LAYERS=4

# ============================================================================
# COMMON TRAINING FUNCTION
# ============================================================================
run_training() {
    local output_dir=$1
    local loss_type=${2:-huber}
    local fusion=${3:-cross-attention}
    local regressor_dims=${4:-"256,64"}
    local lr=${5:-1e-4}
    local scheduler=${6:-warmup_cosine}
    local extra_args=${7:-""}

    echo "============================================================"
    echo "Running: $output_dir"
    echo "  Loss: $loss_type, Fusion: $fusion, Regressor: $regressor_dims"
    echo "  LR: $lr, Scheduler: $scheduler"
    echo "============================================================"

    mkdir -p "$output_dir"

    torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
        --distributed \
        --table "$TABLE" \
        --unifrac-matrix "$UNIFRAC" \
        --metadata "$METADATA" \
        --metadata-column "$TARGET" \
        --categorical-columns "$CATEGORICALS" \
        --categorical-fusion "$fusion" \
        --pretrained-encoder "$PRETRAINED" \
        --freeze-base \
        --target-transform zscore \
        --loss-type "$loss_type" \
        --regressor-hidden-dims "$regressor_dims" \
        --lr "$lr" \
        --scheduler "$scheduler" \
        --epochs $EPOCHS \
        --token-limit $TOKEN_LIMIT \
        --seed $SEED \
        --embedding-dim $EMBEDDING_DIM \
        --attention-heads $ATTENTION_HEADS \
        --attention-layers $ATTENTION_LAYERS \
        --best-metric mae \
        --output-dir "$output_dir" \
        $extra_args \
        2>&1 | tee "$output_dir/training.log"

    echo "Completed: $output_dir"
}

# ============================================================================
# PHASE DEFINITIONS
# ============================================================================

run_phase1() {
    echo "========== PHASE 1: LOSS FUNCTION COMPARISON =========="
    # 1.1 Huber (baseline)
    [[ "$1" == "" || "$1" == "1" ]] && run_training "$OUTPUT_BASE/phase1_loss/huber" "huber"
    # 1.2 MSE
    [[ "$1" == "" || "$1" == "2" ]] && run_training "$OUTPUT_BASE/phase1_loss/mse" "mse"
    # 1.3 MAE
    [[ "$1" == "" || "$1" == "3" ]] && run_training "$OUTPUT_BASE/phase1_loss/mae" "mae"
}

run_phase2() {
    echo "========== PHASE 2: FUSION STRATEGY COMPARISON =========="
    # 2.1 Concat
    [[ "$1" == "" || "$1" == "1" ]] && run_training "$OUTPUT_BASE/phase2_fusion/concat" "huber" "concat"
    # 2.2 Cross-attention (covered in phase 1.1)
    # 2.3 GMU
    [[ "$1" == "" || "$1" == "2" ]] && run_training "$OUTPUT_BASE/phase2_fusion/gmu" "huber" "gmu"
}

run_phase3() {
    echo "========== PHASE 3: REGRESSOR ARCHITECTURE =========="
    # 3.1 Shallow [128]
    [[ "$1" == "" || "$1" == "1" ]] && run_training "$OUTPUT_BASE/phase3_regressor/shallow_128" "huber" "cross-attention" "128"
    # 3.2 Deep [512,256,64] with dropout
    [[ "$1" == "" || "$1" == "2" ]] && run_training "$OUTPUT_BASE/phase3_regressor/deep_512_256_64" "huber" "cross-attention" "512,256,64" "1e-4" "warmup_cosine" "--regressor-dropout 0.1"
    # 3.3 Residual head
    [[ "$1" == "" || "$1" == "3" ]] && run_training "$OUTPUT_BASE/phase3_regressor/residual_256_64" "huber" "cross-attention" "256,64" "1e-4" "warmup_cosine" "--residual-regression-head"
}

run_phase4() {
    echo "========== PHASE 4: LEARNING RATE & SCHEDULER =========="
    # 4.1 Lower LR
    [[ "$1" == "" || "$1" == "1" ]] && run_training "$OUTPUT_BASE/phase4_lr/lr_5e5" "huber" "cross-attention" "256,64" "5e-5"
    # 4.2 Higher LR
    [[ "$1" == "" || "$1" == "2" ]] && run_training "$OUTPUT_BASE/phase4_lr/lr_2e4" "huber" "cross-attention" "256,64" "2e-4"
    # 4.3 Cosine restarts
    [[ "$1" == "" || "$1" == "3" ]] && run_training "$OUTPUT_BASE/phase4_lr/cosine_restarts" "huber" "cross-attention" "256,64" "1e-4" "cosine_restarts"
}

# ============================================================================
# PHASE 5: COMBINATION RUNS (Based on Phase 1-4 Results)
# Best from each phase: mae loss, gmu fusion, residual regressor, lr 5e-5
# ============================================================================

run_phase5() {
    echo "========== PHASE 5: COMBINATION RUNS =========="

    # 5.1 Best combo from survey: mae + gmu + residual + lr 5e-5
    [[ "$1" == "" || "$1" == "1" ]] && run_training \
        "$OUTPUT_BASE/phase5_combo/best_combo" \
        "mae" "gmu" "256,64" "5e-5" "warmup_cosine" \
        "--residual-regression-head"

    # 5.2 Perceiver fusion (new, untested)
    [[ "$1" == "" || "$1" == "2" ]] && run_training \
        "$OUTPUT_BASE/phase5_combo/perceiver" \
        "mae" "perceiver" "256,64" "5e-5" "warmup_cosine" \
        "--residual-regression-head --perceiver-num-latents 64 --perceiver-num-layers 2"

    # 5.3 Category loss weighting
    [[ "$1" == "" || "$1" == "3" ]] && run_training \
        "$OUTPUT_BASE/phase5_combo/category_weights" \
        "mae" "gmu" "256,64" "5e-5" "warmup_cosine" \
        "--residual-regression-head --categorical-loss-weights auto --categorical-loss-weight-column season"

    # 5.4 Higher categorical LR
    [[ "$1" == "" || "$1" == "4" ]] && run_training \
        "$OUTPUT_BASE/phase5_combo/categorical_lr" \
        "mae" "gmu" "256,64" "5e-5" "warmup_cosine" \
        "--residual-regression-head --categorical-lr 5e-4"

    # 5.5 OneCycle scheduler
    [[ "$1" == "" || "$1" == "5" ]] && run_training \
        "$OUTPUT_BASE/phase5_combo/onecycle" \
        "mae" "gmu" "256,64" "5e-5" "onecycle" \
        "--residual-regression-head"
}

# ============================================================================
# PHASE 6: UNFREEZE BASE MODEL (Higher risk, higher reward)
# Use lower LR since we're fine-tuning the full model
# ============================================================================

# ============================================================================
# PHASE 7: SIMPLIFIED BASELINE (Match RF approach)
# Remove complexity: no categoricals, no normalization, abundance sampling
# ============================================================================

run_phase7() {
    echo "========== PHASE 7: SIMPLIFIED BASELINE =========="

    # 7.1 Minimal: no categoricals, minmax normalization, abundance sampling
    # NOTE: Using minmax instead of none - raw targets cause model to predict zeros
    if [[ "$1" == "" || "$1" == "1" ]]; then
        echo "Running simplified baseline (no categoricals, minmax norm)..."
        mkdir -p "$OUTPUT_BASE/phase7_simplified/minimal"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --pretrained-encoder "$PRETRAINED" \
            --freeze-base \
            --target-transform minmax \
            --asv-sampling abundance \
            --loss-type mae \
            --regressor-hidden-dims "64" \
            --lr 5e-5 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase7_simplified/minimal" \
            2>&1 | tee "$OUTPUT_BASE/phase7_simplified/minimal/training.log"
    fi

    # 7.2 Minimal + unfreeze base (highest potential)
    if [[ "$1" == "" || "$1" == "2" ]]; then
        echo "Running simplified baseline with unfrozen base..."
        mkdir -p "$OUTPUT_BASE/phase7_simplified/minimal_unfreeze"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --pretrained-encoder "$PRETRAINED" \
            --target-transform minmax \
            --asv-sampling abundance \
            --loss-type mae \
            --regressor-hidden-dims "64" \
            --lr 1e-5 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase7_simplified/minimal_unfreeze" \
            2>&1 | tee "$OUTPUT_BASE/phase7_simplified/minimal_unfreeze/training.log"
    fi

    # 7.3 With categoricals, minmax normalization (isolate categorical effect)
    if [[ "$1" == "" || "$1" == "3" ]]; then
        echo "Running with categoricals, minmax normalization..."
        mkdir -p "$OUTPUT_BASE/phase7_simplified/minmax_with_cat"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --categorical-columns "$CATEGORICALS" \
            --categorical-fusion gmu \
            --pretrained-encoder "$PRETRAINED" \
            --freeze-base \
            --target-transform minmax \
            --asv-sampling abundance \
            --loss-type mae \
            --regressor-hidden-dims "256,64" \
            --residual-regression-head \
            --lr 5e-5 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase7_simplified/minmax_with_cat" \
            2>&1 | tee "$OUTPUT_BASE/phase7_simplified/minmax_with_cat/training.log"
    fi

    # 7.4 High count penalty (for weighted UniFrac pretrained models)
    if [[ "$1" == "" || "$1" == "4" ]]; then
        echo "Running with high count penalty..."
        mkdir -p "$OUTPUT_BASE/phase7_simplified/high_count_penalty"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --categorical-columns "$CATEGORICALS" \
            --categorical-fusion gmu \
            --pretrained-encoder "$PRETRAINED" \
            --freeze-base \
            --target-transform minmax \
            --asv-sampling abundance \
            --loss-type mae \
            --regressor-hidden-dims "256,64" \
            --residual-regression-head \
            --count-penalty 5.0 \
            --lr 5e-5 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase7_simplified/high_count_penalty" \
            2>&1 | tee "$OUTPUT_BASE/phase7_simplified/high_count_penalty/training.log"
    fi

    # 7.5 Learnable output scale (alternative to normalization)
    if [[ "$1" == "" || "$1" == "5" ]]; then
        echo "Running with learnable output scale (no normalization)..."
        mkdir -p "$OUTPUT_BASE/phase7_simplified/learnable_scale"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --categorical-columns "$CATEGORICALS" \
            --categorical-fusion gmu \
            --pretrained-encoder "$PRETRAINED" \
            --freeze-base \
            --target-transform none \
            --learnable-output-scale \
            --asv-sampling abundance \
            --loss-type mae \
            --regressor-hidden-dims "256,64" \
            --residual-regression-head \
            --lr 5e-5 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase7_simplified/learnable_scale" \
            2>&1 | tee "$OUTPUT_BASE/phase7_simplified/learnable_scale/training.log"
    fi
}

run_phase6() {
    echo "========== PHASE 6: UNFREEZE BASE MODEL =========="

    # 6.1 Unfreeze with very low LR
    if [[ "$1" == "" || "$1" == "1" ]]; then
        echo "Running unfrozen base model training..."
        mkdir -p "$OUTPUT_BASE/phase6_unfreeze/lr_1e5"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --categorical-columns "$CATEGORICALS" \
            --categorical-fusion gmu \
            --pretrained-encoder "$PRETRAINED" \
            --target-transform zscore \
            --loss-type mae \
            --regressor-hidden-dims "256,64" \
            --residual-regression-head \
            --lr 1e-5 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase6_unfreeze/lr_1e5" \
            2>&1 | tee "$OUTPUT_BASE/phase6_unfreeze/lr_1e5/training.log"
    fi

    # 6.2 Unfreeze with even lower LR
    if [[ "$1" == "" || "$1" == "2" ]]; then
        echo "Running unfrozen base model training with lr 5e-6..."
        mkdir -p "$OUTPUT_BASE/phase6_unfreeze/lr_5e6"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --categorical-columns "$CATEGORICALS" \
            --categorical-fusion gmu \
            --pretrained-encoder "$PRETRAINED" \
            --target-transform zscore \
            --loss-type mae \
            --regressor-hidden-dims "256,64" \
            --residual-regression-head \
            --lr 5e-6 \
            --scheduler warmup_cosine \
            --epochs $EPOCHS \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase6_unfreeze/lr_5e6" \
            2>&1 | tee "$OUTPUT_BASE/phase6_unfreeze/lr_5e6/training.log"
    fi

    # 6.3 Unfreeze + more epochs (100)
    if [[ "$1" == "" || "$1" == "3" ]]; then
        echo "Running unfrozen base model with 100 epochs..."
        mkdir -p "$OUTPUT_BASE/phase6_unfreeze/lr_1e5_100ep"

        torchrun --nproc_per_node=$NUM_GPUS -m aam.cli train \
            --distributed \
            --table "$TABLE" \
            --unifrac-matrix "$UNIFRAC" \
            --metadata "$METADATA" \
            --metadata-column "$TARGET" \
            --categorical-columns "$CATEGORICALS" \
            --categorical-fusion gmu \
            --pretrained-encoder "$PRETRAINED" \
            --target-transform zscore \
            --loss-type mae \
            --regressor-hidden-dims "256,64" \
            --residual-regression-head \
            --lr 1e-5 \
            --scheduler warmup_cosine \
            --epochs 100 \
            --token-limit $TOKEN_LIMIT \
            --seed $SEED \
            --embedding-dim $EMBEDDING_DIM \
            --attention-heads $ATTENTION_HEADS \
            --attention-layers $ATTENTION_LAYERS \
            --best-metric mae \
            --output-dir "$OUTPUT_BASE/phase6_unfreeze/lr_1e5_100ep" \
            2>&1 | tee "$OUTPUT_BASE/phase6_unfreeze/lr_1e5_100ep/training.log"
    fi
}

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

PHASE=$1
RUN_ID=$2

case $PHASE in
    1) run_phase1 "$RUN_ID" ;;
    2) run_phase2 "$RUN_ID" ;;
    3) run_phase3 "$RUN_ID" ;;
    4) run_phase4 "$RUN_ID" ;;
    5) run_phase5 "$RUN_ID" ;;
    6) run_phase6 "$RUN_ID" ;;
    7) run_phase7 "$RUN_ID" ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        ;;
    all-new)
        run_phase5
        run_phase6
        run_phase7
        ;;
    status)
        echo "Survey Status:"
        for phase_dir in "$OUTPUT_BASE"/phase*/; do
            echo "$(basename $phase_dir):"
            for run_dir in "$phase_dir"*/; do
                if [ -f "$run_dir/checkpoints/best_model.pt" ]; then
                    best_mae=$(grep "mae=" "$run_dir/training.log" 2>/dev/null | tail -1 | grep -oP "mae=\K[0-9.]+")
                    echo "  $(basename $run_dir): COMPLETE (MAE: ${best_mae:-unknown})"
                elif [ -f "$run_dir/training.log" ]; then
                    echo "  $(basename $run_dir): IN PROGRESS"
                else
                    echo "  $(basename $run_dir): NOT STARTED"
                fi
            done
        done
        ;;
    *)
        echo "Usage: $0 [phase] [run_id]"
        echo ""
        echo "Phases 1-4 (Initial Survey - Complete):"
        echo "  1 [1|2|3]      - Loss function (huber, mse, mae)"
        echo "  2 [1|2]        - Fusion (concat, gmu)"
        echo "  3 [1|2|3]      - Regressor (shallow, deep, residual)"
        echo "  4 [1|2|3]      - Learning rate (5e-5, 2e-4, cosine_restarts)"
        echo ""
        echo "Phases 5-7 (Optimization - Based on Survey Results):"
        echo "  5 [1|2|3|4|5]  - Combinations (best_combo, perceiver, category_weights, categorical_lr, onecycle)"
        echo "  6 [1|2|3]      - Unfreeze base (lr_1e5, lr_5e6, lr_1e5_100ep)"
        echo "  7 [1|2|3|4|5]  - Simplified baseline (minimal, minimal_unfreeze, minmax_with_cat, high_count_penalty, learnable_scale)"
        echo ""
        echo "Batch Commands:"
        echo "  all            - Run phases 1-4 (initial survey)"
        echo "  all-new        - Run phases 5-6 (optimization runs)"
        echo "  status         - Check completion status"
        echo ""
        echo "Examples:"
        echo "  $0 5 1         # Run phase 5, variant 1 (best combo)"
        echo "  $0 6           # Run all phase 6 variants (unfreeze)"
        echo "  $0 all-new     # Run all new optimization runs"
        ;;
esac
