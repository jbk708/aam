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
PRETRAINED=/cosmos/nfs/home/jkirkland/repos/aam/data/all-outdoor-small-model-zscore-5000token/checkpoints/best_model.pt
OUTPUT_BASE=/cosmos/nfs/home/jkirkland/repos/aam/data/survey_add_0c

# Fixed hyperparameters
EPOCHS=50
TOKEN_LIMIT=5000
SEED=42
NUM_GPUS=4

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
# MAIN ENTRY POINT
# ============================================================================

PHASE=$1
RUN_ID=$2

case $PHASE in
    1) run_phase1 "$RUN_ID" ;;
    2) run_phase2 "$RUN_ID" ;;
    3) run_phase3 "$RUN_ID" ;;
    4) run_phase4 "$RUN_ID" ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        ;;
    status)
        echo "Survey Status:"
        for phase_dir in "$OUTPUT_BASE"/phase*/; do
            echo "$(basename $phase_dir):"
            for run_dir in "$phase_dir"*/; do
                if [ -f "$run_dir/checkpoints/best_model.pt" ]; then
                    best_mae=$(grep "val_mae" "$run_dir/training.log" 2>/dev/null | tail -1 | grep -oP "val_mae[=:]\s*\K[0-9.]+")
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
        echo "Phases:"
        echo "  1 [1|2|3]  - Loss function (huber, mse, mae)"
        echo "  2 [1|2]    - Fusion (concat, gmu)"
        echo "  3 [1|2|3]  - Regressor (shallow, deep, residual)"
        echo "  4 [1|2|3]  - Learning rate (5e-5, 2e-4, cosine_restarts)"
        echo "  all        - Run all phases sequentially"
        echo "  status     - Check completion status"
        echo ""
        echo "Examples:"
        echo "  $0 1 1      # Run phase 1, variant 1 (huber)"
        echo "  $0 3        # Run all phase 3 variants"
        echo "  $0 all      # Run complete survey"
        ;;
esac
