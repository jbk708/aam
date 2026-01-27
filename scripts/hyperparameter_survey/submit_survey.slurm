#!/bin/bash
#SBATCH --job-name=aam_survey
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-11
#SBATCH --output=logs/survey_%A_%a.out
#SBATCH --error=logs/survey_%A_%a.err

# ============================================================================
# SLURM Array Job for Hyperparameter Survey
# Submit with: sbatch submit_survey.slurm
# ============================================================================

# Create logs directory
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate aam

# Map array task ID to phase and run
# Array task 1-3:  Phase 1 (loss)
# Array task 4-5:  Phase 2 (fusion)
# Array task 6-8:  Phase 3 (regressor)
# Array task 9-11: Phase 4 (learning rate)

case $SLURM_ARRAY_TASK_ID in
    1) PHASE=1; RUN=1 ;; # huber
    2) PHASE=1; RUN=2 ;; # mse
    3) PHASE=1; RUN=3 ;; # mae
    4) PHASE=2; RUN=1 ;; # concat
    5) PHASE=2; RUN=2 ;; # gmu
    6) PHASE=3; RUN=1 ;; # shallow
    7) PHASE=3; RUN=2 ;; # deep
    8) PHASE=3; RUN=3 ;; # residual
    9) PHASE=4; RUN=1 ;; # lr 5e-5
    10) PHASE=4; RUN=2 ;; # lr 2e-4
    11) PHASE=4; RUN=3 ;; # cosine_restarts
    *)
        echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Running Phase $PHASE, Run $RUN (Array Task $SLURM_ARRAY_TASK_ID)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

# Run the training
cd /cosmos/nfs/home/jkirkland/repos/aam
./data/hyperparameter_survey/run_survey.sh $PHASE $RUN

echo "Completed Phase $PHASE, Run $RUN"
