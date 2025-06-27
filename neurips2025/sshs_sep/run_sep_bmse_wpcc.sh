#!/bin/bash

# --- Configuration based on Python variables ---
SEEDS="456789 42 123 0 9999" # First seed from TRIAL_SEEDS
DATASET="sep" # Assumed from DS_PATH='data/sep_cme' and train.py choices
BATCH_SIZE=200
EPOCHS=5050
MLP_HIDDENS="512 32 256 32 128 32 64 32"
MLP_EMBED_DIM=32
MLP_DROPOUT=0.5
# MLP_SKIP_LAYERS=1 # Default in train.py
# MLP_SKIP_REPR=True # Default in train.py

LR=5e-4 # From START_LR
WEIGHT_DECAY=1 # From WEIGHT_DECAY

# BMSE / GAI specific settings
GMM_FILE="/home/paperspace/Desktop/BalancedMSE/neurips2025/checkpoint/sep_gmm_K8.pkl" # Placeholder - replace with actual path/name if needed
DATA_DIR="/home/paperspace/Desktop/BalancedMSE/neurips2025/data"
# Lower and upper thresholds for label range categorization
# No lower threshold for sep
UPPER_THRESHOLD=2.30258509299

# Create logs directory if it doesn't exist
mkdir -p logs

# Create a datetime code (format: YYYYMMDD_HHMMSS)
DATETIME_CODE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/${DATASET}_bmse_wpcc.${DATETIME_CODE}.log"

# Inform the user
echo "Starting training job for dataset: ${DATASET}, seeds: ${SEEDS}"
echo "Logging output to ${LOG_FILE}"

# Run Training with nohup in the background
nohup python3 train.py \
    --seeds ${SEEDS} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCHS} \
    --mlp_hiddens ${MLP_HIDDENS} \
    --mlp_embed_dim ${MLP_EMBED_DIM} \
    --mlp_dropout ${MLP_DROPOUT} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --bmse \
    --imp gai \
    --gmm_file ${GMM_FILE} \
    --pcc_lambda 0.03 \
    --upper_threshold ${UPPER_THRESHOLD} \
    --gpu 0 > "${LOG_FILE}" 2>&1 &

# Capture and display the PID
PID=$!
echo "Job started with PID ${PID}"
echo "Use 'tail -f ${LOG_FILE}' to monitor progress"