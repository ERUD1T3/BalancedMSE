#!/bin/bash

# --- Configuration based on Python variables ---
SEEDS="456789" # First seed from TRIAL_SEEDS
DATASET="sep" # Assumed from DS_PATH='data/sep_cme' and train.py choices
BATCH_SIZE=200
EPOCHS=100
MLP_HIDDENS="512 32 256 32 128 32 64 32"
MLP_EMBED_DIM=32
MLP_DROPOUT=0.5
# MLP_SKIP_LAYERS=1 # Default in train.py
# MLP_SKIP_REPR=True # Default in train.py

LR=5e-4 # From START_LR
WEIGHT_DECAY=1 # From WEIGHT_DECAY

# BMSE / GAI specific settings
GMM_FILE="/mnt/c/Users/the_3/Documents/github/BalancedMSE/neurips2025/checkpoint/sep_gmm_K8.pkl" # Placeholder - replace with actual path/name if needed
DATA_DIR="/mnt/c/Users/the_3/Documents/github/BalancedMSE/neurips2025/data"
# Lower and upper thresholds for label range categorization
# No lower threshold for sep
UPPER_THRESHOLD=2.30258509299

# --- Run Training ---
echo "Starting training for dataset: ${DATASET}, seeds: ${SEEDS}"

python3 train.py \
    --seeds ${SEEDS} \
    --data-dir ${DATA_DIR} \
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
    --gpu 0 # Uncomment and set GPU ID if needed
    # --schedule 60 80 # Uncomment to use default LR schedule
    # Add other arguments from train.py if needed

echo "Training finished for seeds: ${SEEDS}"