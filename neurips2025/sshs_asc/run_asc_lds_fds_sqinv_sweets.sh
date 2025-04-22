#!/bin/bash

# --- Configuration based on Python variables ---
SEEDS="456789 42 123 0 9999" # First seed from TRIAL_SEEDS
DATASET="asc" # Dataset name
BATCH_SIZE=16384
EPOCHS=6009
MLP_HIDDENS="4096 1024 2048 1024"
MLP_EMBED_DIM=1024
MLP_DROPOUT=0.2
# MLP_SKIP_LAYERS=1 # Default in train.py
# MLP_SKIP_REPR=True # Default in train.py

LR=5e-4 # From START_LR
WEIGHT_DECAY=0.1 # From WEIGHT_DECAY

# BMSE / GAI specific settings
GMM_FILE="C:\Users\kayla\Desktop\BalancedMSE\neurips2025\checkpoint\asc_gmm_K8.pkl" # Placeholder - replace with actual path/name if needed
DATA_DIR="C:\Users\kayla\Desktop\BalancedMSE\neurips2025\data"

# Lower and upper thresholds for label range categorization
LOWER_THRESHOLD=2.30102999566
UPPER_THRESHOLD=4.30102999566

# --- Run Training ---
echo "Starting training for dataset: ${DATASET}, seeds: ${SEEDS}"

python train.py \
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
    --loss mse \
    --lds \
    --fds \
    --reweight sqrt_inv \
    --gpu 0 # Uncomment and set GPU ID if needed
    # --schedule 60 80 # Uncomment to use default LR schedule
    # Add other arguments from train.py if needed

echo "Training finished for seeds: ${SEEDS}"