#!/bin/bash

# --- Configuration based on Python variables ---
SEED=456789 # First seed from TRIAL_SEEDS
DATASET="sep" # Assumed from DS_PATH='data/sep_cme' and train.py choices
DATA_DIR="data" # Default in train.py, assumes data is in data/sep/
BATCH_SIZE=200
EPOCHS=200000
MLP_HIDDENS="512 32 256 32 128 32 64 32"
MLP_EMBED_DIM=32
MLP_DROPOUT=0.5
MLP_SKIP_LAYERS=1
# MLP_SKIP_REPR=True is the default in train.py

LR=5e-4 # From START_LR
WEIGHT_DECAY=1 # From WEIGHT_DECAY
OPTIMIZER="adam" # Default in train.py
LOSS="l1" # Default in train.py

# BMSE / GAI specific settings
INIT_NOISE_SIGMA=0.88 # Mapped from BANDWIDTH
GMM_FILE="gmm.pkl" # Placeholder - replace with actual path/name if needed
SIGMA_LR=0.01 # Default in train.py

# Use balanced metrics for evaluation/model selection
BALANCED_METRIC="--balanced_metric"

# --- Run Training ---
echo "Starting training for dataset: ${DATASET}, seed: ${SEED}"

python train.py \
    --seed ${SEED} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCHS} \
    --mlp_hiddens ${MLP_HIDDENS} \
    --mlp_embed_dim ${MLP_EMBED_DIM} \
    --mlp_dropout ${MLP_DROPOUT} \
    --mlp_skip_layers ${MLP_SKIP_LAYERS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --optimizer ${OPTIMIZER} \
    --loss ${LOSS} \
    --bmse \
    --imp gai \
    --init_noise_sigma ${INIT_NOISE_SIGMA} \
    --gmm_file ${GMM_FILE} \
    --sigma_lr ${SIGMA_LR} \
    ${BALANCED_METRIC} \
    # --gpu 0 # Uncomment and set GPU ID if needed
    # --schedule 60 80 # Uncomment to use default LR schedule
    # Add other arguments from train.py if needed

echo "Training finished for seed: ${SEED}"