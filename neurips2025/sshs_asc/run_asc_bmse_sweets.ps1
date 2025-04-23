# PowerShell script for ASC training with LDS, FDS, and sqrt_inv reweighting

# --- Configuration based on Python variables ---
$SEEDS = "456789 42 123 0 9999" # First seed from TRIAL_SEEDS
$DATASET = "asc" # Dataset name
$BATCH_SIZE = 32768
$EPOCHS = 5031
$MLP_EMBED_DIM = 1024
$MLP_DROPOUT = 0.2
# MLP_SKIP_LAYERS=1 # Default in train.py
# MLP_SKIP_REPR=True # Default in train.py

$LR = "5e-4" # From START_LR
$WEIGHT_DECAY = 0.1 # From WEIGHT_DECAY

# BMSE / GAI specific settings
$GMM_FILE = "C:\Users\kayla\Desktop\BalancedMSE\neurips2025\checkpoint\asc_gmm_K8.pkl" # Placeholder - replace with actual path/name if needed
$DATA_DIR = "C:\Users\kayla\Desktop\BalancedMSE\neurips2025\data"

# Lower and upper thresholds for label range categorization
$LOWER_THRESHOLD = 2.30102999566
$UPPER_THRESHOLD = 4.30102999566

# --- Run Training ---
Write-Host "Starting training for dataset: $DATASET, seeds: $SEEDS"

python train.py `
    --seeds  456789 42 123 0 9999 `
    --data_dir $DATA_DIR `
    --dataset $DATASET `
    --batch_size $BATCH_SIZE `
    --epoch $EPOCHS `
    --mlp_hiddens 4096 1024 2048 1024 `
    --mlp_embed_dim $MLP_EMBED_DIM `
    --mlp_dropout $MLP_DROPOUT `
    --lr $LR `
    --weight_decay $WEIGHT_DECAY `
    --bmse `
    --imp gai `
    --gmm_file $GMM_FILE `
    --gpu 0 `
    # --schedule 60 80 # Uncomment to use default LR schedule
    # Add other arguments from train.py if needed

Write-Host "Training finished for seeds: $SEEDS"