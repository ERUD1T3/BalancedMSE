import time
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import gmean
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger

from loss import *
from neurips2025.tab_ds import TabDS, load_tabular_splits, set_seed
from mlp import create_mlp
from utils import *
from balanaced_mse import *

import os

# Disable KMP warnings
os.environ["KMP_WARNINGS"] = "FALSE"

# Set up command line argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ----- Imbalanced learning related arguments -----
# LDS (Label Distribution Smoothing)
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS (Feature Distribution Smoothing)
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket number for FDS and BNI Loss (TabDS uses DEFAULT_NUM_BINS=100)')
parser.add_argument('--bucket_start', type=int, default=0, help='minimum(starting) bucket for FDS')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# Label value thresholds for data frequency categorization
parser.add_argument('--lower_threshold', type=float, default=None, help='lower threshold value for label range (rare/low-shot data)')
parser.add_argument('--upper_threshold', type=float, default=None, help='upper threshold value for label range (frequent/many-shot data)')

# BMSE (Balanced MSE)
parser.add_argument('--bmse', action='store_true', default=False, help='use Balanced MSE')
parser.add_argument('--imp', type=str, default='gai', choices=['gai', 'bmc', 'bni'], help='implementation options')
parser.add_argument('--gmm_file', type=str, default=None, help='Path to preprocessed GMM file (e.g., sep_gmm_K8.pkl). If None, constructed from dataset/K.')
parser.add_argument('--init_noise_sigma', type=float, default=1., help='initial scale of the noise')
parser.add_argument('--sigma_lr', type=float, default=1e-2, help='learning rate of the noise scale')
parser.add_argument('--balanced_metric', action='store_true', default=False, help='use balanced metric')
parser.add_argument('--fix_noise_sigma', action='store_true', default=False, help='disable joint optimization')

# Re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'],
                    help='cost-sensitive reweighting scheme')
# Two-stage training: RRT (Regressor Re-Training)
parser.add_argument('--retrain_regressor', action='store_true', default=False,
                    help='whether to retrain last regression layer (regressor) of MLP')

# ----- Training/optimization related arguments -----
parser.add_argument('--dataset', type=str, required=True,
                    choices=['sep', 'sarcos', 'onp', 'bf', 'asc', 'ed'],
                    help='Name of the tabular dataset to use.')
parser.add_argument('--data_dir', type=str, default='C:/Users/the_3/Documents/github/BalancedMSE/neurips2025/data', help='Root directory containing dataset subfolders.')
parser.add_argument('--train_split_name', type=str, default='training', help='Name for the training data file/folder.')
parser.add_argument('--val_split_name', type=str, default='validation', help='Name for the validation data file/folder.')
parser.add_argument('--test_split_name', type=str, default='testing', help='Name for the test data file/folder.')

# Added MLP specific args
parser.add_argument('--model', type=str, default='mlp', choices=['mlp'], help='model name')
parser.add_argument('--mlp_hiddens', type=int, nargs='+', default=[100, 100, 100], help='MLP hidden layer sizes')
parser.add_argument('--mlp_embed_dim', type=int, default=128, help='MLP embedding dimension (output of backbone)')
parser.add_argument('--mlp_skip_layers', type=int, default=1, help='MLP skip connection frequency')
parser.add_argument('--mlp_skip_repr', action='store_true', default=True, help='MLP merge skip into final representation')
parser.add_argument('--mlp_dropout', type=float, default=0.1, help='MLP dropout rate')

parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'],
                    help='training loss type')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', type=int, default=50, help='logging frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers used in data loading')
# Checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

args, unknown = parser.parse_known_args()

# Set seed for reproducibility
set_seed(args.seed)

# Initialize training state variables
args.start_epoch, args.best_loss = 0, 1e5

# Build experiment name based on configuration
if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_regressor:
    args.store_name += f'_retrainReg'
if args.bmse:
    args.store_name += f'_{args.imp}_{args.init_noise_sigma}_{args.sigma_lr}'
    if args.imp == 'gai':
        gmm_suffix = args.gmm_file.split('/')[-1].replace('.pkl','')
        args.store_name += f'_{gmm_suffix}'
    if args.fix_noise_sigma:
        args.store_name += '_fixNoise'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_epoch{args.epoch}"
if args.balanced_metric:
    args.store_name += '_balMetric'

# Create folders for storing results
prepare_folders(args)

# Set up logging
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

# Initialize TensorBoard logger
tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)




def main() -> None:
    """
    Main function to run the training/evaluation pipeline.
    
    This function handles:
    1. Setting up the GPU
    2. Loading and preparing datasets
    3. Building the model
    4. Setting up loss functions and optimizers
    5. Training/evaluating the model
    """
    if args.gpu is not None:
        if torch.cuda.is_available():
            print(f"Use GPU: {args.gpu} for training")
            # torch.cuda.set_device(args.gpu) # TODO: revisit to see if AI got it wrong
        else:
            print("CUDA not available, using CPU.")
            args.gpu = None

    # Data preparation
    print('=====> Preparing data...')
    start_time = time.time()
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_splits(
            args.dataset, args.data_dir, args.train_split_name,
            args.val_split_name, args.test_split_name, args.seed
        )
        print(f"Data loaded. Train: {X_train.shape}/{y_train.shape}, Val: {X_val.shape}/{y_val.shape}, Test: {X_test.shape}/{y_test.shape}")
        print(f'Data loading time: {time.time() - start_time:.2f} seconds')
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return

    # Get input dimension from training data
    input_dim = X_train.shape[1]
    print(f"Input feature dimension: {input_dim}")

    # Create TabDS datasets
    print("Creating TabDS datasets...")
    train_dataset = TabDS(
        X=X_train, y=y_train,
        reweight=args.reweight,
        lds=args.lds, 
        lds_kernel=args.lds_kernel, 
        lds_ks=args.lds_ks, 
        lds_sigma=args.lds_sigma,
        bins=args.bucket_num
    )
    val_dataset = TabDS(X=X_val, y=y_val, reweight='none', lds=False)
    test_dataset = TabDS(X=X_test, y=y_test, reweight='none', lds=False)

    # Pass y_train to shot metrics later (needs to be numpy or list)
    train_labels_raw = y_train.tolist()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True if args.gpu is not None else False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True if args.gpu is not None else False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True if args.gpu is not None else False, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model initialization
    print('=====> Building model...')
    model = create_mlp(
        input_dim=input_dim,
        output_dim=1,
        hiddens=args.mlp_hiddens,
        skipped_layers=args.mlp_skip_layers,
        embed_dim=args.mlp_embed_dim,
        skip_repr=args.mlp_skip_repr,
        dropout=args.mlp_dropout,
        fds=args.fds,
        bucket_num=args.bucket_num,
        bucket_start=args.bucket_start,
        start_update=args.start_update,
        start_smooth=args.start_smooth,
        kernel=args.fds_kernel,
        ks=args.fds_ks,
        sigma=args.fds_sigma,
        momentum=args.fds_mmt
    )

    if args.gpu is not None:
        # model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    else:
        model = model.cpu()

    # Evaluation mode - load model and evaluate
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, 'ckpt.best.pth.tar')
        if not os.path.isfile(ckpt_path):
            print(f"Error: Checkpoint file not found at {ckpt_path}")
            return

        print(f"Loading checkpoint for evaluation: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cuda' if args.gpu is not None else 'cpu')

        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict, strict=True)
        print(f"===> Checkpoint '{ckpt_path}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, prefix='Test')
        return

    # For retraining only the final layer (transfer learning)
    if args.retrain_regressor:
        assert args.reweight != 'none' and args.pretrained or args.bmse
        print('===> Retrain last regression layer (output_layer) only!')
        for name, param in model.named_parameters():
            if 'output_layer' not in name:
                param.requires_grad = False
            else:
                print(f"Keeping param trainable: {name}")

    # Set up optimizer
    if not args.retrain_regressor:
        # Optimize all parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # Optimize only the last linear layer parameters
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None,
                            [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Load pretrained weights if specified
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # Only load non-classifier weights
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    # Resume from checkpoint if specified
    if args.resume and not args.retrain_regressor:
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, 'ckpt.latest.pth.tar')

        if os.path.isfile(ckpt_path):
            print(f"===> Loading checkpoint '{ckpt_path}'")
            map_location = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            try:
                args.best_loss = checkpoint['best_loss']
            except KeyError:
                print("Warning: 'best_loss' not found in checkpoint. Initializing to 1e5.")
                args.best_loss = 1e5

            state_dict = checkpoint['state_dict']

            model.load_state_dict(state_dict, strict=False)

            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Optimizer will start from scratch.")

            print(f"===> Loaded checkpoint '{ckpt_path}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{ckpt_path}', starting from scratch.")

    # Enable CUDA optimization
    cudnn.benchmark = True

    # Set up loss function
    if args.bmse:
        if args.imp == 'gai':
            if args.gmm_file is None:
                gmm_filename = f"{args.dataset}_gmm_K{args.gmm_K}.pkl"
                gmm_path = os.path.join(args.store_root, gmm_filename)
                if not os.path.exists(gmm_path):
                    gmm_path_alt = os.path.join(args.data_dir, args.dataset, gmm_filename)
                    if os.path.exists(gmm_path_alt):
                        gmm_path = gmm_path_alt
                    else:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        gmm_path_alt2 = os.path.join(script_dir, gmm_filename)
                        if os.path.exists(gmm_path_alt2):
                            gmm_path = gmm_path_alt2
                        else:
                            raise FileNotFoundError(f"GMM file not found at expected paths: {gmm_path}, {gmm_path_alt}, {gmm_path_alt2}. Please specify --gmm_file or place it correctly.")
            else:
                gmm_path = args.gmm_file

            if not os.path.exists(gmm_path):
                raise FileNotFoundError(f"Specified GMM file not found: {gmm_path}")

            print(f"Loading GMM parameters from: {gmm_path}")
            criterion = GAILoss(args.init_noise_sigma, gmm_path)
        elif args.imp == 'bmc':
            criterion = BMCLoss(args.init_noise_sigma)
        elif args.imp == 'bni':
            print("Fetching bucket info for BNI loss...")
            bucket_centers, bucket_weights = train_dataset.get_bucket_info(
                bins=args.bucket_num,
                lds=args.lds, lds_kernel=args.lds_kernel,
                lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
            print(f"Obtained {len(bucket_centers)} buckets for BNI.")
            criterion = BNILoss(args.init_noise_sigma, bucket_centers, bucket_weights)
        else:
            raise NotImplementedError(f"BMSE implementation '{args.imp}' not supported.")

        # Add noise sigma parameter to optimizer if not fixed
        if not args.fix_noise_sigma:
            if hasattr(criterion, 'noise_sigma'):
                optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': args.sigma_lr, 'name': 'noise_sigma'})
                print(f"Added noise_sigma to optimizer with lr: {args.sigma_lr}")
            else:
                print(f"Warning: BMSE criterion {args.imp} does not have 'noise_sigma' attribute. Cannot optimize it.")
    else:
        # Use standard weighted loss functions
        criterion = globals()[f"weighted_{args.loss}_loss"]

    # Move criterion to GPU if applicable
    if args.gpu is not None and hasattr(criterion, 'cuda'):
        criterion = criterion.cuda()

    # Training loop
    for epoch in range(args.start_epoch, args.epoch):
        # Adjust learning rate according to schedule
        adjust_learning_rate(optimizer, epoch, args)
        
        # Train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, criterion)
        
        # Evaluate on validation set
        (val_loss_mse, val_loss_l1, val_loss_gmean) = validate(
            val_loader, model, prefix='Val')

        # Determine which metric to use for model selection
        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
            
        # Check if current model is the best so far
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best Validation {'MSE' if 'mse' in args.loss else 'L1'} Loss: {args.best_loss:.4f}")
        
        # Save checkpoint
        state_dict_to_save = model.state_dict()
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': state_dict_to_save,
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }, is_best, epoch + 1 == args.epoch)
        
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        # Log metrics to TensorBoard
        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('val/loss_mse', val_loss_mse, epoch)
        tb_logger.log_value('val/loss_l1', val_loss_l1, epoch)
        tb_logger.log_value('val/loss_gmean', val_loss_gmean, epoch)
        
        for i, param_group in enumerate(optimizer.param_groups):
            tb_logger.log_value(f'lr/group_{i}', param_group['lr'], epoch)
        if args.bmse and not args.fix_noise_sigma and hasattr(criterion, 'noise_sigma'):
            tb_logger.log_value('noise_sigma', criterion.noise_sigma.item(), epoch)

    # Test with best checkpoint after training
    print("=" * 120)
    print("Testing best model on testset...")
    best_ckpt_path = os.path.join(args.store_root, args.store_name, 'ckpt.best.pth.tar')
    if not os.path.exists(best_ckpt_path):
        print(f"Error: Best checkpoint not found at {best_ckpt_path}")
        return

    checkpoint = torch.load(best_ckpt_path, map_location='cuda' if args.gpu is not None else 'cpu')
    print(f"Loaded best model from epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")

    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=True)

    # Evaluate on test set
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(
        test_loader, model, prefix='Test')
    
    print(f"Test Results: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")


def train(train_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, criterion: nn.Module) -> float:
    """
    Train the model for one epoch.
    
    Args:
        train_loader: DataLoader for training data
        model: The neural network model
        optimizer: Optimizer for updating model weights
        epoch: Current epoch number
        criterion: Loss function
        
    Returns:
        float: Average training loss for the epoch
    """
    # Initialize meters for tracking time and performance
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    
    loss_name = f'Loss ({args.imp.upper()})' if args.bmse else f'Loss ({args.loss.upper()})'
    losses = AverageMeter(loss_name, ':.4f')
    
    meters_to_display = [batch_time, data_time, losses]

    # Add additional metrics for balanced MSE if enabled
    if args.bmse:
        noise_var = AverageMeter('Noise Var', ':.5f')
        l2 = AverageMeter('L2', ':.5f')
        meters_to_display.extend([noise_var, l2])

    progress = ProgressMeter(len(train_loader), meters_to_display, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    
    # Iterate through batches
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            inputs, targets, weights = \
                inputs.cuda(args.gpu, non_blocking=True), \
                targets.cuda(args.gpu, non_blocking=True), \
                weights.cuda(args.gpu, non_blocking=True)
        else:
            inputs, targets, weights = inputs.cpu(), targets.cpu(), weights.cpu()

        if args.fds:
            predictions, _ = model(inputs, targets, epoch)
        else:
            predictions, _ = model(inputs, targets, epoch)

        predictions = predictions.squeeze(-1) if predictions.ndim > 1 and predictions.shape[-1] == 1 else predictions
        targets = targets.squeeze(-1) if targets.ndim > 1 and targets.shape[-1] == 1 else targets
        weights = weights.squeeze(-1) if weights.ndim > 1 and weights.shape[-1] == 1 else weights

        if args.bmse:
            loss = criterion(predictions, targets)
        else:
            loss = criterion(predictions, targets, weights)

        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        # Update running loss average
        losses.update(loss.item(), inputs.size(0))

        # Track additional metrics for balanced MSE
        if args.bmse:
            if hasattr(criterion, 'noise_sigma') and criterion.noise_sigma is not None:
                noise_var.update(criterion.noise_sigma.item() ** 2)
            l2.update(F.mse_loss(predictions, targets).item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress at specified intervals
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update and hasattr(model, 'fds_module') and model.fds_module is not None:
        print(f"Updating FDS statistics for Epoch [{epoch}]...")
        encodings, labels_list = [], []
        model.eval()
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader, desc="FDS Feature Extraction"):
                if args.gpu is not None:
                    inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
                else:
                    inputs, targets = inputs.cpu(), targets.cpu()

                _, feature = model(inputs, targets, epoch)

                encodings.append(feature.detach().cpu().numpy())
                labels_list.append(targets.detach().cpu().numpy())

        encodings = np.vstack(encodings)
        labels_array = np.concatenate([lbl.squeeze() if lbl.ndim > 1 and lbl.shape[-1] == 1 else lbl for lbl in labels_list])

        encodings_tensor = torch.from_numpy(encodings)
        labels_tensor = torch.from_numpy(labels_array)
        if args.gpu is not None:
            encodings_tensor = encodings_tensor.cuda(args.gpu)
            labels_tensor = labels_tensor.cuda(args.gpu)

        print(f"Calling FDS update with {encodings_tensor.shape} features and {labels_tensor.shape} labels.")
        fds_module = model.fds_module
        fds_module.update_last_epoch_stats(epoch)
        fds_module.update_running_stats(encodings_tensor, labels_tensor, epoch)
        print("FDS statistics updated.")
        model.train()

    return losses.avg


def validate(
        val_loader: DataLoader, 
        model: nn.Module,  
        prefix: str = 'Val'
    ) -> Tuple[float, float, float]:
    """
    Evaluate the model on validation or test data.
    
    Args:
        val_loader: DataLoader for validation/test data
        model: The neural network model
        prefix: Prefix for progress display ('Val' or 'Test')
        
    Returns:
        Tuple containing:
            - MSE loss average
            - L1 loss average
            - Geometric mean of losses
    """
    # Initialize meters for tracking time and performance
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.4f')
    losses_l1 = AverageMeter('Loss (L1)', ':.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    # Define loss functions
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')  # For calculating geometric mean

    # Set model to evaluation mode
    model.eval()
    all_losses_l1_for_gmean = []
    preds_list, labels_list = [], []

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            if args.gpu is not None:
                inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
            else:
                inputs, targets = inputs.cpu(), targets.cpu()

            predictions, _ = model(inputs, labels=None, epoch=None)

            predictions = predictions.squeeze(-1) if predictions.ndim > 1 and predictions.shape[-1] == 1 else predictions
            targets = targets.squeeze(-1) if targets.ndim > 1 and targets.shape[-1] == 1 else targets

            preds_list.append(predictions.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())

            loss_mse = criterion_mse(predictions, targets)
            loss_l1 = criterion_l1(predictions, targets)
            loss_l1_all = criterion_gmean(predictions, targets)
            all_losses_l1_for_gmean.append(loss_l1_all.detach().cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Print progress at specified intervals
            if idx % args.print_freq == 0:
                progress.display(idx)

        # Apply threshold-based metrics
        metrics_dict = threshold_metrics(
            np.hstack(preds_list), 
            np.hstack(labels_list),
            lower_threshold=args.lower_threshold,
            upper_threshold=args.upper_threshold
        )
        
        # Calculate geometric mean of all losses
        loss_gmean = gmean(np.hstack(all_losses_l1_for_gmean), axis=None).astype(float)
        
        # Print detailed performance metrics
        print(f" * Overall: MSE {metrics_dict['overall']['mse']:.3f}\t"
              f"L1 {metrics_dict['overall']['l1']:.3f}\t"
              f"G-Mean {metrics_dict['overall']['gmean']:.3f}\t"
              f"Count {metrics_dict['overall']['count']}")
        print('-' * 40)
        
        # Print metrics for each threshold range if applicable
        if 'below_lower' in metrics_dict:
            print(f" * Below {args.lower_threshold}: MSE {metrics_dict['below_lower']['mse']:.3f}\t"
                  f"L1 {metrics_dict['below_lower']['l1']:.3f}\t"
                  f"G-Mean {metrics_dict['below_lower']['gmean']:.3f}\t"
                  f"Count {metrics_dict['below_lower']['count']}")
        
        if 'middle' in metrics_dict:
            print(f" * Between {args.lower_threshold} and {args.upper_threshold}: MSE {metrics_dict['middle']['mse']:.3f}\t"
                  f"L1 {metrics_dict['middle']['l1']:.3f}\t"
                  f"G-Mean {metrics_dict['middle']['gmean']:.3f}\t"
                  f"Count {metrics_dict['middle']['count']}")
        
        if 'above_upper' in metrics_dict:
            print(f" * Above {args.upper_threshold}: MSE {metrics_dict['above_upper']['mse']:.3f}\t"
                  f"L1 {metrics_dict['above_upper']['l1']:.3f}\t"
                  f"G-Mean {metrics_dict['above_upper']['gmean']:.3f}\t"
                  f"Count {metrics_dict['above_upper']['count']}")
              
    return losses_mse.avg, losses_l1.avg, loss_gmean

def threshold_metrics(
        preds: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different value ranges based on thresholds.
    
    Args:
        preds: Model predictions
        labels: Ground truth labels
        lower_threshold: Lower threshold for label values
        upper_threshold: Upper threshold for label values
        
    Returns:
        Dictionary with metrics for each range (below_lower, middle, above_upper)
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    # Initialize result dictionary
    metrics_dict = defaultdict(dict)
    
    # Calculate overall metrics
    overall_mse = np.mean((preds - labels) ** 2)
    overall_l1 = np.mean(np.abs(preds - labels))
    overall_l1_all = np.abs(preds - labels)
    overall_gmean = gmean(overall_l1_all, axis=None).astype(float)
    
    metrics_dict['overall'] = {
        'mse': overall_mse,
        'l1': overall_l1,
        'gmean': overall_gmean,
        'count': len(labels)
    }
    
    # If thresholds are provided, calculate metrics for each range
    if lower_threshold is not None or upper_threshold is not None:
        # Below lower threshold
        if lower_threshold is not None:
            below_mask = labels < lower_threshold
            if np.any(below_mask):
                below_mse = np.mean((preds[below_mask] - labels[below_mask]) ** 2)
                below_l1 = np.mean(np.abs(preds[below_mask] - labels[below_mask]))
                below_l1_all = np.abs(preds[below_mask] - labels[below_mask])
                below_gmean = gmean(below_l1_all, axis=None).astype(float) if len(below_l1_all) > 0 else 0
                
                metrics_dict['below_lower'] = {
                    'mse': below_mse,
                    'l1': below_l1,
                    'gmean': below_gmean,
                    'count': np.sum(below_mask)
                }
        
        # Above upper threshold
        if upper_threshold is not None:
            above_mask = labels > upper_threshold
            if np.any(above_mask):
                above_mse = np.mean((preds[above_mask] - labels[above_mask]) ** 2)
                above_l1 = np.mean(np.abs(preds[above_mask] - labels[above_mask]))
                above_l1_all = np.abs(preds[above_mask] - labels[above_mask])
                above_gmean = gmean(above_l1_all, axis=None).astype(float) if len(above_l1_all) > 0 else 0
                
                metrics_dict['above_upper'] = {
                    'mse': above_mse,
                    'l1': above_l1,
                    'gmean': above_gmean,
                    'count': np.sum(above_mask)
                }
        
        # Between thresholds
        if lower_threshold is not None and upper_threshold is not None:
            middle_mask = (labels >= lower_threshold) & (labels <= upper_threshold)
            if np.any(middle_mask):
                middle_mse = np.mean((preds[middle_mask] - labels[middle_mask]) ** 2)
                middle_l1 = np.mean(np.abs(preds[middle_mask] - labels[middle_mask]))
                middle_l1_all = np.abs(preds[middle_mask] - labels[middle_mask])
                middle_gmean = gmean(middle_l1_all, axis=None).astype(float) if len(middle_l1_all) > 0 else 0
                
                metrics_dict['middle'] = {
                    'mse': middle_mse,
                    'l1': middle_l1,
                    'gmean': middle_gmean,
                    'count': np.sum(middle_mask)
                }
    
    return metrics_dict





if __name__ == '__main__':
    main()
