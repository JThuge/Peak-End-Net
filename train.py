"""
Peak-End Net Training Script (CLIP ViT-L/14).

Usage:
  # Single GPU (no pretrained encoder)
  python train.py --pretrained_encoder none

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node=4 train.py --pretrained_encoder none
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.log_utils import setup_logger
from models.peak_end_net import (
    PeakAesNetV4,
    PeakAesLossV4,
    OVERALL_INDICES,
    GENERAL_INDICES,
    HUMAN_INDICES,
)
from data.dataloader import (
    PeakAesDatasetV4,
    ALL_SCORE_COLUMNS,
    create_peakaes_v4_dataloader,
    create_divide_dataloader,
    _worker_init_fn,
)
from scipy.optimize import curve_fit

logger = setup_logger("peakaes_net_v4")


def get_args():
    parser = argparse.ArgumentParser(description="Peak-End Net Training")

    # Data paths
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training set CSV")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation set CSV")
    parser.add_argument("--video_paths_json", type=str, required=True,
                        help="Path to video_paths.json mapping video IDs to file paths")
    parser.add_argument("--frame_pseudo_labels_path", type=str, default=None,
                        help="(Deprecated, unused) Path to frame-level pseudo labels")
    parser.add_argument("--extracted_frames_dir", type=str, default=None,
                        help="Directory containing pre-extracted frames (.npz). "
                             "If set, frames are loaded from cache instead of decoding video each epoch.")
    parser.add_argument("--output_dir", type=str, default="output_peakaes_v4")

    # DIVIDE cross-dataset evaluation
    parser.add_argument("--divide_root", type=str, default=None,
                        help="Root directory of DIVIDE dataset (optional, for cross-dataset eval)")
    parser.add_argument("--divide_label_file", type=str, default="val_labels.txt",
                        help="DIVIDE label file name under divide_root")
    parser.add_argument("--divide_extracted_frames_dir", type=str, default=None,
                        help="Directory containing pre-extracted DIVIDE frames (.npz)")
    parser.add_argument("--eval_divide_every", type=int, default=1,
                        help="Evaluate on DIVIDE every N epochs (default: every epoch)")

    # Pretrained encoder
    parser.add_argument("--pretrained_encoder", type=str, default="none",
                        help="Path to pretrained CLIP4Clip model weights from Stage 1. "
                             "Set to 'none' to use raw CLIP weights without pre-training.")
    parser.add_argument("--ava_checkpoint_path", type=str, required=True,
                        help="Path to the AVA-pretrained aesthetic model checkpoint "
                             "(ViT-L/14). Its frozen MLP head scores each frame.")

    # Model parameters
    parser.add_argument("--pretrained_clip_name", type=str, default="ViT-L/14")
    parser.add_argument("--cross_model", type=str, default="cross-base")
    parser.add_argument("--sim_header", type=str, default="meanP")
    parser.add_argument("--linear_patch", type=str, default="3d", choices=["2d", "3d"])
    parser.add_argument("--loose_type", action="store_false")

    # Peak-End Net specific parameters
    parser.add_argument("--frame_context_window", type=int, default=3)
    parser.add_argument("--end_window", type=int, default=3)
    parser.add_argument("--rhythm_dim", type=int, default=64)
    parser.add_argument("--end_ratio", type=float, default=0.25)
    parser.add_argument("--scoring_hidden_dim", type=int, default=256)

    # Training hyperparameters
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_val", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Data parameters
    parser.add_argument("--max_words", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--feature_framerate", type=int, default=1)
    parser.add_argument("--num_thread_reader", type=int, default=2)
    parser.add_argument("--slice_framepos", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--train_frame_order", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--eval_frame_order", type=int, default=0, choices=[0, 1, 2])

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_display", type=int, default=50)
    parser.add_argument("--freeze_layer_num", type=int, default=0)

    args = parser.parse_args()

    # Distributed training
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))
    args.distributed = args.local_rank != -1
    args.n_gpu = 1
    return args


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def is_main_process(args):
    return not args.distributed or args.rank == 0


def setup_distributed(args):
    if not args.distributed:
        return
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    logger.info(f"Distributed training initialized: rank={args.rank}, "
                f"local_rank={args.local_rank}, world_size={args.world_size}")


def cleanup_distributed(args):
    if args.distributed:
        dist.destroy_process_group()


def get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """Warmup + Cosine Annealing learning rate schedule."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        import math
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def evaluate(model, dataloader, device, criterion):
    """Evaluate model on validation set."""
    # Use underlying model to avoid DDP ALLREDUCE sync during forward
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    all_predicted = []
    all_true = []
    all_has_human = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            video, video_mask, scores, has_human, frame_pseudo_labels, video_ids = batch
            video = video.to(device)
            video_mask = video_mask.to(device)
            scores = scores.to(device)
            has_human = has_human.to(device)

            predicted, auxiliary_outputs = eval_model(
                video, video_mask, has_human,
            )

            # Compute loss
            loss, _ = criterion(predicted, scores, has_human)
            total_loss += loss.item()
            num_batches += 1

            all_predicted.append(predicted.cpu().numpy())
            all_true.append(scores.cpu().numpy())
            all_has_human.append(has_human.cpu().numpy())

    all_predicted = np.concatenate(all_predicted, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_has_human = np.concatenate(all_has_human, axis=0)

    avg_loss = total_loss / max(num_batches, 1)

    # Compute per-dimension metrics
    results = {}
    for dim_idx, dim_name in enumerate(ALL_SCORE_COLUMNS):
        pred = all_predicted[:, dim_idx]
        true = all_true[:, dim_idx]

        if dim_idx in HUMAN_INDICES:
            human_mask = all_has_human.astype(bool)
            if human_mask.sum() == 0:
                results[dim_name] = {
                    "MSE": float("nan"), "SROCC": float("nan"),
                    "PLCC": float("nan"), "KRCC": float("nan"),
                    "ACC": float("nan"), "num_samples": 0,
                }
                continue
            pred = pred[human_mask]
            true = true[human_mask]
            num_samples = int(human_mask.sum())
        else:
            num_samples = len(true)

        mse_val = mean_squared_error(true, pred)
        srocc, _ = spearmanr(true, pred)
        plcc, _ = pearsonr(true, pred)
        krcc, _ = kendalltau(true, pred)

        binary_true = (true >= 5.0).astype(int)
        binary_pred = (pred >= 5.0).astype(int)
        accuracy = accuracy_score(binary_true, binary_pred)

        results[dim_name] = {
            "MSE": float(mse_val), "SROCC": float(srocc),
            "PLCC": float(plcc), "KRCC": float(krcc),
            "ACC": float(accuracy), "num_samples": num_samples,
        }
    valid_mses = [r["MSE"] for r in results.values() if not np.isnan(r["MSE"])]
    avg_mse = np.mean(valid_mses) if valid_mses else float("inf")

    return results, avg_mse, avg_loss


# ============================================================
# DIVIDE Cross-Dataset Evaluation
# ============================================================

def _logistic_func(x, a, b, c, d):
    """Logistic function for fitting predictions to ground truth distribution."""
    return a / (1 + np.exp(-b * (x - c))) + d


def _rescale_predictions(predictions, ground_truth):
    """Standardize and rescale predictions to match ground truth distribution."""
    predictions_rescaled = (
        (predictions - np.mean(predictions)) / (np.std(predictions) + 1e-8)
    ) * np.std(ground_truth) + np.mean(ground_truth)
    return predictions_rescaled


def _compute_rmse_with_logistic_fitting(ground_truth, predictions):
    """
    Compute RMSE with rescale + logistic fitting.

    Returns (rmse, fitted_predictions).
    """
    predictions_rescaled = _rescale_predictions(predictions, ground_truth)

    try:
        initial_params = [
            np.max(ground_truth) - np.min(ground_truth),
            1.0,
            np.mean(predictions_rescaled),
            np.min(ground_truth),
        ]
        popt, _ = curve_fit(
            _logistic_func, predictions_rescaled, ground_truth,
            p0=initial_params, maxfev=10000,
        )
        predictions_fitted = _logistic_func(predictions_rescaled, *popt)
    except Exception:
        predictions_fitted = predictions_rescaled

    rmse = np.sqrt(((predictions_fitted - ground_truth) ** 2).mean())
    return rmse, predictions_fitted


def evaluate_on_divide(model, divide_loader, device):
    """
    Evaluate aesthetic dimensions on DIVIDE dataset.

    Model outputs 11-dim scores; index 0 (overall) is used as the prediction
    to compute SROCC/PLCC/KRCC/RMSE against DIVIDE aesthetic dimension (first label column).
    """
    # Use underlying model to avoid DDP ALLREDUCE sync during forward
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    all_predicted_overall = []
    all_true_aesthetic = []

    with torch.no_grad():
        for batch in tqdm(divide_loader, desc="Evaluating on DIVIDE", leave=False):
            video, video_mask, aesthetic_scores, video_names = batch
            video = video.to(device)
            video_mask = video_mask.to(device)

            # DIVIDE has no has_human info; pass all False
            batch_size = video.size(0)
            dummy_has_human = torch.zeros(batch_size, dtype=torch.bool, device=device)

            predicted, _ = eval_model(
                video, video_mask, dummy_has_human,
            )

            predicted_overall = predicted[:, 0].cpu().numpy()
            all_predicted_overall.append(predicted_overall)
            all_true_aesthetic.append(aesthetic_scores.numpy())

    all_predicted_overall = np.concatenate(all_predicted_overall, axis=0)
    all_true_aesthetic = np.concatenate(all_true_aesthetic, axis=0)

    # Compute RMSE (Rescale + Logistic fitting)
    rmse_val, predictions_fitted = _compute_rmse_with_logistic_fitting(
        all_true_aesthetic, all_predicted_overall
    )

    # SROCC/PLCC/KRCC using fitted predictions
    srocc, _ = spearmanr(all_true_aesthetic, predictions_fitted)
    plcc, _ = pearsonr(all_true_aesthetic, predictions_fitted)
    krcc, _ = kendalltau(all_true_aesthetic, predictions_fitted)

    divide_results = {
        "RMSE": float(rmse_val),
        "SROCC": float(srocc),
        "PLCC": float(plcc),
        "KRCC": float(krcc),
        "num_samples": len(all_true_aesthetic),
    }

    return divide_results


def main():
    args = get_args()
    setup_distributed(args)
    set_seed(args.seed + args.rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.distributed:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process(args):
        logger.info("=" * 60)
        logger.info("PeakAes-Net V4 Training")
        logger.info("=" * 60)
        for key, value in sorted(vars(args).items()):
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(args.local_rank if args.distributed else 0)
            gpu_mem = torch.cuda.get_device_properties(
                args.local_rank if args.distributed else 0
            ).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Build model
    if is_main_process(args):
        logger.info("Building Peak-End Net model...")

    model = PeakAesNetV4(
        pretrained_encoder_path=args.pretrained_encoder,
        ava_checkpoint_path=args.ava_checkpoint_path,
        pretrained_clip_name=args.pretrained_clip_name,
        cross_model=args.cross_model,
        linear_patch=args.linear_patch,
        sim_header=args.sim_header,
        max_words=args.max_words,
        max_frames=args.max_frames,
        frame_context_window=args.frame_context_window,
        end_window=args.end_window,
        rhythm_dim=args.rhythm_dim,
        end_ratio=args.end_ratio,
        scoring_hidden_dim=args.scoring_hidden_dim,
    ).to(device)

    if is_main_process(args):
        model.print_model_info()

    # DDP wrapping
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank],
                    output_device=args.local_rank, find_unused_parameters=True)

    actual_model = model.module if args.distributed else model

    # Optimizer
    param_groups = actual_model.get_param_groups(base_lr=args.lr)
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    if is_main_process(args):
        logger.info(f"Trainable parameters: {trainable_params:,}")
        for i, group in enumerate(param_groups):
            group_params = sum(p.numel() for p in group["params"])
            logger.info(f"  Param group {i}: {group_params:,} params, lr={group['lr']}")

    # Loss function
    criterion = PeakAesLossV4()

    # Data loaders
    if is_main_process(args):
        logger.info("Loading datasets...")

    if args.distributed:
        train_dataset = PeakAesDatasetV4(
            csv_path=args.train_csv,
            video_paths_json=args.video_paths_json,
            frame_pseudo_labels_path=args.frame_pseudo_labels_path,
            extracted_frames_dir=args.extracted_frames_dir,
            max_frames=args.max_frames,
            feature_framerate=args.feature_framerate,
            frame_order=args.train_frame_order,
            slice_framepos=args.slice_framepos,
        )
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size_train,
            num_workers=args.num_thread_reader,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=args.num_thread_reader > 0,
        )
        train_size = len(train_dataset)
    else:
        train_loader, train_size = create_peakaes_v4_dataloader(
            args.train_csv, args.video_paths_json, args, is_train=True
        )

    val_loader, val_size = create_peakaes_v4_dataloader(
        args.val_csv, args.video_paths_json, args, is_train=False
    )

    # DIVIDE cross-dataset eval dataloader
    divide_loader, divide_size = create_divide_dataloader(args)

    if is_main_process(args):
        logger.info(f"  Train: {train_size} samples, {len(train_loader)} batches")
        logger.info(f"  Val: {val_size} samples, {len(val_loader)} batches")
        if divide_loader is not None:
            logger.info(f"  DIVIDE: {divide_size} samples, {len(divide_loader)} batches")
        else:
            logger.info(f"  DIVIDE: Not configured (set --divide_root to enable)")

    # Training loop
    best_val_mse = float("inf")
    best_epoch = -1
    training_start_time = time.time()

    if is_main_process(args):
        logger.info("=" * 60)
        logger.info("Starting Peak-End Net training...")
        logger.info("=" * 60)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # Learning rate schedule
        current_lr = get_warmup_cosine_lr(
            epoch, args.warmup_epochs, args.epochs, args.lr
        )
        for param_group in optimizer.param_groups:
            ratio = param_group["lr"] / args.lr if args.lr > 0 else 1.0
            param_group["lr"] = current_lr * ratio

        model.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        num_steps = 0

        if args.distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if is_main_process(args):
            logger.info("")
            logger.info(f"Epoch {epoch+1}/{args.epochs} (lr={current_lr:.6f})")
            pbar = tqdm(total=len(train_loader),
                        desc=f"Train Epoch {epoch+1}",
                        unit="step",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        for step, batch in enumerate(train_loader):
            video, video_mask, scores, has_human, frame_pseudo_labels, video_ids = batch
            video = video.to(device)
            video_mask = video_mask.to(device)
            scores = scores.to(device)
            has_human = has_human.to(device)

            # Forward pass
            predicted, auxiliary_outputs = model(
                video, video_mask, has_human,
            )

            # Main loss (MSE only)
            loss, loss_dict = criterion(predicted, scores, has_human)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )

            optimizer.step()

            total_loss += loss.item()
            total_mse_loss += (loss_dict["overall_mse"] + loss_dict["general_mse"] + loss_dict["human_mse"])
            num_steps += 1

            if is_main_process(args):
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{loss_dict['overall_mse']:.4f}",
                )
                pbar.update(1)

        if is_main_process(args):
            pbar.close()

        train_avg_loss = total_loss / max(num_steps, 1)
        train_avg_mse = total_mse_loss / max(num_steps, 1)

        # Synchronize
        if args.distributed:
            dist.barrier()

        # Validation (all ranks participate in forward to avoid DDP ALLREDUCE deadlock)
        val_results, val_avg_mse, val_avg_loss = evaluate(
            model, val_loader, device, criterion
        )

        # DIVIDE cross-dataset eval (all ranks participate in forward)
        divide_results = None
        if divide_loader is not None and (epoch + 1) % args.eval_divide_every == 0:
            divide_results = evaluate_on_divide(model, divide_loader, device)

        # Logging and checkpoint saving on rank 0 only
        if is_main_process(args):
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} Summary (time: {epoch_time/60:.1f}min):")
            logger.info(f"  Train: loss={train_avg_loss:.4f}, mse={train_avg_mse:.4f}")
            logger.info(f"  Val: loss={val_avg_loss:.4f}, avg_mse={val_avg_mse:.4f}")

            for dim_name, metrics in val_results.items():
                if np.isnan(metrics["MSE"]):
                    continue
                logger.info(
                    f"  [{dim_name}] MSE={metrics['MSE']:.4f} "
                    f"SROCC={metrics['SROCC']:.4f} PLCC={metrics['PLCC']:.4f} "
                    f"KRCC={metrics['KRCC']:.4f} ACC={metrics['ACC']:.4f}"
                )

            if divide_results is not None:
                logger.info(f"  DIVIDE Aesthetic: "
                            f"RMSE={divide_results['RMSE']:.4f} "
                            f"SROCC={divide_results['SROCC']:.4f} "
                            f"PLCC={divide_results['PLCC']:.4f} "
                            f"KRCC={divide_results['KRCC']:.4f} "
                            f"(N={divide_results['num_samples']})")

            # Save checkpoint
            trainable_state = {
                name: param.cpu()
                for name, param in actual_model.named_parameters()
                if param.requires_grad
            }

            checkpoint = {
                "epoch": epoch + 1,
                "trainable_state_dict": trainable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_avg_mse,
                "val_results": val_results,
                "divide_results": divide_results,
                "args": vars(args),
            }

            if val_avg_mse < best_val_mse:
                improvement = best_val_mse - val_avg_mse if best_val_mse != float("inf") else 0
                best_val_mse = val_avg_mse
                best_epoch = epoch + 1
                best_path = os.path.join(args.output_dir, "peakaes_v4_best.pth")
                torch.save(checkpoint, best_path)
                logger.info(f"  ★ New best! MSE improved by {improvement:.4f}, saved to {best_path}")

            epoch_path = os.path.join(args.output_dir, f"peakaes_v4_epoch{epoch+1}.pth")
            torch.save(checkpoint, epoch_path)

        if args.distributed:
            dist.barrier()

    total_time = time.time() - training_start_time
    if is_main_process(args):
        logger.info("=" * 60)
        logger.info("V4 Training completed!")
        logger.info(f"  Best epoch: {best_epoch}/{args.epochs}")
        logger.info(f"  Best Val MSE: {best_val_mse:.4f}")
        logger.info(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
        logger.info(f"  Output: {args.output_dir}")
        logger.info("=" * 60)

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
