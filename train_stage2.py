"""
Peak-End Net Stage 2 Training Script: Gated Fusion (CLIP ViT-L/14).

Stage 2 freezes the entire Stage-1 model (PeakAesNetV4) and trains only the
lightweight GatedFusionModule that fuses the Stage-1 overall score (S_model)
with the average frame-level AVA aesthetic score (S_static):

    S_final = gate * S_model + (1 - gate) * S_static

The gate is supervised with a soft target: whichever of S_model / S_static is
closer to the ground-truth label receives higher trust.

Usage:
  # Single GPU
  python train_stage2.py \\
      --stage1_checkpoint ./output_stage1/peakaes_v4_best.pth \\
      --ava_checkpoint_path .../checkpoints_L/best_model.pth \\
      --train_csv ... --val_csv ... --video_paths_json ...

  # Multi-GPU
  torchrun --nproc_per_node=4 train_stage2.py --stage1_checkpoint ... ...
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.optimize import curve_fit
from tqdm import tqdm

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.log_utils import setup_logger
from models.gated_fusion import GatedFusionPeakAesNetV4
from models.peak_end_net import OVERALL_INDICES, GENERAL_INDICES, HUMAN_INDICES
from data.dataloader import (
    PeakAesDatasetV4,
    ALL_SCORE_COLUMNS,
    create_peakaes_v4_dataloader,
    create_divide_dataloader,
    _worker_init_fn,
)

logger = setup_logger("peakaes_net_stage2")


def get_args():
    parser = argparse.ArgumentParser(description="Peak-End Net Stage 2 (Gated Fusion) Training")

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
    parser.add_argument("--output_dir", type=str, default="output_stage2")

    # DIVIDE cross-dataset evaluation
    parser.add_argument("--divide_root", type=str, default=None,
                        help="Root directory of DIVIDE dataset (optional, for cross-dataset eval)")
    parser.add_argument("--divide_label_file", type=str, default="val_labels.txt",
                        help="DIVIDE label file name under divide_root")
    parser.add_argument("--divide_extracted_frames_dir", type=str, default=None,
                        help="Directory containing pre-extracted DIVIDE frames (.npz)")
    parser.add_argument("--eval_divide_every", type=int, default=1,
                        help="Evaluate on DIVIDE every N epochs (default: every epoch)")

    # Stage-1 checkpoint + pretrained encoder + AVA head
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="Path to the Stage-1 checkpoint (peakaes_v4_best.pth).")
    parser.add_argument("--pretrained_encoder", type=str, default="none",
                        help="Path to pretrained CLIP4Clip encoder weights (must match Stage 1). "
                             "Set to 'none' to use raw CLIP weights.")
    parser.add_argument("--ava_checkpoint_path", type=str, required=True,
                        help="Path to the AVA-pretrained aesthetic model checkpoint "
                             "(ViT-L/14). Its frozen MLP head scores each frame.")

    # Model parameters (must match Stage 1)
    parser.add_argument("--pretrained_clip_name", type=str, default="ViT-L/14")
    parser.add_argument("--cross_model", type=str, default="cross-base")
    parser.add_argument("--sim_header", type=str, default="meanP")
    parser.add_argument("--linear_patch", type=str, default="3d", choices=["2d", "3d"])
    parser.add_argument("--loose_type", action="store_false")

    # Peak-End Net specific parameters (must match Stage 1)
    parser.add_argument("--frame_context_window", type=int, default=3)
    parser.add_argument("--end_window", type=int, default=3)
    parser.add_argument("--rhythm_dim", type=int, default=64)
    parser.add_argument("--end_ratio", type=float, default=0.25)
    parser.add_argument("--scoring_hidden_dim", type=int, default=256)

    # Fusion module parameters
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--gate_loss_weight", type=float, default=1.0,
                        help="Weight for the gate BCE supervision loss.")

    # Training hyperparameters
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_val", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
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


def compute_stage2_loss(final_scores, aux, scores, gate_loss_weight):
    """
    Stage-2 loss = L_fusion (MSE on fused overall) + gate_loss_weight * L_gate (BCE).

    Gate soft target: whichever of S_model / S_static is closer to the label
    should receive higher trust.
        gate_target = static_error / (static_error + model_error + eps)
    (gate -> 1 means trust S_model; when model_error is small, gate_target -> 1).
    """
    target_overall = scores[:, 0]
    S_model = aux["S_model"]
    S_static = aux["S_static"]
    gate = aux["gate"].squeeze(-1)

    # L_fusion: MSE on the fused overall dimension
    L_fusion = F.mse_loss(final_scores[:, 0], target_overall)

    # L_gate: BCE toward soft target
    model_error = torch.abs(S_model - target_overall)
    static_error = torch.abs(S_static - target_overall)
    gate_target = static_error / (static_error + model_error + 1e-6)
    gate_target = gate_target.detach().clamp(0.0, 1.0)
    L_gate = F.binary_cross_entropy(gate.clamp(1e-6, 1 - 1e-6), gate_target)

    total = L_fusion + gate_loss_weight * L_gate
    loss_dict = {
        "total": total.item(),
        "fusion": L_fusion.item(),
        "gate_bce": L_gate.item(),
        "gate_mean": gate.mean().item(),
    }
    return total, loss_dict


def evaluate(model, dataloader, device, gate_loss_weight):
    """Evaluate the fused model on the validation set."""
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    all_predicted = []
    all_true = []
    all_has_human = []
    gate_values = []
    s_model_values = []
    s_static_values = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            video, video_mask, scores, has_human, frame_pseudo_labels, video_ids = batch
            video = video.to(device)
            video_mask = video_mask.to(device)
            scores = scores.to(device)
            has_human = has_human.to(device)

            final_scores, aux = eval_model(video, video_mask, has_human)

            loss, _ = compute_stage2_loss(final_scores, aux, scores, gate_loss_weight)
            total_loss += loss.item()
            num_batches += 1

            gate_values.append(aux["gate"].squeeze(-1).cpu().numpy())
            s_model_values.append(aux["S_model"].cpu().numpy())
            s_static_values.append(aux["S_static"].cpu().numpy())
            all_predicted.append(final_scores.cpu().numpy())
            all_true.append(scores.cpu().numpy())
            all_has_human.append(has_human.cpu().numpy())

    all_predicted = np.concatenate(all_predicted, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_has_human = np.concatenate(all_has_human, axis=0)
    gate_all = np.concatenate(gate_values, axis=0)
    s_model_all = np.concatenate(s_model_values, axis=0)
    s_static_all = np.concatenate(s_static_values, axis=0)
    val_stats = {
        "gate_mean": float(gate_all.mean()),
        "gate_std": float(gate_all.std()),
        "gate_min": float(gate_all.min()),
        "gate_max": float(gate_all.max()),
        "s_model_mean": float(s_model_all.mean()),
        "s_model_std": float(s_model_all.std()),
        "s_static_mean": float(s_static_all.mean()),
        "s_static_std": float(s_static_all.std()),
    }

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

    return results, avg_mse, avg_loss, val_stats


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
    """Compute RMSE with rescale + logistic fitting. Returns (rmse, fitted)."""
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


def _divide_metrics(ground_truth, predictions):
    """Rescale + logistic fit, then RMSE/SROCC/PLCC/KRCC for one predictor."""
    rmse_val, fitted = _compute_rmse_with_logistic_fitting(ground_truth, predictions)
    srocc, _ = spearmanr(ground_truth, fitted)
    plcc, _ = pearsonr(ground_truth, fitted)
    krcc, _ = kendalltau(ground_truth, fitted)
    return {
        "RMSE": float(rmse_val),
        "SROCC": float(srocc),
        "PLCC": float(plcc),
        "KRCC": float(krcc),
    }


def evaluate_on_divide(model, divide_loader, device):
    """Evaluate S_model / S_static / S_final separately on the DIVIDE dataset."""
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    all_S_final = []
    all_S_model = []
    all_S_static = []
    all_gate = []
    all_true_aesthetic = []

    with torch.no_grad():
        for batch in tqdm(divide_loader, desc="Evaluating on DIVIDE", leave=False):
            video, video_mask, aesthetic_scores, video_names = batch
            video = video.to(device)
            video_mask = video_mask.to(device)

            batch_size = video.size(0)
            dummy_has_human = torch.zeros(batch_size, dtype=torch.bool, device=device)

            final_scores, aux = eval_model(video, video_mask, dummy_has_human)

            all_S_final.append(final_scores[:, 0].cpu().numpy())
            all_S_model.append(aux["S_model"].cpu().numpy())
            all_S_static.append(aux["S_static"].cpu().numpy())
            all_gate.append(aux["gate"].squeeze(-1).cpu().numpy())
            all_true_aesthetic.append(aesthetic_scores.numpy())

    gt = np.concatenate(all_true_aesthetic, axis=0)
    s_final = np.concatenate(all_S_final, axis=0)
    s_model = np.concatenate(all_S_model, axis=0)
    s_static = np.concatenate(all_S_static, axis=0)
    gate = np.concatenate(all_gate, axis=0)

    return {
        "S_model": _divide_metrics(gt, s_model),
        "S_static": _divide_metrics(gt, s_static),
        "S_final": _divide_metrics(gt, s_final),
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "num_samples": int(len(gt)),
    }


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
        logger.info("PeakAes-Net Stage 2 (Gated Fusion) Training")
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

    # Build model (frozen Stage-1 + trainable fusion module)
    if is_main_process(args):
        logger.info("Building GatedFusion Peak-End Net model...")

    model = GatedFusionPeakAesNetV4(
        stage1_checkpoint_path=args.stage1_checkpoint,
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
        fusion_hidden_dim=args.fusion_hidden_dim,
    ).to(device)

    if is_main_process(args):
        model.print_model_info()

    # DDP wrapping (only the fusion module has trainable params)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank],
                    output_device=args.local_rank, find_unused_parameters=True)

    actual_model = model.module if args.distributed else model

    # Optimizer: only the fusion module is trainable
    trainable = [p for p in actual_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    trainable_params = sum(p.numel() for p in trainable)
    if is_main_process(args):
        logger.info(f"Trainable parameters (fusion module): {trainable_params:,}")

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
        logger.info("Starting Stage 2 (Gated Fusion) training...")
        logger.info("=" * 60)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        current_lr = get_warmup_cosine_lr(
            epoch, args.warmup_epochs, args.epochs, args.lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        total_loss = 0.0
        total_fusion = 0.0
        total_gate_bce = 0.0
        total_gate_mean = 0.0
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

            final_scores, aux = model(video, video_mask, has_human)

            loss, loss_dict = compute_stage2_loss(
                final_scores, aux, scores, args.gate_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

            optimizer.step()

            total_loss += loss_dict["total"]
            total_fusion += loss_dict["fusion"]
            total_gate_bce += loss_dict["gate_bce"]
            total_gate_mean += loss_dict["gate_mean"]
            num_steps += 1

            if is_main_process(args):
                pbar.set_postfix(
                    loss=f"{loss_dict['total']:.4f}",
                    fusion=f"{loss_dict['fusion']:.4f}",
                    gate=f"{loss_dict['gate_mean']:.3f}",
                )
                pbar.update(1)

        if is_main_process(args):
            pbar.close()

        train_avg_loss = total_loss / max(num_steps, 1)
        train_avg_fusion = total_fusion / max(num_steps, 1)
        train_avg_gate_bce = total_gate_bce / max(num_steps, 1)
        train_avg_gate = total_gate_mean / max(num_steps, 1)

        if args.distributed:
            dist.barrier()

        # Validation
        val_results, val_avg_mse, val_avg_loss, val_stats = evaluate(
            model, val_loader, device, args.gate_loss_weight
        )

        # DIVIDE cross-dataset eval
        divide_results = None
        if divide_loader is not None and (epoch + 1) % args.eval_divide_every == 0:
            divide_results = evaluate_on_divide(model, divide_loader, device)

        # Logging and checkpoint saving on rank 0 only
        if is_main_process(args):
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} Summary (time: {epoch_time/60:.1f}min):")
            logger.info(f"  Train: loss={train_avg_loss:.4f}, fusion={train_avg_fusion:.4f}, "
                        f"gate_bce={train_avg_gate_bce:.4f}, gate_mean={train_avg_gate:.4f}")
            logger.info(f"  Val: loss={val_avg_loss:.4f}, avg_mse={val_avg_mse:.4f}")
            logger.info(f"  Gate stats: mean={val_stats['gate_mean']:.3f}, std={val_stats['gate_std']:.3f}, "
                        f"min={val_stats['gate_min']:.3f}, max={val_stats['gate_max']:.3f}")
            logger.info(f"  S_model: mean={val_stats['s_model_mean']:.3f}, std={val_stats['s_model_std']:.3f}")
            logger.info(f"  S_static: mean={val_stats['s_static_mean']:.3f}, std={val_stats['s_static_std']:.3f}")

            for dim_name, metrics in val_results.items():
                if np.isnan(metrics["MSE"]):
                    continue
                logger.info(
                    f"  [{dim_name}] MSE={metrics['MSE']:.4f} "
                    f"SROCC={metrics['SROCC']:.4f} PLCC={metrics['PLCC']:.4f} "
                    f"KRCC={metrics['KRCC']:.4f} ACC={metrics['ACC']:.4f}"
                )

            if divide_results is not None:
                dfi = divide_results["S_final"]
                logger.info(f"  DIVIDE Aesthetic: "
                            f"RMSE={dfi['RMSE']:.4f} SROCC={dfi['SROCC']:.4f} "
                            f"PLCC={dfi['PLCC']:.4f} KRCC={dfi['KRCC']:.4f} "
                            f"(N={divide_results['num_samples']})")

            # Save checkpoint (only trainable fusion params)
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
                "val_gate_mean": val_stats["gate_mean"],
                "val_results": val_results,
                "divide_results": divide_results,
                "stage1_checkpoint": args.stage1_checkpoint,
                "args": vars(args),
            }

            if val_avg_mse < best_val_mse:
                improvement = best_val_mse - val_avg_mse if best_val_mse != float("inf") else 0
                best_val_mse = val_avg_mse
                best_epoch = epoch + 1
                best_path = os.path.join(args.output_dir, "stage2_best.pth")
                torch.save(checkpoint, best_path)
                logger.info(f"  ★ New best! MSE improved by {improvement:.4f}, saved to {best_path}")

            epoch_path = os.path.join(args.output_dir, f"stage2_epoch{epoch+1}.pth")
            torch.save(checkpoint, epoch_path)

        if args.distributed:
            dist.barrier()

    total_time = time.time() - training_start_time
    if is_main_process(args):
        logger.info("=" * 60)
        logger.info("Stage 2 Training completed!")
        logger.info(f"  Best epoch: {best_epoch}/{args.epochs}")
        logger.info(f"  Best Val MSE: {best_val_mse:.4f}")
        logger.info(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
        logger.info(f"  Output: {args.output_dir}")
        logger.info("=" * 60)

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
