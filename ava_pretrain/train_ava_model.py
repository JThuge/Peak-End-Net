"""
AVA Model Pretraining: CLIP-based image aesthetic score predictor (Multi-GPU DDP).

Trains a frozen CLIP ViT-L/14 backbone with a lightweight 10-class
distribution head on the AVA dataset, using Earth Mover's Distance (EMD) loss.
The resulting checkpoint (best_model.pth) is used as the frozen AVA model
in Peak-End Net Stage 2 (gated fusion).

Usage:
  torchrun --nproc_per_node=8 train_ava_model.py \
      --img_dir /path/to/AVA/images \
      --csv_file /path/to/AVA.txt \
      --clip_model ViT-L/14 \
      --save_dir ./checkpoints
"""

import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip


# ==========================================
# 1. Loss: Earth Mover's Distance (EMD)
# ==========================================
def emd_loss(pred, target, r=2):
    """
    Earth Mover's Distance loss.

    Measures the distance between two probability distributions, suitable for
    aesthetic score distribution prediction.
    """
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    loss = torch.pow(torch.abs(cdf_pred - cdf_target), r)
    return torch.pow(torch.mean(loss), 1.0 / r)


# ==========================================
# 2. Dataset: AVADataset
# ==========================================
class AVADataset(Dataset):
    """
    AVA aesthetic dataset loader.

    Expects a space-separated label file where each row is:
        index image_id count_1 count_2 ... count_10 [extra columns...]
    """

    def __init__(self, csv_file, img_dir, transform=None, debug_size=None):
        self.df = pd.read_csv(csv_file, sep=' ', header=None)
        self.img_dir = img_dir
        self.transform = transform

        if debug_size is not None:
            self.df = self.df.head(debug_size)

        # Pre-compute valid image indices
        self.valid_indices = []
        for idx in tqdm(range(len(self.df)), desc="Validating images", disable=False):
            row = self.df.iloc[idx]
            img_id = str(int(row[1]))
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

        print(f"Found {len(self.valid_indices)} valid images out of {len(self.df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]

        img_id = str(int(row[1]))
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        counts = torch.tensor(row[2:12].values.astype('float32'))
        probs = counts / (counts.sum() + 1e-8)
        mean_score = torch.sum(probs * torch.arange(1, 11).float())

        return image, probs, mean_score


# ==========================================
# 3. Model: CLIP Aesthetic Predictor
# ==========================================
class CLIPAestheticPredictor(nn.Module):
    """
    CLIP-based image aesthetic score predictor.

    This is the AVA model architecture; its state_dict is later loaded by
    AVAModel in Peak-End Net Stage 2.
    """

    def __init__(self, clip_model_name="ViT-L/14", freeze_clip=True):
        super().__init__()

        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        self.feature_dim = self.clip_model.visual.output_dim

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_features=False):
        features = self.clip_model.encode_image(x).float()
        logits = self.head(features)
        probs = self.softmax(logits)

        if return_features:
            return probs, features
        return probs

    def get_mean_score(self, probs):
        weights = torch.arange(1, 11).float().to(probs.device)
        return torch.sum(probs * weights, dim=1)


# ==========================================
# 4. Train / validate helpers
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}") if rank == 0 else dataloader

    for batch in pbar:
        images, labels, _ = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})

    # Synchronize loss across processes
    total_loss_tensor = torch.tensor(total_loss).to(device)
    total_samples_tensor = torch.tensor(total_samples).to(device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    return total_loss_tensor.item() / total_samples_tensor.item()


def validate(model, dataloader, criterion, device, epoch, rank):
    """Validation."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}") if rank == 0 else dataloader

    with torch.no_grad():
        for batch in pbar:
            images, labels, mean_scores = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            pred_scores = model.get_mean_score(outputs)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(pred_scores.cpu().numpy())
            all_labels.extend(mean_scores.numpy())

    # Synchronize loss across processes
    total_loss_tensor = torch.tensor(total_loss).to(device)
    total_samples_tensor = torch.tensor(total_samples).to(device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()

    # Gather predictions from all processes
    preds_tensor = torch.from_numpy(np.array(all_preds)).to(device)
    labels_tensor = torch.from_numpy(np.array(all_labels)).to(device)

    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)

    all_preds = torch.cat(gathered_preds).cpu().numpy()
    all_labels = torch.cat(gathered_labels).cpu().numpy()

    spearman_corr, _ = spearmanr(all_preds, all_labels)
    pearson_corr, _ = pearsonr(all_preds, all_labels)

    return avg_loss, spearman_corr, pearson_corr


# ==========================================
# 5. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='AVA CLIP Aesthetic Model Training (Multi-GPU DDP)')
    parser.add_argument('--img_dir', type=str, default='images', help='Image directory')
    parser.add_argument('--csv_file', type=str, default='AVA.txt', help='AVA label file')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--freeze_clip', action='store_true', default=True, help='Freeze CLIP backbone')
    parser.add_argument('--unfreeze_clip', action='store_false', dest='freeze_clip', help='Unfreeze CLIP backbone')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--debug', type=int, default=None, help='Debug mode: use only N samples')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from')

    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank (auto-set by torchrun)')
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs')

    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("Multi-GPU DDP Training")
        print(f"World Size: {world_size} GPUs")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Total Batch Size: {args.batch_size * world_size}")
        print("=" * 60)
        os.makedirs(args.save_dir, exist_ok=True)

    dist.barrier()

    if rank == 0:
        print(f"Loading CLIP model: {args.clip_model}")

    model = CLIPAestheticPredictor(
        clip_model_name=args.clip_model,
        freeze_clip=args.freeze_clip,
    ).to(device)

    start_epoch = 1
    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # CLIP preprocessing transform
    _, preprocess = clip.load(args.clip_model, device="cpu")

    if rank == 0:
        print("Loading dataset...")

    full_dataset = AVADataset(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        transform=preprocess,
        debug_size=args.debug,
    )

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if rank == 0:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Optimizer
    if args.freeze_clip:
        optimizer = optim.AdamW(model.module.head.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        head_params = list(model.module.head.parameters())
        clip_params = list(model.module.clip_model.parameters())
        optimizer = optim.AdamW([
            {'params': head_params, 'lr': args.lr},
            {'params': clip_params, 'lr': args.lr * 0.1},
        ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = emd_loss

    best_spearman = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*50}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, rank)
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

        val_loss, spearman, pearson = validate(model, val_loader, criterion, device, epoch, rank)
        if rank == 0:
            print(f"Val Loss: {val_loss:.4f}, Spearman: {spearman:.4f}, Pearson: {pearson:.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"Learning Rate: {current_lr:.6f}")

        if rank == 0:
            if spearman > best_spearman:
                best_spearman = spearman
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'spearman': spearman,
                    'pearson': pearson,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Saved best model (Spearman: {spearman:.4f})")

            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    if rank == 0:
        print(f"\nTraining completed! Best Spearman: {best_spearman:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
