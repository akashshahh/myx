"""Fine-tune the PANNs perception head on synthetic degradation labels.

Schedule (per the plan):
  - epoch 0: backbone frozen, train only the 8-dim head (lr 1e-4)
  - epoch >= --unfreeze-epoch: unfreeze backbone, fine-tune all (head lr 1e-4,
    backbone lr 1e-5)
Loss: MSE between sigmoid output and the (8,) severity label vector.
Val: deterministic degradations on the MUSDB test subset; best checkpoint by val
MSE -> checkpoints/perception_best.pth. TensorBoard logs -> runs/.

Device is auto-detected (cuda -> mps -> cpu). AMP (fp16) is enabled on CUDA only;
tuned to fit a 6 GB card with --batch-size 8 --chunk-seconds 6. Bump batch/chunk
on bigger GPUs.

    # 4050 laptop (CUDA):
    python perception/train.py --epochs 12 --batch-size 8 --num-workers 4
    # quick smoke (any device, ~2 batches/epoch, 1 epoch):
    python perception/train.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import DegradedAudioDataset
from data.degradation import ISSUE_DIMENSIONS
from data.musdb_loader import MUSDBLoader
from perception.model import PerceptionModel

PRETRAINED = "checkpoints/Cnn14_mAP=0.431.pth"
BEST_CKPT = "checkpoints/perception_best.pth"


def pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(args) -> tuple[DataLoader, DataLoader]:
    train_ds = DegradedAudioDataset(
        MUSDBLoader(root=args.root, subsets="train", is_wav=args.is_wav, cache_size=args.cache_size),
        chunk_seconds=args.chunk_seconds,
        chunks_per_track=args.chunks_per_track,
    )
    val_ds = DegradedAudioDataset(
        MUSDBLoader(root=args.root, subsets="test", is_wav=args.is_wav, cache_size=args.cache_size),
        chunk_seconds=args.chunk_seconds,
        chunks_per_track=max(2, args.chunks_per_track // 2),
        deterministic=True,
        seed=1234,
    )
    common = dict(
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),
        persistent_workers=(args.num_workers > 0),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_dl, val_dl


def build_optimizer(model: PerceptionModel, head_lr: float, backbone_lr: float) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, p in model.backbone.named_parameters():
        (head if name.startswith(("fc1", "fc_audioset")) else backbone).append(p)
    # Both groups always present; frozen params simply receive no grad until unfreeze.
    return torch.optim.AdamW(
        [{"params": head, "lr": head_lr}, {"params": backbone, "lr": backbone_lr}],
        weight_decay=1e-4,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None) -> float:
    model.eval()
    se, n = 0.0, 0
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x)
        se += torch.sum((pred - y) ** 2).item()
        n += y.numel()
    return se / max(n, 1)


def train(args) -> None:
    device = pick_device(args.device)
    print(f"device: {device}")
    use_amp = device.type == "cuda" and not args.no_amp

    ckpt = PRETRAINED if os.path.exists(PRETRAINED) else None
    if ckpt is None:
        print(f"WARNING: {PRETRAINED} not found — training from random init (much worse).")
    model = PerceptionModel(checkpoint_path=ckpt, freeze_backbone=True, device=device).to(device)

    train_dl, val_dl = make_loaders(args)
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.logdir)
        except Exception as e:  # tensorboard optional
            print(f"(tensorboard unavailable: {e})")

    max_tb = 2 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    best_val = float("inf")
    step = 0

    for epoch in range(epochs):
        if epoch == args.unfreeze_epoch and not args.smoke:
            model.unfreeze_backbone()
            print(f"[epoch {epoch}] unfroze backbone (backbone lr={args.backbone_lr})")

        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", leave=False)
        for bi, (x, y) in enumerate(pbar):
            if max_tb and bi >= max_tb:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)
            seen += x.size(0)
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if writer:
                writer.add_scalar("train/loss", loss.item(), step)

        train_mse = running / max(seen, 1)
        val_mse = evaluate(model, val_dl, device, max_batches=max_tb)
        print(f"epoch {epoch}: train_mse={train_mse:.5f}  val_mse={val_mse:.5f}")
        if writer:
            writer.add_scalar("epoch/train_mse", train_mse, epoch)
            writer.add_scalar("epoch/val_mse", val_mse, epoch)

        if val_mse < best_val and not args.smoke:
            best_val = val_mse
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_mse": val_mse,
                 "issue_dimensions": ISSUE_DIMENSIONS},
                BEST_CKPT,
            )
            print(f"  saved {BEST_CKPT} (val_mse={val_mse:.5f})")

    if writer:
        writer.close()
    if args.smoke:
        print("SMOKE OK — train + val loop ran end-to-end.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-seconds", type=float, default=6.0)
    ap.add_argument("--chunks-per-track", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-epoch", type=int, default=1)
    ap.add_argument("--cache-size", type=int, default=128)
    ap.add_argument("--root", type=str, default=None, help="full MUSDB18 root (default: bundled 7s)")
    ap.add_argument("--is-wav", action="store_true", help="set for MUSDB18-HQ (WAV)")
    ap.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--logdir", type=str, default="runs/perception")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the loop on any device")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
