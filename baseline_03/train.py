"""
Baseline 3: Training Script
Stage 1: Fine-tuning person CNN on 9 action categories (optional).
Stage 2: Training B3 group classifier with MAX-POOLING across all 12 players.
Outputs saved to results/2_B3_Finetuned_Person_Classification/ and logs to logs/train.log.
"""

import os
import random
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import PersonActionCNN, B3Model
from dataset import (
    load_volleyball_metadata, get_crop_transform,
    SinglePlayerCropDataset, Volleyball12PlayersDataset,
    TRAIN_VIDEOS, VAL_VIDEOS, IMAGE_SIZE
)


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B3_Train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_b3(
    videos_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    tracking_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    results_dir: str = "results/2_B3_Finetuned_Person_Classification",
    epochs_stage1: int = 15,
    epochs_stage2: int = 30,
    train_stage1: bool = False,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_workers: int = 2
):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "train.log"))

    logger.info(f"Using device: {device}")

    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, "best_model.pth")
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(local_ckpt_dir, exist_ok=True)
    local_ckpt_path = os.path.join(local_ckpt_dir, "best_model.pth")

    data = load_volleyball_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)

    # Initialize Person Action CNN for Stage 1
    stage1_model = PersonActionCNN(num_actions=9, pretrained=True).to(device)

    if train_stage1:
        logger.info("Starting Stage 1: Fine-tuning Person Action CNN...")
        stage1_train_ds = SinglePlayerCropDataset(train_data, transform=tf)
        stage1_loader = torch.utils.data.DataLoader(
            stage1_train_ds, batch_size=32, shuffle=True,
            num_workers=num_workers, pin_memory=(device == "cuda")
        )
        optimizer1 = optim.AdamW(stage1_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion1 = nn.CrossEntropyLoss()
        scaler1 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        for ep in range(epochs_stage1):
            stage1_model.train()
            s1_loss, s1_corr, s1_tot = 0.0, 0, 0
            for crops, acts in tqdm(stage1_loader, desc=f"Stage 1 Ep {ep+1:02d}/{epochs_stage1}", leave=False):
                crops, acts = crops.to(device, non_blocking=True), acts.to(device, non_blocking=True)
                optimizer1.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = stage1_model(crops)
                    loss = criterion1(out, acts)
                scaler1.scale(loss).backward()
                scaler1.step(optimizer1)
                scaler1.update()
                s1_loss += loss.item() * crops.size(0)
                s1_corr += (out.argmax(1) == acts).sum().item()
                s1_tot += acts.size(0)
            logger.info(f"Stage 1 Ep {ep+1:02d} | Loss: {s1_loss/s1_tot:.4f} Acc: {s1_corr/s1_tot*100:.2f}%")

    # Stage 2: Initialize B3Model with fine-tuned backbone (frozen)
    logger.info("Initializing Stage 2 B3Model with MAX-POOLING over 12 players...")
    b3_model = B3Model(fine_tuned_backbone=stage1_model.backbone, num_classes=8, freeze_backbone=True).to(device)

    b3_train_ds = Volleyball12PlayersDataset(train_data, transform=tf)
    b3_val_ds   = Volleyball12PlayersDataset(val_data, transform=tf)

    loader_kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"), persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(b3_train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = torch.utils.data.DataLoader(b3_val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)

    optimizer2 = optim.AdamW(b3_model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion2 = nn.CrossEntropyLoss()
    scaler2 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    logger.info(f"Starting Stage 2 training for {epochs_stage2} epochs...")
    for epoch in range(epochs_stage2):
        b3_model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{epochs_stage2} [Train]", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer2.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = b3_model(x)
                loss = criterion2(out, y)
            scaler2.scale(loss).backward()
            scaler2.step(optimizer2)
            scaler2.update()
            tr_loss += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()
            tr_tot += y.size(0)

        b3_model.eval()
        val_loss, val_corr, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Ep {epoch+1:02d} [Val]", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = b3_model(x)
                    loss = criterion2(out, y)
                val_loss += loss.item() * x.size(0)
                val_corr += (out.argmax(1) == y).sum().item()
                val_tot += y.size(0)

        scheduler2.step()
        ep_tr_loss = tr_loss / max(1, tr_tot)
        ep_tr_acc = tr_corr / max(1, tr_tot) * 100
        ep_val_loss = val_loss / max(1, val_tot)
        ep_val_acc = val_corr / max(1, val_tot) * 100

        history['train_loss'].append(ep_tr_loss)
        history['val_loss'].append(ep_val_loss)
        history['train_acc'].append(ep_tr_acc)
        history['val_acc'].append(ep_val_acc)

        logger.info(
            f"Epoch {epoch+1:02d}/{epochs_stage2} | "
            f"Train Loss: {ep_tr_loss:.4f} Acc: {ep_tr_acc:.2f}% | "
            f"Val Loss: {ep_val_loss:.4f} Acc: {ep_val_acc:.2f}%"
        )

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            torch.save(b3_model.state_dict(), best_model_path)
            torch.save(b3_model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved best B3 checkpoint (Val Acc: {best_val_acc:.2f}%)")

    # Save history and curves
    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history['train_loss'], label='Train Loss', color='royalblue')
    axs[0].plot(history['val_loss'], label='Val Loss', color='crimson')
    axs[0].set_title('B3 Loss Curve')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(history['train_acc'], label='Train Acc (%)', color='royalblue')
    axs[1].plot(history['val_acc'], label='Val Acc (%)', color='crimson')
    axs[1].set_title('B3 Accuracy Curve')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 3: Fine-tuned Person Classification")
    parser.add_argument("--videos_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--tracking_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
    parser.add_argument("--results_dir", type=str, default="results/2_B3_Finetuned_Person_Classification")
    parser.add_argument("--epochs_stage2", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_stage1", action="store_true")
    args = parser.parse_args()

    train_b3(
        videos_root=args.videos_root,
        tracking_root=args.tracking_root,
        results_dir=args.results_dir,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        train_stage1=args.train_stage1
    )
