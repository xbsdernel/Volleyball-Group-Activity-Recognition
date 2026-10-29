"""
Baseline 6: Training Script
Trains Baseline 6: Two-stage model without LSTM 1.
Frozen fine-tuned person CNN -> max-pool over players per frame -> Group LSTM 2.
Outputs saved to results/5_B6_Two_Stage_No_LSTM1/ and logs to logs/train.log.
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

from model import PersonActionCNN, B6Model
from dataset import (
    load_b6_metadata, get_crop_transform,
    SinglePlayerCropDataset, VolleyballSequencePersonDataset,
    TRAIN_VIDEOS, VAL_VIDEOS, IMAGE_SIZE
)


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B6_Train")
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


def train_b6(
    videos_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    tracking_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    results_dir: str = "results/5_B6_Two_Stage_No_LSTM1",
    epochs_stage1: int = 15,
    epochs_stage2: int = 40,
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

    data = load_b6_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)
    stage1_model = PersonActionCNN(num_actions=9, pretrained=True).to(device)

    if train_stage1:
        logger.info("Training Stage 1 (Person Action CNN)...")
        stage1_ds = SinglePlayerCropDataset(train_data, transform=tf)
        stage1_loader = torch.utils.data.DataLoader(stage1_ds, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
        opt1 = optim.AdamW(stage1_model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit1 = nn.CrossEntropyLoss()
        sc1 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        for ep in range(epochs_stage1):
            stage1_model.train()
            for crops, acts in tqdm(stage1_loader, desc=f"Stage 1 Ep {ep+1:02d}", leave=False):
                crops, acts = crops.to(device), acts.to(device)
                opt1.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    loss = crit1(stage1_model(crops), acts)
                sc1.scale(loss).backward()
                sc1.step(opt1)
                sc1.update()

    # Stage 2: Initialize B6Model
    logger.info("Initializing Stage 2 B6Model (Group LSTM 2 with frozen fine-tuned backbone)...")
    b6_model = B6Model(finetuned_backbone=stage1_model.backbone, num_classes=8, freeze_backbone=True).to(device)

    b6_train_ds = VolleyballSequencePersonDataset(train_data, transform=tf)
    b6_val_ds   = VolleyballSequencePersonDataset(val_data,   transform=tf)

    loader_kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"), persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(b6_train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = torch.utils.data.DataLoader(b6_val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)

    optimizer2 = optim.AdamW(
        list(b6_model.lstm2.parameters()) + list(b6_model.classifier.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion2 = nn.CrossEntropyLoss()
    scaler2 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info(f"Starting Stage 2 training for {epochs_stage2} epochs...")
    for epoch in range(epochs_stage2):
        b6_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{epochs_stage2} [Train]", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer2.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = b6_model(inputs)
                loss = criterion2(out, labels)
            scaler2.scale(loss).backward()
            scaler2.step(optimizer2)
            scaler2.update()
            total_loss += loss.item() * inputs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = total_loss / max(1, total), correct / max(1, total) * 100

        b6_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Ep {epoch+1:02d} [Val]", leave=False):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = b6_model(inputs)
                    loss = criterion2(out, labels)
                total_loss += loss.item() * inputs.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_loss, val_acc = total_loss / max(1, total), correct / max(1, total) * 100
        scheduler2.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch {epoch+1:02d}/{epochs_stage2} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(b6_model.state_dict(), best_model_path)
            torch.save(b6_model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved best B6 checkpoint (Val Acc: {best_val_acc:.2f}%)")

    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
    axs[0].plot(history["val_loss"], label="Val Loss", color="crimson")
    axs[0].set_title("B6 Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["train_acc"], label="Train Acc", color="royalblue")
    axs[1].plot(history["val_acc"], label="Val Acc", color="crimson")
    axs[1].set_title("B6 Accuracy")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 6: Two-stage model without LSTM 1")
    parser.add_argument("--videos_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--tracking_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
    parser.add_argument("--results_dir", type=str, default="results/5_B6_Two_Stage_No_LSTM1")
    parser.add_argument("--epochs_stage2", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_stage1", action="store_true")
    args = parser.parse_args()

    train_b6(
        videos_root=args.videos_root,
        tracking_root=args.tracking_root,
        results_dir=args.results_dir,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        train_stage1=args.train_stage1
    )
