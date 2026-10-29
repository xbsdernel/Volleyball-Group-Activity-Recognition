"""
Baseline 5: Training Script
Trains B5Model (Temporal LSTM on max-pooled person features from off-the-shelf CNN).
Outputs saved to results/4_B5_Temporal_Model_Person_Features/ and logs to logs/train.log.
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

from model import B5Model
from dataset import build_b5_dataloaders


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B5_Train")
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


def train_b5(
    videos_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    tracking_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    results_dir: str = "results/4_B5_Temporal_Model_Person_Features",
    epochs: int = 40,
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

    logger.info("Building person sequence DataLoaders...")
    train_loader, val_loader, test_loader = build_b5_dataloaders(
        videos_root=videos_root, tracking_root=tracking_root,
        batch_size=batch_size, num_workers=num_workers
    )

    b5_model = B5Model(num_classes=8, hidden_dim=500, dropout=0.5).to(device)

    # Train only LSTM and classifier
    trainable_params = list(b5_model.lstm.parameters()) + list(b5_model.classifier.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        b5_model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = b5_model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()
            tr_tot += y.size(0)

        b5_model.eval()
        val_loss, val_corr, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} [Val]", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = b5_model(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_corr += (out.argmax(1) == y).sum().item()
                val_tot += y.size(0)

        scheduler.step()
        ep_tr_loss = tr_loss / max(1, tr_tot)
        ep_tr_acc = tr_corr / max(1, tr_tot) * 100
        ep_val_loss = val_loss / max(1, val_tot)
        ep_val_acc = val_corr / max(1, val_tot) * 100

        history['train_loss'].append(ep_tr_loss)
        history['val_loss'].append(ep_val_loss)
        history['train_acc'].append(ep_tr_acc)
        history['val_acc'].append(ep_val_acc)

        logger.info(
            f"Ep {epoch+1:02d}/{epochs} | "
            f"Train Loss: {ep_tr_loss:.4f} Acc: {ep_tr_acc:.2f}% | "
            f"Val Loss: {ep_val_loss:.4f} Acc: {ep_val_acc:.2f}%"
        )

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            torch.save(b5_model.state_dict(), best_model_path)
            torch.save(b5_model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved best B5 checkpoint (Val Acc: {best_val_acc:.2f}%)")

    # Save artifacts
    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history['train_loss'], label='Train Loss', color='royalblue')
    axs[0].plot(history['val_loss'], label='Val Loss', color='crimson')
    axs[0].set_title('Loss Curves (B5)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(history['train_acc'], label='Train Acc (%)', color='royalblue')
    axs[1].plot(history['val_acc'], label='Val Acc (%)', color='crimson')
    axs[1].set_title('Accuracy Curves (B5)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 5: Temporal Model with Person Features")
    parser.add_argument("--videos_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--tracking_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
    parser.add_argument("--results_dir", type=str, default="results/4_B5_Temporal_Model_Person_Features")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_b5(
        videos_root=args.videos_root,
        tracking_root=args.tracking_root,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
    )
