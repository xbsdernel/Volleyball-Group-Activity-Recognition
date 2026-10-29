"""
Baseline 1: Training Script
Trains ResNet-50 on single middle frames using AMP, AdamW, differential learning rates,
and Cosine Annealing with Warmup scheduler.
Outputs artifacts to results/1_B1_Image_Classification/ and logs to logs/train.log.
"""

import os
import random
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from model import B1ImageClassifier
from dataset import build_dataloaders, CLASSES


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B1_Train")
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train_b1(
    data_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    results_dir: str = "results/1_B1_Image_Classification",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    early_stop_patience: int = 7,
    early_stop_min_delta: float = 0.005,
    num_workers: int = 2
):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "train.log"))

    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    model_save_path = os.path.join(results_dir, "best_model.pth")
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(local_ckpt_dir, exist_ok=True)
    local_ckpt_path = os.path.join(local_ckpt_dir, "best_model.pth")

    logger.info("Building datasets and DataLoaders...")
    train_loader, val_loader, test_loader, label_encoder = build_dataloaders(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers
    )

    logger.info(f"Classes ({len(CLASSES)}): {CLASSES}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = B1ImageClassifier(num_classes=len(CLASSES), pretrained=True, dropout=0.3).to(device)

    # Differential learning rates: 10x lower for pretrained backbone
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    head_params = list(model.backbone.fc.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params,     'lr': lr}
    ], weight_decay=1e-4)

    _warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    _cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[_warmup, _cosine], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0
    early_stop_counter = 0

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss_train, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_train += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss_train / max(1, total_train)
        avg_train_acc = correct_train / max(1, total_train) * 100

        # Validation
        model.eval()
        total_loss_val, correct_val, total_val = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                total_loss_val += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_loss_val / max(1, total_val)
        avg_val_acc = correct_val / max(1, total_val) * 100
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['val_f1'].append(val_f1)

        logger.info(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}% F1: {val_f1:.4f}"
        )

        # Early stopping and model checkpointing
        if val_f1 > (best_val_f1 + early_stop_min_delta):
            best_val_f1 = val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            torch.save(model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved new best checkpoint (Val F1: {best_val_f1:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                logger.info(f"[Early Stopping] No improvement for {early_stop_patience} consecutive epochs.")
                break

    # Save training history & curves
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history['train_loss'], label='Train Loss', color='royalblue')
    axs[0].plot(history['val_loss'], label='Val Loss', color='crimson')
    axs[0].set_title('B1 Loss Curve')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(history['train_acc'], label='Train Acc', color='royalblue')
    axs[1].plot(history['val_acc'], label='Val Acc', color='crimson')
    axs[1].plot([f * 100 for f in history['val_f1']], label='Val F1 (%)', color='forestgreen', linestyle='--')
    axs[1].set_title('B1 Accuracy & F1 Curve')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Percentage (%)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info(f"Training completed. Checkpoints and curves saved to {results_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 1: Single Image Classification")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--results_dir", type=str, default="results/1_B1_Image_Classification")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_b1(
        data_root=args.data_root,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
    )
