"""
Baseline 4: Training Script
Trains Baseline 4 (ResNet-50 + Temporal LSTM on 9-frame sequences).
Supports end-to-end training with gradient accumulation, AMP, and early stopping.
Outputs saved to results/3_B4_Temporal_Model_Image_Features/ and logs to logs/train.log.
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
from sklearn.metrics import f1_score

from model import B4Model
from dataset import build_sequence_dataloaders, CLASSES


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B4_Train")
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


def train_b4(
    data_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    results_dir: str = "results/3_B4_Temporal_Model_Image_Features",
    epochs: int = 40,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.005,
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
    model_save_path = os.path.join(results_dir, "best_model.pth")
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(local_ckpt_dir, exist_ok=True)
    local_ckpt_path = os.path.join(local_ckpt_dir, "best_model.pth")

    logger.info("Building sequence DataLoaders...")
    train_loader, val_loader, test_loader, label_encoder = build_sequence_dataloaders(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers
    )

    model = B4Model(num_classes=len(CLASSES), hidden_size=256, num_layers=2, dropout=0.5).to(device)

    # Backbone parameters receive lower LR, LSTM head gets base LR
    backbone_params = list(model.backbone.parameters())
    lstm_params = list(model.lstm_classifier.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': lstm_params,     'lr': lr}
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
        total_loss_train, correct_train, seen_train = 0.0, 0, 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{epochs} [Train]", leave=False)
        for i, (inputs, labels) in enumerate(train_bar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss_train += loss.item() * gradient_accumulation_steps * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            seen_train += labels.size(0)

        avg_train_loss = total_loss_train / max(1, seen_train)
        avg_train_acc = correct_train / max(1, seen_train) * 100

        # Validate
        model.eval()
        total_loss_val, correct_val, seen_val = 0.0, 0, 0
        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                total_loss_val += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                seen_val += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = total_loss_val / max(1, seen_val)
        avg_val_acc = correct_val / max(1, seen_val) * 100
        val_f1 = f1_score(all_labels_list, all_preds, average='weighted', zero_division=0)

        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['val_f1'].append(val_f1)

        logger.info(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}% | F1: {val_f1:.4f}"
        )

        if val_f1 > (best_val_f1 + early_stop_min_delta):
            best_val_f1 = val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            torch.save(model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved new best B4 checkpoint (Val F1: {best_val_f1:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                logger.info(f"[Early Stopping] Triggered at epoch {epoch+1}.")
                break

    # Save artifacts
    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axs[0].plot(history['val_loss'], label='Val Loss', color='tomato')
    axs[0].set_title('B4 Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(history['train_acc'], label='Train Acc', color='steelblue')
    axs[1].plot(history['val_acc'], label='Val Acc', color='tomato')
    axs[1].set_title('B4 Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 4: Temporal Model with Image Features")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--results_dir", type=str, default="results/3_B4_Temporal_Model_Image_Features")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_b4(
        data_root=args.data_root,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
    )
