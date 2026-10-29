"""
Baseline 7: Training Script
Trains Baseline 7: Two-stage model without LSTM 2.
Person CNN + LSTM 1 (frozen) -> max-pool last hidden state over 12 players -> Linear classifier.
Outputs saved to results/6_B7_Two_Stage_No_LSTM2/ and logs to logs/train.log.
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

from model import PersonLSTM1, B7Model
from dataset import (
    load_b7_metadata, get_crop_transform,
    PersonCropDataset, HierarchicalDataset,
    TRAIN_VIDEOS, VAL_VIDEOS, IMAGE_SIZE
)


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B7_Train")
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


def train_b7(
    videos_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    tracking_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    results_dir: str = "results/6_B7_Two_Stage_No_LSTM2",
    epochs_stage1: int = 15,
    epochs_stage2: int = 40,
    train_stage1: bool = False,
    batch_size: int = 8,
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

    data = load_b7_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)
    person_lstm1 = PersonLSTM1(num_actions=9, hidden_dim=1024, pretrained=True).to(device)

    # Initialize Stage 2 B7Model with frozen PersonLSTM1
    logger.info("Initializing Stage 2 B7Model (Classifier on max-pooled LSTM 1 representations)...")
    b7_model = B7Model(person_lstm1=person_lstm1, num_classes=8, hidden_dim=1024, freeze_stage1=True).to(device)

    b7_train_ds = HierarchicalDataset(train_data, transform=tf)
    b7_val_ds   = HierarchicalDataset(val_data,   transform=tf)

    loader_kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"), persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(b7_train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = torch.utils.data.DataLoader(b7_val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)

    optimizer2 = optim.AdamW(b7_model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion2 = nn.CrossEntropyLoss()
    scaler2 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info(f"Starting Stage 2 training for {epochs_stage2} epochs...")
    for epoch in range(epochs_stage2):
        b7_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{epochs_stage2} [Train]", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer2.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = b7_model(inputs)
                loss = criterion2(out, labels)
            scaler2.scale(loss).backward()
            scaler2.step(optimizer2)
            scaler2.update()
            total_loss += loss.item() * inputs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = total_loss / max(1, total), correct / max(1, total) * 100

        b7_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Ep {epoch+1:02d} [Val]", leave=False):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = b7_model(inputs)
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
            torch.save(b7_model.state_dict(), best_model_path)
            torch.save(b7_model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved best B7 checkpoint (Val Acc: {best_val_acc:.2f}%)")

    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
    axs[0].plot(history["val_loss"], label="Val Loss", color="crimson")
    axs[0].set_title("B7 Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["train_acc"], label="Train Acc", color="royalblue")
    axs[1].plot(history["val_acc"], label="Val Acc", color="crimson")
    axs[1].set_title("B7 Accuracy")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 7: Two-stage model without LSTM 2")
    parser.add_argument("--videos_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--tracking_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
    parser.add_argument("--results_dir", type=str, default="results/6_B7_Two_Stage_No_LSTM2")
    parser.add_argument("--epochs_stage2", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_stage1", action="store_true")
    args = parser.parse_args()

    train_b7(
        videos_root=args.videos_root,
        tracking_root=args.tracking_root,
        results_dir=args.results_dir,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        train_stage1=args.train_stage1
    )
