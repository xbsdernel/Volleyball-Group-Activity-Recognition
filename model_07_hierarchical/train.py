"""
Model 7: Training Script
Trains TwoStageHierarchicalModel (Full Two-Stage Hierarchical Model from Ibrahim et al. CVPR 2016).
Stage 1: Person CNN + Person LSTM 1 on action labels (optional).
Stage 2: FC projection (3000) + Group LSTM 2 (500) + Group classifier.
Outputs saved to results/7_Model7_Hierarchical_Full/ and logs to logs/train.log.
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

from model import TwoStageHierarchicalModel
from dataset import (
    load_hierarchical_metadata, get_crop_transform,
    PersonTrackletDataset, HierarchicalDataset,
    TRAIN_VIDEOS, VAL_VIDEOS, IMAGE_SIZE
)


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("Model7_Train")
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


def train_hierarchical(
    videos_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    tracking_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    results_dir: str = "results/7_Model7_Hierarchical_Full",
    epochs_s1: int = 15,
    epochs_s2: int = 40,
    train_stage1: bool = False,
    batch_size: int = 4,
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

    data = load_hierarchical_metadata(videos_root, tracking_root)
    train_data = [d for d in data if d["video_id"] in TRAIN_VIDEOS]
    val_data   = [d for d in data if d["video_id"] in VAL_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)

    hier_model = TwoStageHierarchicalModel(
        h1=1024, proj=3000, h2=500, num_classes=8, num_actions=9, dropout=0.5, pretrained_backbone=True
    ).to(device)

    # Optional Stage 1 Training
    if train_stage1:
        logger.info("Starting Stage 1: Person-level tracklet training...")
        stage1_train_ds = PersonTrackletDataset(train_data, transform=tf)
        stage1_loader = torch.utils.data.DataLoader(
            stage1_train_ds, batch_size=16, shuffle=True,
            num_workers=num_workers, pin_memory=(device == "cuda")
        )
        stage1_params = (
            list(hier_model.person_cnn.parameters()) +
            list(hier_model.lstm1.parameters()) +
            list(hier_model.person_action_head.parameters())
        )
        opt1 = optim.AdamW(stage1_params, lr=1e-4, weight_decay=1e-4)
        crit1 = nn.CrossEntropyLoss()
        sc1 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        for ep in range(epochs_s1):
            hier_model.train()
            correct, total = 0, 0
            for tracklets, acts in tqdm(stage1_loader, desc=f"Stage 1 Ep {ep+1:02d}", leave=False):
                tracklets, acts = tracklets.to(device, non_blocking=True), acts.to(device, non_blocking=True)
                opt1.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = hier_model.forward_stage1(tracklets)
                    B, T, C_act = logits.shape
                    loss = crit1(logits.view(B * T, C_act), acts.unsqueeze(1).expand(B, T).reshape(B * T))
                sc1.scale(loss).backward()
                sc1.step(opt1)
                sc1.update()
                correct += (logits[:, -1, :].argmax(1) == acts).sum().item()
                total += B
            logger.info(f"Stage 1 Ep {ep+1:02d} | Action Acc: {correct / max(1, total) * 100:.2f}%")

    # Freeze Stage 1 components for Stage 2
    logger.info("Freezing Stage 1 components (Person CNN + LSTM 1) for Stage 2...")
    for p in hier_model.person_cnn.parameters():
        p.requires_grad = False
    for p in hier_model.lstm1.parameters():
        p.requires_grad = False
    for p in hier_model.person_action_head.parameters():
        p.requires_grad = False

    stage2_params = (
        list(hier_model.fc_proj.parameters()) +
        list(hier_model.lstm2.parameters()) +
        list(hier_model.group_classifier.parameters())
    )

    s2_train_ds = HierarchicalDataset(train_data, transform=tf)
    s2_val_ds   = HierarchicalDataset(val_data,   transform=tf)

    loader_kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"), persistent_workers=(num_workers > 0))
    train_loader = torch.utils.data.DataLoader(s2_train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = torch.utils.data.DataLoader(s2_val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)

    optimizer2 = optim.AdamW(stage2_params, lr=lr, weight_decay=1e-4)
    criterion2 = nn.CrossEntropyLoss()
    scaler2 = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_s2, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info(f"Starting Stage 2 hierarchical training for {epochs_s2} epochs...")
    for epoch in range(epochs_s2):
        hier_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{epochs_s2} [Train Hier]", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer2.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = hier_model.forward_stage2(inputs)
                loss = criterion2(out, labels)
            scaler2.scale(loss).backward()
            scaler2.step(optimizer2)
            scaler2.update()
            total_loss += loss.item() * inputs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = total_loss / max(1, total), correct / max(1, total) * 100

        hier_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Ep {epoch+1:02d} [Val]", leave=False):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = hier_model.forward_stage2(inputs)
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

        logger.info(f"Ep {epoch+1:02d}/{epochs_s2} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(hier_model.state_dict(), best_model_path)
            torch.save(hier_model.state_dict(), local_ckpt_path)
            logger.info(f"  --> Saved best hierarchical checkpoint (Val Acc: {best_val_acc:.2f}%)")

    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, "training_history.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
    axs[0].plot(history["val_loss"], label="Val Loss", color="crimson")
    axs[0].set_title("Hierarchical Model — Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["train_acc"], label="Train Acc", color="royalblue")
    axs[1].plot(history["val_acc"], label="Val Acc", color="crimson")
    axs[1].set_title("Hierarchical Model — Accuracy")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.suptitle("Model #7: Two-Stage Hierarchical Deep Temporal Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info("Hierarchical model training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 7: Two-Stage Hierarchical Deep Temporal Model")
    parser.add_argument("--videos_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--tracking_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
    parser.add_argument("--results_dir", type=str, default="results/7_Model7_Hierarchical_Full")
    parser.add_argument("--epochs_s2", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_stage1", action="store_true")
    args = parser.parse_args()

    train_hierarchical(
        videos_root=args.videos_root,
        tracking_root=args.tracking_root,
        results_dir=args.results_dir,
        epochs_s2=args.epochs_s2,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        train_stage1=args.train_stage1
    )
