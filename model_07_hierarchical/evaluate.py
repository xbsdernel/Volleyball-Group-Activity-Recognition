"""
Model 7: Evaluation Script
Evaluates TwoStageHierarchicalModel on the test split.
Generates metrics.json, classification_report.csv, and confusion_matrix.png.
Outputs saved to results/7_Model7_Hierarchical_Full/ and logs to logs/evaluate.log.
"""

import os
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from model import TwoStageHierarchicalModel
from dataset import (
    load_hierarchical_metadata, get_crop_transform,
    HierarchicalDataset, TEST_VIDEOS, CLASSES, IMAGE_SIZE
)


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("Model7_Eval")
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


def evaluate_hierarchical(
    videos_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    tracking_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    results_dir: str = "results/7_Model7_Hierarchical_Full",
    checkpoint_path: str = None,
    batch_size: int = 4,
    num_workers: int = 2
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "evaluate.log"))

    logger.info(f"Using device: {device}")

    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    if checkpoint_path is None:
        primary = os.path.join(results_dir, "best_model.pth")
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_model.pth")
        checkpoint_path = primary if os.path.isfile(primary) else fallback

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    data = load_hierarchical_metadata(videos_root, tracking_root)
    test_data = [d for d in data if d["video_id"] in TEST_VIDEOS]

    tf = get_crop_transform(IMAGE_SIZE)
    test_ds = HierarchicalDataset(test_data, transform=tf)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )

    hier_model = TwoStageHierarchicalModel(
        h1=1024, proj=3000, h2=500, num_classes=8, num_actions=9, dropout=0.5, pretrained_backbone=False
    ).to(device)

    if os.path.isfile(checkpoint_path):
        hier_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Loaded hierarchical checkpoint.")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Running with uninitialized weights.")

    hier_model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Hierarchical Test Evaluation]"):
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = hier_model.forward_stage2(inputs)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(labels.numpy())

    test_acc = accuracy_score(targets, preds)
    test_f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)
    test_f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
    test_prec = precision_score(targets, preds, average='weighted', zero_division=0)
    test_rec = recall_score(targets, preds, average='weighted', zero_division=0)

    logger.info(f"Test Accuracy        : {test_acc * 100:.2f}%")
    logger.info(f"Test F1 (Weighted)  : {test_f1_weighted:.4f}")
    logger.info(f"Test F1 (Macro)     : {test_f1_macro:.4f}")
    logger.info(f"Test Precision      : {test_prec:.4f}")
    logger.info(f"Test Recall         : {test_rec:.4f}")

    metrics_dict = {
        "model": "Model #7 - Two-stage Hierarchical Model (Full)",
        "test_accuracy": float(test_acc),
        "test_f1_weighted": float(test_f1_weighted),
        "test_f1_macro": float(test_f1_macro),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "checkpoint": checkpoint_path
    }

    metrics_file = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Saved metrics to {metrics_file}")

    rep_dict = classification_report(targets, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    rep_csv = os.path.join(metrics_dir, "classification_report.csv")
    pd.DataFrame(rep_dict).transpose().to_csv(rep_csv)
    logger.info(f"Saved classification report to {rep_csv}")

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Model #7: Hierarchical Confusion Matrix (Test Split)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model 7: Two-Stage Hierarchical Deep Temporal Model")
    parser.add_argument("--videos_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--tracking_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
    parser.add_argument("--results_dir", type=str, default="results/7_Model7_Hierarchical_Full")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    evaluate_hierarchical(
        videos_root=args.videos_root,
        tracking_root=args.tracking_root,
        results_dir=args.results_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
