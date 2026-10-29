"""
Baseline 1: Evaluation Script
Evaluates trained ResNet-50 on the test split.
Generates metrics.json, classification_report.csv, and confusion_matrix.png.
Outputs saved to results/1_B1_Image_Classification/ and logs to logs/evaluate.log.
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

from model import B1ImageClassifier
from dataset import build_dataloaders, CLASSES


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B1_Eval")
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


def evaluate_b1(
    data_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    results_dir: str = "results/1_B1_Image_Classification",
    checkpoint_path: str = None,
    batch_size: int = 64,
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

    _, _, test_loader, label_encoder = build_dataloaders(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers
    )

    model = B1ImageClassifier(num_classes=len(CLASSES), pretrained=False, dropout=0.3).to(device)
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Successfully loaded model checkpoint.")
    else:
        logger.warning(f"Checkpoint {checkpoint_path} not found. Running with initial weights.")

    model.eval()
    test_preds, test_targets = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Test Split Evaluation]"):
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1_weighted = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
    test_f1_macro = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    test_prec = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
    test_rec = recall_score(test_targets, test_preds, average='weighted', zero_division=0)

    logger.info(f"Test Accuracy        : {test_acc * 100:.2f}%")
    logger.info(f"Test F1 (Weighted)  : {test_f1_weighted:.4f}")
    logger.info(f"Test F1 (Macro)     : {test_f1_macro:.4f}")
    logger.info(f"Test Precision      : {test_prec:.4f}")
    logger.info(f"Test Recall         : {test_rec:.4f}")

    metrics_dict = {
        "model": "B1 - Image Classification",
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

    rep_dict = classification_report(test_targets, test_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    rep_csv = os.path.join(metrics_dir, "classification_report.csv")
    pd.DataFrame(rep_dict).transpose().to_csv(rep_csv)
    logger.info(f"Saved classification report to {rep_csv}")

    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('B1 Confusion Matrix (Test Split)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline 1: Single Image Classification")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--results_dir", type=str, default="results/1_B1_Image_Classification")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    evaluate_b1(
        data_root=args.data_root,
        results_dir=args.results_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
