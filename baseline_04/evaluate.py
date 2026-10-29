"""
Baseline 4: Evaluation Script
Evaluates B4Model (ResNet-50 + Temporal LSTM) on test split sequences.
Generates metrics.json, classification_report.csv, and confusion_matrix.png.
Outputs saved to results/3_B4_Temporal_Model_Image_Features/ and logs to logs/evaluate.log.
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

from model import B4Model
from dataset import build_sequence_dataloaders, CLASSES


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("B4_Eval")
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


def evaluate_b4(
    data_root: str = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    results_dir: str = "results/3_B4_Temporal_Model_Image_Features",
    checkpoint_path: str = None,
    batch_size: int = 2,
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

    _, _, test_loader, label_encoder = build_sequence_dataloaders(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers
    )

    model = B4Model(num_classes=len(CLASSES), hidden_size=256, num_layers=2, dropout=0.5, pretrained_backbone=False).to(device)

    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Loaded model checkpoint successfully.")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Running with uninitialized weights.")

    model.eval()
    all_preds_test, all_labels_test = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[B4 Test Evaluation]"):
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.numpy())

    test_accuracy  = accuracy_score(all_labels_test, all_preds_test)
    test_precision = precision_score(all_labels_test, all_preds_test, average='weighted', zero_division=0)
    test_recall    = recall_score(all_labels_test,    all_preds_test, average='weighted', zero_division=0)
    test_f1        = f1_score(all_labels_test,        all_preds_test, average='weighted', zero_division=0)
    test_f1_macro  = f1_score(all_labels_test,        all_preds_test, average='macro', zero_division=0)

    logger.info(f"Test Accuracy        : {test_accuracy * 100:.2f}%")
    logger.info(f"Test F1 (Weighted)  : {test_f1:.4f}")
    logger.info(f"Test F1 (Macro)     : {test_f1_macro:.4f}")
    logger.info(f"Test Precision      : {test_precision:.4f}")
    logger.info(f"Test Recall         : {test_recall:.4f}")

    metrics_dict = {
        "model": "B4 - Temporal Model with Image Features",
        "test_accuracy": float(test_accuracy),
        "test_f1_weighted": float(test_f1),
        "test_f1_macro": float(test_f1_macro),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "checkpoint": checkpoint_path
    }

    metrics_file = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Saved metrics to {metrics_file}")

    rep_dict = classification_report(all_labels_test, all_preds_test, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    rep_csv = os.path.join(metrics_dir, "classification_report.csv")
    pd.DataFrame(rep_dict).transpose().to_csv(rep_csv)
    logger.info(f"Saved classification report to {rep_csv}")

    cm = confusion_matrix(all_labels_test, all_preds_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('B4 Confusion Matrix (Test Split)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline 4: Temporal Model with Image Features")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos")
    parser.add_argument("--results_dir", type=str, default="results/3_B4_Temporal_Model_Image_Features")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    evaluate_b4(
        data_root=args.data_root,
        results_dir=args.results_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
