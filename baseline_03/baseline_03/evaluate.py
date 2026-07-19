import os

from configs import config
from datasets.prepare_dataset import prepare_data
from datasets.dataloaders import get_datasets, get_dataloaders
from models.b3_model import build_model
from engine.test import evaluate_test
from utils.device import get_device
from utils.checkpoint import load_checkpoint
from utils.plots import plot_confusion_matrix


def main():
    device = get_device()

    train_data, val_data, test_data = prepare_data()
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_data, val_data, test_data
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    model = build_model(device)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    if os.path.isfile(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, device)
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}. "
              f"Evaluating with current model weights.")

    test_accuracy, test_f1, true_labels, predicted_labels = evaluate_test(
        model, test_loader, test_dataset, device
    )

    plot_confusion_matrix(
        true_labels,
        predicted_labels,
        config.LABEL_MAP,
        save_path="plots/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
