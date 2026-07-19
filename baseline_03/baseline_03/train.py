import torch
import torch.nn as nn
import torch.optim as optim

from configs import config
from datasets.prepare_dataset import prepare_data
from datasets.dataloaders import get_datasets, get_dataloaders
from models.b3_model import build_model
from engine.train import train_one_epoch
from engine.validate import validate_one_epoch
from utils.device import get_device
from utils.plots import plot_metrics
from utils.checkpoint import save_checkpoint
from utils.helpers import ensure_dir


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )

    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []
    total_f1_validation_plot = []

    ensure_dir(config.CHECKPOINT_DIR)
    ensure_dir("plots")

    for epoch in range(config.EPOCHS):
        avg_train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config.EPOCHS
        )

        avg_val_loss, val_accuracy, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device, epoch, config.EPOCHS
        )

        scheduler.step()

        total_loss_train_plot.append(round(avg_train_loss, 4))
        total_loss_validation_plot.append(round(avg_val_loss, 4))
        total_acc_train_plot.append(round(train_accuracy, 4))
        total_acc_validation_plot.append(round(val_accuracy, 4))
        total_f1_validation_plot.append(round(val_f1, 4))

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS}, "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Validation Loss: {avg_val_loss:.4f} "
            f"Validation Accuracy: {val_accuracy:.2f}%"
            f"Validation F1: {val_f1:.4f}"
        )
        print("=" * 25)

    save_checkpoint(model)

    plot_metrics(
        total_loss_train_plot,
        total_loss_validation_plot,
        total_acc_train_plot,
        total_acc_validation_plot,
        total_f1_validation_plot,
        save_path="plots/training_curves.png"
    )

    return model, test_loader, test_dataset


if __name__ == "__main__":
    main()
