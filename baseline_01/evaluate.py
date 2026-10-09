import os

from configs import config
from datasets.prepare_dataset import prepare_data
from datasets.dataloaders import get_datasets, get_dataloaders
from models.resnet50 import build_model
from engine.test import evaluate_test
from utils.device import get_device
from utils.checkpoint import load_checkpoint


def main():
    device = get_device()

    train_data, val_data, test_data = prepare_data()
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_data, val_data, test_data
    )
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    model = build_model(device=device, use_channels_last=config.USE_CHANNELS_LAST)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    if os.path.isfile(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, device)
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}. "
              f"Evaluating with current model weights.")

    evaluate_test(model, test_loader, test_dataset, device,
                  use_amp=config.USE_AMP, use_channels_last=config.USE_CHANNELS_LAST)


if __name__ == "__main__":
    main()

