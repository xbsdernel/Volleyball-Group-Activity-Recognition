import argparse
import os

import torch
from PIL import Image

from configs import config
from datasets.custom_dataset import label_encoder
from datasets.transforms import get_transforms
from models.resnet50 import build_model
from utils.device import get_device
from utils.checkpoint import load_checkpoint


def predict(image_path, model, device, transform, use_amp=config.USE_AMP, use_channels_last=config.USE_CHANNELS_LAST):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    if use_channels_last:
        image = image.to(memory_format=torch.channels_last)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(image)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    if hasattr(label_encoder, "classes_") and len(label_encoder.classes_) > 0:
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    else:
        predicted_label = predicted_idx

    return predicted_label


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME),
        help="Path to the model checkpoint."
    )
    args = parser.parse_args()

    device = get_device()
    transform = get_transforms()

    model = build_model(device=device, use_channels_last=config.USE_CHANNELS_LAST)

    if os.path.isfile(args.checkpoint):
        model = load_checkpoint(model, args.checkpoint, device)
    else:
        print(f"Warning: checkpoint not found at {args.checkpoint}. "
              f"Using untrained model weights.")

    prediction = predict(args.image, model, device, transform)
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()

