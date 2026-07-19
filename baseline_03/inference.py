import argparse
import os

import torch
from PIL import Image

from configs import config
from datasets.tracking import load_tracking_annot
from datasets.transforms import get_transforms
from models.b3_model import build_model
from utils.device import get_device
from utils.checkpoint import load_checkpoint


def predict(image_path, tracking_path, model, device, transform):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    frame_boxes = load_tracking_annot(tracking_path)
    frame_id = list(frame_boxes.keys())[0]

    player_crops = []

    for box_info in frame_boxes[frame_id][:config.NUM_PLAYERS]:
        x1, y1, x2, y2 = box_info.box
        crop = image.crop((x1, y1, x2, y2))
        crop = transform(crop)
        player_crops.append(crop)

    while len(player_crops) < config.NUM_PLAYERS:
        player_crops.append(torch.zeros(3, *config.IMAGE_SIZE))

    player_crops = torch.stack(player_crops).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(player_crops)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    inverse_label_map = {v: k for k, v in config.LABEL_MAP.items()}
    predicted_label = inverse_label_map[predicted_idx]

    return predicted_label


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--tracking", type=str, required=True, help="Path to the tracking annotation file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME),
        help="Path to the model checkpoint."
    )
    args = parser.parse_args()

    device = get_device()
    transform = get_transforms()

    model = build_model(device)

    if os.path.isfile(args.checkpoint):
        model = load_checkpoint(model, args.checkpoint, device)
    else:
        print(f"Warning: checkpoint not found at {args.checkpoint}. "
              f"Using untrained model weights.")

    prediction = predict(args.image, args.tracking, model, device, transform)
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
