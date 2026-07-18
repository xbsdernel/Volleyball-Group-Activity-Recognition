import os

import torch

from configs import config


def save_checkpoint(model, checkpoint_dir=config.CHECKPOINT_DIR,
                     checkpoint_name=config.CHECKPOINT_NAME):

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, checkpoint_path, device):

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model
