import os
import torch
from configs import config


def save_checkpoint(model, optimizer=None, scheduler=None, scaler=None,
                    epoch=0, best_val_loss=float("inf"),
                    checkpoint_dir=config.CHECKPOINT_DIR,
                    checkpoint_name=config.CHECKPOINT_NAME):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    temp_path = checkpoint_path + ".tmp"

    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint_dict = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }

    # Atomic write to protect against session interruption
    torch.save(checkpoint_dict, temp_path)
    os.replace(temp_path, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, checkpoint_path, device, optimizer=None, scheduler=None, scaler=None):
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return model

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        if optimizer is not None and checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix if loading DP/DDP state_dict into non-module model
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    target_model = model.module if hasattr(model, 'module') else model
    target_model.load_state_dict(new_state_dict)

    return model

