import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from configs import config
from utils.metrics import compute_correct


def validate_one_epoch(model, val_loader, criterion, device, epoch, epochs,
                       use_amp=config.USE_AMP, use_channels_last=config.USE_CHANNELS_LAST,
                       is_main_process=True, is_distributed=False):
    model.eval()
    total_loss_val = 0.0
    total_acc_val = 0
    seen_val = 0

    if is_main_process:
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False)
    else:
        val_bar = val_loader

    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)


            batch_size = labels.size(0)
            batch_loss = val_loss.detach()
            batch_correct = compute_correct(outputs, labels)

            total_loss_val += batch_loss.item() * batch_size
            total_acc_val += batch_correct
            seen_val += batch_size

            if is_main_process and hasattr(val_bar, "set_postfix"):
                val_bar.set_postfix(
                    loss=f"{batch_loss.item():.4f}",
                    acc=f"{total_acc_val / seen_val * 100:.2f}%",
                )

    if is_distributed and dist.is_initialized():
        stats_tensor = torch.tensor([total_loss_val, float(total_acc_val), float(seen_val)], device=device)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        total_loss_val = stats_tensor[0].item()
        total_acc_val = stats_tensor[1].item()
        seen_val = int(stats_tensor[2].item())

    avg_val_loss = total_loss_val / max(1, seen_val)
    val_accuracy = (total_acc_val / max(1, seen_val)) * 100

    return avg_val_loss, val_accuracy

