import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from configs import config
from utils.metrics import compute_correct


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs,
                    scaler=None, use_amp=config.USE_AMP, use_channels_last=config.USE_CHANNELS_LAST,
                    is_main_process=True, is_distributed=False):
    model.train()
    total_acc_train = 0
    total_loss_train = 0.0
    seen_train = 0

    if is_main_process:
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
    else:
        train_bar = train_loader

    for inputs, labels in train_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)


        if scaler is not None and use_amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        batch_loss = train_loss.detach()
        batch_correct = compute_correct(outputs, labels)

        total_loss_train += batch_loss.item() * batch_size
        total_acc_train += batch_correct
        seen_train += batch_size

        if is_main_process and hasattr(train_bar, "set_postfix"):
            train_bar.set_postfix(
                loss=f"{batch_loss.item():.4f}",
                acc=f"{total_acc_train / seen_train * 100:.2f}%",
            )

    if is_distributed and dist.is_initialized():
        stats_tensor = torch.tensor([total_loss_train, float(total_acc_train), float(seen_train)], device=device)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        total_loss_train = stats_tensor[0].item()
        total_acc_train = stats_tensor[1].item()
        seen_train = int(stats_tensor[2].item())

    avg_train_loss = total_loss_train / max(1, seen_train)
    train_accuracy = (total_acc_train / max(1, seen_train)) * 100

    return avg_train_loss, train_accuracy

