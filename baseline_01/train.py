import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from configs import config
from datasets.prepare_dataset import prepare_data
from datasets.dataloaders import get_datasets, get_dataloaders
from models.resnet50 import build_model
from engine.train import train_one_epoch
from engine.validate import validate_one_epoch
from utils.device import get_device
from utils.early_stopping import EarlyStopping
from utils.plots import plot_metrics
from utils.helpers import ensure_dir
from utils.seed import set_seed


def train_worker(rank, world_size, train_data, val_data, test_data):
    is_distributed = world_size > 1

    if is_distributed:
        backend = "gloo" if sys.platform == "win32" else "nccl"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(get_device())

    is_main_process = (rank == 0)

    if is_main_process:
        set_seed(config.SEED)
        print(f"[Info] Running training with World Size = {world_size} on device: {device}")
        print(f"[Info] Optimizations enabled: AMP={config.USE_AMP}, ChannelsLast={config.USE_CHANNELS_LAST}, "
              f"cuDNN Benchmark={config.CUDNN_BENCHMARK}, DDP={is_distributed}")

    if config.CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Prepare datasets and dataloaders
    train_dataset, val_dataset, test_dataset = get_datasets(train_data, val_data, test_data)
    train_loader, val_loader, test_loader, train_sampler = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size
    )

    # Build model and wrap in DDP if distributed
    model = build_model(device=device, use_channels_last=config.USE_CHANNELS_LAST)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Loss, optimizer, scheduler, AMP scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    # Metric tracking
    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    early_stopping = EarlyStopping(patience=config.PATIENCE)

    if is_main_process:
        ensure_dir(config.CHECKPOINT_DIR)
        ensure_dir("plots")

    for epoch in range(config.EPOCHS):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=config.EPOCHS,
            scaler=scaler,
            use_amp=config.USE_AMP,
            use_channels_last=config.USE_CHANNELS_LAST,
            is_main_process=is_main_process,
            is_distributed=is_distributed
        )

        avg_val_loss, val_accuracy = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=config.EPOCHS,
            use_amp=config.USE_AMP,
            use_channels_last=config.USE_CHANNELS_LAST,
            is_main_process=is_main_process,
            is_distributed=is_distributed
        )

        scheduler.step()

        if is_main_process:
            total_loss_train_plot.append(round(avg_train_loss, 4))
            total_loss_validation_plot.append(round(avg_val_loss, 4))
            total_acc_train_plot.append(round(train_accuracy, 4))
            total_acc_validation_plot.append(round(val_accuracy, 4))

            print(
                f"Epoch {epoch + 1}/{config.EPOCHS}, "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Train Accuracy: {train_accuracy:.2f}% "
                f"Validation Loss: {avg_val_loss:.4f} "
                f"Validation Accuracy: {val_accuracy:.2f}%"
            )
            print("=" * 25)

        should_stop = early_stopping.step(
            avg_val_loss=avg_val_loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            is_main_process=is_main_process
        )

        # Synchronize early stopping decision across DDP ranks
        if is_distributed:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            if stop_tensor.item() == 1:
                break
        elif should_stop:
            break

    if is_main_process:
        plot_metrics(
            total_loss_train_plot,
            total_loss_validation_plot,
            total_acc_train_plot,
            total_acc_validation_plot,
            save_path="plots/training_curves.png"
        )

    if is_distributed:
        dist.destroy_process_group()

    return model, test_loader, test_dataset


def main():
    set_seed(config.SEED)
    train_data, val_data, test_data = prepare_data()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if config.USE_DDP and num_gpus > 1:
        world_size = num_gpus
        print(f"[Main] Spawning {world_size} DDP worker processes...")
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, train_data, val_data, test_data))
    else:
        print("[Main] Running single-process execution...")
        train_worker(0, 1, train_data, val_data, test_data)


if __name__ == "__main__":
    main()

