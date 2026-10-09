import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from configs import config
from datasets.prepare_dataset import prepare_data
from datasets.dataloaders import get_datasets, get_dataloaders
from models.resnet50 import build_model
from utils.device import get_device
from utils.seed import set_seed


def run_benchmark_steps(model, loader, criterion, optimizer, scaler, device,
                        num_steps=30, warmup_steps=5, use_amp=True, use_channels_last=True):
    model.train()
    step_times = []
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    data_iter = iter(loader)

    # Warmup steps
    for _ in range(warmup_steps):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # Timed benchmarking steps
    start_total_time = time.perf_counter()

    for step in range(num_steps):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)

        t0 = time.perf_counter()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()

        step_times.append(t1 - t0)
        total_samples += labels.size(0)

    end_total_time = time.perf_counter()
    total_duration = end_total_time - start_total_time
    samples_per_sec = total_samples / total_duration
    avg_step_ms = (sum(step_times) / len(step_times)) * 1000

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "samples_per_sec": samples_per_sec,
        "avg_step_ms": avg_step_ms,
        "total_duration": total_duration,
        "peak_vram_mb": peak_vram_mb
    }


def main():
    print("=" * 60)
    print("PyTorch Performance Benchmarking Suite")
    print("=" * 60)

    set_seed(config.SEED)
    train_data, val_data, test_data = prepare_data()
    train_dataset, val_dataset, test_dataset = get_datasets(train_data, val_data, test_data)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected System GPUs: {num_gpus}")

    # Config 1: Baseline Single GPU (FP32, Batch 32, num_workers 2)
    print("\n[Benchmarking] 1. Baseline Single GPU (FP32, Batch 32)...")
    train_loader_base, _, _, _ = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=32, num_workers=2, is_distributed=False
    )
    device = get_device()
    model_base = build_model(device=device, use_channels_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer_base = optim.AdamW(model_base.parameters(), lr=1e-3)

    res_base = run_benchmark_steps(
        model_base, train_loader_base, criterion, optimizer_base, scaler=None,
        device=device, num_steps=30, warmup_steps=5, use_amp=False, use_channels_last=False
    )

    print(f"  Throughput: {res_base['samples_per_sec']:.2f} samples/sec")
    print(f"  Avg Step Time: {res_base['avg_step_ms']:.2f} ms")
    print(f"  Peak VRAM: {res_base['peak_vram_mb']:.2f} MB")

    # Config 2: Optimized Single GPU (AMP FP16, ChannelsLast, Batch 64, cuDNN Benchmark)
    print("\n[Benchmarking] 2. Optimized Single GPU (AMP FP16, ChannelsLast, Batch 64)...")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_loader_opt, _, _, _ = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=64, num_workers=4, is_distributed=False
    )
    model_opt = build_model(device=device, use_channels_last=True)
    optimizer_opt = optim.AdamW(model_opt.parameters(), lr=1e-3)
    scaler_opt = torch.cuda.amp.GradScaler(enabled=True)

    res_opt = run_benchmark_steps(
        model_opt, train_loader_opt, criterion, optimizer_opt, scaler=scaler_opt,
        device=device, num_steps=30, warmup_steps=5, use_amp=True, use_channels_last=True
    )

    print(f"  Throughput: {res_opt['samples_per_sec']:.2f} samples/sec")
    print(f"  Avg Step Time: {res_opt['avg_step_ms']:.2f} ms")
    print(f"  Peak VRAM: {res_opt['peak_vram_mb']:.2f} MB")
    speedup = res_opt['samples_per_sec'] / max(1e-5, res_base['samples_per_sec'])
    print(f"  Speedup vs Baseline: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("Benchmarking Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
