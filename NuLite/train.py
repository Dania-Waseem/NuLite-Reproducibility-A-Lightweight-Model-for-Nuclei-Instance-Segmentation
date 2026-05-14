"""
my_train.py
Baseline training script for NuLite (Experiment 0 / Reproduction).
Reproduces NuLite paper baseline with exact hyperparameters.
Runs for 2 epochs as in the reproduction study.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from my_dataset import PanNukeDataset, compute_total_loss
from my_model import NuLite

# ─── Config (exact NuLite paper hyperparameters) ──────────────────────────────
CONFIG = {
    "data_dir"      : "./data_processed",
    "train_folds"   : [0],
    "val_folds"     : [1],
    "epochs"        : 2,
    "batch_size"    : 4,        # reduced for CPU (paper used 16 on GPU)
    "lr"            : 3e-4,
    "weight_decay"  : 1e-4,
    "betas"         : (0.85, 0.95),
    "scheduler_gamma": 0.85,
    "num_classes"   : 6,
    "log_dir"       : "./logs/baseline",
    "results_dir"   : "./results",
    "checkpoint_name": "baseline_best.pth",
    "experiment"    : "Baseline (2 epoch reproduction)",
}


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, batch in enumerate(loader):
        img        = batch["image"].to(device)
        gt_binary  = batch["binary_map"].to(device)
        gt_hv      = batch["hv_map"].to(device)
        gt_type    = batch["type_map"].to(device)

        optimizer.zero_grad()
        pred_binary, pred_hv, pred_type = model(img)

        loss, loss_dict = compute_total_loss(
            pred_binary, pred_hv, pred_type,
            gt_binary, gt_hv, gt_type
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            avg = total_loss / (i + 1)
            print(f"  Epoch {epoch} [{i+1}/{n_batches}] Loss: {avg:.4f}")

    return total_loss / n_batches


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            img       = batch["image"].to(device)
            gt_binary = batch["binary_map"].to(device)
            gt_hv     = batch["hv_map"].to(device)
            gt_type   = batch["type_map"].to(device)

            pred_binary, pred_hv, pred_type = model(img)
            loss, _ = compute_total_loss(
                pred_binary, pred_hv, pred_type,
                gt_binary, gt_hv, gt_type
            )
            total_loss += loss.item()

    return total_loss / len(loader)


def save_loss_curve(train_losses, val_losses, save_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses,
             "b-o", label="Train Loss", linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses,
             "r-s", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Loss curve saved: {save_path}")


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"NuLite Training — {CONFIG['experiment']}")
    print(f"Device: {device}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"{'='*60}\n")

    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading datasets...")
    train_dataset = PanNukeDataset(
        CONFIG["data_dir"], CONFIG["train_folds"], augment=True)
    val_dataset   = PanNukeDataset(
        CONFIG["data_dir"], CONFIG["val_folds"],   augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(
        val_dataset,   batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False)

    # Model (NuLite-T baseline, no attention gate)
    print("\nInitializing NuLite-T model...")
    model = NuLite(num_classes=CONFIG["num_classes"],
                   use_attention_gate=False).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer: AdamW with exact NuLite paper hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=CONFIG["betas"]
    )

    # Scheduler: Exponential LR with gamma=0.85
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=CONFIG["scheduler_gamma"])

    # Training loop
    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_epoch   = 0

    # CSV log
    log_file = Path(CONFIG["log_dir"]) / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "lr"])

    print("\nStarting training...\n")
    t_start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{CONFIG['epochs']}  LR={lr_now:.6f}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch)
        val_loss   = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            ckpt_path  = Path(CONFIG["log_dir"]) / CONFIG["checkpoint_name"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : val_loss,
                "config"     : CONFIG,
            }, ckpt_path)
            print(f"  ** New best model saved (epoch {epoch})")

        # Log
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, round(train_loss, 4),
                 round(val_loss, 4), round(lr_now, 6)])

        scheduler.step()
        print()

    t_end = time.time()
    print(f"\nTraining complete in {(t_end-t_start)/60:.1f} minutes")
    print(f"Best val loss: {best_val:.4f} at epoch {best_epoch}")

    # Save loss curve
    save_loss_curve(
        train_losses, val_losses,
        Path(CONFIG["results_dir"]) / "baseline_loss_curve.png",
        f"NuLite Baseline Training Loss ({CONFIG['epochs']} Epochs)"
    )

    # Save summary
    summary = {
        "experiment"  : CONFIG["experiment"],
        "epochs"      : CONFIG["epochs"],
        "best_val_loss": round(best_val, 4),
        "best_epoch"  : best_epoch,
        "n_params"    : n_params,
        "train_losses": [round(l, 4) for l in train_losses],
        "val_losses"  : [round(l, 4) for l in val_losses],
    }
    with open(Path(CONFIG["log_dir"]) / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLogs saved to: {CONFIG['log_dir']}")
    print(f"Checkpoint: {Path(CONFIG['log_dir']) / CONFIG['checkpoint_name']}")


if __name__ == "__main__":
    main()