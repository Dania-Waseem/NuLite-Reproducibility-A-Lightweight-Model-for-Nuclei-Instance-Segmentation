"""
my_train_exp2.py
Experiment 2: Attention Gate Architecture Modification (15 epochs).
Adds AttentionGate module to final skip connection of NuLite decoder.
Uses same loss functions as baseline (no boundary loss).
As described in Section III.D of the submitted paper.
"""

import torch
from torch.utils.data import DataLoader
import csv
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from my_dataset import PanNukeDataset, compute_total_loss
from my_model import NuLite

CONFIG = {
    "data_dir"       : "./data_processed",
    "train_folds"    : [0],
    "val_folds"      : [1],
    "epochs"         : 1, 
    "batch_size"     : 4,
    "lr"             : 3e-4,
    "weight_decay"   : 1e-4,
    "betas"          : (0.85, 0.95),
    "scheduler_gamma": 0.85,
    "num_classes"    : 6,
    "log_dir"        : "./logs/exp2",
    "results_dir"    : "./results",
    "checkpoint_name": "exp2_best.pth",
    "experiment"     : "Experiment 2: Attention Gate (15 epochs)",
}


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n = len(loader)
    for i, batch in enumerate(loader):
        img       = batch["image"].to(device)
        gt_binary = batch["binary_map"].to(device)
        gt_hv     = batch["hv_map"].to(device)
        gt_type   = batch["type_map"].to(device)

        optimizer.zero_grad()
        pred_binary, pred_hv, pred_type = model(img)
        loss, _ = compute_total_loss(
            pred_binary, pred_hv, pred_type,
            gt_binary, gt_hv, gt_type)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Epoch {epoch} [{i+1}/{n}] Loss: {total_loss/(i+1):.4f}")

    return total_loss / n


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
                gt_binary, gt_hv, gt_type)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"NuLite — {CONFIG['experiment']}")
    print(f"Device: {device} | Epochs: {CONFIG['epochs']}")
    print(f"Attention Gate: ENABLED at final skip connection")
    print(f"{'='*60}\n")

    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)

    train_ds = PanNukeDataset(
        CONFIG["data_dir"], CONFIG["train_folds"], augment=True)
    val_ds   = PanNukeDataset(
        CONFIG["data_dir"], CONFIG["val_folds"],   augment=False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    # NuLite with Attention Gate enabled
    model = NuLite(num_classes=CONFIG["num_classes"],
                   use_attention_gate=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"], betas=CONFIG["betas"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=CONFIG["scheduler_gamma"])

    train_losses, val_losses = [], []
    best_val, best_epoch = float("inf"), 0

    log_file = Path(CONFIG["log_dir"]) / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","lr"])

    print("Starting Experiment 2 training...\n")
    t0 = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{CONFIG['epochs']}  LR={lr_now:.6f}")

        tl = train_one_epoch(model, train_loader, optimizer, device, epoch)
        vl = validate(model, val_loader, device)
        train_losses.append(tl)
        val_losses.append(vl)

        print(f"  Train: {tl:.4f}  Val: {vl:.4f}")

        if vl < best_val:
            best_val, best_epoch = vl, epoch
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": vl, "config": CONFIG,
            }, Path(CONFIG["log_dir"]) / CONFIG["checkpoint_name"])
            print(f"  ** Best model saved (epoch {epoch})")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, round(tl,4), round(vl,4), round(lr_now,6)])

        scheduler.step()
        print()

    print(f"Done in {(time.time()-t0)/60:.1f} min | Best val: {best_val:.4f} @ ep {best_epoch}")

    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(train_losses)+1), train_losses, "b-o", label="Train")
    plt.plot(range(1,len(val_losses)+1),   val_losses,   "r-s", label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Experiment 2: Attention Gate (15 Epochs)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(Path(CONFIG["results_dir"]) / "exp2_loss_curve.png", dpi=120)
    plt.close()

    with open(Path(CONFIG["log_dir"]) / "training_summary.json", "w") as f:
        json.dump({"experiment": CONFIG["experiment"],
                   "epochs": CONFIG["epochs"],
                   "best_val_loss": round(best_val,4),
                   "best_epoch": best_epoch,
                   "n_params": n_params,
                   "train_losses": [round(l,4) for l in train_losses],
                   "val_losses":   [round(l,4) for l in val_losses]}, f, indent=2)

    print(f"Logs: {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()