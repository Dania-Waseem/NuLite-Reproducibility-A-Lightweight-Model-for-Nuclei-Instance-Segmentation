"""
my_train_exp1.py
Experiment 1: Loss Function Modification (20 epochs).
Modifications from NuLite paper:
1. Emphasized FocalTversky Loss in binary branch (alpha=0.7, beta=0.3)
2. Custom morphological Boundary Loss added to HV map branch
As described in Section III.C of the submitted paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import csv
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from my_dataset import PanNukeDataset, dice_loss, focal_tversky_loss
from my_dataset import mc_focal_tversky_loss, mse_loss_maps, msge_loss_maps
from my_dataset import xentropy_loss
from my_model import NuLite

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir"       : "./data_processed",
    "train_folds"    : [0],
    "val_folds"      : [1],
    "epochs"         : 20,
    "batch_size"     : 4,
    "lr"             : 3e-4,
    "weight_decay"   : 1e-4,
    "betas"          : (0.85, 0.95),
    "scheduler_gamma": 0.85,
    "num_classes"    : 6,
    "log_dir"        : "./logs/exp1",
    "results_dir"    : "./results",
    "checkpoint_name": "exp1_best.pth",
    "experiment"     : "Experiment 1: Loss Modification (20 epochs)",
    # Experiment 1 specific
    "ft_alpha"       : 0.7,   # emphasize FN (missed nuclei)
    "ft_beta"        : 0.3,
    "boundary_lambda": 1.0,   # weight for boundary loss
}


# ─── Boundary Loss (Experiment 1 modification) ────────────────────────────────

def morphological_boundary(binary_map, kernel_size=3):
    """
    Compute boundary mask B = Dilate(M, k) - Erode(M, k)
    using max-pooling for dilation and min-pooling for erosion.
    As in Equation 5 of the paper.
    """
    pad = kernel_size // 2
    # Dilation via max pooling
    dilated = F.max_pool2d(
        binary_map, kernel_size=kernel_size,
        stride=1, padding=pad)
    # Erosion via min pooling (negate, max pool, negate)
    eroded = -F.max_pool2d(
        -binary_map, kernel_size=kernel_size,
        stride=1, padding=pad)
    boundary = dilated - eroded
    return boundary.clamp(0, 1)


def boundary_loss(pred_hv, gt_binary):
    """
    Morphological Boundary Loss for HV map branch.
    BCE between predicted boundary and true boundary.
    Equation 6 of the paper:
    L_boundary = -1/|B| * sum[B*log(B_hat) + (1-B)*log(1-B_hat)]
    """
    # Compute true boundary from ground truth binary map
    B_true = morphological_boundary(gt_binary)  # [B, 1, 256, 256]

    # Use magnitude of HV prediction as boundary prediction
    # |HV| is large at boundaries where distance maps change rapidly
    hv_mag = torch.sqrt(
        pred_hv[:, 0:1] ** 2 + pred_hv[:, 1:2] ** 2)  # [B,1,256,256]
    # Normalize to [0,1]
    hv_max = hv_mag.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-6
    B_pred  = hv_mag / hv_max

    # BCE only at boundary pixels
    B_true_flat = B_true.view(-1)
    B_pred_flat = B_pred.view(-1).clamp(1e-6, 1 - 1e-6)

    boundary_mask = B_true_flat > 0.1
    if boundary_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_hv.device)

    loss = -(
        B_true_flat[boundary_mask] * torch.log(B_pred_flat[boundary_mask]) +
        (1 - B_true_flat[boundary_mask]) *
        torch.log(1 - B_pred_flat[boundary_mask])
    ).mean()
    return loss


def compute_exp1_loss(pred_binary, pred_hv, pred_type,
                      gt_binary, gt_hv, gt_type, cfg):
    """
    Experiment 1 total loss with:
    - Emphasized FocalTversky (alpha=0.7, beta=0.3) in binary branch
    - Additional Boundary Loss in HV branch
    """
    # Binary branch: Dice + FocalTversky (emphasized)
    loss_bin_dice = dice_loss(pred_binary, gt_binary)
    loss_bin_ft   = focal_tversky_loss(
        pred_binary, gt_binary,
        alpha=cfg["ft_alpha"], beta=cfg["ft_beta"])

    # HV branch: MSE + MSGE + Boundary Loss
    loss_hv_mse      = mse_loss_maps(pred_hv, gt_hv, gt_binary)
    loss_hv_msge     = msge_loss_maps(pred_hv, gt_hv, gt_binary)
    loss_hv_boundary = boundary_loss(pred_hv, gt_binary)

    # Type branch: BCE + Dice + MCFocalTversky (unchanged)
    loss_type_bce  = xentropy_loss(pred_type, gt_type)
    loss_type_dice = dice_loss(
        pred_type.view(-1, 1, *pred_type.shape[2:]),
        gt_type.view(-1, 1, *gt_type.shape[2:])
    )
    loss_type_mcft = mc_focal_tversky_loss(pred_type, gt_type)

    total = (
        loss_bin_dice + loss_bin_ft +
        loss_hv_mse + loss_hv_msge +
        cfg["boundary_lambda"] * loss_hv_boundary +
        loss_type_bce + loss_type_dice + loss_type_mcft
    )

    return total, {
        "binary_dice"   : loss_bin_dice.item(),
        "binary_ft"     : loss_bin_ft.item(),
        "hv_mse"        : loss_hv_mse.item(),
        "hv_msge"       : loss_hv_msge.item(),
        "hv_boundary"   : loss_hv_boundary.item(),
        "type_bce"      : loss_type_bce.item(),
        "type_dice"     : loss_type_dice.item(),
        "type_mcft"     : loss_type_mcft.item(),
    }


def train_one_epoch(model, loader, optimizer, device, epoch, cfg):
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
        loss, _ = compute_exp1_loss(
            pred_binary, pred_hv, pred_type,
            gt_binary, gt_hv, gt_type, cfg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Epoch {epoch} [{i+1}/{n}] Loss: {total_loss/(i+1):.4f}")

    return total_loss / n


def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            img       = batch["image"].to(device)
            gt_binary = batch["binary_map"].to(device)
            gt_hv     = batch["hv_map"].to(device)
            gt_type   = batch["type_map"].to(device)
            pred_binary, pred_hv, pred_type = model(img)
            loss, _ = compute_exp1_loss(
                pred_binary, pred_hv, pred_type,
                gt_binary, gt_hv, gt_type, cfg)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"NuLite — {CONFIG['experiment']}")
    print(f"Device: {device} | Epochs: {CONFIG['epochs']}")
    print(f"FocalTversky alpha={CONFIG['ft_alpha']} beta={CONFIG['ft_beta']}")
    print(f"Boundary Loss lambda={CONFIG['boundary_lambda']}")
    print(f"{'='*60}\n")

    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)

    train_ds = PanNukeDataset(CONFIG["data_dir"], CONFIG["train_folds"], augment=True)
    val_ds   = PanNukeDataset(CONFIG["data_dir"], CONFIG["val_folds"],   augment=False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    model = NuLite(num_classes=CONFIG["num_classes"],
                   use_attention_gate=False).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    print("Starting Experiment 1 training...\n")
    t0 = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{CONFIG['epochs']}  LR={lr_now:.6f}")

        tl = train_one_epoch(model, train_loader, optimizer, device, epoch, CONFIG)
        vl = validate(model, val_loader, device, CONFIG)
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

    # Loss curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(train_losses)+1), train_losses, "b-o", label="Train")
    plt.plot(range(1,len(val_losses)+1),   val_losses,   "r-s", label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Experiment 1: Loss Modification (20 Epochs)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(Path(CONFIG["results_dir"]) / "exp1_loss_curve.png", dpi=120)
    plt.close()

    with open(Path(CONFIG["log_dir"]) / "training_summary.json", "w") as f:
        json.dump({"experiment": CONFIG["experiment"],
                   "epochs": CONFIG["epochs"],
                   "best_val_loss": round(best_val,4),
                   "best_epoch": best_epoch,
                   "train_losses": [round(l,4) for l in train_losses],
                   "val_losses": [round(l,4) for l in val_losses]}, f, indent=2)

    print(f"Logs: {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()