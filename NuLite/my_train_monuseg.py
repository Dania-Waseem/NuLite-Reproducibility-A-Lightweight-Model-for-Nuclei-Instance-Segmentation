"""
my_train_monuseg.py
Train NuLite on MoNuSeg dataset.
MoNuSeg has only binary masks (no cell type labels).
So only the binary branch and HV branch are trained.
Type branch is disabled for MoNuSeg training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from my_model import NuLite
from my_dataset import dice_loss, focal_tversky_loss, mse_loss_maps, msge_loss_maps
from my_dataset import compute_hv_map


CONFIG = {
    "data_dir"       : "./data_processed/monuseg",
    "epochs"         : 10,
    "batch_size"     : 4,
    "lr"             : 3e-4,
    "weight_decay"   : 1e-4,
    "betas"          : (0.85, 0.95),
    "scheduler_gamma": 0.85,
    "log_dir"        : "./logs/monuseg",
    "results_dir"    : "./results",
    "checkpoint_name": "monuseg_best.pth",
    "experiment"     : "MoNuSeg Training (10 epochs)",
}


# ── MoNuSeg Dataset ────────────────────────────────────────────────────────────

class MoNuSegDataset(Dataset):
    def __init__(self, data_dir, split="train", augment=False):
        self.augment  = augment
        self.img_dir  = Path(data_dir) / split / "images"
        self.mask_dir = Path(data_dir) / split / "masks"
        self.samples  = sorted(list(self.img_dir.glob("*.png")))
        print(f"  MoNuSeg {split}: {len(self.samples)} patches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path  = self.samples[idx]
        mask_path = self.mask_dir / (img_path.stem + ".npy")

        img  = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)  # binary [256,256]

        # Normalize image
        img_norm = (img / 255.0 - 0.5) / 0.5
        img_t    = torch.tensor(img_norm).permute(2, 0, 1).float()

        # Binary map
        binary_map = torch.tensor(mask).unsqueeze(0)  # [1,256,256]

        # HV map from binary mask (treat whole mask as single instance)
        inst_map = ndimage.label(mask)[0]
        hv_map   = torch.tensor(compute_hv_map(inst_map)).float()  # [2,256,256]

        return {
            "image"     : img_t,
            "binary_map": binary_map,
            "hv_map"    : hv_map,
        }


def compute_monuseg_loss(pred_binary, pred_hv, gt_binary, gt_hv):
    """Loss for MoNuSeg: binary + HV branches only."""
    loss_dice = dice_loss(pred_binary, gt_binary)
    loss_ft   = focal_tversky_loss(pred_binary, gt_binary)
    loss_mse  = mse_loss_maps(pred_hv, gt_hv, gt_binary)
    loss_msge = msge_loss_maps(pred_hv, gt_hv, gt_binary)
    total = loss_dice + loss_ft + loss_mse + loss_msge
    return total


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total = 0.0
    n = len(loader)
    for i, batch in enumerate(loader):
        img       = batch["image"].to(device)
        gt_binary = batch["binary_map"].to(device)
        gt_hv     = batch["hv_map"].to(device)

        optimizer.zero_grad()
        pred_binary, pred_hv, _ = model(img)
        loss = compute_monuseg_loss(pred_binary, pred_hv, gt_binary, gt_hv)
        loss.backward()
        optimizer.step()
        total += loss.item()

        if (i+1) % 20 == 0 or (i+1) == n:
            print(f"  Epoch {epoch} [{i+1}/{n}] Loss: {total/(i+1):.4f}")

    return total / n


def validate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            img       = batch["image"].to(device)
            gt_binary = batch["binary_map"].to(device)
            gt_hv     = batch["hv_map"].to(device)
            pred_binary, pred_hv, _ = model(img)
            loss = compute_monuseg_loss(pred_binary, pred_hv, gt_binary, gt_hv)
            total += loss.item()
    return total / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"NuLite — {CONFIG['experiment']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(exist_ok=True)

    train_ds = MoNuSegDataset(CONFIG["data_dir"], "train", augment=True)
    test_ds  = MoNuSegDataset(CONFIG["data_dir"], "test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(test_ds,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    model = NuLite(num_classes=6, use_attention_gate=False).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"],
                                  betas=CONFIG["betas"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=CONFIG["scheduler_gamma"])

    train_losses, val_losses = [], []
    best_val, best_epoch = float("inf"), 0

    log_file = Path(CONFIG["log_dir"]) / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","lr"])

    print("Starting MoNuSeg training...\n")
    t0 = time.time()

    for epoch in range(1, CONFIG["epochs"]+1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{CONFIG['epochs']}  LR={lr_now:.6f}")

        tl = train_one_epoch(model, train_loader, optimizer, device, epoch)
        vl = validate(model, val_loader, device)
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"  Train: {tl:.4f}  Val: {vl:.4f}")

        if vl < best_val:
            best_val, best_epoch = vl, epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": vl, "config": CONFIG},
                       Path(CONFIG["log_dir"]) / CONFIG["checkpoint_name"])
            print(f"  ** Best saved (epoch {epoch})")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, round(tl,4), round(vl,4), round(lr_now,6)])

        scheduler.step()
        print()

    print(f"Done in {(time.time()-t0)/60:.1f} min | Best: {best_val:.4f} @ ep {best_epoch}")

    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(train_losses)+1), train_losses, "b-o", label="Train")
    plt.plot(range(1,len(val_losses)+1),   val_losses,   "r-s", label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("MoNuSeg Training Loss (10 Epochs)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(Path(CONFIG["results_dir"]) / "monuseg_loss_curve.png", dpi=120)
    plt.close()
    print(f"Loss curve saved. Logs: {CONFIG['log_dir']}")

    with open(Path(CONFIG["log_dir"]) / "training_summary.json", "w") as f:
        json.dump({"experiment": CONFIG["experiment"],
                   "best_val_loss": round(best_val,4),
                   "best_epoch": best_epoch,
                   "train_losses": [round(l,4) for l in train_losses],
                   "val_losses":   [round(l,4) for l in val_losses]}, f, indent=2)


if __name__ == "__main__":
    main()