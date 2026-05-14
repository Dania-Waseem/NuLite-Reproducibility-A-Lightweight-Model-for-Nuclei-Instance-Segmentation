"""
my_dataset.py
PanNuke dataset loader and all loss functions for NuLite.
Loss functions exactly as described in the NuLite paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import csv
import random


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PanNukeDataset(Dataset):
    def __init__(self, data_dir, folds, augment=False):
        self.augment   = augment
        self.samples   = []

        for fold in folds:
            img_dir   = Path(data_dir) / f"fold{fold}" / "images"
            label_dir = Path(data_dir) / f"fold{fold}" / "labels"
            for img_path in sorted(img_dir.glob("*.png")):
                label_path = label_dir / (img_path.stem + ".npy")
                if label_path.exists():
                    self.samples.append((img_path, label_path))

        print(f"  Dataset: {len(self.samples)} samples from folds {folds}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        img = np.array(Image.open(img_path)).astype(np.float32)

        # Normalize: mean=0.5, std=0.5 (as in NuLite paper)
        img = (img / 255.0 - 0.5) / 0.5
        img = torch.tensor(img).permute(2, 0, 1).float()  # [3, 256, 256]

        # Load label
        label = np.load(label_path)  # [2, 256, 256]
        instance_map = label[0]      # instance IDs
        class_map    = label[1]      # class IDs 1-5

        # Binary nucleus map
        binary_map = (instance_map > 0).astype(np.float32)  # [256, 256]

        # HV maps (horizontal/vertical distance maps)
        hv_map = compute_hv_map(instance_map)  # [2, 256, 256]

        # Type map: one-hot [6, 256, 256] (5 classes + background)
        type_map = np.zeros((6, 256, 256), dtype=np.float32)
        for c in range(1, 6):
            type_map[c] = (class_map == c).astype(np.float32)
        type_map[0] = (class_map == 0).astype(np.float32)  # background

        # Augmentation
        if self.augment:
            img, binary_map, hv_map, type_map = random_augment(
                img, binary_map, hv_map, type_map)

        return {
            "image"      : img,
            "binary_map" : torch.tensor(binary_map).unsqueeze(0),  # [1,256,256]
            "hv_map"     : torch.tensor(hv_map).float(),            # [2,256,256]
            "type_map"   : torch.tensor(type_map).float(),          # [6,256,256]
        }


def compute_hv_map(instance_map):
    """Compute horizontal and vertical distance maps from instance map."""
    H, W = instance_map.shape
    hv = np.zeros((2, H, W), dtype=np.float32)

    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        mask = (instance_map == inst_id)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx = xs.mean()
        cy = ys.mean()
        # Normalize to [-1, 1]
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        hv[0][mask] = (xs - cx) / (x_range / 2 + 1e-6)
        hv[1][mask] = (ys - cy) / (y_range / 2 + 1e-6)

    return np.clip(hv, -1, 1)


def random_augment(img, binary_map, hv_map, type_map):
    """Simple augmentation: flips and 90-degree rotations."""
    # Horizontal flip
    if random.random() > 0.5:
        img        = torch.flip(img, dims=[2])
        binary_map = np.fliplr(binary_map).copy()
        hv_map     = np.flip(hv_map, axis=2).copy()
        hv_map[0]  = -hv_map[0]
        type_map   = np.flip(type_map, axis=2).copy()

    # Vertical flip
    if random.random() > 0.5:
        img        = torch.flip(img, dims=[1])
        binary_map = np.flipud(binary_map).copy()
        hv_map     = np.flip(hv_map, axis=1).copy()
        hv_map[1]  = -hv_map[1]
        type_map   = np.flip(type_map, axis=1).copy()

    # 90 degree rotation
    if random.random() > 0.5:
        k          = random.choice([1, 2, 3])
        img        = torch.rot90(img, k=k, dims=[1, 2])
        binary_map = np.rot90(binary_map, k=k).copy()
        hv_map     = np.rot90(hv_map, k=k, axes=(1, 2)).copy()
        type_map   = np.rot90(type_map, k=k, axes=(1, 2)).copy()

    return img, binary_map, hv_map, type_map


# ─── Loss Functions (exactly as in NuLite paper) ──────────────────────────────

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation."""
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice   = (2 * inter + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_tversky_loss(pred, target, alpha=0.5, beta=0.5, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky Loss as in NuLite paper.
    TI = TP / (TP + alpha*FN + beta*FP)
    LFT = (1 - TI)^gamma
    """
    pred   = torch.sigmoid(pred)
    tp     = (pred * target).sum(dim=(2, 3))
    fn     = ((1 - pred) * target).sum(dim=(2, 3))
    fp     = (pred * (1 - target)).sum(dim=(2, 3))
    ti     = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return ((1 - ti) ** gamma).mean()


def mc_focal_tversky_loss(pred, target, num_classes=6,
                           alpha=0.5, beta=0.5, gamma=0.75):
    """Multi-class Focal Tversky Loss for type map."""
    pred_soft = torch.softmax(pred, dim=1)
    loss = 0.0
    for c in range(num_classes):
        p = pred_soft[:, c:c+1]
        t = target[:, c:c+1]
        loss += focal_tversky_loss(
            p, t, alpha=alpha, beta=beta, gamma=gamma)
    return loss / num_classes


def mse_loss_maps(pred, target, binary_map):
    """MSE loss applied only within nucleus regions."""
    mask  = binary_map.expand_as(pred)
    diff  = (pred - target) ** 2
    loss  = (diff * mask).sum() / (mask.sum() + 1e-6)
    return loss


def msge_loss_maps(pred, target, binary_map):
    """
    Mean Squared Gradient Error loss.
    Penalizes incorrect spatial gradients at instance boundaries.
    """
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx, dy

    pred_dx, pred_dy   = gradient(pred)
    target_dx, target_dy = gradient(target)
    mask = binary_map.expand_as(pred)

    loss_x = ((pred_dx - target_dx) ** 2 * mask).sum() / (mask.sum() + 1e-6)
    loss_y = ((pred_dy - target_dy) ** 2 * mask).sum() / (mask.sum() + 1e-6)
    return (loss_x + loss_y) / 2


def xentropy_loss(pred, target):
    """Binary cross entropy for type map."""
    return F.binary_cross_entropy_with_logits(pred, target)


def compute_total_loss(pred_binary, pred_hv, pred_type,
                       gt_binary, gt_hv, gt_type,
                       weights=None):
    """
    Total NuLite loss combining all branches.
    Branch weights as in the original NuLite config (all=1).
    """
    if weights is None:
        weights = {
            "binary_dice": 1.0, "binary_ft": 1.0,
            "hv_mse": 1.0,      "hv_msge": 1.0,
            "type_bce": 1.0,    "type_dice": 1.0, "type_mcft": 1.0,
        }

    # Binary branch
    loss_bin_dice = dice_loss(pred_binary, gt_binary)
    loss_bin_ft   = focal_tversky_loss(pred_binary, gt_binary)

    # HV branch
    loss_hv_mse  = mse_loss_maps(pred_hv, gt_hv, gt_binary)
    loss_hv_msge = msge_loss_maps(pred_hv, gt_hv, gt_binary)

    # Type branch
    loss_type_bce  = xentropy_loss(pred_type, gt_type)
    loss_type_dice = dice_loss(
        pred_type.view(-1, 1, *pred_type.shape[2:]),
        gt_type.view(-1, 1, *gt_type.shape[2:])
    )
    loss_type_mcft = mc_focal_tversky_loss(pred_type, gt_type)

    total = (
        weights["binary_dice"] * loss_bin_dice +
        weights["binary_ft"]   * loss_bin_ft   +
        weights["hv_mse"]      * loss_hv_mse   +
        weights["hv_msge"]     * loss_hv_msge  +
        weights["type_bce"]    * loss_type_bce +
        weights["type_dice"]   * loss_type_dice +
        weights["type_mcft"]   * loss_type_mcft
    )
    return total, {
        "binary_dice": loss_bin_dice.item(),
        "binary_ft"  : loss_bin_ft.item(),
        "hv_mse"     : loss_hv_mse.item(),
        "hv_msge"    : loss_hv_msge.item(),
        "type_bce"   : loss_type_bce.item(),
        "type_dice"  : loss_type_dice.item(),
        "type_mcft"  : loss_type_mcft.item(),
    }