"""
src/utils.py
Utility functions for NuLite project.
Includes metric computation, visualization, and result saving helpers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
from scipy import ndimage
from PIL import Image


CLASS_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

CLASS_COLORS = {
    0: [1.0, 0.0, 0.0],
    1: [0.0, 0.8, 0.0],
    2: [0.0, 0.4, 1.0],
    3: [1.0, 0.9, 0.0],
    4: [1.0, 0.0, 1.0],
}

PAPER_RESULTS = {
    "mPQ"             : 0.4762,
    "bPQ"             : 0.6817,
    "F1 Detection"    : 0.8204,
    "Binary Dice"     : 0.7103,
    "PQ Neoplastic"   : 0.5121,
    "PQ Inflammatory" : 0.4213,
    "PQ Connective"   : 0.3984,
    "PQ Dead"         : 0.2011,
    "PQ Epithelial"   : 0.4491,
}


# ── Metric Functions ───────────────────────────────────────────────────────────

def get_instances(pred_binary, threshold=0.5):
    """Convert binary prediction to instance map using connected components."""
    import torch
    binary  = (torch.sigmoid(pred_binary) > threshold).squeeze().cpu().numpy()
    labeled, n = ndimage.label(binary)
    return labeled, n


def compute_dice(pred_binary, gt_binary, threshold=0.5):
    """Compute Binary Dice coefficient."""
    import torch
    pred  = (torch.sigmoid(pred_binary) > threshold).float()
    gt    = gt_binary.float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0
    return (2 * inter / union).item()


def compute_panoptic_quality(pred_instances, gt_instances, iou_threshold=0.5):
    """Compute Panoptic Quality (PQ = SQ * DQ)."""
    pred_ids = np.unique(pred_instances); pred_ids = pred_ids[pred_ids != 0]
    gt_ids   = np.unique(gt_instances);  gt_ids   = gt_ids[gt_ids != 0]

    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0, 0.0, 0.0

    matched_iou, matched_gt = [], set()

    for p_id in pred_ids:
        pred_mask = (pred_instances == p_id)
        best_iou, best_gt = 0.0, -1
        for g_id in gt_ids:
            if g_id in matched_gt:
                continue
            gt_mask = (gt_instances == g_id)
            inter   = (pred_mask & gt_mask).sum()
            union   = (pred_mask | gt_mask).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou, best_gt = iou, g_id
        if best_iou >= iou_threshold:
            matched_iou.append(best_iou)
            matched_gt.add(best_gt)

    TP = len(matched_iou)
    FP = len(pred_ids) - TP
    FN = len(gt_ids)   - TP
    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-6)
    sq = np.mean(matched_iou) if matched_iou else 0.0
    return sq * dq, sq, dq


def compute_f1_detection(pred_instances, gt_instances, dist_threshold=12):
    """Compute F1 Detection score using centroid matching."""
    pred_ids = np.unique(pred_instances); pred_ids = pred_ids[pred_ids != 0]
    gt_ids   = np.unique(gt_instances);  gt_ids   = gt_ids[gt_ids != 0]

    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0

    pred_centroids = []
    for p_id in pred_ids:
        ys, xs = np.where(pred_instances == p_id)
        pred_centroids.append((xs.mean(), ys.mean()))

    gt_centroids = []
    for g_id in gt_ids:
        ys, xs = np.where(gt_instances == g_id)
        gt_centroids.append((xs.mean(), ys.mean()))

    matched_gt, tp = set(), 0
    for pc in pred_centroids:
        for j, gc in enumerate(gt_centroids):
            if j in matched_gt:
                continue
            if np.sqrt((pc[0]-gc[0])**2 + (pc[1]-gc[1])**2) < dist_threshold:
                tp += 1
                matched_gt.add(j)
                break

    fp = len(pred_centroids) - tp
    fn = len(gt_centroids)   - tp
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


# ── Visualization Functions ────────────────────────────────────────────────────

def draw_nucleus_overlay(img, instance_map, type_map, alpha=0.45):
    """Draw colored nucleus boundaries on image."""
    overlay = img.copy().astype(np.float32) / 255.0
    result  = overlay.copy()
    inst_ids = np.unique(instance_map); inst_ids = inst_ids[inst_ids != 0]

    for inst_id in inst_ids:
        mask   = (instance_map == inst_id)
        vals   = type_map[mask]
        if len(vals) == 0:
            continue
        cls_id = int(np.bincount(vals.astype(int)).argmax())
        color  = np.array(CLASS_COLORS.get(cls_id, [0.5, 0.5, 0.5]))
        for c in range(3):
            result[:,:,c][mask] = alpha*color[c] + (1-alpha)*overlay[:,:,c][mask]
        eroded   = ndimage.binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded
        for c in range(3):
            result[:,:,c][boundary] = color[c]

    return (result * 255).astype(np.uint8)


def plot_metrics_comparison(all_results, save_path="results/metrics_comparison.png"):
    """Bar chart comparing all experiments against paper results."""
    metrics = ["mPQ", "bPQ", "F1 Detection", "Binary Dice"]
    exp_names = list(all_results.keys())
    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]
    x = np.arange(len(metrics))
    total_width = 0.7
    width = total_width / (len(exp_names) + 1)
    offsets = np.linspace(-total_width/2, total_width/2, len(exp_names) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    paper_vals = [PAPER_RESULTS[m] for m in metrics]
    ax.bar(x + offsets[0], paper_vals, width, label="Paper (130ep)",
           color="gold", edgecolor="black", linewidth=0.7)

    for i, (name, results) in enumerate(all_results.items()):
        vals = [results.get(m, 0) for m in metrics]
        ax.bar(x + offsets[i+1], vals, width, label=name,
               color=colors[i], edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Metrics Comparison — All Experiments vs Paper",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_curve(train_losses, val_losses, title, save_path):
    """Plot training and validation loss curves."""
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
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")


def get_legend_patches():
    """Return matplotlib legend patches for nucleus classes."""
    return [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i+1])
        for i in range(5)
    ]


# ── Result Saving Functions ────────────────────────────────────────────────────

def save_metrics_json(metrics, experiment_name, save_path):
    """Save evaluation metrics to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = {"experiment": experiment_name, "metrics": metrics}
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {save_path}")


def print_metrics_table(results, experiment_name):
    """Print formatted metrics table to terminal."""
    print(f"\n{'='*55}")
    print(f"Results: {experiment_name}")
    print(f"{'─'*55}")
    print(f"{'Metric':<22} {'Paper':>8} {'Ours':>8} {'Diff':>8}")
    print(f"{'─'*55}")
    for metric, val in results.items():
        paper_val = PAPER_RESULTS.get(metric, 0.0)
        diff = val - paper_val
        print(f"{metric:<22} {paper_val:>8.4f} {val:>8.4f} {diff:>+8.4f}")
    print(f"{'='*55}")


def load_image_for_inference(img_path):
    """Load and preprocess a single image for model inference."""
    import torch
    img = np.array(Image.open(img_path).convert("RGB"))
    if img.shape[0] != 256 or img.shape[1] != 256:
        img = np.array(
            Image.fromarray(img.astype(np.uint8)).resize((256, 256)))
    img_norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor   = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float()
    return img.astype(np.uint8), tensor