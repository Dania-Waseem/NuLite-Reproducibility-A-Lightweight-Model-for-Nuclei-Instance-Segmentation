"""
my_evaluate.py
Final evaluation script for all NuLite experiments on PanNuke Fold 2.
Computes all metrics reported in the paper:
- mPQ, bPQ, F1 Detection, Binary Dice
- Per-class PQ: Neoplastic, Inflammatory, Connective, Dead, Epithelial
Generates comparison plots and table images.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from my_dataset import PanNukeDataset
from my_model import NuLite

CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

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


def get_instances(pred_binary, threshold=0.5):
    binary  = (torch.sigmoid(pred_binary) > threshold).squeeze().cpu().numpy()
    labeled, n = ndimage.label(binary)
    return labeled, n


def compute_panoptic_quality(pred_instances, gt_instances, iou_threshold=0.5):
    pred_ids = np.unique(pred_instances); pred_ids = pred_ids[pred_ids != 0]
    gt_ids   = np.unique(gt_instances);  gt_ids   = gt_ids[gt_ids != 0]

    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0, 0.0, 0.0

    matched_iou, matched_gt, matched_pred = [], set(), set()

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
            matched_pred.add(p_id)

    TP = len(matched_iou)
    FP = len(pred_ids) - TP
    FN = len(gt_ids)   - TP
    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-6)
    sq = np.mean(matched_iou) if matched_iou else 0.0
    return sq * dq, sq, dq


def compute_dice(pred_binary, gt_binary, threshold=0.5):
    pred  = (torch.sigmoid(pred_binary) > threshold).float()
    gt    = gt_binary.float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0
    return (2 * inter / union).item()


def compute_f1_detection(pred_instances, gt_instances, dist_threshold=12):
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


def evaluate(checkpoint_path, use_attention_gate=False,experiment_name="Model", results_dir="./results"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")

    model = NuLite(num_classes=6,
                   use_attention_gate=use_attention_gate).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    test_ds     = PanNukeDataset("./data_processed", folds=[2], augment=False)
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    all_dice, all_f1_det, all_bpq, all_mpq = [], [], [], []
    all_class_pq = {c: [] for c in CLASS_NAMES}

    print("Running evaluation...")
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            img       = batch["image"].to(device)
            gt_binary = batch["binary_map"]
            gt_type   = batch["type_map"]

            pred_binary, pred_hv, pred_type = model(img)
            pred_binary = pred_binary.cpu()
            pred_type   = pred_type.cpu()

            all_dice.append(compute_dice(pred_binary, gt_binary))

            pred_inst, _ = get_instances(pred_binary)

            label     = np.load(test_ds.samples[idx][1])
            gt_inst   = label[0]
            gt_class  = label[1]

            all_f1_det.append(compute_f1_detection(pred_inst, gt_inst))

            bpq, _, _ = compute_panoptic_quality(pred_inst, gt_inst)
            all_bpq.append(bpq)

            pred_type_map = torch.argmax(
                torch.softmax(pred_type, dim=1), dim=1
            ).squeeze().numpy()

            class_pqs = []
            for c_idx, c_name in enumerate(CLASS_NAMES):
                c_label = c_idx + 1
                pred_c  = (pred_inst * (pred_type_map == c_idx)).astype(np.int32)
                gt_c    = (gt_inst   * (gt_class == c_label)).astype(np.int32)
                pq, _, _ = compute_panoptic_quality(pred_c, gt_c)
                all_class_pq[c_name].append(pq)
                class_pqs.append(pq)

            all_mpq.append(np.mean(class_pqs))

            if (idx + 1) % 300 == 0:
                print(f"  {idx+1}/{len(test_loader)} done...")

    results = {
        "mPQ"             : round(float(np.mean(all_mpq)),    4),
        "bPQ"             : round(float(np.mean(all_bpq)),    4),
        "F1 Detection"    : round(float(np.mean(all_f1_det)), 4),
        "Binary Dice"     : round(float(np.mean(all_dice)),   4),
        "PQ Neoplastic"   : round(float(np.mean(all_class_pq["Neoplastic"])),   4),
        "PQ Inflammatory" : round(float(np.mean(all_class_pq["Inflammatory"])), 4),
        "PQ Connective"   : round(float(np.mean(all_class_pq["Connective"])),   4),
        "PQ Dead"         : round(float(np.mean(all_class_pq["Dead"])),         4),
        "PQ Epithelial"   : round(float(np.mean(all_class_pq["Epithelial"])),   4),
    }

    print(f"\n{'─'*55}")
    print(f"{'Metric':<22} {'Paper':>8} {'Ours':>8} {'Diff':>8}")
    print(f"{'─'*55}")
    for metric, val in results.items():
        paper_val = PAPER_RESULTS.get(metric, 0.0)
        diff      = val - paper_val
        print(f"{metric:<22} {paper_val:>8.4f} {val:>8.4f} {diff:>+8.4f}")
    print(f"{'─'*55}")

    Path(results_dir).mkdir(exist_ok=True)
    safe = experiment_name.replace(" ","_").replace(":","").replace("(","").replace(")","")
    with open(Path(results_dir) / f"{safe}_metrics.json", "w") as f:
        json.dump({"experiment": experiment_name, "metrics": results}, f, indent=2)

    return results


def plot_all_results(all_results, results_dir="./results"):
    metrics_main  = ["mPQ", "bPQ", "F1 Detection", "Binary Dice"]
    metrics_class = ["PQ Neoplastic","PQ Inflammatory","PQ Connective",
                     "PQ Dead","PQ Epithelial"]

    exp_names = list(all_results.keys())
    colors    = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    # ── Overall metrics ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(metrics_main))
    n_exp = len(exp_names)
    total_width = 0.7
    width = total_width / (n_exp + 1)

    paper_vals = [PAPER_RESULTS[m] for m in metrics_main]
    offsets    = np.linspace(-total_width/2, total_width/2, n_exp + 1)

    bars = ax.bar(x + offsets[0], paper_vals, width,
                  label="Paper (130ep)", color="gold",
                  edgecolor="black", linewidth=0.7)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    for i, (exp_name, results) in enumerate(all_results.items()):
        vals = [results[m] for m in metrics_main]
        bars = ax.bar(x + offsets[i+1], vals, width,
                      label=exp_name, color=colors[i],
                      edgecolor="black", linewidth=0.7)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_main, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Overall Metrics — All Methods (PanNuke Fold 2)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "overall_metrics_comparison.png", dpi=130)
    plt.close()
    print("Saved: overall_metrics_comparison.png")

    # ── Per-class PQ ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics_class))
    offsets = np.linspace(-total_width/2, total_width/2, n_exp + 1)

    paper_class = [PAPER_RESULTS[m] for m in metrics_class]
    bars = ax.bar(x + offsets[0], paper_class, width,
                  label="Paper (130ep)", color="gold",
                  edgecolor="black", linewidth=0.7)

    for i, (exp_name, results) in enumerate(all_results.items()):
        vals = [results[m] for m in metrics_class]
        ax.bar(x + offsets[i+1], vals, width,
               label=exp_name, color=colors[i],
               edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("PQ ","") for m in metrics_class], fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Panoptic Quality (PQ)", fontsize=11)
    ax.set_title("Per-Class PQ — All Methods (PanNuke Fold 2)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "perclass_pq_comparison.png", dpi=130)
    plt.close()
    print("Saved: perclass_pq_comparison.png")

    # ── Full results table image ───────────────────────────────────────────────
    all_metrics = metrics_main + metrics_class
    col_labels  = ["Metric", "Paper (130ep)"] + list(all_results.keys())
    table_data  = []
    for m in all_metrics:
        row = [m, f"{PAPER_RESULTS[m]:.4f}"]
        for results in all_results.values():
            val  = results[m]
            diff = val - PAPER_RESULTS[m]
            row.append(f"{val:.4f} ({diff:+.4f})")
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF3FB")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Full Results: All Experiments vs Paper (PanNuke Fold 2)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "full_results_table.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: full_results_table.png")


if __name__ == "__main__":
    results_dir = "./results"
    Path(results_dir).mkdir(exist_ok=True)
    all_results = {}

    checkpoints = [
        ("./logs/baseline/baseline_best.pth", False, "A2 Baseline (2ep)"),
        ("./logs/exp1/exp1_best.pth",         False, "Exp1: Loss Mod (20ep)"),
        ("./logs/exp2/exp2_best.pth",         True,  "Exp2: Attn Gate (15ep)"),
    ]

    for ckpt_path, use_ag, exp_name in checkpoints:
        if Path(ckpt_path).exists():
            all_results[exp_name] = evaluate(
                ckpt_path, use_ag, exp_name, results_dir)
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    if all_results:
        print("\nGenerating comparison plots...")
        plot_all_results(all_results, results_dir)
        print(f"\nAll evaluation complete. Results in: {results_dir}/")
    else:
        print("No checkpoints found.")