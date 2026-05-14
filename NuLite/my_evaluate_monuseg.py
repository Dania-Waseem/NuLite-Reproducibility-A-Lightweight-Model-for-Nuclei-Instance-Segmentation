"""
my_evaluate_monuseg.py
Evaluates trained models on MoNuSeg test set.
Reports Dice and Jaccard (no PQ since MoNuSeg has no cell type labels).
Two modes:
1. MoNuSeg-trained model on MoNuSeg test (normal evaluation)
2. Exp1 PanNuke checkpoint on MoNuSeg test (zero-shot generalization)
   -- as described in Section III.E of the submitted paper
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from my_model import NuLite
from my_train_monuseg import MoNuSegDataset


def compute_dice_jaccard(pred_binary, gt_binary, threshold=0.5):
    """Compute Dice and Jaccard for binary segmentation."""
    pred = (torch.sigmoid(pred_binary) > threshold).float().cpu()
    gt   = gt_binary.float().cpu()

    inter = (pred * gt).sum().item()
    pred_sum = pred.sum().item()
    gt_sum   = gt.sum().item()
    union    = pred_sum + gt_sum - inter

    dice    = (2 * inter) / (pred_sum + gt_sum + 1e-6)
    jaccard = inter / (union + 1e-6)
    return dice, jaccard


def evaluate_on_monuseg(checkpoint_path, use_attention_gate=False,
                         experiment_name="Model", results_dir="./results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Evaluating on MoNuSeg: {experiment_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    model = NuLite(num_classes=6,
                   use_attention_gate=use_attention_gate).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded from epoch {ckpt['epoch']}")

    test_ds     = MoNuSegDataset("./data_processed/monuseg", "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, num_workers=0)
    print(f"Test patches: {len(test_ds)}")

    all_dice, all_jacc = [], []

    with torch.no_grad():
        for batch in test_loader:
            img       = batch["image"].to(device)
            gt_binary = batch["binary_map"]
            pred_binary, _, _ = model(img)
            pred_binary = pred_binary.cpu()
            dice, jacc  = compute_dice_jaccard(pred_binary, gt_binary)
            all_dice.append(dice)
            all_jacc.append(jacc)

    mean_dice = round(float(np.mean(all_dice)), 4)
    std_dice  = round(float(np.std(all_dice)),  4)
    mean_jacc = round(float(np.mean(all_jacc)), 4)
    std_jacc  = round(float(np.std(all_jacc)),  4)

    print(f"\n{'─'*45}")
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"{'─'*45}")
    print(f"{'Dice':<20} {mean_dice:>10.4f} {std_dice:>10.4f}")
    print(f"{'Jaccard':<20} {mean_jacc:>10.4f} {std_jacc:>10.4f}")
    print(f"{'─'*45}")

    results = {
        "experiment": experiment_name,
        "Dice"      : mean_dice,
        "Jaccard"   : mean_jacc,
        "Dice_std"  : std_dice,
        "Jaccard_std": std_jacc,
        "n_patches" : len(test_ds),
    }

    Path(results_dir).mkdir(exist_ok=True)
    safe = experiment_name.replace(" ", "_").replace(":", "")
    with open(Path(results_dir) / f"monuseg_{safe}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_monuseg_comparison(all_results, results_dir="./results"):
    """Bar chart comparing all models on MoNuSeg."""
    names = list(all_results.keys())
    dices = [all_results[n]["Dice"]    for n in names]
    jaccs = [all_results[n]["Jaccard"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    colors_d = ["steelblue", "coral", "mediumseagreen"]
    colors_j = ["#5b9bd5", "#f4b183", "#70ad47"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, dices, w, label="Dice",
                   color=colors_d[:len(names)], edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(x + w/2, jaccs, w, label="Jaccard",
                   color=colors_j[:len(names)], edgecolor="black", linewidth=0.7,
                   alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("MoNuSeg Results: Dice and Jaccard Comparison",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "monuseg_comparison.png", dpi=130)
    plt.close()
    print("Saved: monuseg_comparison.png")

    # Table image
    col_labels = ["Model", "Dice", "Dice Std", "Jaccard", "Jaccard Std"]
    table_data = [
        [n,
         f"{all_results[n]['Dice']:.4f}",
         f"{all_results[n]['Dice_std']:.4f}",
         f"{all_results[n]['Jaccard']:.4f}",
         f"{all_results[n]['Jaccard_std']:.4f}"]
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(11, 3 + len(names)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF3FB")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("MoNuSeg Evaluation Results",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "monuseg_results_table.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: monuseg_results_table.png")


def visualize_monuseg_predictions(checkpoint_path, use_attention_gate=False,
                                   experiment_name="Model", n_samples=6,
                                   results_dir="./results"):
    """Show visual predictions on MoNuSeg test patches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NuLite(num_classes=6,
                    use_attention_gate=use_attention_gate).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = MoNuSegDataset("./data_processed/monuseg", "test", augment=False)
    samples = test_ds.samples[:n_samples]

    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples*3, 9))
    fig.suptitle(f"MoNuSeg Predictions — {experiment_name}",
                 fontsize=13, fontweight="bold")

    for j, img_path in enumerate(samples):
        img_orig = np.array(Image.open(img_path))
        mask_path = test_ds.mask_dir / (img_path.stem + ".npy")
        gt_mask   = np.load(mask_path)

        img_norm = (img_orig.astype(np.float32)/255.0 - 0.5) / 0.5
        tensor   = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_binary, _, _ = model(tensor)

        pred_mask = (torch.sigmoid(pred_binary) > 0.5).squeeze().cpu().numpy()

        axes[0,j].imshow(img_orig)
        axes[0,j].set_title(f"Image {j+1}", fontsize=8)
        axes[0,j].axis("off")

        axes[1,j].imshow(gt_mask, cmap="Reds")
        axes[1,j].set_title("GT Mask", fontsize=8)
        axes[1,j].axis("off")

        axes[2,j].imshow(pred_mask, cmap="Blues")
        dice, _ = compute_dice_jaccard(pred_binary.cpu(), torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0))
        axes[2,j].set_title(f"Pred (Dice={dice:.3f})", fontsize=8)
        axes[2,j].axis("off")

    plt.tight_layout()
    safe = experiment_name.replace(" ","_").replace(":","")
    out  = Path(results_dir) / f"monuseg_predictions_{safe}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    results_dir  = "./results"
    all_results  = {}

    # 1. MoNuSeg-trained model on MoNuSeg test
    monuseg_ckpt = "./logs/monuseg/monuseg_best.pth"
    if Path(monuseg_ckpt).exists():
        all_results["MoNuSeg Trained"] = evaluate_on_monuseg(
            monuseg_ckpt,
            use_attention_gate=False,
            experiment_name="MoNuSeg Trained",
            results_dir=results_dir
        )
        visualize_monuseg_predictions(
            monuseg_ckpt, False, "MoNuSeg Trained",
            n_samples=6, results_dir=results_dir)
    else:
        print(f"MoNuSeg checkpoint not found: {monuseg_ckpt}")

    # 2. Exp1 PanNuke checkpoint on MoNuSeg (zero-shot generalization)
    # As described in paper Section III.E
    exp1_ckpt = "./logs/exp1/exp1_best.pth"
    if Path(exp1_ckpt).exists():
        all_results["Exp1 Zero-Shot"] = evaluate_on_monuseg(
            exp1_ckpt,
            use_attention_gate=False,
            experiment_name="Exp1 Zero-Shot",
            results_dir=results_dir
        )
        visualize_monuseg_predictions(
            exp1_ckpt, False, "Exp1 Zero-Shot",
            n_samples=6, results_dir=results_dir)
    else:
        print(f"Exp1 checkpoint not found: {exp1_ckpt}")

    # 3. Plot comparison
    if all_results:
        plot_monuseg_comparison(all_results, results_dir)
        print("\nMoNuSeg evaluation complete.")
        print(f"Results saved in: {results_dir}/")
    else:
        print("No checkpoints found. Run training scripts first.")