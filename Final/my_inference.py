"""
my_inference.py
Inference script with interactive options.
Shows original image, detected nuclei with colored boundaries,
class count chart, and optionally side-by-side model comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from scipy import ndimage
from my_model import NuLite

CLASS_NAMES = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}

CLASS_COLORS = {
    0: [1.0, 0.0, 0.0],
    1: [0.0, 0.8, 0.0],
    2: [0.0, 0.4, 1.0],
    3: [1.0, 0.9, 0.0],
    4: [1.0, 0.0, 1.0],
}

CHECKPOINTS = {
    "1": ("./logs/baseline/baseline_best.pth", False, "Baseline (2ep)"),
    "2": ("./logs/exp1/exp1_best.pth",         False, "Exp1: Loss Mod (20ep)"),
    "3": ("./logs/exp2/exp2_best.pth",         True,  "Exp2: Attn Gate (15ep)"),
    "4": ("./logs/monuseg/monuseg_best.pth",   False, "MoNuSeg Trained"),
}


def load_model(ckpt_path, use_attention_gate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NuLite(num_classes=6, use_attention_gate=use_attention_gate).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded: {ckpt_path} (epoch {ckpt['epoch']})")
    return model, device


def preprocess(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    if img.shape[0] != 256 or img.shape[1] != 256:
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize((256,256)))
    img_norm = (img.astype(np.float32)/255.0 - 0.5) / 0.5
    tensor   = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float()
    return img.astype(np.uint8), tensor


def get_instances(pred_binary, threshold=0.5):
    binary  = (torch.sigmoid(pred_binary) > threshold).squeeze().cpu().numpy()
    labeled, n = ndimage.label(binary)
    return labeled, n


def draw_overlay(img, instance_map, type_map, alpha=0.45):
    overlay = img.copy().astype(np.float32) / 255.0
    result  = overlay.copy()
    inst_ids = np.unique(instance_map); inst_ids = inst_ids[inst_ids != 0]
    for inst_id in inst_ids:
        mask   = (instance_map == inst_id)
        vals   = type_map[mask]
        if len(vals) == 0:
            continue
        cls_id = int(np.bincount(vals.astype(int)).argmax())
        color  = np.array(CLASS_COLORS.get(cls_id, [0.5,0.5,0.5]))
        for c in range(3):
            result[:,:,c][mask] = alpha*color[c] + (1-alpha)*overlay[:,:,c][mask]
        eroded   = ndimage.binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded
        for c in range(3):
            result[:,:,c][boundary] = color[c]
    return (result*255).astype(np.uint8)


def run_single(model, device, img_path, exp_name, save_dir):
    img_orig, tensor = preprocess(img_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        pred_binary, pred_hv, pred_type = model(tensor)

    pred_binary   = pred_binary.cpu()
    pred_type_map = torch.argmax(
        torch.softmax(pred_type.cpu(), dim=1), dim=1
    ).squeeze().numpy().astype(np.uint8)

    instance_map, n_inst = get_instances(pred_binary)
    overlay = draw_overlay(img_orig, instance_map, pred_type_map)

    class_counts = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        ids = np.unique(instance_map[pred_type_map == cls_id])
        class_counts[cls_name] = len(ids[ids != 0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"NuLite Inference — {exp_name} — {Path(img_path).name}",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(img_orig)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Detected Nuclei ({n_inst} instances)", fontsize=11)
    axes[1].axis("off")

    axes[2].bar(list(class_counts.keys()), list(class_counts.values()),
                color=[CLASS_COLORS[i] for i in range(5)],
                edgecolor="black", linewidth=0.7)
    axes[2].set_title("Nucleus Class Counts", fontsize=11)
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=25)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    safe = exp_name.replace(" ","_").replace(":","").replace("(","").replace(")","")
    out  = Path(save_dir) / f"inference_{safe}_{Path(img_path).stem}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Total nuclei: {n_inst}")
    for name, cnt in class_counts.items():
        if cnt > 0:
            print(f"    {name}: {cnt}")
    return overlay, class_counts


def run_comparison(model1, device1, name1, model2, device2, name2,
                   img_path, save_dir):
    """Side-by-side comparison of two models on the same image."""
    img_orig, t1 = preprocess(img_path)
    _, t2 = preprocess(img_path)

    with torch.no_grad():
        pb1, _, pt1 = model1(t1.to(device1))
        pb2, _, pt2 = model2(t2.to(device2))

    inst1, n1 = get_instances(pb1.cpu())
    inst2, n2 = get_instances(pb2.cpu())

    tm1 = torch.argmax(torch.softmax(pt1.cpu(),dim=1),dim=1).squeeze().numpy().astype(np.uint8)
    tm2 = torch.argmax(torch.softmax(pt2.cpu(),dim=1),dim=1).squeeze().numpy().astype(np.uint8)

    ov1 = draw_overlay(img_orig, inst1, tm1)
    ov2 = draw_overlay(img_orig, inst2, tm2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Model Comparison — {Path(img_path).name}",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(img_orig)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(ov1)
    axes[1].set_title(f"{name1}\n({n1} nuclei)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(ov2)
    axes[2].set_title(f"{name2}\n({n2} nuclei)", fontsize=10)
    axes[2].axis("off")

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out = Path(save_dir) / f"comparison_{Path(img_path).stem}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def run_on_test_fold(model, device, exp_name, n_samples=6, save_dir="./results/inference"):
    """Run inference on sample images from PanNuke test fold."""
    img_dir  = Path("./data_processed/fold2/images")
    img_paths = sorted(list(img_dir.glob("*.png")))[:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 7))
    fig.suptitle(f"NuLite Inference on Test Fold — {exp_name}",
                 fontsize=13, fontweight="bold")

    for j, img_path in enumerate(img_paths):
        img_orig, tensor = preprocess(img_path)
        with torch.no_grad():
            pred_binary, _, pred_type = model(tensor.to(device))
        inst_map, n_inst = get_instances(pred_binary.cpu())
        type_map = torch.argmax(
            torch.softmax(pred_type.cpu(),dim=1),dim=1
        ).squeeze().numpy().astype(np.uint8)
        overlay = draw_overlay(img_orig, inst_map, type_map)

        axes[0,j].imshow(img_orig)
        axes[0,j].set_title(f"Image {j+1}", fontsize=8)
        axes[0,j].axis("off")

        axes[1,j].imshow(overlay)
        axes[1,j].set_title(f"{n_inst} nuclei", fontsize=8)
        axes[1,j].axis("off")

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5,-0.02))

    plt.tight_layout()
    safe = exp_name.replace(" ","_").replace(":","").replace("(","").replace(")","")
    out  = Path(save_dir) / f"test_fold_samples_{safe}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    save_dir = "./results/inference"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("NuLite Inference")
    print("="*60)
    print("\nAvailable options:")
    print("  1. Run inference on test fold samples (all available checkpoints)")
    print("  2. Compare two models side by side")
    print("  3. Run on a specific image file")
    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        print("\nRunning inference on test fold samples for all checkpoints...")
        for key, (ckpt_path, use_ag, exp_name) in CHECKPOINTS.items():
            if Path(ckpt_path).exists():
                print(f"\n{exp_name}")
                model, device = load_model(ckpt_path, use_ag)
                run_on_test_fold(model, device, exp_name,
                                 n_samples=6, save_dir=save_dir)
            else:
                print(f"  Skipping {exp_name} (checkpoint not found)")

    elif choice == "2":
        print("\nChoose Model 1:")
        for k, (_, _, name) in CHECKPOINTS.items():
            print(f"  {k}. {name}")
        k1 = input("Enter number: ").strip()

        print("\nChoose Model 2:")
        for k, (_, _, name) in CHECKPOINTS.items():
            print(f"  {k}. {name}")
        k2 = input("Enter number: ").strip()

        if k1 in CHECKPOINTS and k2 in CHECKPOINTS:
            ckpt1, ag1, name1 = CHECKPOINTS[k1]
            ckpt2, ag2, name2 = CHECKPOINTS[k2]

            if Path(ckpt1).exists() and Path(ckpt2).exists():
                model1, device1 = load_model(ckpt1, ag1)
                model2, device2 = load_model(ckpt2, ag2)

                img_dir   = Path("./data_processed/fold2/images")
                img_paths = sorted(list(img_dir.glob("*.png")))[:3]
                print(f"\nComparing {name1} vs {name2} on 3 test images...")
                for img_path in img_paths:
                    run_comparison(model1, device1, name1,
                                   model2, device2, name2,
                                   img_path, save_dir)
            else:
                print("One or both checkpoints not found.")
        else:
            print("Invalid choice.")

    elif choice == "3":
        img_path = input("\nEnter full path to image file: ").strip().strip('"')
        if not Path(img_path).exists():
            print(f"Image not found: {img_path}")
        else:
            print("\nChoose checkpoint:")
            for k, (_, _, name) in CHECKPOINTS.items():
                print(f"  {k}. {name}")
            k = input("Enter number: ").strip()
            if k in CHECKPOINTS:
                ckpt_path, use_ag, exp_name = CHECKPOINTS[k]
                if Path(ckpt_path).exists():
                    model, device = load_model(ckpt_path, use_ag)
                    run_single(model, device, img_path, exp_name, save_dir)
                else:
                    print(f"Checkpoint not found: {ckpt_path}")
            else:
                print("Invalid choice.")
    else:
        print("Invalid choice. Running option 1 by default...")
        for key, (ckpt_path, use_ag, exp_name) in CHECKPOINTS.items():
            if Path(ckpt_path).exists():
                print(f"\n{exp_name}")
                model, device = load_model(ckpt_path, use_ag)
                run_on_test_fold(model, device, exp_name,
                                 n_samples=6, save_dir=save_dir)

    print(f"\nAll inference results saved in: {save_dir}/")