"""
my_preprocess.py
Preprocessing script for PanNuke dataset.
Converts .npy files into PNG images and numpy label files
following the exact structure described in the NuLite paper.
"""

import numpy as np
import os
import csv
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TISSUE_TYPES = {
    0: "Adrenal_gland", 1: "Bile-duct", 2: "Bladder",
    3: "Breast", 4: "Cervix", 5: "Colon", 6: "Esophagus",
    7: "HeadNeck", 8: "Kidney", 9: "Liver", 10: "Lung",
    11: "Ovarian", 12: "Pancreatic", 13: "Prostate",
    14: "Skin", 15: "Stomach", 16: "Testis",
    17: "Thyroid", 18: "Uterus"
}

NUCLEUS_CLASSES = {
    0: "Neoplastic", 1: "Inflammatory",
    2: "Connective", 3: "Dead", 4: "Epithelial"
}

CLASS_COLORS = {
    0: (255, 0, 0),    # Neoplastic - red
    1: (0, 255, 0),    # Inflammatory - green
    2: (0, 0, 255),    # Connective - blue
    3: (255, 255, 0),  # Dead - yellow
    4: (255, 0, 255),  # Epithelial - magenta
}


def process_fold(fold_idx, input_path, output_path):
    fold_dir = Path(input_path) / f"fold{fold_idx}"
    out_dir  = Path(output_path) / f"fold{fold_idx}"

    img_dir   = out_dir / "images"
    label_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFold {fold_idx}: Loading numpy files...")
    images = np.load(fold_dir / "images.npy")   # [N, 256, 256, 3]
    masks  = np.load(fold_dir / "masks.npy")    # [N, 256, 256, 6]
    types  = np.load(fold_dir / "types.npy")    # [N]

    N = images.shape[0]
    print(f"  Found {N} images, shape: {images.shape}")

    types_rows    = []
    cell_count_rows = []

    print(f"  Processing images...")
    for i in range(N):
        img_name = f"{fold_idx}_{i:05d}.png"

        # Save image as PNG
        img = images[i].astype(np.uint8)
        Image.fromarray(img).save(img_dir / img_name)

        # Process mask: [256, 256, 6]
        # channels 0-4 are nucleus classes, channel 5 is background
        mask = masks[i]  # [256, 256, 6]

        instance_map = np.zeros((256, 256), dtype=np.int32)
        class_map    = np.zeros((256, 256), dtype=np.int32)

        global_id = 1
        cell_counts = [0] * 5

        for c in range(5):  # 5 nucleus classes
            ch = mask[:, :, c]
            instance_ids = np.unique(ch)
            instance_ids = instance_ids[instance_ids != 0]
            for inst_id in instance_ids:
                px = (ch == inst_id)
                instance_map[px] = global_id
                class_map[px]    = c + 1  # 1-indexed
                cell_counts[c]  += 1
                global_id += 1

        # Save label
        label_name = f"{fold_idx}_{i:05d}.npy"
        np.save(label_dir / label_name,
                np.stack([instance_map, class_map], axis=0))

        # types.csv row
        t = types[i]
        tissue_name = t if isinstance(t, str) else TISSUE_TYPES.get(int(t), "Unknown")
        types_rows.append([img_name, tissue_name])

        # cell_count.csv row
        cell_count_rows.append(
            [img_name] + cell_counts
        )

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{N}")

    # Write types.csv
    with open(out_dir / "types.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "tissue_type"])
        writer.writerows(types_rows)

    # Write cell_count.csv
    with open(out_dir / "cell_count.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "Neoplastic", "Inflammatory",
                         "Connective", "Dead", "Epithelial"])
        writer.writerows(cell_count_rows)

    print(f"  Fold {fold_idx} done. Saved to {out_dir}")
    return types_rows, cell_count_rows


def show_sample_images(input_path, output_path, n_samples=6):
    """Show sample images with their masks visually."""
    print("\nGenerating sample visualization...")
    fold_dir = Path(output_path) / "fold0" / "images"
    label_dir = Path(output_path) / "fold0" / "labels"

    imgs = sorted(list(fold_dir.glob("*.png")))[:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    fig.suptitle("PanNuke Preprocessing — Sample Images and Masks",
                 fontsize=13, fontweight="bold")

    for j, img_path in enumerate(imgs):
        img = np.array(Image.open(img_path))
        label_path = label_dir / (img_path.stem + ".npy")
        label = np.load(label_path)
        class_map = label[1]

        # Image
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"Image {j}", fontsize=8)
        axes[0, j].axis("off")

        # Colored mask
        color_mask = np.zeros((*class_map.shape, 3), dtype=np.uint8)
        for c, color in CLASS_COLORS.items():
            color_mask[class_map == c + 1] = color

        overlay = img.copy()
        mask_area = class_map > 0
        overlay[mask_area] = (
            0.5 * img[mask_area] +
            0.5 * color_mask[mask_area]
        ).astype(np.uint8)

        axes[1, j].imshow(overlay)
        axes[1, j].set_title("Mask overlay", fontsize=8)
        axes[1, j].axis("off")

    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=NUCLEUS_CLASSES[i])
        for i, c in CLASS_COLORS.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    Path("results").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/preprocessing_samples.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("  Saved: results/preprocessing_samples.png")


if __name__ == "__main__":
    INPUT  = "./data_raw"
    OUTPUT = "./data_processed"

    for fold in range(3):
        process_fold(fold, INPUT, OUTPUT)

    show_sample_images(INPUT, OUTPUT)
    print("\nPreprocessing complete.")