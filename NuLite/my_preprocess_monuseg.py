"""
my_preprocess_monuseg.py
Preprocesses MoNuSeg dataset:
- Parses XML annotations to extract nucleus polygon boundaries
- Converts whole-slide .tif images into 256x256 patches
- Saves binary masks (no cell type labels in MoNuSeg)
- Saves PNG images and numpy mask files
As described in Section IV.B of the submitted paper:
"We processed the 14 whole-slide images into 126 patches of 256x256 pixels
by parsing the XML annotations and extracting binary masks."
"""

import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_xml_annotations(xml_path):
    """
    Parse MoNuSeg XML annotation file.
    Returns list of polygon coordinates for each nucleus.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []

    # MoNuSeg XML structure: Annotation > Regions > Region > Vertices > Vertex
    for region in root.iter("Region"):
        vertices = []
        for vertex in region.iter("Vertex"):
            x = float(vertex.get("X", 0))
            y = float(vertex.get("Y", 0))
            vertices.append((x, y))
        if len(vertices) >= 3:
            polygons.append(vertices)

    return polygons


def create_binary_mask(polygons, img_size):
    """Create binary mask from polygon annotations."""
    mask = Image.new("L", (img_size[1], img_size[0]), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 3:
            flat = [(int(x), int(y)) for x, y in poly]
            draw.polygon(flat, fill=255)
    return np.array(mask)


def extract_patches(img_array, mask_array, patch_size=256, stride=256):
    """Extract non-overlapping patches from image and mask."""
    H, W = img_array.shape[:2]
    patches_img  = []
    patches_mask = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch  = img_array[y:y+patch_size, x:x+patch_size]
            mask_patch = mask_array[y:y+patch_size, x:x+patch_size]

            # Skip patches with no nuclei
            if mask_patch.max() == 0:
                continue

            patches_img.append(img_patch)
            patches_mask.append(mask_patch)

    return patches_img, patches_mask


def process_split(split_dir, output_dir, split_name):
    """Process train or test split."""
    split_dir  = Path(split_dir)
    output_dir = Path(output_dir) / split_name
    img_dir    = output_dir / "images"
    mask_dir   = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(split_dir.glob("*.tif"))
    print(f"\n{split_name}: Found {len(tif_files)} WSI files")

    total_patches = 0
    patch_info    = []

    for tif_path in tif_files:
        xml_path = tif_path.with_suffix(".xml")
        if not xml_path.exists():
            print(f"  No XML for {tif_path.name}, skipping")
            continue

        print(f"  Processing {tif_path.name}...")

        # Load image
        try:
            img = np.array(Image.open(tif_path).convert("RGB"))
        except Exception as e:
            print(f"    Error loading image: {e}")
            continue

        # Parse annotations
        try:
            polygons = parse_xml_annotations(xml_path)
        except Exception as e:
            print(f"    Error parsing XML: {e}")
            continue

        # Create binary mask
        mask = create_binary_mask(polygons, img.shape[:2])

        # Extract patches
        img_patches, mask_patches = extract_patches(img, mask)
        print(f"    Image shape: {img.shape} | Polygons: {len(polygons)} | Patches: {len(img_patches)}")

        # Save patches
        stem = tif_path.stem
        for i, (img_p, mask_p) in enumerate(zip(img_patches, mask_patches)):
            name = f"{stem}_{i:03d}"
            Image.fromarray(img_p.astype(np.uint8)).save(img_dir / f"{name}.png")
            np.save(mask_dir / f"{name}.npy", (mask_p > 0).astype(np.uint8))
            patch_info.append(name)
            total_patches += 1

    print(f"\n{split_name} total patches saved: {total_patches}")
    return total_patches, patch_info


def show_monuseg_samples(output_dir, split_name="test", n=6):
    """Show sample patches with binary masks overlaid."""
    img_dir  = Path(output_dir) / split_name / "images"
    mask_dir = Path(output_dir) / split_name / "masks"

    imgs = sorted(list(img_dir.glob("*.png")))[:n]
    if not imgs:
        print("No images found for visualization.")
        return

    fig, axes = plt.subplots(2, len(imgs), figsize=(len(imgs)*3, 6))
    fig.suptitle(f"MoNuSeg Preprocessing — {split_name} Samples",
                 fontsize=13, fontweight="bold")

    for j, img_path in enumerate(imgs):
        img  = np.array(Image.open(img_path))
        mask = np.load(mask_dir / (img_path.stem + ".npy"))

        axes[0, j].imshow(img)
        axes[0, j].set_title(f"Patch {j+1}", fontsize=8)
        axes[0, j].axis("off")

        overlay = img.copy().astype(np.float32)
        overlay[mask > 0, 0] = 255
        overlay[mask > 0, 1] = overlay[mask > 0, 1] * 0.4
        overlay[mask > 0, 2] = overlay[mask > 0, 2] * 0.4
        axes[1, j].imshow(overlay.astype(np.uint8))
        axes[1, j].set_title("Binary mask", fontsize=8)
        axes[1, j].axis("off")

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/monuseg_preprocessing_samples.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: results/monuseg_preprocessing_samples.png")


if __name__ == "__main__":
    INPUT_TRAIN  = "./data_raw/monuseg/train"
    INPUT_TEST   = "./data_raw/monuseg/test"
    OUTPUT       = "./data_processed/monuseg"

    n_train, _ = process_split(INPUT_TRAIN, OUTPUT, "train")
    n_test,  _ = process_split(INPUT_TEST,  OUTPUT, "test")

    print(f"\nTotal train patches: {n_train}")
    print(f"Total test patches : {n_test}")

    show_monuseg_samples(OUTPUT, split_name="test")
    print("\nMoNuSeg preprocessing complete.")