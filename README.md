# NuLite

## Overview

This repository contains our implementation and extension of the **NuLite model** for nuclei instance segmentation and classification in histopathology images. The work builds on a verified baseline and explores two modifications to study model behavior under limited training conditions.

The project includes:

* Baseline reproduction of NuLite
* Two experimental modifications
* Evaluation on in-domain and cross-domain datasets
* Inference pipelines and saved outputs

---

## Experiments

### Baseline

* Reproduced original NuLite pipeline
* Verified training, losses, and metrics
* Used as reference for comparison with later experiments

---

### Experiment 1 — Loss Function Modification

* Modified the **nuclei binary detection branch**
* Increased emphasis on **FocalTversky Loss**
* Other branches (HV map, type map, tissue classification) kept unchanged

**Goal:**
Improve nucleus detection under class imbalance and analyze its effect on segmentation quality.

**Additional Evaluation:**

* Cross-domain generalization tested on a different dataset (MoNuSeg) without retraining

---

### Experiment 2 — Attention Gate Addition

* Introduced an **Attention Gate** in the decoder
* Applied before final feature fusion (skip connection)

**Goal:**
Improve feature selection by suppressing background and enhancing nucleus-relevant regions, especially for difficult classes.

---

## Generalization

Generalization was explicitly evaluated by:

* Training on PanNuke dataset
* Testing on:

  * PanNuke test split (in-domain)
  * MoNuSeg dataset (cross-domain, zero-shot)

This measures how well the model transfers to unseen tissues and imaging conditions.

---

## Evaluation Metrics

Performance is measured using:

* **mPQ (mean Panoptic Quality)** — overall segmentation + classification quality
* **bPQ (binary PQ)** — segmentation quality without class labels
* **Dice Score** — pixel-level overlap
* **Jaccard Index (IoU)** — stricter overlap metric
* **F1 Detection** — nucleus detection accuracy
* **Per-class PQ** — performance across nucleus types

---

## Outputs

The repository includes:

* Trained model checkpoints
* Training logs and loss curves
* Inference outputs (JSON / CSV)
* Notebook containing:

  * Visualizations
  * Metric summaries
  * Experiment comparisons

---

## Datasets

* **PanNuke**

  * Used for training, validation, and testing
  * Multi-class nuclei annotations across tissue types

* **MoNuSeg**

  * Used for cross-domain evaluation
  * Binary nucleus segmentation (no type labels)

---

## References

* NuLite architecture and pretrained weights
* PanNuke dataset
* MoNuSeg dataset
* Focal Tversky Loss
* Attention U-Net / Attention Gates
* HoVer-Net and related nuclei segmentation work

---

## Contributors

* Dania Waseem
* Wajeeha Khalid
* Taiba Tariq
