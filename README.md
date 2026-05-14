# ⚠️ IMPORTANT NOTE

**👉 ONLY DOWNLOAD / USE THE `NuLite/` FOLDER FOR INFERENCE AND MODEL TESTING.**  
All required code, checkpoints, configs, and inference scripts are inside this folder.

---

## 📁 NuLite Project Structure

```text
NuLite/
├── checkpoints/
│   ├── baseline_best.pth
│   ├── exp1_best.pth
│   ├── exp2_best.pth
│   └── monuseg_best.pth
├── data/
│   └── sample_data.csv
├── data_processed/
│   ├── fold2/
│   └── monuseg/
│       └── test/
├── notebooks/
│   ├── results/
│   │   └── inference/
│   └── 01_inference_demo.ipynb
├── results/
│   ├── inference/
│   ├── logs/
│   ├── A2_Baseline_2ep_metrics.json
│   ├── Exp1_Loss_Mod_20ep_metrics.json
│   ├── Exp2_Attn_Gate_15ep_metrics.json
│   ├── monuseg_Exp1_Zero-Shot.json
│   ├── monuseg_MoNuSeg_Trained.json
│   ├── monuseg_predictions_Exp1_Zero-Shot.json
│   ├── monuseg_predictions_MoNuSeg_Trained.json
│   ├── monuseg_preprocessing_samples.png
│   └── preprocessing_samples.png
├── src/
│   ├── __pycache__/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── utils.py
├── config.yaml
├── inference.py
├── my_dataset.py
├── my_evaluate.py
├── my_evaluate_monuseg.py
├── my_model.py
├── my_preprocess.py
├── my_preprocess_monuseg.py
├── my_train_exp1.py
├── my_train_exp2.py
├── my_train_monuseg.py
├── requirements.txt
├── summary.py
└── train.py

---

## Experiments

### Baseline Reproduction
Reproduced the original NuLite-T pipeline on PanNuke.
Verified training loop, loss functions, metrics, and output shapes.
Used as the reference point for all experimental comparisons.

---

### Experiment 1 — Loss Function Modification

Two changes made to the loss configuration:

| Branch | Original | Experiment 1 |
|--------|----------|--------------|
| nuclei_binary_map | Dice + FocalTversky | Dice + FocalTversky **(emphasized)** |
| hv_map | MSE + MSGE | MSE + MSGE + **Boundary Loss** |
| nuclei_type_map | BCE + Dice + MCFocalTversky | Unchanged |
| tissue_types | CrossEntropy | Unchanged |

**Goal:** Improve nucleus detection under class imbalance and sharpen
instance boundary separation for the watershed post-processing step.

Also includes cross-domain zero-shot evaluation on MoNuSeg without
any retraining or fine-tuning.

---

### Experiment 2 — Attention Gate Addition

An `AttentionGate` module inserted at the final skip connection of the
NuLite decoder, between `decoder0` and the concatenation step.

**Goal:** Suppress background encoder features and amplify nucleus-relevant
spatial regions, improving performance on rare and visually subtle classes
such as Inflammatory and Epithelial nuclei.

---

## Results Summary

### PanNuke Fold 2 — All Methods

| Metric | Paper (130ep) | Baseline (2ep) | Exp 1 (20ep) | Exp 2 (15ep) |
|--------|:-------------:|:--------------:|:------------:|:------------:|
| mPQ | 0.4762 | 0.3097 | 0.2117 | **0.3965** |
| bPQ | 0.6817 | 0.4982 | 0.4787 | **0.5703** |
| F1 Detection | 0.8204 | 0.7441 | 0.7235 | **0.7761** |
| Binary Dice | 0.7103 | 0.7396 | 0.7331 | **0.7758** |
| PQ Neoplastic | 0.5121 | 0.3674 | 0.3940 | **0.4758** |
| PQ Inflammatory | 0.4213 | 0.3310 | 0.0144 | **0.3922** |
| PQ Connective | 0.3984 | 0.2690 | 0.2123 | **0.3132** |
| PQ Dead | 0.2011 | 0.0000 | 0.0000 | 0.0000 |
| PQ Epithelial | 0.4491 | 0.1900 | 0.0014 | **0.4040** |

### Cross-Domain Generalization — MoNuSeg (Experiment 1, Zero-Shot)

| Dataset | Dice | Jaccard | Std Dice | Std Jaccard |
|---------|:----:|:-------:|:--------:|:-----------:|
| PanNuke Fold 2 (in-domain) | 0.7331 | 0.6195 | — | — |
| MoNuSeg (cross-domain) | **0.7473** | 0.6005 | 0.0632 | 0.0773 |

---

## Datasets

### PanNuke
- 7,901 image patches from 19 tissue types
- 256×256 pixels with instance-level nucleus annotations
- Five nucleus classes: Neoplastic, Inflammatory, Connective, Dead, Epithelial
- Split: Fold 0 (train) · Fold 1 (val) · Fold 2 (test)
- Download: [Warwick TIA Centre](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)

### MoNuSeg
- 14 whole-slide tissue images across 8 organs
- Binary nucleus masks only — no cell type labels
- Used exclusively for cross-domain zero-shot evaluation
- Organs: Breast · Liver · Kidney · Prostate · Colon · Bladder · Stomach · Brain
- Download: [Grand Challenge](https://monuseg.grand-challenge.org/)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mPQ** | Mean Panoptic Quality — primary metric combining detection and classification across all five nucleus classes |
| **bPQ** | Binary Panoptic Quality — segmentation quality independent of class labels |
| **F1 Detection** | Harmonic mean of centroid detection precision and recall |
| **Binary Dice** | Pixel-level overlap between predicted and ground truth nucleus masks |
| **Per-class PQ** | Panoptic Quality broken down for each of the five nucleus types |
| **Jaccard (IoU)** | Stricter overlap metric used on MoNuSeg evaluation |

---

## Outputs

The repository includes the following saved outputs:

- Trained model checkpoints for all experiments
- Per-epoch training and validation loss logs
- PanNuke evaluation results (all experiments)
- MoNuSeg cross-domain evaluation results (Experiment 1)
- Notebook with visualizations, metric summaries, and experiment comparisons

---

## References

1. Tommasino et al., *NuLite: Lightweight and Fast Model for Nuclei Instance Segmentation*, BSPC 2025
2. Gamper et al., *PanNuke Dataset*, ECDP 2019
3. Graham et al., *HoVer-Net*, Medical Image Analysis 2019
4. Vasu et al., *FastViT*, ICCV 2023
5. Oktay et al., *Attention U-Net*, MIDL 2018
6. Abraham & Khan, *Focal Tversky Loss*, ISBI 2019
7. Kumar et al., *MoNuSeg*, IEEE TMI 2017
8. Horst et al., *CellViT*, Medical Image Analysis 2024

---

## Contributors

| Name | Student ID |
|------|-----------|
| Dania Waseem | 23i-2622 |
| Wajeeha Khalid | 23i-2610 |
| Taiba Tariq | 23i-2618 |
