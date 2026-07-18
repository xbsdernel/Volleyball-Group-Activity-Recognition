# Group Activity Recognition — Baseline Implementations

This repository contains from-scratch implementations of multiple baseline models from existing research papers on **Group Activity Recognition (GAR)**. The goal is to reproduce, standardize, and compare these baselines under a common, modular training pipeline.

Eight baselines are planned in total. The repository is under active development, with baselines being added incrementally.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Implemented Baselines](#implemented-baselines)
- [Current Implementation](#current-implementation)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Project Overview

**Group Activity Recognition (GAR)** is the task of classifying the collective activity performed by a group of interacting individuals in a video (e.g., a volleyball team executing a "spike" or "set" play), as opposed to recognizing the action of a single actor in isolation. GAR requires modeling both:

- **Individual-level dynamics** — the appearance and motion of each person in the scene.
- **Group-level context** — how individuals' actions relate to and influence one another over time.

This makes GAR substantially more challenging than single-person action recognition, and it has motivated a range of architectural approaches — from hierarchical recurrent models to graph-based and transformer-based relational reasoning.

The purpose of this repository is to:

1. **Reproduce** published baseline methods for GAR under a consistent codebase and dataset split.
2. **Isolate the contribution of each architectural idea** (temporal modeling, person-level reasoning, relational modeling, etc.) by comparing baselines under identical training conditions.
3. **Provide a reference implementation** that is clean, modular, and easy to extend with new baselines or datasets.

All baselines are evaluated on the same benchmark dataset and split protocol to ensure fair comparison.

---


## Current Implementation

### Baseline 1 — Hierarchical Deep Temporal Models for Group Activity Recognition

**Main idea**
The original paper proposes a two-level hierarchical temporal architecture: individual person dynamics are modeled first, then aggregated into a group-level representation that captures the collective temporal evolution of the scene, which is finally used to classify the overall group activity.

**High-level architecture (target design)**

```
Frame → Per-Person CNN Features → Person-level LSTM
                                        │
                                pooling over persons
                                        ▼
                                Scene-level LSTM → Group Activity Classification
```

**Temporal modeling approach**
The full paper models temporal dynamics at two levels — a person-level LSTM that captures each individual's motion over time, and a scene-level LSTM that aggregates pooled person representations across frames to capture group-level temporal dynamics.

> **Current scope:** the present implementation covers the **CNN visual feature classification stage** of this baseline — a ResNet-50 backbone fine-tuned to classify group activity directly from single frames. This serves as the feature-extraction / frame-level baseline stage of the pipeline. The person-level and scene-level LSTM temporal stages described above are the next milestone for this baseline and are not yet included in the current codebase.

**Training pipeline**

1. Parse per-video annotation files into `(image_path, activity_label)` pairs.
2. Split data by **video ID** into train / validation / test sets (no frame leakage across splits).
3. Apply standard preprocessing: resize to 128×128, normalize with ImageNet statistics.
4. Fine-tune a pretrained **ResNet-50** with a new classification head (8 activity classes).
5. Optimize with **AdamW** + **Cosine Annealing** learning rate schedule.
6. Track training/validation loss and accuracy each epoch.
7. Apply **early stopping** on validation loss with checkpointing of the best model.
8. Evaluate final accuracy on the held-out test split.

**Main code components**

| Component | File |
|---|---|
| Config / hyperparameters | `configs/config.py` |
| Annotation parsing & split | `datasets/prepare_dataset.py` |
| Dataset class | `datasets/custom_dataset.py` |
| Preprocessing | `datasets/transforms.py` |
| Model definition | `models/resnet50.py` |
| Train / validate / test loops | `engine/train.py`, `engine/validate.py`, `engine/test.py` |
| Checkpointing & early stopping | `utils/checkpoint.py`, `utils/early_stopping.py` |
| Entry points | `train.py`, `evaluate.py`, `inference.py` |

---

## Repository Structure

```
group-activity-recognition-baselines/
│
├── README.md
├── requirements.txt
│
├── baselines/
│   ├── baseline_01_hierarchical_deep_temporal/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── configs/
│   │   ├── datasets/
│   │   ├── models/
│   │   ├── engine/
│   │   ├── utils/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── plots/
│   │
│   ├── baseline_02_.../          # planned
│   ├── baseline_03_.../          # planned
│   ├── ...
│   └── baseline_08_.../          # planned
│
├── docs/
│   └── papers/                   # reference papers / notes per baseline
│
└── notebooks/                    # exploratory / original prototyping notebooks
```

---

## How to Run

### Installation

```bash
git clone https://github.com/<your-username>/group-activity-recognition-baselines.git
cd group-activity-recognition-baselines/baselines/baseline_01_hierarchical_deep_temporal
pip install -r requirements.txt
```

Update the dataset path in `configs/config.py` (`ROOT_DIR`) before running.

### Training

```bash
python train.py
```

Trains the model and saves the best checkpoint to `checkpoints/best_model.pth`, along with loss/accuracy curves in `plots/`.

### Evaluation

```bash
python evaluate.py
```

Loads the best checkpoint and reports accuracy on the held-out test split.

---

## Results

Results will be populated as each baseline is completed, using the same dataset split and evaluation protocol.

| # | Method                                   | Accuracy |
| - | ---------------------------------------- | -------- |
| 1 | B1 - Image Classification                |      |
| 2 | B2 - Person Classification               |      |
| 3 | B3 - Fine-tuned Person Classification    |      |
| 4 | B4 - Temporal Model with Image Features  |      |
| 5 | B5 - Temporal Model with Person Features |      |
| 6 | B6 - Two-stage Model without LSTM 1      |      |
| 7 | B7 - Two-stage Model without LSTM 2      |      |
| 8 | Two-stage Hierarchical Model             |      |


---
