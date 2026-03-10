# 02 — Clustering Pre-Training

## Research Question

Can a differentiable clustering loss — one that operates *within* the gradient flow — learn representations that support classification better than the non-differentiable centroid projection from [Experiment 01](../01_centroid_projection/README.md)? And does pre-training a backbone with geometric objectives actually help, or does the model reach the same place training from scratch?

## Background

Experiment 01 showed that direct geometric manipulation (CentroidProjection) hurts accuracy at every strength tested. The fundamental problem was the straight-through estimator: the nudge shifted activations but was invisible to backpropagation, creating a train/test distribution shift. Finding #7 from that experiment noted that a loss-based approach would avoid this entirely by putting the geometric objective inside the gradient flow.

This experiment tests that hypothesis with two-phase training:
1. **Phase 1 (Clustering):** Train a backbone with a geometric loss — no classification head
2. **Phase 2 (Classification):** Freeze the backbone, attach new trainable layers, train with CrossEntropyLoss

## Shared Setup

All models trained on MNIST (48k train / 12k val / 10k test), `seed=42`, `batch_size=512`, Adam `lr=0.001`.

Backbone architecture: `784 → fc1(128, BN, Dropout) → fc2(64, LN, Dropout)`

## Key Differences from Experiment 01

| Aspect | 01: CentroidProjection | 02: Clustering Pre-Training |
|--------|----------------------|---------------------------|
| Mechanism | Direct geometric nudge (straight-through) | Differentiable loss (full gradient flow) |
| When applied | During forward pass alongside classification | Separate pre-training phase before classification |
| Centroids | Pre-computed per epoch from full dataset | Moving batch centroids (NB01) or fixed simplex (NB02) |
| Classification | Trained jointly with projection | Trained separately on frozen features |

---

## Phase 1: Clustering Backbone Training (Notebooks 01–02)

Two approaches to learning geometrically structured representations, both training the same 784→128→64 backbone.

### NB01 — Moving Centroid Loss

Differentiable clustering loss: intra-class pull (attract to own centroid) + inter-class push (repel other centroids via margin hinge loss). Centroids are recomputed per batch.

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Margin | 10.0 |
| Intra weight | 1.0 |
| Inter weight | 0.5 |
| L1 lambda | 1e-4 |
| Dropout | 0.3 |
| Epochs (Phase 1) | 5 |

**Phase 1 Training (Backbone):**

| Epoch | Intra Loss | Inter Loss | Total Loss |
|-------|-----------|-----------|-----------|
| 0 | 9.1353 | 8.6157 | 13.4431 |
| 1 | 2.5257 | 9.1866 | 7.1190 |
| 2 | 0.8598 | 9.3144 | 5.5170 |
| 3 | 0.3125 | 9.3721 | 4.9986 |
| 4 | 0.1865 | 9.3944 | 4.8837 |

Intra-class loss dropped from 9.1 to 0.2 — points condensed toward their centroids. But inter-class loss barely moved (8.6→9.4), meaning centroids never pushed far enough apart to create clean separation.

**Phase 2 Training (Classification Head: 64→10, frozen backbone):**

Trainable: **650** / 109,770 total params (just the linear head).

| Epoch | Train Loss |
|-------|-----------|
| 0 | 2.2124 |
| 1 | 2.0520 |
| 2 | 1.9198 |
| 3 | 1.8086 |
| 4 | 1.7154 |

**Final val accuracy: 0.5447**
**Final test accuracy: 0.5435**

A failure. The representations learned by the moving centroid loss were not linearly separable — 54% is barely better than random for a 10-class problem. The loss succeeded at condensing clusters but the centroids themselves were too close together and the representation geometry wasn't structured for a linear classifier.

### NB02 — Fixed Simplex Targets

Instead of moving centroids, define 10 target points as vertices of a regular simplex in 64D space (all equidistant, scale=5.0). Train the backbone with MSE loss to push each class's activations toward its fixed target vertex.

**Pairwise target distances:** min=7.4536, max=7.4536, mean=7.4536 (perfectly equidistant by construction).

**Hyperparameters:** Same as NB01 except the clustering loss is MSE to simplex targets.

**Phase 1 Training (Backbone):**

| Epoch | MSE Loss |
|-------|---------|
| 0 | 0.4477 |
| 1 | 0.2303 |
| 2 | 0.1985 |
| 3 | 0.1892 |
| 4 | 0.1853 |

MSE dropped to 0.19 — the backbone learned to map each class near its target vertex.

**Phase 2 Training (Classification Head: 64→10, frozen backbone):**

Trainable: **650** / 109,770 total params.

| Epoch | Train Loss |
|-------|-----------|
| 0 | 2.1427 |
| 1 | 1.7385 |
| 2 | 1.4485 |
| 3 | 1.2521 |
| 4 | 1.1318 |

**Final val accuracy: 0.9417**
**Final test accuracy: 0.9464**

Dramatic improvement over NB01 (54% → 94.6%). Fixed simplex targets gave the backbone a stable, well-separated objective. The pre-defined equidistant geometry ensured clean inter-class separation that the moving centroids in NB01 failed to achieve.

This backbone checkpoint is saved to `checkpoints/simplex_backbone.pt` and reused in NB03–05.

### Phase 1 Summary

| NB | Clustering Method | Phase 2 Test Acc |
|----|------------------|-----------------|
| 01 | Moving centroids (intra pull + inter push) | 0.5435 |
| 02 | Fixed simplex targets (MSE) | 0.9464 |

**Key insight:** The clustering objective matters enormously. Moving centroids condensed points but left centroids too close together. Fixed simplex targets guaranteed equidistant separation by construction, producing linearly separable representations.

---

## Phase 2: Extended Models (Notebooks 03–05)

NB02's frozen backbone + linear head reached 94.6%. Can we do better by adding trainable hidden layers on top of the frozen backbone? These notebooks load the simplex backbone from NB02's checkpoint, freeze it, and train new layers.

### NB03 — Extended Simplex (Heavy Regularization)

Adds two new hidden layers: 64→48→32→10, with BN, LN, Dropout(0.3), L1(1e-4).

**Trainable: 5,178** / 114,298 total params.

| Epoch | Train Loss | Val Loss | Val Acc | Test Acc |
|-------|-----------|---------|---------|----------|
| 0 | 1.6104 | 0.4351 | 0.9586 | 0.9638 |
| 1 | 1.0933 | 0.2402 | 0.9575 | 0.9638 |
| 2 | 0.9989 | 0.2041 | 0.9571 | 0.9636 |
| 3 | 0.9590 | 0.1964 | 0.9561 | 0.9629 |
| 4 | 0.9381 | 0.1974 | 0.9567 | 0.9631 |

**Final val accuracy: 0.9567**
**Final test accuracy: 0.9631**

A +1.7% improvement over NB02's linear head (94.6% → 96.3%). But the train/val loss gap is striking: train loss (~0.94) is **~4.7x** higher than val loss (~0.20). This is *underfitting*, not overfitting — Dropout(0.3) and L1(1e-4) are too aggressive for only 5K trainable params. The regularization is adding ~0.7 to training loss for no generalization benefit.

### NB04 — Extended Simplex (Reduced Regularization)

Same architecture as NB03, but reduced Dropout (0.3→0.1) and L1 (1e-4→1e-5).

**Trainable: 5,178** / 114,298 total params.

**Final val accuracy: 0.9578**
**Final test accuracy: 0.9636** (peak 0.9668 at best checkpoint)

Slight improvement over NB03. The train/val loss gap narrowed to ~2.5x. Reducing regularization helped but didn't fully close the gap — the remaining inflation comes from residual dropout and L1.

### NB05 — Extended Simplex (No Regularization on Head)

Removes all explicit regularization on the new layers: head dropout 0.0 (backbone stays 0.1), L1 removed entirely. Wider fc3 (48→64, removes bottleneck). 15 epochs for more convergence time.

**Trainable: 6,762** / 115,882 total params.

**Final val accuracy: 0.9454**
**Final test accuracy: 0.9486**

Worse than NB03 and NB04. With the wider fc3 and no regularization, the head has more capacity but is still bottlenecked by frozen 64D backbone features. The longer training and zero regularization didn't help — the frozen backbone's representations are the ceiling, and the extra capacity found no useful signal beyond what the regularized versions already captured.

### Extended Model Summary

| NB | Head Config | Trainable | Test Acc | Notes |
|----|------------|-----------|----------|-------|
| 02 | Linear 64→10 | 650 | 0.9464 | Simple head |
| 03 | 64→48→32→10, heavy reg | 5,178 | 0.9631 | Underfitting (4.7x gap) |
| 04 | 64→48→32→10, low reg | 5,178 | 0.9636 | Gap narrowed (2.5x) |
| 05 | 64→64→32→10, no reg | 6,762 | 0.9486 | Wider but worse |

**Key insight:** Adding capacity on top of frozen features helps (94.6% → 96.4%), but the frozen backbone is the bottleneck. Regularization tuning has diminishing returns when the representation itself is fixed.

---

## Phase 3: From-Scratch Control (Notebook 06)

The critical question: does the simplex pre-training actually help, or would the same architecture reach a higher accuracy if all parameters were trainable from the start?

### NB06 — From-Scratch Control

Identical architecture to NB05 (784→128→64→64→32→10, same dropout pattern, same hyperparameters). Random init, **all** params trainable. No simplex backbone, no frozen layers.

**Trainable: 115,882** / 115,882 total params (vs. 6,762 in NB05).

**Final val accuracy: 0.9786**
**Final test accuracy: 0.9785**

The from-scratch model substantially outperforms every frozen backbone variant:

| Model | Test Acc | Δ vs NB06 |
|-------|----------|----------|
| NB02 (simplex + linear head) | 0.9464 | -3.2% |
| NB03 (simplex + extended, heavy reg) | 0.9631 | -1.5% |
| NB04 (simplex + extended, low reg) | 0.9636 | -1.5% |
| NB05 (simplex + extended, no reg) | 0.9486 | -3.0% |
| **NB06 (from scratch)** | **0.9785** | **—** |

---

## Findings

### 1. Fixed Simplex Targets >> Moving Centroids

NB01's moving centroid loss produced representations that were barely classifiable (54%). NB02's fixed simplex targets reached 94.6% with the same backbone and training setup. The difference is architectural: fixed targets guarantee equidistant inter-class separation by construction, while moving centroids can converge to degenerate configurations where classes overlap.

### 2. Differentiable Clustering Produces Usable (But Not Optimal) Representations

NB02's simplex backbone produced representations that a linear head could classify at 94.6% — far better than Experiment 01's centroid projection, which *degraded* accuracy from 96.3% to 95.2% (best case). Putting the geometric objective inside the gradient flow works: the backbone learned to organize activations toward the target geometry, and those representations transferred to classification.

### 3. Adding Capacity Helps, But the Frozen Backbone Is the Ceiling

Replacing the linear head with two hidden layers improved accuracy from 94.6% to 96.4% (NB04). But all frozen-backbone variants plateaued in the 94–97% range regardless of head size, regularization, or training time. The backbone's fixed 64D representation is the bottleneck.

### 4. Heavy Regularization on Small Param Counts Causes Underfitting

NB03's train/val loss gap of 4.7x (train loss *higher* than val loss) is a textbook underfitting signal. With only 5,178 trainable params, Dropout(0.3) and L1(1e-4) were far too aggressive — they added noise and penalty for no generalization benefit. Reducing regularization (NB04) helped, but even minimal regularization was unnecessary at this param count.

### 5. The Simplex Pre-Training Doesn't Help

NB06 (from scratch, 97.9%) beat every frozen backbone variant by 1.5–3.2%. End-to-end training with backpropagation learns representations that are at least as geometrically structured as the simplex-trained backbone — and more useful for classification, because the full model can co-adapt all layers to the classification objective.

This mirrors Experiment 01's conclusion: at optimal settings, the projection was a wash. The model learns equivalent geometric structure on its own through standard cross-entropy optimization.

### 6. The Core Research Answer

Neither non-differentiable manipulation (Experiment 01) nor differentiable pre-training (Experiment 02) improves on what end-to-end training achieves. Standard backpropagation with cross-entropy loss already produces well-clustered, linearly separable representations. Explicitly forcing geometric structure — whether through activation nudging or loss-based pre-training — adds constraints that limit the model rather than helping it.

---

## Architecture Reference

**Backbone (NB01–02, frozen in NB03–05):**
```
Input (784) → fc1(128) → BatchNorm → ReLU → Dropout(0.3)
            → fc2(64)  → LayerNorm → ReLU → Dropout(0.3)
```

**Extended head (NB03–04):**
```
→ fc3(48)  → BatchNorm → ReLU → Dropout
→ fc4(32)  → LayerNorm → ReLU → Dropout
→ fc5(10)  → Output
```

**Extended head (NB05–06, wider fc3):**
```
→ fc3(64)  → BatchNorm → ReLU → Dropout(0.0)
→ fc4(32)  → LayerNorm → ReLU → Dropout(0.0)
→ fc5(10)  → Output
```

## Hyperparameters

### Shared (Phase 1)

| Parameter | Value |
|-----------|-------|
| LR | 0.001 |
| Weight Decay | 1e-4 |
| L1 Lambda | 1e-4 |
| Dropout | 0.3 |
| Batch Size | 512 |
| Phase 1 Epochs | 5 |

### NB01 — Moving Centroids

| Parameter | Value |
|-----------|-------|
| Margin | 10.0 |
| Intra Weight | 1.0 |
| Inter Weight | 0.5 |

### NB02 — Simplex Targets

| Parameter | Value |
|-----------|-------|
| Target Scale | 5.0 |

### Extended Models (NB03–06)

| Parameter | NB03 | NB04 | NB05 | NB06 |
|-----------|------|------|------|------|
| Head Dropout | 0.3 | 0.1 | 0.0 | 0.0 |
| Backbone Dropout | 0.3 | 0.1 | 0.1 | 0.1 |
| L1 Lambda | 1e-4 | 1e-5 | 0 | 0 |
| fc3 Width | 48 | 48 | 64 | 64 |
| Epochs | 5 | 5 | 15 | 15 |
| Backbone Frozen | Yes | Yes | Yes | **No** |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) | same | same | same |

## Tracker Runs

- **`cluster_pretrain`** — NB01 Phase 1 (moving centroid backbone)
- **`cluster_classify`** — NB01 Phase 2 (classification on moving centroid features)
- **`simplex_pretrain`** — NB02 Phase 1 (simplex target backbone)
- **`simplex_classify`** — NB02 Phase 2 (classification on simplex features)
- **`simplex_extended`** — NB03 (extended model on frozen simplex backbone)
- **`simplex_extended_lowreg`** — NB04 (same architecture, reduced dropout + L1)
- **`simplex_extended_noreg`** — NB05 (wider fc3, no dropout/L1 on head, 15 epochs)
- **`from_scratch_control`** — NB06 (same architecture as 05, random init, all params trainable)

## Reproduction

Each notebook is self-contained. Run cells in order. Seed is fixed (`torch.manual_seed(42)`). NB02 saves the backbone checkpoint to `checkpoints/simplex_backbone.pt`; NB03–05 load it.

```
experimentation/02_clustering_pretrain/
├── notebooks/
│   ├── 01_cluster_then_classify.ipynb   # Moving centroids → classify (failed: 54%)
│   ├── 02_simplex_targets.ipynb         # Simplex MSE → classify (94.6%), saves backbone
│   ├── 03_extended_simplex.ipynb        # Frozen backbone + extended head, heavy reg (96.3%)
│   ├── 04_extended_low_reg.ipynb        # Reduced reg (96.4%)
│   ├── 05_extended_no_reg.ipynb         # No head reg, wider fc3 (94.9%)
│   └── 06_from_scratch_control.ipynb    # Same arch, from scratch, all trainable (97.9%)
├── checkpoints/
│   └── simplex_backbone.pt              # Saved by NB02, loaded by NB03–05
└── README.md
```
