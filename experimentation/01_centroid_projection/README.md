# 01 — Centroid Projection: Inference-Time Activation Nudging

## Research Question

Can we improve neural network representations by directly manipulating hidden-layer activations toward class centroids — and if so, can this manipulation be extended to inference time without access to ground-truth labels?

## Background

Standard neural networks learn representations implicitly through backpropagation. The final hidden layer's activations *tend* to cluster by class, but this is a side effect of optimizing cross-entropy loss, not an explicit objective. We hypothesized that directly nudging activations toward their class centroids during training could produce more structured, linearly separable representations.

**CentroidProjection** is a custom module that sits between hidden layers and the classifier. At each training step, it:

1. Computes per-class centroids from the full training set (updated each epoch)
2. Attracts each sample's activation toward its own-class centroid (controlled by `alpha`)
3. Repels centroids away from each other (controlled by `beta`)
4. Uses a straight-through estimator — the nudge shifts values but gradients flow through unchanged

This is *not* a learned transformation. It's direct geometric manipulation of the activation space, invisible to the backward pass.

## Shared Setup

All models trained on MNIST (48k train / 12k val / 10k test), `seed=42`, `batch_size=512`, Adam `lr=0.001`.

Architecture: `784 → fc1(128) → fc2(64) → fc3(10)`

## Phase 1: Baseline Stack (Notebooks 01–06)

Before introducing centroid projection, we built up a standard MLP incrementally to establish a strong baseline. Each notebook adds one technique on top of the previous.

### NB01 — Baseline MLP

Vanilla 784→128→64→10 with ReLU activations, no regularization.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 0.5830 |
| 1 | 0.2312 |
| 2 | 0.1742 |
| 3 | 0.1390 |
| 4 | 0.1100 |

**Test accuracy: 0.9598**

Reaches 96.0% in 5 epochs. Training loss (0.110) is well below validation loss, suggesting the model has capacity to spare and would benefit from regularization.

### NB02 — + Weight Decay (L2)

Adds `weight_decay=1e-4` to Adam.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 0.5828 |
| 1 | 0.2307 |
| 2 | 0.1738 |
| 3 | 0.1386 |
| 4 | 0.1099 |

**Test accuracy: 0.9604** (+0.06% vs NB01)

Negligible effect. At this scale, L2 regularization alone doesn't meaningfully constrain the model — training loss barely changed.

### NB03 — + BatchNorm

Adds `BatchNorm1d` after both hidden layers.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 0.6419 |
| 1 | 0.1845 |
| 2 | 0.1071 |
| 3 | 0.0726 |
| 4 | 0.0512 |

**Test accuracy: 0.9704** (+1.0% vs NB02)

The single biggest improvement in the entire baseline stack. Training loss dropped dramatically faster (0.64→0.05 across 5 epochs vs. 0.58→0.11 without BN), confirming that BatchNorm's internal normalization accelerates convergence and improves generalization.

### NB04 — + Dropout (0.3)

Adds `Dropout(0.3)` after both hidden activations.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 0.8358 |
| 1 | 0.3298 |
| 2 | 0.2379 |
| 3 | 0.1946 |
| 4 | 0.1652 |

**Test accuracy: 0.9643** (-0.6% vs NB03)

Accuracy *decreased* from NB03. Training loss was much higher at epoch 4 (0.165 vs. 0.051), showing dropout is actively slowing learning. At 5 epochs, the regularization penalty hasn't been offset by enough training. However, when trained for 10 epochs (NB11's Model A), this architecture reaches 97.5% — the regularization pays off with longer training.

### NB05 — + L1 Regularization (Elastic Net)

Adds `l1_lambda=1e-4` penalty to the loss.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 1.0173 |
| 1 | 0.4815 |
| 2 | 0.3827 |
| 3 | 0.3375 |
| 4 | 0.3090 |

**Test accuracy: 0.9659** (+0.16% vs NB04)

Marginal improvement. The L1 term encourages weight sparsity on top of L2 shrinkage, but the effect is small. Note that train loss is now dominated by the L1 penalty (~0.2 from regularization alone), making raw loss comparison to earlier notebooks misleading.

### NB06 — Replace BN(h2) with LayerNorm

Replaces `BatchNorm1d` on hidden_2 with `LayerNorm`.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 0.9822 |
| 1 | 0.4641 |
| 2 | 0.3870 |
| 3 | 0.3463 |
| 4 | 0.3218 |

**Test accuracy: 0.9630** (-0.3% vs NB05)

Slight accuracy loss. LayerNorm normalizes per-sample across features (not per-batch), eliminating the train/eval behavior difference that BatchNorm introduces. This consistency became important for the centroid projection experiments, where train/eval mismatch was already a concern. This architecture (BN on h1, LN on h2, Dropout, elastic net) is the baseline for all centroid projection work.

### Baseline Stack Summary

| NB | Technique | Cumulative Stack | Test Acc | Δ |
|----|-----------|------------------|----------|---|
| 01 | Baseline MLP | — | 0.9598 | — |
| 02 | + Weight Decay (1e-4) | L2 | 0.9604 | +0.06% |
| 03 | + BatchNorm | L2 + BN | 0.9704 | +1.00% |
| 04 | + Dropout (0.3) | L2 + BN + DO | 0.9643 | -0.61% |
| 05 | + L1 (1e-4) | L2 + BN + DO + L1 | 0.9659 | +0.16% |
| 06 | BN(h1) + LN(h2) | L2 + BN + LN + DO + L1 | 0.9630 | -0.29% |

**Key takeaway:** BatchNorm was the largest single gain. Dropout hurts at 5 epochs but helps at 10 (see NB11). The fully-regularized NB06 architecture at **96.3% / 5 epochs** and **97.5% / 10 epochs** is the baseline for all projection experiments.

---

## Phase 2: Training-Time Projection (Notebooks 07–08)

Applied CentroidProjection during training only (`alpha=0.3, beta=0.1`). At eval, the projection is a no-op since no ground-truth labels are available.

### NB07 — Centroid Projection (Last Layer Only)

Projection on hidden_2, applied during training when labels are available.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 1.1596 |
| 1 | 0.2439 |
| 2 | 0.1782 |
| 3 | 0.1492 |
| 4 | 0.1314 |

**Test accuracy: 0.9516** (-1.1% vs NB06)

Training loss dropped faster than the baseline (0.131 vs. 0.322 at epoch 4) — the model is fitting *easier*, pre-sorted data. But test accuracy was 1.1% worse. The classifier learned to depend on pre-sorted activations; at eval, when the projection was a no-op, the raw activations didn't match what the classifier expected.

### NB08 — Centroid Projection (Both Layers)

Projection on both hidden_1 and hidden_2.

**Training:**
| Epoch | Train Loss |
|-------|-----------|
| 0 | 1.1406 |
| 1 | 0.1990 |
| 2 | 0.1473 |
| 3 | 0.1187 |
| 4 | 0.1045 |

**Test accuracy: 0.8605** (-10.3% vs NB06)

Catastrophic degradation. The training loss kept decreasing (0.10 at epoch 4) while eval accuracy collapsed. The model perfectly fit the doubly-nudged training distribution while losing all ability to generalize to un-nudged eval activations. The more layers that are externally organized, the more the model's learned weights become dependent on that organization.

### Training-Time Projection Summary

| NB | Configuration | Test Acc | vs. NB06 |
|----|--------------|----------|----------|
| 07 | Projection last layer | 0.9516 | -1.1% |
| 08 | Projection both layers | 0.8605 | -10.3% |

---

## Phase 3: Inference-Time Projection (Notebooks 09–10)

NB07–08 had a fundamental train/test mismatch: the projection was a no-op at eval since no labels were available. NB09–10 close this gap by keeping the projection active at inference, using nearest-centroid assignment (`torch.cdist`) as pseudo-labels.

Importantly, 07/09 and 08/10 train identically — `infer` mode only affects eval. The only difference is what happens at test time.

### NB09 — Centroid Inference (Last Layer Only)

Inference-time projection on hidden_2 via nearest-centroid pseudo-labels.

**Training:** Identical to NB07 (same seed, same training-time behavior).

**Test accuracy: 0.9497** (-1.3% vs NB06, -0.2% vs NB07)

Closing the train/test gap didn't help — accuracy was slightly *worse* than NB07's no-op eval. The nearest-centroid pseudo-labels were reasonably accurate, but applying the nudge at inference added noise without giving the classifier anything useful. The model had already adapted its weights to the nudged distribution during training; applying a noisy approximation of that nudge at eval was no better than omitting it.

### NB10 — Centroid Inference (Both Layers)

Inference-time projection on both hidden_1 and hidden_2.

**Training:** Identical to NB08.

**Test accuracy: 0.7484** (-21.5% vs NB06, -11.2% vs NB08)

The worst result in the entire series. Accuracy degraded every epoch, falling from 92% (epoch 0) to **75%** (epoch 4). The dual-layer inference projection created a compounding error loop: incorrect pseudo-label assignments at layer 1 nudged activations toward wrong centroids, distorting the representations flowing into layer 2, which further degraded pseudo-label accuracy at layer 2. Each layer's errors amplified the other's.

### Inference-Time Projection Summary

| NB | Configuration | Test Acc | vs. NB06 | vs. train-only twin |
|----|--------------|----------|----------|---------------------|
| 09 | Inference (last layer) | 0.9497 | -1.3% | -0.2% vs NB07 |
| 10 | Inference (both layers) | 0.7484 | -21.5% | -11.2% vs NB08 |

---

## Phase 4: Controlled Quantitative Comparison (Notebook 11)

NB11 is the controlled comparison. Three models trained from scratch with the same seed, same data, same 10-epoch schedule, at ~3x gentler projection strength (`alpha=0.1, beta=0.03`) than NB07–10:

| Model | Description | Test Acc |
|-------|------------|----------|
| A (NB06-style) | No projection — baseline | 0.9754 |
| B (NB07-style) | Projection during training only | 0.9728 |
| C (NB09-style) | Projection during training + inference | 0.9717 |

Full-dimensional (64D) hidden_2 activations were extracted for the entire test set and evaluated with logistic regression (linear separability) and silhouette score (cluster compactness):

```
Layer                               LogReg Acc  Silhouette
----------------------------------------------------------
Model A (06) — hidden_2                 0.9777      0.5432
Model B (07) — hidden_2                 0.9757      0.5017
Model B (07) — projected                0.9757      0.5017  (no-op at eval)
Model C (09) — hidden_2                 0.9757      0.5017
Model C (09) — projected                0.9743      0.5438
```

**Deltas vs Model A baseline:**

| Representation | LogReg Δ | Silhouette Δ |
|----------------|----------|-------------|
| B — hidden_2 | -0.0020 | -0.0416 |
| B — projected | -0.0020 | -0.0416 |
| C — hidden_2 | -0.0020 | -0.0416 |
| C — projected | -0.0033 | +0.0006 |

**Within-model projection effect (Model C hidden_2 → projected):**
- LogReg: **-0.0013** (within noise)
- Silhouette: **+0.0421** (meaningful improvement)

The projection improved geometric cluster quality but slightly hurt linear separability. Model C's projected representations (silhouette 0.5438) essentially matched Model A's natural representations (0.5432) — the projection merely recovered what was lost by training with the nudge.

Models B and C have identical hidden_2 representations (both 0.5017 silhouette) since they train identically — the `infer` flag only affects eval. Both are worse than Model A's unprojected hidden_2 (0.5432), meaning training *with* the projection actively degraded the representations the model learned on its own.

---

## Findings

### 1. Centroid Projection Hurts More Than It Helps

At every strength tested, projection either degraded accuracy or was neutral. At full strength (alpha=0.3), test accuracy dropped 1–21%. At reduced strength (alpha=0.1), the damage was within noise (-0.4%), but so was any benefit. The projection never improved on what standard cross-entropy training achieves on its own.

### 2. Multi-Layer Projection Is Catastrophic

Applying projection to both hidden layers (NB08, NB10) caused dramatic accuracy collapse. NB08 diverged mid-training with accuracy falling from 93% to 86% during a single epoch. NB10 was worse — accuracy fell to 75%, below what the model achieved after its very first epoch. The dual projection amplifies the distribution mismatch between train and eval and, when combined with inference-time pseudo-labels, creates a compounding error feedback loop.

### 3. The Train/Test Distribution Shift Is Fundamental

During training, the classifier (fc3) sees pre-sorted activations — nudged toward centroids. The gradients that update fc1 and fc2 optimize for a world where activations arrive pre-organized. At eval without the nudge, the learned weights encounter raw activations they weren't adapted for. This is analogous to teacher forcing in sequence models.

Closing the gap with inference-time projection (NB09–10) didn't solve this — it introduced its own noise from pseudo-label assignment and the fundamental issue remained: the model didn't learn to self-organize.

### 4. Silhouette and Linear Separability Optimize Different Geometry

The projection consistently improved silhouette scores (cluster compactness) while hurting logistic regression accuracy (linear separability). These metrics reward different geometric properties:

- **Silhouette** rewards points being close to their own centroid and far from others — a radial, distance-based criterion
- **Logistic regression** needs well-placed hyperplanes — a directional, margin-based criterion

Compressing points toward centroids helps the first but collapses the within-class variance that hyperplanes need to find good decision boundaries.

### 5. At Optimal Strength, the Projection Is a Wash

When tuned to alpha=0.1, beta=0.03, the projection barely helps and barely hurts. Model C's projected silhouette (0.5438) matched Model A's natural silhouette (0.5432). The model without projection simply learns equivalent geometric structure on its own — standard cross-entropy loss already produces well-clustered representations.

### 6. Regularization Trades Short-Term Accuracy for Long-Term Generalization

The Phase 1 baseline stack revealed that BatchNorm alone (NB03) achieved the highest 5-epoch accuracy (97.0%). Adding Dropout on top (NB04) actually *hurt* at 5 epochs (96.4%) because the stochastic masking slows convergence. But with 10 epochs of training (NB11), the fully-regularized architecture reaches 97.5% — the regularization pays off given sufficient training time.

### 7. Direct Manipulation vs. Loss-Based Approaches

The fundamental issue is that CentroidProjection operates outside the gradient flow. It shifts activations but the model can't learn *from* those shifts. A loss-based alternative (e.g. center loss, simplex target MSE) achieves the same clustering objective through gradients, with no train/test mismatch. This motivated [Experiment 02](../02_clustering_pretrain/README.md), which tests differentiable clustering pre-training.

---

## Complete Results Table

| NB | Config | Epochs | Test Acc | vs. Baseline |
|----|--------|--------|----------|-------------|
| 01 | Baseline MLP | 5 | 0.9598 | — |
| 02 | + Weight Decay | 5 | 0.9604 | +0.06% |
| 03 | + BatchNorm | 5 | 0.9704 | +1.06% |
| 04 | + Dropout (0.3) | 5 | 0.9643 | +0.45% |
| 05 | + L1 (1e-4) | 5 | 0.9659 | +0.61% |
| 06 | + LayerNorm(h2) | 5 | 0.9630 | +0.32% |
| 07 | + Projection (last, train) | 5 | 0.9516 | -0.82% |
| 08 | + Projection (both, train) | 5 | 0.8605 | -9.93% |
| 09 | + Projection (last, infer) | 5 | 0.9497 | -1.01% |
| 10 | + Projection (both, infer) | 5 | 0.7484 | -21.14% |
| 11A | No projection | 10 | 0.9754 | +1.56% |
| 11B | Projection (train only) | 10 | 0.9728 | +1.30% |
| 11C | Projection (train+infer) | 10 | 0.9717 | +1.19% |

## Architecture Reference

```
Input (784) → fc1(128) → BatchNorm → ReLU → Dropout(0.3)
            → fc2(64)  → LayerNorm → ReLU → Dropout(0.3)
            → [CentroidProjection] → fc3(10) → Output
```

Shared: Adam optimizer, lr=0.001, weight_decay=1e-4, l1_lambda=1e-4, batch_size=512.
NB01–10: 5 epochs. NB11: 10 epochs.
NB07–10: alpha=0.3, beta=0.1. NB11: alpha=0.1, beta=0.03.

## Tracker Runs

- **`mnist_baseline`** — NB01
- **`mnist_weight_decay`** — NB02
- **`mnist_batchnorm`** — NB03
- **`mnist_dropout`** — NB04
- **`mnist_elastic_net`** — NB05
- **`mnist_layernorm`** — NB06
- **`mnist_centroid_last`** — NB07
- **`mnist_centroid_all`** — NB08
- **`mnist_centroid_infer_last`** — NB09
- **`mnist_centroid_infer_all`** — NB10
- NB11 — analysis-only (no tracker)

## Reproduction

Each notebook is self-contained. Run cells in order. Seed is fixed (`torch.manual_seed(42)`) for reproducibility. NB01–10 export visualization data via `ExperimentTracker`; NB11 is analysis-only.

```
experimentation/01_centroid_projection/notebooks/
├── 01_baseline.ipynb              # Vanilla MLP
├── 02_weight_decay.ipynb          # + L2 regularization
├── 03_batchnorm.ipynb             # + BatchNorm
├── 04_dropout.ipynb               # + Dropout (0.3)
├── 05_elastic_net.ipynb           # + L1 regularization
├── 06_layer_normalization.ipynb   # + LayerNorm on hidden_2
├── 07_centroid_proj_last.ipynb    # + CentroidProjection (hidden_2)
├── 08_centroid_proj_all.ipynb     # + CentroidProjection (both layers)
├── 09_centroid_infer_last.ipynb   # + Inference-mode projection (hidden_2)
├── 10_centroid_infer_all.ipynb    # + Inference-mode projection (both)
└── 11_embedding_regression.ipynb  # Quantitative comparison: LogReg + Silhouette
```
