# Centroid Projection: Inference-Time Activation Nudging

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

## Experiment Series

### Phase 1: Baseline Stack (Notebooks 01–06)

Built up a standard MLP incrementally to establish a strong baseline:

| Notebook | Addition | Test Acc |
|----------|----------|----------|
| 01 | Baseline MLP (784→128→64→10) | ~0.96 |
| 02 | + Weight Decay (L2) | ~0.96 |
| 03 | + BatchNorm (hidden_1) | ~0.97 |
| 04 | + Dropout (0.3) | ~0.97 |
| 05 | + L1 Regularization (Elastic Net) | ~0.97 |
| 06 | + LayerNorm (hidden_2) | ~0.975 |

By notebook 06, the model achieves **97.5% test accuracy** with well-regularized representations. This became the baseline for all centroid projection experiments.

### Phase 2: Centroid Projection During Training (Notebooks 07–08)

**Notebook 07 — Centroid Projection (Last Hidden Layer)**
- Added CentroidProjection after hidden_2 only
- `alpha=0.3, beta=0.1` (attraction and repulsion strengths)
- Training loss dropped much faster (0.105 vs 0.273 by epoch 10)
- But test accuracy *decreased* to ~0.962

**Notebook 08 — Centroid Projection (All Hidden Layers)**
- Added CentroidProjection after *both* hidden_1 and hidden_2
- Same hyperparameters, double the geometric regularization

**Key observation:** The projection made training loss look impressive, but test accuracy was worse than the baseline. The model was learning to rely on the nudge rather than developing its own discriminative representations.

### Phase 3: Inference-Time Projection (Notebooks 09–10)

The projection was a no-op at inference — `forward()` returned activations unchanged when labels weren't provided. This created a train/test distribution shift: the model trained with pre-sorted activations but tested on raw ones.

**Notebook 09 — Centroid Inference (Last Layer)**
- Modified CentroidProjection with an `infer` mode
- At eval time, assigns each sample to its nearest centroid via `torch.cdist`
- Uses the nearest-centroid index as a pseudo-label for the nudge
- This closes the train/test gap by keeping the projection active at inference

**Notebook 10 — Centroid Inference (All Layers)**
- Same inference-mode extension applied to both hidden layers

### Phase 4: Quantitative Evaluation (Notebook 11)

Notebook 11 is the rigorous comparison. It trains three models from scratch with the same seed and data:

- **Model A (06-style):** No projection — baseline
- **Model B (07-style):** Projection during training only (no-op at eval)
- **Model C (09-style):** Projection during training AND inference

After training, it extracts full-dimensional (64D) hidden_2 activations for the entire test set and evaluates:

1. **Logistic Regression accuracy** — Can a linear classifier separate the representations? Higher = more linearly separable.
2. **Silhouette score** — How tight and well-separated are the class clusters? Higher = better geometric structure.

#### Iteration 1: Original Strength (alpha=0.3, beta=0.1)

```
Layer                               LogReg Acc  Silhouette
----------------------------------------------------------
Model A (06) — hidden_2                 0.9777      0.5432
Model B (07) — hidden_2                 0.9700      0.4581
Model C (09) — projected                0.9647      0.5825
```

The projection dramatically improved silhouette (+0.1244 within Model C), but *hurt* logistic regression accuracy (-0.0053). Tighter clusters, worse linear boundaries. The nudge was compressing within-class variance that the linear classifier needed.

#### Iteration 2: L2 Normalization

We hypothesized that magnitude distortion was the issue. Added L2 normalization before the projection (work on the unit sphere) and restored original magnitude after. This required careful handling of the straight-through gradient — computing the normalized delta inside `torch.no_grad()` while applying it to `x` outside to preserve gradient flow.

Results were similar but more pronounced: even stronger silhouette gains, even worse logistic regression. Constraining movement to the hypersphere surface made centroid attraction more aggressive, amplifying the fundamental tension.

#### Iteration 3: Reduced Strength (alpha=0.1, beta=0.03)

Backed off to ~3x gentler nudge, without normalization:

```
Layer                               LogReg Acc  Silhouette
----------------------------------------------------------
Model A (06) — hidden_2                 0.9777      0.5432
Model B (07) — hidden_2                 0.9757      0.5017
Model C (09) — projected                0.9743      0.5438

Projection effect (within Model C):
  hidden_2 → projected:  LogReg -0.0013, Silhouette +0.0421
```

The logistic regression damage nearly vanished (-0.0013, within noise). The silhouette gain was moderate (+0.0421), bringing Model C's projected representations to 0.5438 — essentially matching Model A's 0.5432.

## Findings

### The Projection Creates a Train/Test Distribution Shift

During training, `fc3` (the classifier) sees nudged activations — pre-sorted by class. The gradients that update `fc1` and `fc2` optimize for a world where activations arrive pre-organized. At test time without the nudge, the learned weights aren't adapted for raw activations. This is analogous to teacher forcing in sequence models.

### Silhouette and Linear Separability Are Different Objectives

The projection consistently improved silhouette scores (cluster compactness) while hurting logistic regression accuracy (linear separability). These metrics optimize for different geometric properties:

- **Silhouette** rewards points being close to their own centroid and far from other centroids — a radial, distance-based criterion
- **Logistic regression** needs well-placed hyperplanes — a directional, margin-based criterion

Compressing points toward centroids helps the first but can collapse the variance that hyperplanes need to find good decision boundaries.

### At Optimal Strength, the Projection Is a Wash

When tuned down to `alpha=0.1, beta=0.03`, the projection barely helps and barely hurts. It recovers cluster quality that was lost by training with the nudge, landing essentially at the baseline. The model without projection simply learns equivalent representations on its own.

### Direct Activation Manipulation vs. Loss-Based Approaches

The fundamental issue is that CentroidProjection operates outside the gradient flow. It shifts activations but the model can't learn *from* those shifts. A loss-based alternative like center loss:

```python
center_loss = (h - centroids[labels]).pow(2).sum(1).mean()
loss = ce_loss + lambda_center * center_loss
```

...achieves the same clustering objective through gradients, with no train/test mismatch. The model learns to self-organize rather than being externally organized.

## Architecture Reference

```
Input (784) → fc1 → BatchNorm → ReLU → Dropout(0.3)
           → fc2 → LayerNorm → ReLU → Dropout(0.3)
           → [CentroidProjection] → fc3 → Output (10)
```

All models: Adam optimizer, lr=0.001, weight_decay=1e-4, l1_lambda=1e-4, 10 epochs, batch_size=512.

## Reproduction

Each notebook is self-contained. Run cells in order. Seed is fixed (`torch.manual_seed(42)`) for reproducibility. Notebooks 01–10 export visualization data via `ExperimentTracker`; notebook 11 is analysis-only.

```
experimentation/01_centroid_projection/notebooks/
├── 01_baseline.ipynb              # Vanilla MLP
├── 02_weight_decay.ipynb          # + L2 regularization
├── 03_batchnorm.ipynb             # + BatchNorm on hidden_1
├── 04_dropout.ipynb               # + Dropout (0.3)
├── 05_elastic_net.ipynb           # + L1 regularization
├── 06_layer_normalization.ipynb   # + LayerNorm on hidden_2
├── 07_centroid_proj_last.ipynb    # + CentroidProjection (hidden_2)
├── 08_centroid_proj_all.ipynb     # + CentroidProjection (both layers)
├── 09_centroid_infer_last.ipynb   # + Inference-mode projection (hidden_2)
├── 10_centroid_infer_all.ipynb    # + Inference-mode projection (both)
└── 11_embedding_regression.ipynb  # Quantitative comparison: LogReg + Silhouette
```
