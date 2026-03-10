# Experimentation

Each experiment lives in its own numbered directory with a dedicated README and notebooks.

| # | Experiment | Description |
|---|-----------|-------------|
| [01](01_centroid_projection/) | Centroid Projection | Can direct geometric manipulation of hidden-layer activations toward class centroids improve representations? Builds an MLP incrementally (baseline through regularization), then tests centroid projection during training and inference. |
| [02](02_clustering_pretrain/) | Clustering Pre-Training | Can a differentiable clustering loss learn representations that support classification? Two-phase: train backbone with intra/inter centroid loss, then freeze and train classifier head. |

## Structure

```
experimentation/
├── 01_centroid_projection/
│   ├── notebooks/       # 11 notebooks (baseline → centroid projection → evaluation)
│   └── README.md        # Full experiment writeup and results
├── 02_clustering_pretrain/
│   ├── notebooks/       # 6 notebooks (moving centroids, simplex targets, extended, from-scratch control)
│   └── README.md        # Experiment writeup
├── runs/                # Training run outputs (gitignored)
└── README.md            # This file
```

## Run Data

The `runs/` directory stores training outputs (checkpoints, metrics) and is gitignored. Each notebook writes its run data here via `ExperimentTracker`. Run the notebooks locally to regenerate.
