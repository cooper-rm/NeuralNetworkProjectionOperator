# Experimentation

Each experiment lives in its own numbered directory with a dedicated README and notebooks.

| # | Experiment | Description |
|---|-----------|-------------|
| [01](01_centroid_projection/) | Centroid Projection | Can direct geometric manipulation of hidden-layer activations toward class centroids improve representations? Builds an MLP incrementally (baseline through regularization), then tests centroid projection during training and inference. |

## Structure

```
experimentation/
├── 01_centroid_projection/
│   ├── notebooks/       # 11 notebooks (baseline → centroid projection → evaluation)
│   └── README.md        # Full experiment writeup and results
├── runs/                # Training run outputs (gitignored)
└── README.md            # This file
```

## Run Data

The `runs/` directory stores training outputs (checkpoints, metrics) and is gitignored. Each notebook writes its run data here via `ExperimentTracker`. Run the notebooks locally to regenerate.
