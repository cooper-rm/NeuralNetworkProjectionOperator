# Neural Network Projection Operator

Investigating whether direct geometric manipulation of neural network activations can improve learned representations — starting with centroid projection on MNIST.

## Motivation

Standard neural networks learn representations implicitly through backpropagation. Hidden-layer activations tend to cluster by class, but this is a side effect of the loss function, not an explicit objective. This project explores what happens when you directly intervene in the activation space — nudging representations toward geometric targets — and whether that helps, hurts, or makes no difference.

## Repository Structure

```
experimentation/          # Experiment series, each with notebooks and a writeup
├── 01_centroid_projection/   # Centroid projection: 11 notebooks, baseline → evaluation
└── runs/                     # Training outputs (gitignored)

scripts/
├── viz_export.py         # ExperimentTracker — forward hooks, auto-versioning, JSON export
└── generate_mock_data.py # Synthetic data generator for visualization dev

visualization/            # React/Three.js app for exploring training runs
├── src/                  # App source
└── public/data/          # Run data consumed by the app (gitignored)

exploration/              # Early-stage notebooks and scratch work
```

## Centroid Projection (Experiment 01)

The first experiment series tests **CentroidProjection**, a module that attracts hidden-layer activations toward their class centroids and repels centroids from each other, using a straight-through estimator (no gradient signal from the nudge).

Key finding: the projection consistently improves cluster compactness (silhouette score) but hurts linear separability (logistic regression accuracy). At optimal strength, it's a wash — the baseline network learns equivalent representations on its own. The fundamental issue is that the manipulation operates outside gradient flow, creating a train/test distribution shift analogous to teacher forcing.

Full writeup: [`experimentation/01_centroid_projection/README.md`](experimentation/01_centroid_projection/README.md)

## Usage

### Running Experiments

Each notebook is self-contained. Open in Jupyter and run cells in order. Seeds are fixed for reproducibility. Notebooks export visualization data via `ExperimentTracker` from `scripts/viz_export.py`.

### Visualization

The visualization app reads JSON run data from `visualization/public/data/`. To generate mock data for development:

```bash
python scripts/generate_mock_data.py
```

To run the visualization app:

```bash
cd visualization && npm install && npm run dev
```

## Note on Data

Training run data (`experimentation/runs/`, `visualization/public/data/`) is gitignored. Run the notebooks or mock data script locally to generate it.
