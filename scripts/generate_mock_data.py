"""Generate mock neural network checkpoint data for visualization.

Produces multiple runs with different architectures, each with:
  - runs.json (index of all runs)
  - <run_id>/run_config.json (architecture, hyperparams, metadata, input_samples)
  - <run_id>/step_NNN.json (lightweight checkpoints with activations)
"""

import json
import os
import shutil
from datetime import datetime

import numpy as np

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed, using PCA-like projection instead")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization", "public", "data")
NUM_SAMPLES = 500
NUM_CLASSES = 10
NUM_INPUT_SAMPLES = 5

# Define runs with different architectures
RUNS = [
    {
        "id": "mlp_128_64_run1",
        "model_name": "MLP 128-64",
        "description": "2-hidden-layer MLP on MNIST",
        "layers": ["input", "hidden_1", "hidden_2", "output"],
        "sizes": [784, 128, 64, 10],
        "hyperparameters": {"lr": 0.001, "batch_size": 64, "epochs": 10},
        "total_steps": 30,
    },
    {
        "id": "mlp_256_128_run2",
        "model_name": "MLP 256-128",
        "description": "Wider 2-hidden-layer MLP on MNIST",
        "layers": ["input", "hidden_1", "hidden_2", "output"],
        "sizes": [784, 256, 128, 10],
        "hyperparameters": {"lr": 0.0005, "batch_size": 128, "epochs": 10},
        "total_steps": 30,
    },
]


def random_projection_3d(data):
    """Project high-dim data to 3D using random projection (fallback)."""
    d = data.shape[1]
    proj = np.random.randn(d, 3) / np.sqrt(d)
    return data @ proj


def project_to_3d(data):
    """Project data to 3D using UMAP if available, else random projection."""
    if HAS_UMAP and data.shape[1] > 3:
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        return reducer.fit_transform(data)
    elif data.shape[1] > 3:
        return random_projection_3d(data)
    else:
        return data


def generate_embeddings(step, total_steps, layer_names, layer_sizes):
    """Generate embeddings that separate more as training progresses."""
    t = step / max(total_steps - 1, 1)
    labels = np.repeat(np.arange(NUM_CLASSES), NUM_SAMPLES // NUM_CLASSES)
    np.random.seed(step * 100)

    embeddings = {}
    for layer_name, layer_size in zip(layer_names, layer_sizes):
        dim = min(layer_size, 50)

        centers = np.random.randn(NUM_CLASSES, dim) * (1.0 + t * 3.0)

        noise_scale = 3.0 * (1.0 - t * 0.7)
        if layer_name == "input":
            noise_scale = 3.0
        elif layer_name == "output":
            noise_scale = 3.0 * (1.0 - t * 0.85)

        points_hd = np.zeros((NUM_SAMPLES, dim))
        for c in range(NUM_CLASSES):
            mask = labels == c
            n = mask.sum()
            points_hd[mask] = centers[c] + np.random.randn(n, dim) * noise_scale

        points_3d = project_to_3d(points_hd)

        pmin, pmax = points_3d.min(), points_3d.max()
        if pmax - pmin > 0:
            points_3d = (points_3d - pmin) / (pmax - pmin) * 10 - 5

        embeddings[layer_name] = {
            "points": np.round(points_3d, 4).tolist(),
            "labels": labels.tolist()
        }

    return embeddings


def generate_weight_matrices(step, total_steps, layer_names, layer_sizes):
    """Generate weight matrices as numpy arrays (for forward pass and serialization)."""
    t = step / max(total_steps - 1, 1)
    np.random.seed(42)

    weight_arrays = {}
    for i in range(1, len(layer_names)):
        name = layer_names[i]
        fan_in = layer_sizes[i - 1]
        fan_out = layer_sizes[i]

        std = np.sqrt(2.0 / (fan_in + fan_out))
        base = np.random.randn(fan_in, fan_out) * std

        np.random.seed(step * 1000 + hash(name) % 10000)
        evolution = np.random.randn(fan_in, fan_out) * std * t * 0.5

        weight_arrays[name] = base + evolution

    return weight_arrays


def weights_to_json(weight_arrays):
    """Convert numpy weight arrays to JSON-serializable format."""
    weights = {}
    for name, arr in weight_arrays.items():
        weights[name] = {
            "shape": list(arr.shape),
            "values": np.round(arr, 5).tolist()
        }
    return weights


def generate_input_samples():
    """Generate 5 synthetic 28x28 MNIST-like input images with distinct patterns."""
    np.random.seed(777)
    samples = []

    def _make_image(draw_fn):
        img = np.zeros((28, 28), dtype=np.float64)
        draw_fn(img)
        img = np.clip(img, 0.0, 1.0)
        # Add slight noise for realism
        img += np.random.randn(28, 28) * 0.03
        return np.clip(img, 0.0, 1.0)

    # Digit 0: oval ring
    def draw_zero(img):
        cy, cx = 14, 14
        for y in range(28):
            for x in range(28):
                d = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                if 7 < d < 11:
                    img[y, x] = 1.0 - abs(d - 9) / 2.0

    # Digit 1: vertical bar
    def draw_one(img):
        img[4:24, 12:16] = 0.9
        img[4:7, 10:14] = 0.7  # serif top
        img[22:25, 9:19] = 0.7  # base

    # Digit 3: two bumps on right
    def draw_three(img):
        for y in range(28):
            for x in range(28):
                # Top arc
                d1 = np.sqrt((y - 9) ** 2 + (x - 14) ** 2)
                if 5 < d1 < 8 and x >= 12:
                    img[y, x] = max(img[y, x], 0.9 - abs(d1 - 6.5) / 2.5)
                # Bottom arc
                d2 = np.sqrt((y - 19) ** 2 + (x - 14) ** 2)
                if 5 < d2 < 8 and x >= 12:
                    img[y, x] = max(img[y, x], 0.9 - abs(d2 - 6.5) / 2.5)
        # Connecting spine
        img[4:24, 10:13] = np.maximum(img[4:24, 10:13], 0.5)

    # Digit 7: horizontal top + diagonal
    def draw_seven(img):
        img[4:7, 6:22] = 0.9  # top bar
        for i in range(18):
            y = 6 + i
            x = int(20 - i * 0.6)
            if 0 <= y < 28 and 0 <= x < 26:
                img[y, max(0, x - 1):min(28, x + 2)] = 0.85

    # Digit 5: S-like shape
    def draw_five(img):
        img[4:7, 6:20] = 0.9   # top bar
        img[4:14, 6:9] = 0.9   # left vertical
        img[12:15, 6:20] = 0.9  # middle bar
        img[14:24, 17:20] = 0.9 # right vertical
        img[22:25, 6:20] = 0.9  # bottom bar

    drawers = [
        (0, draw_zero),
        (1, draw_one),
        (3, draw_three),
        (7, draw_seven),
        (5, draw_five),
    ]

    for label, draw_fn in drawers:
        img = _make_image(draw_fn)
        samples.append({
            "label": int(label),
            "pixels": np.round(img, 3).tolist()
        })

    return samples


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def generate_activations(input_samples, weight_arrays, layer_names, layer_sizes):
    """Compute forward-pass activations for each input sample through the network.

    Returns dict: {"sample_0": {"input": [...], "hidden_1": [...], ...}, ...}
    """
    activations = {}

    for si, sample in enumerate(input_samples):
        pixels = np.array(sample["pixels"]).flatten()  # 784
        sample_acts = {"input": np.round(pixels, 4).tolist()}

        current = pixels
        for li in range(1, len(layer_names)):
            layer_name = layer_names[li]
            W = weight_arrays[layer_name]
            z = current @ W

            if layer_name == "output":
                current = softmax(z)
            else:
                current = np.maximum(0, z)  # ReLU

            sample_acts[layer_name] = np.round(current, 5).tolist()

        activations[f"sample_{si}"] = sample_acts

    return activations


def generate_gradients(step, total_steps, layer_names, layer_sizes):
    """Generate synthetic gradient data simulating training dynamics.

    - Gradient magnitude decays over training (loss decreasing)
    - Earlier layers get smaller gradients (mild vanishing gradient simulation)
    """
    t = step / max(total_steps - 1, 1)
    np.random.seed(step * 200 + 7)

    # Overall gradient scale decays as training progresses
    global_scale = 0.05 * np.exp(-1.5 * t) + 0.002

    gradients = {}
    flow_layers = []
    flow_weight_norms = []
    flow_bias_norms = []

    # Only layers with weights (skip input)
    weight_layers = [(layer_names[i], layer_sizes[i - 1], layer_sizes[i])
                     for i in range(1, len(layer_names))]
    num_weight_layers = len(weight_layers)

    for li, (name, fan_in, fan_out) in enumerate(weight_layers):
        # Earlier layers get smaller gradients (vanishing gradient effect)
        depth_factor = 0.3 + 0.7 * (li / max(num_weight_layers - 1, 1))
        layer_scale = global_scale * depth_factor

        # Weight gradients
        wg = np.random.randn(fan_in, fan_out) * layer_scale
        w_norm = float(np.linalg.norm(wg))

        # Bias gradients
        bg = np.random.randn(fan_out) * layer_scale * 0.5
        b_norm = float(np.linalg.norm(bg))

        gradients[name] = {
            "weight": {
                "shape": [fan_in, fan_out],
                "values": np.round(wg, 5).tolist(),
            },
            "bias": {
                "shape": [fan_out],
                "values": np.round(bg, 5).tolist(),
            },
            "stats": {
                "weight_norm": round(w_norm, 6),
                "weight_mean": round(float(wg.mean()), 6),
                "weight_std": round(float(wg.std()), 6),
                "weight_min": round(float(wg.min()), 6),
                "weight_max": round(float(wg.max()), 6),
                "bias_norm": round(b_norm, 6),
                "bias_mean": round(float(bg.mean()), 6),
                "bias_std": round(float(bg.std()), 6),
            },
        }

        flow_layers.append(name)
        flow_weight_norms.append(round(w_norm, 6))
        flow_bias_norms.append(round(b_norm, 6))

    gradients["_flow"] = {
        "layers": flow_layers,
        "weight_grad_norms": flow_weight_norms,
        "bias_grad_norms": flow_bias_norms,
    }

    return gradients


def generate_run(run_config, input_samples):
    """Generate all data for a single run."""
    run_id = run_config["id"]
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    total_steps = run_config["total_steps"]
    layer_names = run_config["layers"]
    layer_sizes = run_config["sizes"]

    # Write run_config.json (now includes input_samples)
    config = {
        "run_id": run_id,
        "model_name": run_config["model_name"],
        "description": run_config["description"],
        "architecture": {
            "layers": layer_names,
            "sizes": layer_sizes,
        },
        "hyperparameters": run_config["hyperparameters"],
        "total_steps": total_steps,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_samples": input_samples,
        "gradient_capture": {"enabled": True, "full_matrices": True},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Generate checkpoint files
    for step in range(total_steps):
        epoch = step // 3
        t = step / max(total_steps - 1, 1)
        loss = round(2.3 * np.exp(-2.5 * t) + 0.05, 4)
        accuracy = round(0.1 + 0.85 * (1 - np.exp(-3.0 * t)), 4)

        # Generate weight matrices as numpy arrays
        weight_arrays = generate_weight_matrices(step, total_steps, layer_names, layer_sizes)

        # Compute activations via forward pass
        activations = generate_activations(
            input_samples, weight_arrays, layer_names, layer_sizes
        )

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "embeddings": generate_embeddings(step, total_steps, layer_names, layer_sizes),
            "weights": weights_to_json(weight_arrays),
            "activations": activations,
            "gradients": generate_gradients(step, total_steps, layer_names, layer_sizes),
        }

        filename = f"step_{step:03d}.json"
        filepath = os.path.join(run_dir, filename)
        with open(filepath, "w") as f:
            json.dump(checkpoint, f)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filename} (epoch={epoch}, loss={loss:.4f}, acc={accuracy:.4f}, size={size_mb:.1f}MB)")

    print(f"  run_config.json written")


def main():
    # Clean old flat files (step_NNN.json in root data dir)
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, f)
            if f.startswith("step_") and f.endswith(".json"):
                os.remove(fpath)
                print(f"Removed old {f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate shared input samples
    input_samples = generate_input_samples()
    print(f"Generated {len(input_samples)} input samples (labels: {[s['label'] for s in input_samples]})")

    # Write runs.json
    runs_index = {
        "runs": [
            {"id": r["id"], "model_name": r["model_name"], "description": r["description"]}
            for r in RUNS
        ]
    }
    with open(os.path.join(OUTPUT_DIR, "runs.json"), "w") as f:
        json.dump(runs_index, f, indent=2)
    print("Generated runs.json")

    # Generate each run
    for run_config in RUNS:
        print(f"\nGenerating run: {run_config['id']}")
        generate_run(run_config, input_samples)

    print(f"\nDone! {len(RUNS)} runs written to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
