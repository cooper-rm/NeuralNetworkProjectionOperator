"""Experiment tracker that uses forward hooks to capture activations/weights from any PyTorch model.

Usage:
    from scripts.viz_export import ExperimentTracker

    model = MyModel()
    tracker = ExperimentTracker(
        run_id="mnist_mlp_128_64",
        model_name="MNIST MLP 128-64",
        description="2-hidden-layer MLP on MNIST",
        hyperparameters={"lr": 0.001, "batch_size": 64, "epochs": 10},
        model=model,
    )
    tracker.track("input", size=784)
    tracker.track("hidden_1", model.fc1, size=128)
    tracker.track("hidden_2", model.fc2, size=64)
    tracker.track("output", model.fc3, size=10)

    tracker.set_input_samples(images, labels)
    tracker.set_viz_samples(images, labels)

    for epoch in range(10):
        ...
        tracker.save_checkpoint(step=epoch, epoch=epoch, loss=loss, accuracy=acc)

    tracker.finalize()
"""

import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import torch

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "experimentation", "runs")


class ExperimentTracker:
    def __init__(self, run_id, model_name, description, hyperparameters, model=None):
        self.base_run_id = run_id
        self.model_name = model_name
        self.description = description
        self.hyperparameters = hyperparameters
        self._model = model

        self.run_id, self.run_dir = self._resolve_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)

        self._layers = []          # list of (name, module_or_None, size)
        self._hooks = []           # registered hook handles
        self._hook_outputs = {}    # name -> captured tensor

        self._input_samples = None
        self._input_labels = None
        self._viz_samples = None
        self._viz_labels = None
        self._pass_labels = False  # if True, pass labels= to model forward

        self._gradient_capture = False
        self._gradient_full_matrices = False
        self._captured_gradients = {}
        self._gradient_path = []       # accumulated per-batch gradient norms

        self._landscape_enabled = False
        self._landscape_grid_size = 41
        self._landscape_range = 1.0
        self._landscape_directions = None
        self._landscape_data = None

        self._step_count = 0
        self._prev_embeddings = {}  # layer_name -> previous step's 3D points (for alignment)
        print(f"ExperimentTracker: will write to {self.run_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, name, module=None, size=None):
        if size is None:
            raise ValueError(f"size is required for layer '{name}'")
        self._layers.append((name, module, size))
        if module is not None:
            handle = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def set_input_samples(self, images, labels):
        if isinstance(images, torch.Tensor):
            self._input_samples = images.detach().cpu()
        else:
            self._input_samples = torch.stack([t.detach().cpu() for t in images])
        self._input_labels = list(labels)

    def set_viz_samples(self, images, labels):
        if isinstance(images, torch.Tensor):
            self._viz_samples = images.detach().cpu()
        else:
            self._viz_samples = torch.stack([t.detach().cpu() for t in images])
        self._viz_labels = list(labels)

    def enable_gradient_capture(self, full_matrices=True):
        """Opt in to gradient capture. Call once during setup."""
        self._gradient_capture = True
        self._gradient_full_matrices = full_matrices

    def capture_gradients(self):
        """Snapshot gradients from all tracked layers.

        Must be called after loss.backward() and before optimizer.zero_grad().
        """
        if not self._gradient_capture:
            return

        gradients = {}
        flow_layers = []
        flow_weight_norms = []
        flow_bias_norms = []

        for name, module, _ in self._layers:
            if module is None:
                continue
            if not hasattr(module, 'weight') or module.weight.grad is None:
                continue

            wg = module.weight.grad.detach().cpu().numpy()
            layer_data = {}

            # Stats for weight gradients
            stats = {
                "weight_norm": round(float(np.linalg.norm(wg)), 6),
                "weight_mean": round(float(wg.mean()), 6),
                "weight_std": round(float(wg.std()), 6),
                "weight_min": round(float(wg.min()), 6),
                "weight_max": round(float(wg.max()), 6),
            }

            # Full weight gradient matrix (transposed to [fan_in, fan_out])
            if self._gradient_full_matrices:
                if wg.ndim == 4:
                    wg_t = wg.reshape(wg.shape[0], -1).T
                else:
                    wg_t = wg.T
                layer_data["weight"] = {
                    "shape": list(wg_t.shape),
                    "values": np.round(wg_t, 5).tolist(),
                }

            # Bias gradients
            if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is not None:
                bg = module.bias.grad.detach().cpu().numpy()
                stats["bias_norm"] = round(float(np.linalg.norm(bg)), 6)
                stats["bias_mean"] = round(float(bg.mean()), 6)
                stats["bias_std"] = round(float(bg.std()), 6)
                if self._gradient_full_matrices:
                    layer_data["bias"] = {
                        "shape": list(bg.shape),
                        "values": np.round(bg, 5).tolist(),
                    }
                flow_bias_norms.append(round(float(np.linalg.norm(bg)), 6))
            else:
                flow_bias_norms.append(0.0)

            layer_data["stats"] = stats
            gradients[name] = layer_data

            flow_layers.append(name)
            flow_weight_norms.append(stats["weight_norm"])

        gradients["_flow"] = {
            "layers": flow_layers,
            "weight_grad_norms": flow_weight_norms,
            "bias_grad_norms": flow_bias_norms,
        }

        self._captured_gradients = gradients

        # Append lightweight stats to the gradient path
        self._gradient_path.append({
            "layers": flow_layers,
            "weight_grad_norms": flow_weight_norms,
            "bias_grad_norms": flow_bias_norms,
        })

    def enable_forward_labels(self):
        """Pass labels= to model forward during capture (needed for CentroidProjection)."""
        self._pass_labels = True

    def enable_loss_landscape(self, grid_size=41, range_val=1.0):
        """Enable loss landscape computation. Call once during setup."""
        self._landscape_enabled = True
        self._landscape_grid_size = grid_size
        self._landscape_range = range_val
        self._landscape_directions = None  # generated lazily on first compute

    def compute_loss_landscape(self, data_x, data_y, criterion):
        """Compute a 2D loss landscape around the current parameters.

        Uses two filter-normalized random directions (Li et al. 2018).
        Call in the training loop before save_checkpoint, same pattern as capture_gradients.
        """
        if not self._landscape_enabled or self._model is None:
            return

        # Lazily generate directions on first call
        if self._landscape_directions is None:
            self._landscape_directions = self._generate_landscape_directions()

        d1, d2 = self._landscape_directions
        grid_size = self._landscape_grid_size
        r = self._landscape_range
        alphas = np.linspace(-r, r, grid_size)
        betas = np.linspace(-r, r, grid_size)

        # Save original parameters
        orig_params = [p.data.clone() for p in self._model.parameters()]

        self._model.eval()
        landscape = np.zeros((grid_size, grid_size))

        if isinstance(data_x, np.ndarray):
            data_x = torch.tensor(data_x, dtype=torch.float32)
        if isinstance(data_y, np.ndarray):
            data_y = torch.tensor(data_y, dtype=torch.long)

        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    # Perturb: param = orig + alpha*d1 + beta*d2
                    for p, orig, dir1, dir2 in zip(
                        self._model.parameters(), orig_params, d1, d2
                    ):
                        p.data.copy_(orig + alpha * dir1 + beta * dir2)

                    output = self._model(data_x)
                    loss = criterion(output, data_y)
                    landscape[i, j] = loss.item()

        # Restore original parameters
        for p, orig in zip(self._model.parameters(), orig_params):
            p.data.copy_(orig)

        self._model.train()

        self._landscape_data = {
            "grid_size": grid_size,
            "alpha_range": [-r, r],
            "beta_range": [-r, r],
            "values": np.round(landscape, 5).tolist(),
        }

    def _generate_landscape_directions(self):
        """Generate two filter-normalized random directions in parameter space."""
        gen = torch.Generator().manual_seed(42)
        d1 = []
        d2 = []
        for p in self._model.parameters():
            dir1 = torch.randn(p.shape, generator=gen)
            dir2 = torch.randn(p.shape, generator=gen)
            # Filter-normalize: scale each direction tensor to match the param's norm
            p_norm = p.data.norm()
            if p_norm > 0:
                dir1 = dir1 * (p_norm / dir1.norm())
                dir2 = dir2 * (p_norm / dir2.norm())
            d1.append(dir1)
            d2.append(dir2)
        return d1, d2

    def save_checkpoint(self, step, epoch, loss=None, accuracy=None, metrics=None):
        activations = self._compute_activations()
        embeddings = self._compute_embeddings()
        weights = self._extract_weights()

        if metrics is None:
            metrics = {}
            if loss is not None:
                metrics["train_loss"] = round(float(loss), 4)
            if accuracy is not None:
                metrics["test_accuracy"] = round(float(accuracy), 4)

        top_loss = metrics.get("val_loss", metrics.get("train_loss", loss or 0))
        top_acc = metrics.get("val_accuracy", metrics.get("test_accuracy", accuracy or 0))

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "loss": round(float(top_loss), 4),
            "accuracy": round(float(top_acc), 4),
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "embeddings": embeddings,
            "weights": weights,
            "activations": activations,
        }

        if self._gradient_capture:
            if self._captured_gradients:
                checkpoint["gradients"] = self._captured_gradients
                self._captured_gradients = {}
            else:
                print("  Warning: gradient capture enabled but no gradients captured. "
                      "Call capture_gradients() after loss.backward().")
            if self._gradient_path:
                checkpoint["gradient_path"] = self._gradient_path
                self._gradient_path = []

        if self._landscape_enabled and self._landscape_data is not None:
            checkpoint["loss_landscape"] = self._landscape_data
            self._landscape_data = None

        filename = f"step_{step:03d}.json"
        filepath = os.path.join(self.run_dir, filename)
        with open(filepath, "w") as f:
            json.dump(checkpoint, f)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filename} (epoch={epoch}, loss={top_loss:.4f}, acc={top_acc:.4f}, size={size_mb:.1f}MB)")
        self._step_count += 1

        # Keep run_config.json and runs.json always up to date
        self._write_run_config()
        if self._step_count == 1:
            self._update_runs_json()

    def finalize(self):
        self._write_run_config()
        self._remove_hooks()
        print(f"Finalized run '{self.run_id}' with {self._step_count} checkpoints")

    # ------------------------------------------------------------------
    # Hook machinery
    # ------------------------------------------------------------------

    def _make_hook(self, name):
        def hook_fn(_module, _input, output):
            self._hook_outputs[name] = output.detach().cpu()
        return hook_fn

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Activations (5 input samples)
    # ------------------------------------------------------------------

    def _compute_activations(self):
        if self._input_samples is None or self._model is None:
            return {}

        activations = {}
        self._model.eval()

        for si in range(len(self._input_samples)):
            sample = self._input_samples[si]
            self._hook_outputs.clear()

            with torch.no_grad():
                kwargs = {}
                if self._pass_labels and self._input_labels is not None:
                    kwargs['labels'] = torch.tensor([self._input_labels[si]])
                self._model(sample.unsqueeze(0), **kwargs)

            sample_acts = {}
            for name, module, _ in self._layers:
                if module is None:
                    sample_acts[name] = np.round(sample.numpy().flatten(), 4).tolist()
                else:
                    act = self._hook_outputs.get(name)
                    if act is not None:
                        act_np = act.squeeze(0).numpy()
                        if act_np.ndim > 1:
                            act_np = act_np.flatten()
                        sample_acts[name] = np.round(act_np, 5).tolist()

            activations[f"sample_{si}"] = sample_acts

        return activations

    # ------------------------------------------------------------------
    # Embeddings (N viz samples -> UMAP 3D)
    # ------------------------------------------------------------------

    def _compute_embeddings(self):
        if self._viz_samples is None or self._model is None:
            return {}

        self._model.eval()
        self._hook_outputs.clear()

        with torch.no_grad():
            kwargs = {}
            if self._pass_labels and self._viz_labels is not None:
                kwargs['labels'] = torch.tensor(self._viz_labels)
            self._model(self._viz_samples, **kwargs)

        embeddings = {}
        for name, module, _ in self._layers:
            if module is None:
                data = self._viz_samples.numpy()
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
            else:
                act = self._hook_outputs.get(name)
                if act is None:
                    continue
                data = act.numpy()
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)

            points_3d = self._project_to_3d(name, data)
            embeddings[name] = {
                "points": np.round(points_3d, 4).tolist(),
                "labels": self._viz_labels,
            }

        return embeddings

    def _project_to_3d(self, layer_name, data):
        """Project to 3D with UMAP + Procrustes alignment to previous step."""
        if data.shape[1] <= 3:
            points = data.copy()
        elif HAS_UMAP:
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
            points = reducer.fit_transform(data)
        else:
            d = data.shape[1]
            np.random.seed(42)
            proj = np.random.randn(d, 3) / np.sqrt(d)
            points = data @ proj

        # Normalize to [-5, 5]
        pmin, pmax = points.min(), points.max()
        if pmax - pmin > 0:
            points = (points - pmin) / (pmax - pmin) * 10 - 5

        # Procrustes: rotate/reflect to best match previous step
        if layer_name in self._prev_embeddings:
            points = self._procrustes_align(self._prev_embeddings[layer_name], points)

        self._prev_embeddings[layer_name] = points.copy()
        return points

    @staticmethod
    def _procrustes_align(reference, target):
        """Align target points to reference via Procrustes (rotation + reflection).

        Finds the orthogonal matrix R that minimizes ||reference - target @ R||.
        Does not scale — only rotates/reflects to preserve the true spread.
        """
        # Center both
        ref_mean = reference.mean(axis=0)
        tgt_mean = target.mean(axis=0)
        ref_c = reference - ref_mean
        tgt_c = target - tgt_mean

        # SVD of cross-covariance
        M = tgt_c.T @ ref_c
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Correct reflection if needed (ensure proper rotation)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # Apply: rotate target around its mean, then shift to reference mean
        aligned = (target - tgt_mean) @ R + ref_mean
        return aligned

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _extract_weights(self):
        weights = {}
        for name, module, _ in self._layers:
            if module is None:
                continue
            if hasattr(module, 'weight'):
                w = module.weight.data.detach().cpu().numpy()
                if w.ndim == 4:
                    # Conv2d: [out, in, kH, kW] -> [in*kH*kW, out]
                    w = w.reshape(w.shape[0], -1).T
                else:
                    w = w.T  # Linear: [fan_out, fan_in] -> [fan_in, fan_out]
                weights[name] = {
                    "shape": list(w.shape),
                    "values": np.round(w, 5).tolist(),
                }
        return weights

    # ------------------------------------------------------------------
    # Auto-versioning
    # ------------------------------------------------------------------

    def _resolve_run_dir(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(self.base_run_id)}_v(\d+)$")
        max_version = 0
        for entry in os.listdir(DATA_DIR):
            m = pattern.match(entry)
            if m:
                max_version = max(max_version, int(m.group(1)))
        next_version = max_version + 1
        run_id = f"{self.base_run_id}_v{next_version}"
        return run_id, os.path.join(DATA_DIR, run_id)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _write_run_config(self):
        input_samples_json = []
        if self._input_samples is not None:
            for i in range(len(self._input_samples)):
                pixels = self._input_samples[i].numpy()
                if pixels.ndim == 1:
                    side = int(np.sqrt(pixels.shape[0]))
                    if side * side == pixels.shape[0]:
                        pixels = pixels.reshape(side, side)
                input_samples_json.append({
                    "label": int(self._input_labels[i]),
                    "pixels": np.round(pixels, 3).tolist(),
                })

        config = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "description": self.description,
            "architecture": {
                "layers": [name for name, _, _ in self._layers],
                "sizes": [size for _, _, size in self._layers],
            },
            "hyperparameters": self.hyperparameters,
            "total_steps": self._step_count,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_samples": input_samples_json,
        }

        if self._gradient_capture:
            config["gradient_capture"] = {
                "enabled": True,
                "full_matrices": self._gradient_full_matrices,
            }

        if self._landscape_enabled:
            config["loss_landscape"] = {
                "enabled": True,
                "grid_size": self._landscape_grid_size,
                "range": self._landscape_range,
            }

        with open(os.path.join(self.run_dir, "run_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _update_runs_json(self):
        runs_path = os.path.join(DATA_DIR, "runs.json")
        if os.path.exists(runs_path):
            with open(runs_path) as f:
                data = json.load(f)
        else:
            data = {"runs": []}

        data["runs"] = [r for r in data["runs"] if r["id"] != self.run_id]
        data["runs"].append({
            "id": self.run_id,
            "model_name": self.model_name,
            "description": self.description,
        })

        with open(runs_path, "w") as f:
            json.dump(data, f, indent=2)
