import * as THREE from 'three';
import { createOrbitControls } from '../utils/controls.js';
import { activationToColor, activationToEmissiveIntensity } from '../utils/colors.js';

const MAX_NODES_PER_LAYER = 20;
const NODE_RADIUS = 0.3;
const LAYER_SPACING = 8;
const FLOW_DURATION = 0.8; // seconds for wavefront to traverse entire network

export class NetworkView {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 200);
    this.camera.position.set(0, 10, 30);

    this.renderer = null;
    this.controls = null;
    this.group = null;
    this.checkpoint = null;
    this.config = null;
    this._animId = null;

    // Activation state
    this._sampleIndex = 0;
    this._nodeMeshes = [];    // [layerIndex][nodeIndex] -> Mesh
    this._nodeIndices = [];   // [layerIndex][nodeIndex] -> original index in full layer
    this._edgeLines = [];     // [{line, fromLayer, fromNode, toLayer, toNode}]
    this._layerPositions = [];

    // Input plane
    this._inputPlane = null;
    this._inputCanvas = null;
    this._inputTexture = null;

    // Output display
    this._outputGroup = null;
    this._outputBars = [];
    this._outputLabels = [];

    // Flow animation
    this._flowActive = false;
    this._flowElapsed = 0;
    this._lastTime = 0;

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(5, 15, 10);
    this.scene.add(dirLight);
  }

  mount(container) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);

    // Enable ACES filmic tone mapping for natural glow
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;

    container.appendChild(this.renderer.domElement);
    this.controls = createOrbitControls(this.camera, this.renderer.domElement);
    this.resize(container.clientWidth, container.clientHeight);
    this._lastTime = performance.now() / 1000;
    this._startLoop();
  }

  unmount() {
    this._stopLoop();
    if (this.renderer) {
      this.renderer.domElement.remove();
      this.renderer.dispose();
      this.renderer = null;
    }
    if (this.controls) {
      this.controls.dispose();
      this.controls = null;
    }
  }

  resize(w, h) {
    if (!this.renderer) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  update(checkpoint, config) {
    this.checkpoint = checkpoint;
    this.config = config;
    this._rebuildNetwork();
    this._applyAllActivations();
  }

  /** Public: change which input sample is visualized. */
  setSample(index) {
    this._sampleIndex = index;
    this._applyAllActivations();
    this._startFlowAnimation();
  }

  // ── Build network geometry ────────────────────────────

  _rebuildNetwork() {
    // Dispose old
    if (this.group) {
      this.scene.remove(this.group);
      this.group.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (obj.material.map) obj.material.map.dispose();
          obj.material.dispose();
        }
      });
    }
    if (this._inputPlane) {
      this.scene.remove(this._inputPlane);
      this._inputPlane = null;
    }
    if (this._outputGroup) {
      this.scene.remove(this._outputGroup);
      this._outputGroup = null;
    }

    this._nodeMeshes = [];
    this._nodeIndices = [];
    this._edgeLines = [];
    this._layerPositions = [];
    this._outputBars = [];
    this._outputLabels = [];

    if (!this.config) return;

    this.group = new THREE.Group();
    const { layers, sizes } = this.config.architecture;
    const numLayers = layers.length;

    const totalWidth = (numLayers - 1) * LAYER_SPACING;
    const startX = -totalWidth / 2;

    const nodeSphere = new THREE.SphereGeometry(NODE_RADIUS, 16, 12);

    for (let li = 0; li < numLayers; li++) {
      const layerName = layers[li];
      const fullSize = sizes[li];
      const displayCount = Math.min(fullSize, MAX_NODES_PER_LAYER);
      const x = startX + li * LAYER_SPACING;

      const totalHeight = (displayCount - 1) * (NODE_RADIUS * 2.5);
      const startY = -totalHeight / 2;

      const positions = [];
      const meshes = [];
      const indices = [];

      // Compute which original indices we subsample to
      for (let ni = 0; ni < displayCount; ni++) {
        const origIndex = fullSize <= MAX_NODES_PER_LAYER
          ? ni
          : Math.round(ni * (fullSize - 1) / (displayCount - 1));

        const y = startY + ni * (NODE_RADIUS * 2.5);
        const pos = new THREE.Vector3(x, y, 0);
        positions.push(pos);
        indices.push(origIndex);

        const material = new THREE.MeshPhongMaterial({
          color: 0x111111,
          emissive: 0x000000,
          emissiveIntensity: 0,
          shininess: 60,
        });
        const mesh = new THREE.Mesh(nodeSphere, material);
        mesh.position.copy(pos);
        meshes.push(mesh);
        this.group.add(mesh);
      }

      this._layerPositions.push(positions);
      this._nodeMeshes.push(meshes);
      this._nodeIndices.push(indices);

      // Layer label
      const label = this._makeLabel(`${layerName} (${fullSize})`, x, startY - 2);
      this.group.add(label);
    }

    // Draw edges between adjacent layers
    for (let li = 0; li < numLayers - 1; li++) {
      const fromPositions = this._layerPositions[li];
      const toPositions = this._layerPositions[li + 1];

      const maxEdges = 200;
      const totalPossible = fromPositions.length * toPositions.length;
      const step = Math.max(1, Math.ceil(totalPossible / maxEdges));

      let edgeCount = 0;
      for (let fi = 0; fi < fromPositions.length; fi++) {
        for (let ti = 0; ti < toPositions.length; ti++) {
          if (edgeCount % step !== 0) {
            edgeCount++;
            continue;
          }
          edgeCount++;

          const points = [fromPositions[fi], toPositions[ti]];
          const geom = new THREE.BufferGeometry().setFromPoints(points);
          const material = new THREE.LineBasicMaterial({
            color: 0x333333,
            transparent: true,
            opacity: 0.15,
          });
          const line = new THREE.Line(geom, material);
          this.group.add(line);

          this._edgeLines.push({
            line,
            fromLayer: li,
            fromNode: fi,
            toLayer: li + 1,
            toNode: ti,
          });
        }
      }
    }

    this.scene.add(this.group);

    // Build input plane and output display
    this._buildInputPlane(startX);
    this._buildOutputDisplay(startX + (numLayers - 1) * LAYER_SPACING);
  }

  // ── Input image billboard ─────────────────────────────

  _buildInputPlane(startX) {
    this._inputCanvas = document.createElement('canvas');
    this._inputCanvas.width = 28;
    this._inputCanvas.height = 28;

    this._inputTexture = new THREE.CanvasTexture(this._inputCanvas);
    this._inputTexture.magFilter = THREE.NearestFilter;
    this._inputTexture.minFilter = THREE.NearestFilter;

    const planeGeom = new THREE.PlaneGeometry(5, 5);
    const planeMat = new THREE.MeshBasicMaterial({
      map: this._inputTexture,
      side: THREE.DoubleSide,
    });
    this._inputPlane = new THREE.Mesh(planeGeom, planeMat);
    this._inputPlane.position.set(startX - 6, 0, 0);
    this.scene.add(this._inputPlane);
  }

  _updateInputPlane() {
    if (!this._inputCanvas || !this.config || !this.config.input_samples) return;

    const sample = this.config.input_samples[this._sampleIndex];
    if (!sample) return;

    const ctx = this._inputCanvas.getContext('2d');
    const imgData = ctx.createImageData(28, 28);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = sample.pixels[y][x];
        const idx = (y * 28 + x) * 4;
        // Use fire-ish tint for nonzero pixels
        const r = Math.min(255, Math.round(v * 255 * 1.2));
        const g = Math.min(255, Math.round(v * 200));
        const b = Math.round(v * 80);
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    this._inputTexture.needsUpdate = true;
  }

  // ── Output prediction bar chart ───────────────────────

  _buildOutputDisplay(outputX) {
    this._outputGroup = new THREE.Group();
    this._outputGroup.position.set(outputX + 6, 0, 0);
    this._outputBars = [];
    this._outputLabels = [];

    const barWidth = 0.4;
    const barSpacing = 0.6;
    const totalHeight = 10 * barSpacing;
    const startY = -totalHeight / 2;

    for (let i = 0; i < 10; i++) {
      // Bar
      const geom = new THREE.BoxGeometry(1, barWidth, barWidth);
      const mat = new THREE.MeshPhongMaterial({ color: 0x444444 });
      const bar = new THREE.Mesh(geom, mat);
      bar.position.set(0, startY + i * barSpacing, 0);
      bar.scale.x = 0.01; // will be updated
      this._outputGroup.add(bar);
      this._outputBars.push(bar);

      // Label
      const label = this._makeLabel(String(i), -1.5, startY + i * barSpacing);
      label.scale.set(1.5, 0.5, 1);
      this._outputGroup.add(label);
      this._outputLabels.push(label);
    }

    this.scene.add(this._outputGroup);
  }

  _updateOutputDisplay() {
    if (!this.checkpoint || !this.checkpoint.activations || !this._outputBars.length) return;

    const sampleKey = `sample_${this._sampleIndex}`;
    const sampleActs = this.checkpoint.activations[sampleKey];
    if (!sampleActs || !sampleActs.output) return;

    const probs = sampleActs.output;
    const maxProb = Math.max(...probs);
    const predictedClass = probs.indexOf(maxProb);

    const maxBarLength = 4;

    for (let i = 0; i < 10; i++) {
      const bar = this._outputBars[i];
      const prob = probs[i];

      bar.scale.x = Math.max(0.01, prob * maxBarLength);
      // Offset so bars grow from left edge
      bar.position.x = bar.scale.x / 2;

      if (i === predictedClass) {
        bar.material.color.setHex(0xffd700); // gold
        bar.material.emissive.setHex(0x664400);
        bar.material.emissiveIntensity = 0.5;
      } else {
        const c = activationToColor(prob);
        bar.material.color.copy(c);
        bar.material.emissive.setHex(0x000000);
        bar.material.emissiveIntensity = 0;
      }
    }
  }

  // ── Activation application ────────────────────────────

  _applyAllActivations() {
    this._applyActivations();
    this._applyEdgeActivations();
    this._updateInputPlane();
    this._updateOutputDisplay();
  }

  _applyActivations() {
    if (!this.checkpoint || !this.checkpoint.activations || !this.config) return;

    const sampleKey = `sample_${this._sampleIndex}`;
    const sampleActs = this.checkpoint.activations[sampleKey];
    if (!sampleActs) return;

    const { layers } = this.config.architecture;

    for (let li = 0; li < layers.length; li++) {
      const layerName = layers[li];
      const acts = sampleActs[layerName];
      if (!acts) continue;

      const meshes = this._nodeMeshes[li];
      const indices = this._nodeIndices[li];

      // Per-layer normalization for visualization
      let maxAct = 0;
      for (let ni = 0; ni < indices.length; ni++) {
        const origIdx = indices[ni];
        if (origIdx < acts.length) {
          maxAct = Math.max(maxAct, Math.abs(acts[origIdx]));
        }
      }
      if (maxAct === 0) maxAct = 1;

      for (let ni = 0; ni < meshes.length; ni++) {
        const origIdx = indices[ni];
        const rawVal = origIdx < acts.length ? acts[origIdx] : 0;
        const normVal = Math.abs(rawVal) / maxAct;

        const color = activationToColor(normVal);
        const emissiveIntensity = activationToEmissiveIntensity(normVal);

        meshes[ni].material.color.copy(color);
        meshes[ni].material.emissive.copy(color);
        meshes[ni].material.emissiveIntensity = emissiveIntensity;
      }
    }
  }

  _applyEdgeActivations() {
    if (!this.checkpoint || !this.checkpoint.activations || !this.config) return;

    const sampleKey = `sample_${this._sampleIndex}`;
    const sampleActs = this.checkpoint.activations[sampleKey];
    if (!sampleActs) return;

    const { layers } = this.config.architecture;
    const weights = this.checkpoint.weights;

    // Precompute per-layer signal ranges for normalization
    const layerMaxSignal = [];
    for (let li = 0; li < layers.length - 1; li++) {
      const fromActs = sampleActs[layers[li]];
      const weightKey = layers[li + 1];
      const layerWeights = weights ? weights[weightKey] : null;

      let maxSig = 0;
      if (fromActs && layerWeights) {
        const fromIndices = this._nodeIndices[li];
        const toIndices = this._nodeIndices[li + 1];
        for (const fi of fromIndices) {
          for (const ti of toIndices) {
            if (fi < fromActs.length && fi < layerWeights.values.length && ti < layerWeights.values[fi].length) {
              const signal = Math.abs(fromActs[fi] * layerWeights.values[fi][ti]);
              if (signal > maxSig) maxSig = signal;
            }
          }
        }
      }
      layerMaxSignal.push(maxSig || 1);
    }

    for (const edge of this._edgeLines) {
      const { line, fromLayer, fromNode, toLayer, toNode } = edge;
      const fromLayerName = layers[fromLayer];
      const fromActs = sampleActs[fromLayerName];
      const weightKey = layers[toLayer];
      const layerWeights = weights ? weights[weightKey] : null;

      let normSignal = 0;
      if (fromActs && layerWeights) {
        const fi = this._nodeIndices[fromLayer][fromNode];
        const ti = this._nodeIndices[toLayer][toNode];
        if (fi < fromActs.length && fi < layerWeights.values.length && ti < layerWeights.values[fi].length) {
          const signal = Math.abs(fromActs[fi] * layerWeights.values[fi][ti]);
          normSignal = signal / layerMaxSignal[fromLayer];
        }
      }

      const color = activationToColor(normSignal);
      line.material.color.copy(color);
      line.material.opacity = 0.05 + normSignal * 0.5;
    }
  }

  // ── Flow animation ────────────────────────────────────

  _startFlowAnimation() {
    this._flowActive = true;
    this._flowElapsed = 0;
  }

  _applyFlowVisuals(dt) {
    if (!this._flowActive || !this.config) return;

    this._flowElapsed += dt;
    const progress = this._flowElapsed / FLOW_DURATION;

    if (progress >= 1.0) {
      this._flowActive = false;
      // Restore full activations
      this._applyAllActivations();
      return;
    }

    const { layers } = this.config.architecture;
    const numLayers = layers.length;
    // Wavefront position: 0..numLayers-1
    const wavefrontPos = progress * (numLayers - 1);

    for (let li = 0; li < numLayers; li++) {
      const meshes = this._nodeMeshes[li];
      if (!meshes) continue;

      let dimFactor;
      if (li < wavefrontPos - 0.5) {
        // Behind wavefront: fully revealed
        dimFactor = 1.0;
      } else if (li > wavefrontPos + 0.5) {
        // Ahead of wavefront: dimmed
        dimFactor = 0.1;
      } else {
        // At wavefront: pulse bright
        dimFactor = 1.3;
      }

      for (const mesh of meshes) {
        mesh.material.emissiveIntensity *= dimFactor;
      }
    }

    // Dim/brighten edges based on wavefront
    for (const edge of this._edgeLines) {
      const midLayer = (edge.fromLayer + edge.toLayer) / 2;
      if (midLayer < wavefrontPos - 0.5) {
        // revealed
      } else if (midLayer > wavefrontPos + 0.5) {
        edge.line.material.opacity *= 0.1;
      } else {
        edge.line.material.opacity = Math.min(1, edge.line.material.opacity * 1.5);
      }
    }
  }

  // ── Helpers ───────────────────────────────────────────

  _makeLabel(text, x, y) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.font = '24px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, 128, 40);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(x, y, 0);
    sprite.scale.set(4, 1, 1);
    return sprite;
  }

  _startLoop() {
    const loop = () => {
      this._animId = requestAnimationFrame(loop);

      const now = performance.now() / 1000;
      const dt = Math.min(now - this._lastTime, 0.1); // cap delta
      this._lastTime = now;

      if (this._flowActive) {
        // Re-apply base activations, then overlay flow visuals
        this._applyAllActivations();
        this._applyFlowVisuals(dt);
      }

      if (this.controls) this.controls.update();
      if (this.renderer) this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  _stopLoop() {
    if (this._animId != null) {
      cancelAnimationFrame(this._animId);
      this._animId = null;
    }
  }
}
