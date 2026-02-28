import * as THREE from 'three';
import { createOrbitControls } from '../utils/controls.js';
import { weightToColor } from '../utils/colors.js';

const MAX_SURFACE_DIM = 64; // downsample large matrices for rendering

export class WeightsView {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 200);
    this.camera.position.set(20, 15, 20);

    this.renderer = null;
    this.controls = null;
    this.mesh = null;
    this.currentLayer = null;
    this.checkpoint = null;
    this.config = null;
    this._animId = null;
    this._valueRange = null;

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(10, 20, 10);
    this.scene.add(dirLight);

    this._grid = null;
    this._axisGroup = null;
  }

  mount(container) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);
    this.controls = createOrbitControls(this.camera, this.renderer.domElement);
    this.resize(container.clientWidth, container.clientHeight);
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

  update(checkpoint, config, layerName) {
    this.checkpoint = checkpoint;
    this.config = config;
    const weightLayers = Object.keys(checkpoint.weights);
    this.currentLayer = layerName && checkpoint.weights[layerName]
      ? layerName
      : weightLayers[0];
    this._rebuildSurface();
  }

  setLayer(layerName) {
    if (this.currentLayer === layerName) return;
    this.currentLayer = layerName;
    this._rebuildSurface();
  }

  /** Returns { min, max } of the currently displayed weight values. */
  getValueRange() {
    return this._valueRange;
  }

  _rebuildSurface() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }

    if (this._axisGroup) {
      this.scene.remove(this._axisGroup);
      this._axisGroup = null;
    }
    if (this._grid) {
      this.scene.remove(this._grid);
      this._grid = null;
    }

    this._valueRange = null;

    if (!this.checkpoint || !this.currentLayer) return;

    const w = this.checkpoint.weights[this.currentLayer];
    if (!w) return;

    // Subsample the weight matrix
    const values = w.values;
    const rows = values.length;
    const cols = values[0].length;
    const stepR = Math.max(1, Math.ceil(rows / MAX_SURFACE_DIM));
    const stepC = Math.max(1, Math.ceil(cols / MAX_SURFACE_DIM));

    const sampledRows = [];
    for (let r = 0; r < rows; r += stepR) {
      const row = [];
      for (let c = 0; c < cols; c += stepC) {
        row.push(values[r][c]);
      }
      sampledRows.push(row);
    }

    const nR = sampledRows.length;
    const nC = sampledRows[0].length;

    // Find min/max for color mapping
    let minVal = Infinity, maxVal = -Infinity;
    for (const row of sampledRows) {
      for (const v of row) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }

    this._valueRange = { min: minVal, max: maxVal };

    // Build geometry
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const indices = [];

    // Proportional sizing: larger dim gets 30 units, smaller scales accordingly
    const maxDim = Math.max(rows, cols);
    const extentX = 30 * (cols / maxDim);
    const extentZ = 30 * (rows / maxDim);
    const scaleX = extentX / nC;
    const scaleZ = extentZ / nR;
    const halfX = extentX / 2;
    const halfZ = extentZ / 2;
    const scaleY = 10 / Math.max(Math.abs(minVal), Math.abs(maxVal), 0.01);

    for (let r = 0; r < nR; r++) {
      for (let c = 0; c < nC; c++) {
        const v = sampledRows[r][c];
        const x = c * scaleX - halfX;
        const y = v * scaleY;
        const z = r * scaleZ - halfZ;
        positions.push(x, y, z);

        const color = weightToColor(v, minVal, maxVal);
        colors.push(color.r, color.g, color.b);
      }
    }

    for (let r = 0; r < nR - 1; r++) {
      for (let c = 0; c < nC - 1; c++) {
        const i = r * nC + c;
        indices.push(i, i + 1, i + nC);
        indices.push(i + 1, i + nC + 1, i + nC);
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 30,
      flatShading: false,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.mesh);

    // Grid sized to match the surface
    const gridSize = Math.max(extentX, extentZ);
    this._grid = new THREE.GridHelper(gridSize, 30, 0x333333, 0x222222);
    this._grid.position.y = -5;
    this.scene.add(this._grid);

    this._buildAxes(rows, cols, minVal, maxVal, scaleY, halfX, halfZ);
  }

  _buildAxes(rows, cols, minVal, maxVal, scaleY, halfX, halfZ) {
    const group = new THREE.Group();
    const axisColor = 0x888888;
    const gridY = -5;

    const lineMat = new THREE.LineBasicMaterial({ color: axisColor });

    // X axis (output neurons) — along front edge
    const xLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfX, gridY, halfZ),
        new THREE.Vector3(halfX, gridY, halfZ),
      ]),
      lineMat,
    );
    group.add(xLine);

    // Z axis (input neurons) — along left edge
    const zLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfX, gridY, -halfZ),
        new THREE.Vector3(-halfX, gridY, halfZ),
      ]),
      lineMat,
    );
    group.add(zLine);

    // Y axis (weight value) — vertical at back-left corner
    const yTop = Math.max(Math.abs(minVal), Math.abs(maxVal), 0.01) * scaleY;
    const yBot = -yTop;
    const yLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfX, Math.min(gridY, yBot), -halfZ),
        new THREE.Vector3(-halfX, yTop, -halfZ),
      ]),
      lineMat,
    );
    group.add(yLine);

    // Labels
    group.add(this._makeLabel(`Output Neurons (${cols})`, 0, gridY - 1.5, halfZ + 3));
    group.add(this._makeLabel(`Input Neurons (${rows})`, -halfX - 3, gridY - 1.5, 0));
    group.add(this._makeLabel('Weight Value', -halfX - 3, yTop * 0.5, -halfZ));

    this._axisGroup = group;
    this.scene.add(group);
  }

  _makeLabel(text, x, y, z) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, 256, 64);
    ctx.font = '24px sans-serif';
    ctx.fillStyle = '#aaaaaa';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(x, y, z);
    sprite.scale.set(10, 2.5, 1);
    return sprite;
  }

  _startLoop() {
    const loop = () => {
      this._animId = requestAnimationFrame(loop);
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
