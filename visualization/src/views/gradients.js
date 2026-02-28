import * as THREE from 'three';
import { createOrbitControls } from '../utils/controls.js';
import { landscapeToColor, gradientFlowColor } from '../utils/colors.js';

export class GradientsView {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 200);
    this.camera.position.set(25, 20, 25);

    this.renderer = null;
    this.controls = null;
    this.checkpoint = null;
    this.config = null;
    this._animId = null;
    this._valueRange = null;

    // Surface + decorations
    this._surfaceGroup = null;
    this._ball = null;
    this._gridLines = null;
    this._axisGroup = null;
    this._flowGroup = null;
    this._noDataLabel = null;

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(10, 20, 10);
    this.scene.add(dirLight);
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

  update(checkpoint, config, _layerName) {
    this.checkpoint = checkpoint;
    this.config = config;

    if (!checkpoint.loss_landscape && !checkpoint.gradients) {
      this._clearScene();
      this._showNoDataLabel();
      return;
    }

    this._rebuildLandscape();
    this._rebuildFlowChart();
  }

  setLayer(_layerName) {
    // No-op: landscape is global, not per-layer
  }

  getValueRange() {
    return this._valueRange;
  }

  _clearScene() {
    if (this._surfaceGroup) {
      this._disposeGroup(this._surfaceGroup);
      this.scene.remove(this._surfaceGroup);
      this._surfaceGroup = null;
    }
    if (this._ball) {
      this.scene.remove(this._ball);
      this._ball.geometry.dispose();
      this._ball.material.dispose();
      this._ball = null;
    }
    if (this._gridLines) {
      this._disposeGroup(this._gridLines);
      this.scene.remove(this._gridLines);
      this._gridLines = null;
    }
    if (this._axisGroup) {
      this._disposeGroup(this._axisGroup);
      this.scene.remove(this._axisGroup);
      this._axisGroup = null;
    }
    if (this._flowGroup) {
      this._disposeGroup(this._flowGroup);
      this.scene.remove(this._flowGroup);
      this._flowGroup = null;
    }
    if (this._noDataLabel) {
      this.scene.remove(this._noDataLabel);
      this._noDataLabel = null;
    }
    this._valueRange = null;
  }

  _showNoDataLabel() {
    this._noDataLabel = this._makeLabel('No landscape data — run a notebook with enable_loss_landscape()', 0, 2, 0);
    this._noDataLabel.scale.set(20, 4, 1);
    this.scene.add(this._noDataLabel);
  }

  _rebuildLandscape() {
    // Clean previous surface + ball + grid lines + axes (but not flow chart)
    if (this._surfaceGroup) {
      this._disposeGroup(this._surfaceGroup);
      this.scene.remove(this._surfaceGroup);
      this._surfaceGroup = null;
    }
    if (this._ball) {
      this.scene.remove(this._ball);
      this._ball.geometry.dispose();
      this._ball.material.dispose();
      this._ball = null;
    }
    if (this._gridLines) {
      this._disposeGroup(this._gridLines);
      this.scene.remove(this._gridLines);
      this._gridLines = null;
    }
    if (this._axisGroup) {
      this._disposeGroup(this._axisGroup);
      this.scene.remove(this._axisGroup);
      this._axisGroup = null;
    }
    if (this._noDataLabel) {
      this.scene.remove(this._noDataLabel);
      this._noDataLabel = null;
    }
    this._valueRange = null;

    if (!this.checkpoint || !this.checkpoint.loss_landscape) {
      this._showNoDataLabel();
      return;
    }

    const landscape = this.checkpoint.loss_landscape;
    const values = landscape.values;
    const nR = values.length;
    const nC = values[0].length;

    // Find min/max loss
    let minVal = Infinity, maxVal = -Infinity;
    for (const row of values) {
      for (const v of row) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    this._valueRange = { min: minVal, max: maxVal };

    // Surface extent
    const extent = 30;
    const scaleX = extent / (nC - 1);
    const scaleZ = extent / (nR - 1);
    const halfX = extent / 2;
    const halfZ = extent / 2;
    const lossRange = maxVal - minVal || 1;
    const scaleY = 15 / lossRange;

    // ── Surface mesh ──
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const indices = [];

    for (let r = 0; r < nR; r++) {
      for (let c = 0; c < nC; c++) {
        const v = values[r][c];
        const x = c * scaleX - halfX;
        const y = (v - minVal) * scaleY;
        const z = r * scaleZ - halfZ;
        positions.push(x, y, z);

        const color = landscapeToColor(v, minVal, maxVal);
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
      shininess: 40,
      flatShading: false,
    });

    const surfaceGroup = new THREE.Group();
    surfaceGroup.add(new THREE.Mesh(geometry, material));
    this._surfaceGroup = surfaceGroup;
    this.scene.add(surfaceGroup);

    // ── Grid lines on surface ──
    const gridGroup = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.25, transparent: true });
    const yOffset = 0.05;
    const gridStep = 5;

    // Row lines (constant r)
    for (let r = 0; r < nR; r += gridStep) {
      const pts = [];
      for (let c = 0; c < nC; c++) {
        const x = c * scaleX - halfX;
        const y = (values[r][c] - minVal) * scaleY + yOffset;
        const z = r * scaleZ - halfZ;
        pts.push(new THREE.Vector3(x, y, z));
      }
      const lineGeom = new THREE.BufferGeometry().setFromPoints(pts);
      gridGroup.add(new THREE.Line(lineGeom, lineMat));
    }

    // Column lines (constant c)
    for (let c = 0; c < nC; c += gridStep) {
      const pts = [];
      for (let r = 0; r < nR; r++) {
        const x = c * scaleX - halfX;
        const y = (values[r][c] - minVal) * scaleY + yOffset;
        const z = r * scaleZ - halfZ;
        pts.push(new THREE.Vector3(x, y, z));
      }
      const lineGeom = new THREE.BufferGeometry().setFromPoints(pts);
      gridGroup.add(new THREE.Line(lineGeom, lineMat));
    }

    this._gridLines = gridGroup;
    this.scene.add(gridGroup);

    // ── Ball at center (current optimizer position) ──
    const centerR = Math.floor(nR / 2);
    const centerC = Math.floor(nC / 2);
    const centerLoss = values[centerR][centerC];
    const ballRadius = 0.6;
    const ballGeom = new THREE.SphereGeometry(ballRadius, 16, 16);
    const ballMat = new THREE.MeshPhongMaterial({
      color: 0xff6600,
      emissive: 0xff3300,
      emissiveIntensity: 0.4,
      shininess: 60,
    });
    this._ball = new THREE.Mesh(ballGeom, ballMat);
    this._ball.position.set(
      centerC * scaleX - halfX,
      (centerLoss - minVal) * scaleY + ballRadius,
      centerR * scaleZ - halfZ,
    );
    this.scene.add(this._ball);

    // ── Axis labels ──
    this._buildAxes(minVal, maxVal, scaleY, halfX, halfZ);
  }

  _buildAxes(minVal, maxVal, scaleY, halfX, halfZ) {
    const group = new THREE.Group();
    const axisColor = 0x888888;
    const baseY = 0;
    const topY = (maxVal - minVal) * scaleY;

    const lineMat = new THREE.LineBasicMaterial({ color: axisColor });

    // X axis (alpha)
    const xLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfX, baseY, halfZ + 1),
        new THREE.Vector3(halfX, baseY, halfZ + 1),
      ]),
      lineMat,
    );
    group.add(xLine);

    // Z axis (beta)
    const zLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfX - 1, baseY, -halfZ),
        new THREE.Vector3(-halfX - 1, baseY, halfZ),
      ]),
      lineMat,
    );
    group.add(zLine);

    // Y axis (loss)
    const yLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-halfX - 1, baseY, -halfZ),
        new THREE.Vector3(-halfX - 1, topY, -halfZ),
      ]),
      lineMat,
    );
    group.add(yLine);

    // Labels
    const alphaLabel = this._makeLabel('alpha direction', 0, baseY - 2, halfZ + 3);
    group.add(alphaLabel);

    const betaLabel = this._makeLabel('beta direction', -halfX - 3, baseY - 2, 0);
    group.add(betaLabel);

    const lossMinLabel = this._makeLabel(minVal.toFixed(2), -halfX - 4, baseY, -halfZ);
    lossMinLabel.scale.set(6, 1.5, 1);
    group.add(lossMinLabel);

    const lossMaxLabel = this._makeLabel(maxVal.toFixed(2), -halfX - 4, topY, -halfZ);
    lossMaxLabel.scale.set(6, 1.5, 1);
    group.add(lossMaxLabel);

    const lossTitle = this._makeLabel('Loss', -halfX - 4, topY * 0.5, -halfZ);
    group.add(lossTitle);

    this._axisGroup = group;
    this.scene.add(group);
  }

  _rebuildFlowChart() {
    if (this._flowGroup) {
      this._disposeGroup(this._flowGroup);
      this.scene.remove(this._flowGroup);
      this._flowGroup = null;
    }

    if (!this.checkpoint || !this.checkpoint.gradients || !this.checkpoint.gradients._flow) return;

    const flow = this.checkpoint.gradients._flow;
    const { layers, weight_grad_norms } = flow;
    if (!layers || layers.length === 0) return;

    const group = new THREE.Group();
    const maxNorm = Math.max(...weight_grad_norms, 0.001);
    const maxBarHeight = 12;
    const barWidth = 1.5;
    const barDepth = 1.5;
    const spacing = 3.5;

    // Position flow chart offset to the right of the main surface
    const offsetX = 25;
    const baseY = -5;

    // Title label
    const title = this._makeLabel('Gradient Flow', offsetX, baseY + maxBarHeight + 3, 0);
    title.scale.set(10, 2.5, 1);
    group.add(title);

    for (let i = 0; i < layers.length; i++) {
      const norm = weight_grad_norms[i];
      const height = Math.max(0.1, (norm / maxNorm) * maxBarHeight);

      const geom = new THREE.BoxGeometry(barWidth, height, barDepth);
      const color = gradientFlowColor(norm, maxNorm);
      const mat = new THREE.MeshPhongMaterial({ color });
      const bar = new THREE.Mesh(geom, mat);

      const x = offsetX + (i - (layers.length - 1) / 2) * spacing;
      bar.position.set(x, baseY + height / 2, 0);
      group.add(bar);

      // Layer name label below
      const label = this._makeLabel(layers[i], x, baseY - 1.5, 0);
      label.scale.set(6, 1.5, 1);
      group.add(label);

      // Norm value label above bar
      const normLabel = this._makeLabel(norm.toFixed(4), x, baseY + height + 1, 0);
      normLabel.scale.set(6, 1.5, 1);
      group.add(normLabel);
    }

    this._flowGroup = group;
    this.scene.add(group);
  }

  _disposeGroup(group) {
    group.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    });
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
