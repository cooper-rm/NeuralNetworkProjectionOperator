import * as THREE from 'three';
import { createOrbitControls } from '../utils/controls.js';
import { getClassColorThree } from '../utils/colors.js';

export class EmbeddingsView {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 200);
    this.camera.position.set(8, 6, 8);

    this.renderer = null;
    this.controls = null;
    this.points = null;
    this._centroidObjects = [];  // centroid spheres + axis lines
    this.currentLayer = null;
    this.checkpoint = null;
    this.config = null;
    this._animId = null;

    // Axes helper
    const axes = new THREE.AxesHelper(5);
    this.scene.add(axes);

    // Soft ambient + directional light
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight.position.set(5, 10, 5);
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

  update(checkpoint, config, layerName) {
    this.checkpoint = checkpoint;
    this.config = config;
    this.currentLayer = layerName || config.architecture.layers[0];
    this._rebuildPoints();
  }

  setLayer(layerName) {
    if (this.currentLayer === layerName) return;
    this.currentLayer = layerName;
    this._rebuildPoints();
  }

  _rebuildPoints() {
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    }
    for (const obj of this._centroidObjects) {
      this.scene.remove(obj);
      obj.geometry.dispose();
      obj.material.dispose();
    }
    this._centroidObjects = [];

    if (!this.checkpoint || !this.currentLayer) return;

    const embed = this.checkpoint.embeddings[this.currentLayer];
    if (!embed) return;

    const count = embed.points.length;

    // ── Individual sample spheres ──
    const sphereGeo = new THREE.SphereGeometry(0.06, 8, 6);
    const material = new THREE.MeshPhongMaterial({
      transparent: true,
      opacity: 0.65,
    });

    const mesh = new THREE.InstancedMesh(sphereGeo, material, count);
    const dummy = new THREE.Object3D();

    // Accumulate centroids per class
    const classSums = {};  // label -> {x, y, z, count}

    for (let i = 0; i < count; i++) {
      const [x, y, z] = embed.points[i];
      const label = embed.labels[i];

      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, getClassColorThree(label));

      if (!classSums[label]) {
        classSums[label] = { x: 0, y: 0, z: 0, count: 0 };
      }
      classSums[label].x += x;
      classSums[label].y += y;
      classSums[label].z += z;
      classSums[label].count++;
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    this.points = mesh;
    this.scene.add(this.points);

    // ── Centroids (bright sphere + 3 axis lines per class) ──
    const classes = Object.keys(classSums).map(Number).sort((a, b) => a - b);
    const centroidGeo = new THREE.SphereGeometry(0.12, 12, 8);
    const armLen = 0.4;

    for (let i = 0; i < classes.length; i++) {
      const label = classes[i];
      const s = classSums[label];
      const cx = s.x / s.count;
      const cy = s.y / s.count;
      const cz = s.z / s.count;

      const classColor = getClassColorThree(label);
      const brightColor = classColor.clone().lerp(new THREE.Color(0xffffff), 0.35);

      // Bright sphere
      const mat = new THREE.MeshPhongMaterial({
        color: brightColor,
        emissive: classColor,
        emissiveIntensity: 0.5,
      });
      const sphere = new THREE.Mesh(centroidGeo, mat);
      sphere.position.set(cx, cy, cz);
      this.scene.add(sphere);
      this._centroidObjects.push(sphere);

      // 3 small axis lines protruding from centroid
      const center = new THREE.Vector3(cx, cy, cz);
      const dirs = [
        new THREE.Vector3(armLen, 0, 0),
        new THREE.Vector3(0, armLen, 0),
        new THREE.Vector3(0, 0, armLen),
      ];
      for (const dir of dirs) {
        const geo = new THREE.BufferGeometry().setFromPoints([
          center.clone().sub(dir),
          center.clone().add(dir),
        ]);
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: brightColor }));
        this.scene.add(line);
        this._centroidObjects.push(line);
      }
    }
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
