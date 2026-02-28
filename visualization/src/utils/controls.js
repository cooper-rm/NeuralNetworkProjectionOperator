import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Create OrbitControls for a camera/renderer pair.
 * Returns the controls instance.
 */
export function createOrbitControls(camera, domElement) {
  const controls = new OrbitControls(camera, domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;
  controls.zoomSpeed = 1.0;
  controls.panSpeed = 0.6;
  controls.minDistance = 2;
  controls.maxDistance = 100;
  return controls;
}
