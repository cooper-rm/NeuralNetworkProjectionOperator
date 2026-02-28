import * as THREE from 'three';

// 10 distinct colors for MNIST digit classes 0-9
const CLASS_COLORS = [
  '#e6194b', // 0 - red
  '#3cb44b', // 1 - green
  '#4363d8', // 2 - blue
  '#f58231', // 3 - orange
  '#911eb4', // 4 - purple
  '#42d4f4', // 5 - cyan
  '#f032e6', // 6 - magenta
  '#bfef45', // 7 - lime
  '#fabed4', // 8 - pink
  '#dcbeff', // 9 - lavender
];

export function getClassColor(label) {
  return CLASS_COLORS[label % CLASS_COLORS.length];
}

export function getClassColorThree(label) {
  return new THREE.Color(getClassColor(label));
}

export function getAllClassColors() {
  return CLASS_COLORS;
}

/** Blue (negative) -> white (zero) -> red (positive) */
export function weightToColor(value, minVal, maxVal) {
  const range = Math.max(Math.abs(minVal), Math.abs(maxVal)) || 1;
  const t = value / range; // -1 to 1
  if (t < 0) {
    return new THREE.Color(1 + t, 1 + t, 1); // blue side
  } else {
    return new THREE.Color(1, 1 - t, 1 - t); // red side
  }
}

/**
 * Fire-scale color for activation values (0..1).
 * 5-stop gradient: black(0) -> dark red(0.33) -> orange(0.66) -> yellow(0.85) -> white(1.0)
 */
const FIRE_STOPS = [
  { t: 0.0,  r: 0.0,  g: 0.0,  b: 0.0  },  // black
  { t: 0.33, r: 0.55, g: 0.0,  b: 0.0  },  // dark red
  { t: 0.66, r: 1.0,  g: 0.45, b: 0.0  },  // orange
  { t: 0.85, r: 1.0,  g: 0.9,  b: 0.1  },  // yellow
  { t: 1.0,  r: 1.0,  g: 1.0,  b: 1.0  },  // white
];

export function activationToColor(value) {
  const v = Math.max(0, Math.min(1, value));
  // Find the two stops we're between
  for (let i = 0; i < FIRE_STOPS.length - 1; i++) {
    const a = FIRE_STOPS[i];
    const b = FIRE_STOPS[i + 1];
    if (v <= b.t) {
      const f = (v - a.t) / (b.t - a.t);
      return new THREE.Color(
        a.r + (b.r - a.r) * f,
        a.g + (b.g - a.g) * f,
        a.b + (b.b - a.b) * f
      );
    }
  }
  return new THREE.Color(1, 1, 1);
}

/** Purple (negative) -> white (zero) -> green (positive) diverging scale for gradients. */
export function gradientValueToColor(value, minVal, maxVal) {
  const range = Math.max(Math.abs(minVal), Math.abs(maxVal)) || 1;
  const t = value / range; // -1 to 1
  if (t < 0) {
    // purple side
    const s = -t;
    return new THREE.Color(0.6 + 0.4 * (1 - s), 1 - s, 0.8 + 0.2 * (1 - s));
  } else {
    // green side
    const s = t;
    return new THREE.Color(0.27 + 0.73 * (1 - s), 0.8 + 0.2 * (1 - s), 0.27 + 0.73 * (1 - s));
  }
}

/**
 * Health indicator for gradient flow bars.
 * Red (vanishing, < 1% of max) -> yellow (small) -> green (healthy) -> red (exploding, > 200% of max).
 */
export function gradientFlowColor(norm, maxNorm) {
  if (maxNorm <= 0) return new THREE.Color(0.5, 0.5, 0.5);
  const ratio = norm / maxNorm;
  if (ratio < 0.01) {
    // Vanishing: red
    return new THREE.Color(0.9, 0.15, 0.15);
  } else if (ratio < 0.2) {
    // Small: interpolate red -> yellow
    const t = (ratio - 0.01) / 0.19;
    return new THREE.Color(0.9, 0.15 + 0.75 * t, 0.15 * (1 - t));
  } else if (ratio <= 1.0) {
    // Healthy: interpolate yellow -> green
    const t = (ratio - 0.2) / 0.8;
    return new THREE.Color(0.9 - 0.6 * t, 0.9, 0.15 + 0.35 * t);
  } else if (ratio <= 2.0) {
    // Getting large: green -> orange
    const t = (ratio - 1.0) / 1.0;
    return new THREE.Color(0.3 + 0.6 * t, 0.9 - 0.5 * t, 0.5 - 0.35 * t);
  } else {
    // Exploding: red
    return new THREE.Color(0.9, 0.15, 0.15);
  }
}

/**
 * Viridis-like colormap for loss landscape values.
 * 5-stop gradient: deep purple -> blue -> teal -> green -> yellow
 */
const VIRIDIS_STOPS = [
  { t: 0.0,  r: 0.267, g: 0.004, b: 0.329 },  // deep purple
  { t: 0.25, r: 0.192, g: 0.408, b: 0.557 },  // blue
  { t: 0.5,  r: 0.129, g: 0.569, b: 0.549 },  // teal
  { t: 0.75, r: 0.369, g: 0.788, b: 0.384 },  // green
  { t: 1.0,  r: 0.992, g: 0.906, b: 0.145 },  // yellow
];

export function landscapeToColor(value, minVal, maxVal) {
  const range = maxVal - minVal || 1;
  const v = Math.max(0, Math.min(1, (value - minVal) / range));
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    const a = VIRIDIS_STOPS[i];
    const b = VIRIDIS_STOPS[i + 1];
    if (v <= b.t) {
      const f = (v - a.t) / (b.t - a.t);
      return new THREE.Color(
        a.r + (b.r - a.r) * f,
        a.g + (b.g - a.g) * f,
        a.b + (b.b - a.b) * f
      );
    }
  }
  return new THREE.Color(0.992, 0.906, 0.145);
}

/** Nonlinear emissive intensity for activation glow. */
export function activationToEmissiveIntensity(value) {
  const v = Math.max(0, Math.min(1, value));
  return Math.pow(v, 0.7) * 0.8;
}
