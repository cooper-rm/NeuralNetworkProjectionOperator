import { loadRuns, refreshRuns, loadRunConfig, loadCheckpoint } from './data/loader.js';
import { EmbeddingsView } from './views/embeddings.js';
import { WeightsView } from './views/weights.js';
import { GradientsView } from './views/gradients.js';
import { NetworkView } from './views/network.js';
import { getAllClassColors } from './utils/colors.js';

// ── State ──────────────────────────────────────────────
let currentRunId = null;
let currentConfig = null;
let currentStep = 0;
let currentData = null;
let activeTab = 'embeddings';
let selectedLayer = null;
let selectedSampleIndex = 0;

// Playback state
let isPlaying = false;
let playbackInterval = null;
let playbackSpeed = 1000; // ms between steps

const views = {
  embeddings: new EmbeddingsView(),
  weights: new WeightsView(),
  gradients: new GradientsView(),
  network: new NetworkView(),
};

// ── DOM refs ───────────────────────────────────────────
const viewport = document.getElementById('viewport');
const slider = document.getElementById('step-slider');
const stepLabel = document.getElementById('step-label');
const epochLabel = document.getElementById('epoch-label');
const trainLossLabel = document.getElementById('train-loss-label');
const valLossLabel = document.getElementById('val-loss-label');
const valAccLabel = document.getElementById('val-acc-label');
const testAccLabel = document.getElementById('test-acc-label');
const tabButtons = document.querySelectorAll('.tab');

// Run selector
const runSelect = document.getElementById('run-select');

// Layer panel
const layerPanel = document.getElementById('layer-panel');
const layerList = document.getElementById('layer-list');

// Sample selector panel
const sampleSelector = document.getElementById('sample-selector');
const sampleSelect = document.getElementById('sample-select');

// Color scale panel
const colorScalePanel = document.getElementById('color-scale-panel');
const colorLegend = document.getElementById('color-legend');
const weightGradient = document.getElementById('weight-gradient');
const gradientMin = document.getElementById('gradient-min');
const gradientMax = document.getElementById('gradient-max');
const activationGradient = document.getElementById('activation-gradient');
const gradValueGradient = document.getElementById('grad-value-gradient');
const gradValMin = document.getElementById('grad-val-min');
const gradValMax = document.getElementById('grad-val-max');

// Playback controls
const btnPrev = document.getElementById('btn-prev');
const btnPlay = document.getElementById('btn-play');
const btnNext = document.getElementById('btn-next');
const speedSelect = document.getElementById('speed-select');

// ── Color legend (class colors for embeddings) ────────
function buildClassLegend() {
  colorLegend.innerHTML = '';
  const colors = getAllClassColors();
  for (let i = 0; i < 10; i++) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-swatch" style="background:${colors[i]}"></span><span>${i}</span>`;
    colorLegend.appendChild(item);
  }
}

buildClassLegend();

// ── Color scale management ────────────────────────────
function updateColorScale() {
  if (activeTab === 'embeddings') {
    colorScalePanel.style.display = 'block';
    colorLegend.style.display = 'block';
    weightGradient.style.display = 'none';
    activationGradient.style.display = 'none';
    gradValueGradient.style.display = 'none';
  } else if (activeTab === 'weights') {
    colorScalePanel.style.display = 'block';
    colorLegend.style.display = 'none';
    weightGradient.style.display = 'block';
    activationGradient.style.display = 'none';
    gradValueGradient.style.display = 'none';
    const range = views.weights.getValueRange();
    if (range) {
      gradientMin.textContent = range.min.toFixed(3);
      gradientMax.textContent = range.max.toFixed(3);
    }
  } else if (activeTab === 'gradients') {
    colorScalePanel.style.display = 'block';
    colorLegend.style.display = 'none';
    weightGradient.style.display = 'none';
    activationGradient.style.display = 'none';
    gradValueGradient.style.display = 'block';
    const range = views.gradients.getValueRange();
    if (range) {
      gradValMin.textContent = range.min.toFixed(2);
      gradValMax.textContent = range.max.toFixed(2);
    }
  } else if (activeTab === 'network') {
    colorScalePanel.style.display = 'block';
    colorLegend.style.display = 'none';
    weightGradient.style.display = 'none';
    activationGradient.style.display = 'block';
    gradValueGradient.style.display = 'none';
  } else {
    colorScalePanel.style.display = 'none';
  }
}

// ── Sample selector management ────────────────────────
function populateSampleSelector() {
  sampleSelect.innerHTML = '';
  if (!currentConfig || !currentConfig.input_samples) return;

  currentConfig.input_samples.forEach((sample, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Sample ${i} (digit ${sample.label})`;
    sampleSelect.appendChild(opt);
  });

  sampleSelect.value = selectedSampleIndex;
}

sampleSelect.addEventListener('change', () => {
  selectedSampleIndex = parseInt(sampleSelect.value, 10);
  views.network.setSample(selectedSampleIndex);
});

// ── Layer panel management ────────────────────────────
function updateLayerPanel() {
  layerList.innerHTML = '';

  if (!currentConfig) return;

  let layers = [];
  if (activeTab === 'embeddings') {
    layers = currentConfig.architecture.layers;
  } else if (activeTab === 'weights') {
    layers = currentConfig.architecture.layers.filter((_, i) => i > 0);
  } else if (activeTab === 'gradients') {
    // Landscape is global — no layer selection needed
    layers = [];
  }

  // Show/hide panels based on active tab
  if (activeTab === 'network') {
    layerPanel.style.display = 'none';
    sampleSelector.style.display = 'block';
    return;
  } else {
    sampleSelector.style.display = 'none';
  }

  if (layers.length === 0) {
    layerPanel.style.display = 'none';
    return;
  }

  layerPanel.style.display = 'block';

  // Default to first layer if current selection isn't valid
  if (!selectedLayer || !layers.includes(selectedLayer)) {
    selectedLayer = layers[0];
  }

  for (const name of layers) {
    const li = document.createElement('li');
    li.textContent = name;
    li.classList.toggle('active', name === selectedLayer);
    li.addEventListener('click', () => {
      selectedLayer = name;
      updateLayerPanel();
      if (activeTab === 'embeddings') {
        views.embeddings.setLayer(name);
      } else if (activeTab === 'weights') {
        views.weights.setLayer(name);
        updateColorScale();
      } else if (activeTab === 'gradients') {
        views.gradients.setLayer(name);
        updateColorScale();
      }
    });
    layerList.appendChild(li);
  }
}

// ── Tab switching ──────────────────────────────────────
function switchTab(tabName) {
  views[activeTab].unmount();
  activeTab = tabName;

  tabButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === tabName);
  });

  views[activeTab].mount(viewport);
  resizeActiveView();

  updateLayerPanel();
  updateColorScale();

  if (currentData && currentConfig) {
    updateActiveView(currentData, currentConfig);
  }
}

tabButtons.forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.view));
});

// ── Update views with data ─────────────────────────────
function updateActiveView(data, config) {
  if (activeTab === 'embeddings') {
    views.embeddings.update(data, config, selectedLayer);
  } else if (activeTab === 'weights') {
    views.weights.update(data, config, selectedLayer);
    updateColorScale();
  } else if (activeTab === 'gradients') {
    views.gradients.update(data, config, selectedLayer);
    updateColorScale();
  } else if (activeTab === 'network') {
    views.network.update(data, config);
  }
}

function updateBottomBar(data) {
  stepLabel.textContent = `Step ${data.step}`;
  epochLabel.textContent = `Epoch ${data.epoch}`;

  const m = data.metrics || {};
  const trainLoss = m.train_loss ?? data.loss ?? null;
  const valLoss = m.val_loss ?? null;
  const valAcc = m.val_accuracy ?? null;
  const testAcc = m.test_accuracy ?? data.accuracy ?? null;

  trainLossLabel.textContent = trainLoss !== null ? `Train: ${trainLoss.toFixed(4)}` : 'Train: ---';
  valLossLabel.textContent = valLoss !== null ? `Val: ${valLoss.toFixed(4)}` : 'Val: ---';
  valAccLabel.textContent = valAcc !== null ? `Val Acc: ${(valAcc * 100).toFixed(1)}%` : 'Val Acc: ---';
  testAccLabel.textContent = testAcc !== null ? `Test Acc: ${(testAcc * 100).toFixed(1)}%` : 'Test Acc: ---';
}

// ── Step loading ──────────────────────────────────────
slider.addEventListener('input', async () => {
  currentStep = parseInt(slider.value, 10);
  await loadStep(currentStep);
});

async function loadStep(step) {
  if (!currentRunId) return;
  try {
    currentData = await loadCheckpoint(currentRunId, step);
    updateBottomBar(currentData);
    updateActiveView(currentData, currentConfig);
  } catch (err) {
    console.error('Failed to load step', step, err);
  }
}

// ── Run selector ──────────────────────────────────────
runSelect.addEventListener('change', async () => {
  await switchRun(runSelect.value);
});

// ── Refresh runs button ──────────────────────────────
const btnRefresh = document.getElementById('btn-refresh-runs');
btnRefresh.addEventListener('click', async () => {
  btnRefresh.classList.add('spinning');
  try {
    const prevId = currentRunId;
    const runsData = await refreshRuns();
    runSelect.innerHTML = '';
    for (const run of runsData.runs) {
      const opt = document.createElement('option');
      opt.value = run.id;
      opt.textContent = `${run.model_name} (${run.id})`;
      runSelect.appendChild(opt);
    }
    if (runsData.runs.some(r => r.id === prevId)) {
      runSelect.value = prevId;
      await switchRun(prevId);
    } else if (runsData.runs.length > 0) {
      await switchRun(runsData.runs[0].id);
    }
  } finally {
    btnRefresh.classList.remove('spinning');
  }
});

async function switchRun(runId) {
  currentRunId = runId;
  currentConfig = await loadRunConfig(runId);
  currentStep = 0;

  // Update slider range
  slider.max = currentConfig.total_steps - 1;
  slider.value = 0;

  // Reset layer selection
  selectedLayer = null;
  updateLayerPanel();

  // Populate sample selector from config
  selectedSampleIndex = 0;
  populateSampleSelector();

  // Load first step
  await loadStep(0);
}

// ── Playback controls ─────────────────────────────────
function startPlayback() {
  if (isPlaying) return;
  isPlaying = true;
  btnPlay.classList.add('playing');
  btnPlay.innerHTML = '&#9646;&#9646;'; // pause icon

  playbackInterval = setInterval(async () => {
    const maxStep = parseInt(slider.max, 10);
    if (currentStep >= maxStep) {
      currentStep = 0; // loop back
    } else {
      currentStep++;
    }
    slider.value = currentStep;
    await loadStep(currentStep);
  }, playbackSpeed);
}

function stopPlayback() {
  if (!isPlaying) return;
  isPlaying = false;
  btnPlay.classList.remove('playing');
  btnPlay.innerHTML = '&#9654;'; // play icon
  clearInterval(playbackInterval);
  playbackInterval = null;
}

btnPlay.addEventListener('click', () => {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

btnPrev.addEventListener('click', async () => {
  stopPlayback();
  if (currentStep > 0) {
    currentStep--;
    slider.value = currentStep;
    await loadStep(currentStep);
  }
});

btnNext.addEventListener('click', async () => {
  stopPlayback();
  const maxStep = parseInt(slider.max, 10);
  if (currentStep < maxStep) {
    currentStep++;
    slider.value = currentStep;
    await loadStep(currentStep);
  }
});

speedSelect.addEventListener('change', () => {
  playbackSpeed = parseInt(speedSelect.value, 10);
  if (isPlaying) {
    stopPlayback();
    startPlayback();
  }
});

// ── Resize handling ────────────────────────────────────
function resizeActiveView() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  views[activeTab].resize(w, h);
}

window.addEventListener('resize', resizeActiveView);

// ── Init ───────────────────────────────────────────────
async function init() {
  let runsData;
  try {
    runsData = await loadRuns();
  } catch {
    viewport.innerHTML = '<p style="padding:2rem;color:#f66;">No data found. Run a notebook in <code>experimentation/notebooks/</code> first.</p>';
    return;
  }

  if (!runsData.runs || runsData.runs.length === 0) {
    viewport.innerHTML = '<p style="padding:2rem;color:#f66;">No runs found in runs.json.</p>';
    return;
  }

  // Populate run dropdown
  for (const run of runsData.runs) {
    const opt = document.createElement('option');
    opt.value = run.id;
    opt.textContent = `${run.model_name} (${run.id})`;
    runSelect.appendChild(opt);
  }

  // Mount the default tab
  views[activeTab].mount(viewport);
  resizeActiveView();

  // Load the first run
  await switchRun(runsData.runs[0].id);
}

init();
