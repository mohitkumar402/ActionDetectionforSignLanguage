/**
 * SignSense — AI Sign Language Interpreter
 * Architecture: MediaPipe runs IN BROWSER → sends 126 keypoint floats to server
 * Server only runs LSTM inference → returns prediction JSON
 * Result: ~10x faster than sending JPEG frames
 */

'use strict';

// ─── Config ──────────────────────────────────────────────────────────────────
const DEFAULT_CONFIG = {
  backendUrl: 'ws://127.0.0.1:8001',
  threshold: 0.5,
  fpsLimit: 24,
  showAnnotated: true,
  autoTts: false,
  showLandmarks: true,
};

const GESTURES = [
  { name: 'hello',    emoji: '👋', label: 'HELLO',  description: 'Wave your hand as a greeting' },
  { name: 'thanks',   emoji: '🙏', label: 'THANKS', description: 'Press palms together' },
  { name: 'iloveyou', emoji: '🤟', label: 'ILY',    description: 'Thumb, index & pinky extended' },
  { name: 'one',      emoji: '☝️', label: 'ONE',    description: 'Raise index finger' },
  { name: 'two',      emoji: '✌️', label: 'TWO',    description: 'V shape fingers' },
  { name: 'three',    emoji: '🤙', label: 'THREE',  description: 'Raise three fingers' },
];

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  config: { ...DEFAULT_CONFIG },
  ws: null,
  sessionId: generateSessionId(),
  isRunning: false,
  isPaused: false,
  isFlipped: false,

  // MediaPipe camera utility handle
  mpCamera: null,
  mpHolistic: null,

  // stats
  totalPredictions: 0,
  gestureUsage: Object.fromEntries(GESTURES.map(g => [g.name, 0])),
  confidenceSum: 0,
  startTime: null,
  uptimeTimer: null,

  // TTS
  ttsEnabled: false,
  speechSynth: window.speechSynthesis || null,
};

// Local frame tracking (not sent to server)
let _localFrameCount = 0;
let _localFpsCount   = 0;
let _lastFpsTime     = Date.now();
let _canvasCtx       = null;

// ─── DOM ─────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dom = {
  video:          $('videoInput'),
  annotated:      $('annotatedFeed'),
  canvas:         $('overlayCanvas'),
  statusPill:     $('connectionStatus'),
  statusText:     null,
  statusDot:      null,
  fpsValue:       $('fpsValue'),

  startBtn:       $('startBtn'),
  pauseBtn:       $('pauseBtn'),
  stopBtn:        $('stopBtn'),
  flipBtn:        $('flipBtn'),

  ttsBtn:         $('ttsBtn'),
  copyBtn:        $('copyBtn'),
  clearBtn:       $('clearBtn'),

  sentenceDisplay:  $('sentenceDisplay'),
  wordCount:        $('wordCount'),
  lastGesture:      $('lastGesture'),

  confidenceBars:   $('confidenceBars'),
  usageBars:        $('usageBars'),
  gestureGrid:      $('gestureGrid'),
  historyList:      $('historyList'),
  detectionBadge:   $('detectionBadge'),

  predictionEmoji:  $('predictionEmoji'),
  predictionLabel:  $('predictionLabel'),
  ringFill:         $('ringFill'),
  confValue:        $('confValue'),

  totalPredictions: $('totalPredictions'),
  avgConfidence:    $('avgConfidence'),
  uptime:           $('uptime'),
  frameCount:       $('frameCount'),

  scanOverlay:      $('scanOverlay'),
  scanLabel:        $('scanLabel'),
  sequenceBarFill:  $('sequenceBarFill'),
  sequenceBarLabel: $('sequenceBarLabel'),
  landmarkIndicator: $('landmarkIndicator'),
  landmarkText:     $('landmarkText'),

  settingsBtn:      $('settingsBtn'),
  helpBtn:          $('helpBtn'),
  settingsModal:    $('settingsModal'),
  closeSettings:    $('closeSettings'),
  saveSettings:     $('saveSettings'),
  resetSettings:    $('resetSettings'),
  backendUrl:       $('backendUrl'),
  thresholdSlider:  $('thresholdSlider'),
  thresholdValue:   $('thresholdValue'),
  fpsLimit:         $('fpsLimit'),
  showAnnotated:    $('showAnnotated'),
  autoTts:          $('autoTts'),
  showLandmarks:    $('showLandmarks'),
  clearHistoryBtn:  $('clearHistoryBtn'),
  toastContainer:   $('toastContainer'),
};

// ─── Init ─────────────────────────────────────────────────────────────────────
function init() {
  dom.statusText = dom.statusPill.querySelector('.status-text');
  dom.statusDot  = dom.statusPill.querySelector('.status-dot');
  _canvasCtx     = dom.canvas.getContext('2d');

  loadConfig();
  buildConfidenceBars();
  buildGestureGrid();
  buildUsageBars();
  bindEvents();
  updateUptime();
  setConnectionStatus('disconnected', 'Not connected');
  showToast('info', '🤟 Welcome to SignSense! Click Start Camera to begin.', 3000);
}

// ─── Config ───────────────────────────────────────────────────────────────────
function loadConfig() {
  try {
    const stored = localStorage.getItem('signsense_config');
    if (stored) state.config = { ...DEFAULT_CONFIG, ...JSON.parse(stored) };
  } catch (_) {}
  applyConfigToForm();
}
function saveConfig() {
  try { localStorage.setItem('signsense_config', JSON.stringify(state.config)); } catch (_) {}
}
function applyConfigToForm() {
  dom.backendUrl.value           = state.config.backendUrl;
  dom.thresholdSlider.value      = state.config.threshold;
  dom.thresholdValue.textContent = parseFloat(state.config.threshold).toFixed(2);
  dom.fpsLimit.value             = state.config.fpsLimit;
  dom.showAnnotated.checked      = state.config.showAnnotated;
  dom.autoTts.checked            = state.config.autoTts;
  dom.showLandmarks.checked      = state.config.showLandmarks;
}

// ─── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  const url = `${state.config.backendUrl}/ws/${state.sessionId}`;
  setConnectionStatus('connecting', 'Connecting...');

  try { state.ws = new WebSocket(url); }
  catch (e) {
    setConnectionStatus('disconnected', 'Failed');
    showToast('error', '❌ Cannot connect to backend. Is the server running?', 5000);
    return;
  }

  state.ws.onopen = () => {
    setConnectionStatus('connected', 'Connected');
    showToast('success', '✅ Connected — initializing MediaPipe...');
    startMediaPipe();
  };

  state.ws.onmessage = (ev) => {
    try { handleServerMessage(JSON.parse(ev.data)); } catch (_) {}
  };

  state.ws.onclose = () => {
    setConnectionStatus('disconnected', 'Disconnected');
    if (state.isRunning) {
      showToast('error', '🔌 Connection lost. Reconnecting in 3s...', 3000);
      setTimeout(() => { if (state.isRunning) connectWS(); }, 3000);
    }
  };

  state.ws.onerror = () => {
    setConnectionStatus('disconnected', 'Error');
    showToast('error', '❌ WebSocket error. Check backend URL in Settings.', 4000);
  };
}

function handleServerMessage(msg) {
  if (msg.type === 'prediction') {
    // FPS from server (confirmation)
    if (msg.fps) dom.fpsValue.textContent = msg.fps;

    // Buffer bar from server
    if (msg.buffer !== undefined) updateSequenceBar(msg.buffer);
    if (msg.frame_count !== undefined) dom.frameCount.textContent = msg.frame_count;

    // Confidence bars
    if (msg.probabilities && Object.keys(msg.probabilities).length > 0) {
      updateConfidenceBars(msg.probabilities);
    }

    // Prediction hero + history
    if (msg.action && msg.confidence > state.config.threshold) {
      updatePredictionHero(msg.action, msg.confidence);
      updateDetectionBadge(true);

      if (state.config.autoTts) speakText(msg.action);

      addHistoryItem(msg.action, msg.confidence, msg.timestamp);

      state.gestureUsage[msg.action] = (state.gestureUsage[msg.action] || 0) + 1;
      state.totalPredictions++;
      state.confidenceSum += msg.confidence;
      updateUsageBars();
      updateStats();
    } else {
      updateDetectionBadge(false);
    }

    // Sentence
    if (msg.sentence && Array.isArray(msg.sentence)) {
      updateSentenceDisplay(msg.sentence);
    }

  } else if (msg.type === 'sentence_reset') {
    updateSentenceDisplay([]);
  }
}

// ─── MediaPipe (Browser-side) ─────────────────────────────────────────────────
function startMediaPipe() {
  // Check MediaPipe libs loaded
  if (typeof Holistic === 'undefined' || typeof Camera === 'undefined') {
    showToast('error', '⚠️ MediaPipe not loaded — check internet connection', 6000);
    return;
  }

  // Destroy previous instance
  stopMediaPipe();

  // Build Holistic model — complexity 0 = LITE (fastest)
  state.mpHolistic = new Holistic({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });

  state.mpHolistic.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  state.mpHolistic.onResults(onMediaPipeResults);

  // Camera utility — handles video loop at native frame rate
  state.mpCamera = new Camera(dom.video, {
    onFrame: async () => {
      if (!state.isRunning || state.isPaused || !state.mpHolistic) return;
      await state.mpHolistic.send({ image: dom.video });
    },
    width: 640,
    height: 480,
  });

  // Show video
  dom.video.style.display = 'block';
  dom.annotated.style.display = 'none';

  state.mpCamera.start()
    .then(() => {
      dom.scanOverlay.classList.add('active');
      dom.scanLabel.textContent = 'SCANNING';
      showToast('success', '⚡ Real-time mode active (browser MediaPipe)', 3000);
    })
    .catch(err => {
      showToast('error', `📷 Camera error: ${err.message}`, 5000);
    });
}

function stopMediaPipe() {
  if (state.mpCamera) {
    try { state.mpCamera.stop(); } catch (_) {}
    state.mpCamera = null;
  }
  if (state.mpHolistic) {
    try { state.mpHolistic.close(); } catch (_) {}
    state.mpHolistic = null;
  }
}

// ─── MediaPipe Results Handler ────────────────────────────────────────────────
function onMediaPipeResults(results) {
  // ── Local FPS counter
  _localFpsCount++;
  const now = Date.now();
  const elapsed = (now - _lastFpsTime) / 1000;
  if (elapsed >= 1.0) {
    dom.fpsValue.textContent = Math.round(_localFpsCount / elapsed);
    _localFpsCount = 0;
    _lastFpsTime = now;
  }

  _localFrameCount++;
  dom.frameCount.textContent = _localFrameCount;

  // ── Draw landmarks on overlay canvas (local — zero network cost)
  const W = dom.video.videoWidth  || 640;
  const H = dom.video.videoHeight || 480;
  if (dom.canvas.width !== W) dom.canvas.width = W;
  if (dom.canvas.height !== H) dom.canvas.height = H;

  _canvasCtx.save();
  _canvasCtx.clearRect(0, 0, W, H);

  if (state.config.showLandmarks) {
    // Right hand — cyan
    if (results.rightHandLandmarks && typeof drawConnectors !== 'undefined') {
      drawConnectors(_canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
        { color: '#00f5d4', lineWidth: 2 });
      drawLandmarks(_canvasCtx, results.rightHandLandmarks,
        { color: '#00f5d4', fillColor: '#00f5d420', lineWidth: 1, radius: 4 });
    }
    // Left hand — magenta
    if (results.leftHandLandmarks && typeof drawConnectors !== 'undefined') {
      drawConnectors(_canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
        { color: '#ec4899', lineWidth: 2 });
      drawLandmarks(_canvasCtx, results.leftHandLandmarks,
        { color: '#ec4899', fillColor: '#ec489920', lineWidth: 1, radius: 4 });
    }
  }
  _canvasCtx.restore();

  // ── Landmark indicator
  const handsDetected = !!(results.leftHandLandmarks || results.rightHandLandmarks);
  if (handsDetected) {
    dom.landmarkIndicator.classList.add('detected');
    dom.landmarkText.textContent = 'Hands detected ✓';
  } else {
    dom.landmarkIndicator.classList.remove('detected');
    dom.landmarkText.textContent = 'No hands detected';
  }

  // ── Extract keypoints — 126 floats (63 lh + 63 rh)
  const lh = results.leftHandLandmarks
    ? results.leftHandLandmarks.flatMap(l => [l.x, l.y, l.z])
    : new Array(63).fill(0);
  const rh = results.rightHandLandmarks
    ? results.rightHandLandmarks.flatMap(l => [l.x, l.y, l.z])
    : new Array(63).fill(0);

  // ── Send ONLY keypoints to server (~500 bytes vs ~100KB JPEG)
  if (state.ws && state.ws.readyState === WebSocket.OPEN && !state.isPaused) {
    // Backpressure: skip frame if send buffer backing up
    if (state.ws.bufferedAmount < 30_000) {
      state.ws.send(JSON.stringify({ type: 'keypoints', data: [...lh, ...rh] }));
    }
  }
}

// ─── Stop everything ──────────────────────────────────────────────────────────
function stopAll() {
  state.isRunning = false;
  state.isPaused  = false;
  stopMediaPipe();
  if (state.ws) {
    try { state.ws.close(); } catch (_) {}
    state.ws = null;
  }
  _canvasCtx && _canvasCtx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
}

// ─── UI Builders ──────────────────────────────────────────────────────────────
function buildConfidenceBars() {
  dom.confidenceBars.innerHTML = '';
  GESTURES.forEach(g => {
    const el = document.createElement('div');
    el.className = 'conf-bar-item';
    el.innerHTML = `
      <div class="conf-bar-label">
        <span class="conf-bar-name">${g.emoji} ${g.label}</span>
        <span class="conf-bar-pct" id="confPct_${g.name}">0%</span>
      </div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill" id="confFill_${g.name}" style="width:0%"></div>
      </div>`;
    dom.confidenceBars.appendChild(el);
  });
}

function buildGestureGrid() {
  dom.gestureGrid.innerHTML = '';
  GESTURES.forEach(g => {
    const el = document.createElement('div');
    el.className = 'gesture-card';
    el.id = `gestureCard_${g.name}`;
    el.innerHTML = `
      <div class="gesture-card-emoji">${g.emoji}</div>
      <div class="gesture-card-name">${g.label}</div>
      <div class="gesture-card-desc">${g.description}</div>`;
    dom.gestureGrid.appendChild(el);
  });
}

function buildUsageBars() {
  dom.usageBars.innerHTML = '';
  GESTURES.forEach(g => {
    const el = document.createElement('div');
    el.className = 'usage-bar-item';
    el.innerHTML = `
      <span class="usage-emoji">${g.emoji}</span>
      <div class="usage-bar-content">
        <div class="usage-bar-header">
          <span class="usage-bar-name">${g.label}</span>
          <span class="usage-bar-count" id="usageCount_${g.name}">0</span>
        </div>
        <div class="usage-track">
          <div class="usage-fill" id="usageFill_${g.name}" style="width:0%"></div>
        </div>
      </div>`;
    dom.usageBars.appendChild(el);
  });
}

// ─── UI Updaters ──────────────────────────────────────────────────────────────
const _lastConfPct = {};
function updateConfidenceBars(probs) {
  GESTURES.forEach(g => {
    const pct = Math.round((probs[g.name] || 0) * 100);
    if (_lastConfPct[g.name] === pct) return;
    _lastConfPct[g.name] = pct;
    const fill  = $(`confFill_${g.name}`);
    const label = $(`confPct_${g.name}`);
    if (fill)  fill.style.width = `${pct}%`;
    if (fill)  fill.className = `conf-bar-fill${pct >= 70 ? ' high' : pct >= 40 ? ' medium' : ''}`;
    if (label) label.textContent = `${pct}%`;
  });
}

function updateUsageBars() {
  const maxVal = Math.max(1, ...Object.values(state.gestureUsage));
  GESTURES.forEach(g => {
    const count = state.gestureUsage[g.name] || 0;
    const pct = Math.round((count / maxVal) * 100);
    const fill = $(`usageFill_${g.name}`);
    const cnt  = $(`usageCount_${g.name}`);
    if (fill) fill.style.width = `${pct}%`;
    if (cnt)  cnt.textContent = count;
  });
}

function updatePredictionHero(action, confidence) {
  const g = GESTURES.find(x => x.name === action);
  if (!g) return;
  dom.predictionEmoji.textContent = g.emoji;
  dom.predictionLabel.textContent = g.label;
  const pct = Math.round(confidence * 100);
  dom.confValue.textContent = `${pct}%`;
  const circumference = 201;
  dom.ringFill.style.strokeDashoffset = circumference - (pct / 100) * circumference;
  dom.ringFill.className = `ring-fill${pct >= 80 ? ' high' : ''}`;
  document.querySelectorAll('.gesture-card').forEach(c => c.classList.remove('active'));
  const card = $(`gestureCard_${action}`);
  if (card) card.classList.add('active');
  dom.lastGesture.textContent = `${g.emoji} ${g.label}`;
}

function updateDetectionBadge(active) {
  dom.detectionBadge.textContent = active ? 'DETECTING' : 'Idle';
  dom.detectionBadge.className   = `card-badge${active ? ' active' : ''}`;
}

function updateSentenceDisplay(words) {
  if (!words || words.length === 0) {
    dom.sentenceDisplay.innerHTML = '<span class="placeholder-text">Start camera and make gestures to see translation here...</span>';
    dom.wordCount.textContent = '0 words';
    return;
  }
  dom.sentenceDisplay.innerHTML = words.map(w => {
    const g = GESTURES.find(x => x.name === w);
    return `<span class="word-chip">${g ? g.emoji + ' ' : ''}${w.toUpperCase()}</span>`;
  }).join('');
  dom.wordCount.textContent = `${words.length} word${words.length !== 1 ? 's' : ''}`;
}

function addHistoryItem(gesture, confidence, timestamp) {
  const g = GESTURES.find(x => x.name === gesture);
  if (!g) return;
  const empty = dom.historyList.querySelector('.empty-state');
  if (empty) empty.remove();
  const el = document.createElement('div');
  el.className = 'history-item';
  const t = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
  el.innerHTML = `
    <div class="history-gesture"><span class="history-emoji">${g.emoji}</span><span>${g.label}</span></div>
    <span class="history-conf">${Math.round(confidence * 100)}%</span>
    <span class="history-time">${t}</span>`;
  dom.historyList.insertBefore(el, dom.historyList.firstChild);
  while (dom.historyList.children.length > 20) dom.historyList.removeChild(dom.historyList.lastChild);
}

let _lastStatUpdate = 0;
function updateStats() {
  const now = Date.now();
  if (now - _lastStatUpdate < 1000) return;
  _lastStatUpdate = now;
  dom.totalPredictions.textContent = state.totalPredictions;
  const avg = state.totalPredictions > 0
    ? Math.round((state.confidenceSum / state.totalPredictions) * 100) : 0;
  dom.avgConfidence.textContent = `${avg}%`;
}

function updateUptime() {
  state.uptimeTimer = setInterval(() => {
    if (!state.isRunning || !state.startTime) { dom.uptime.textContent = '0s'; return; }
    const s = Math.floor((Date.now() - state.startTime) / 1000);
    if (s < 60) dom.uptime.textContent = `${s}s`;
    else if (s < 3600) dom.uptime.textContent = `${Math.floor(s/60)}m${s%60}s`;
    else dom.uptime.textContent = `${Math.floor(s/3600)}h${Math.floor((s%3600)/60)}m`;
  }, 1000);
}

function updateSequenceBar(count) {
  const pct = Math.round((count / 30) * 100);
  dom.sequenceBarFill.style.width  = `${pct}%`;
  dom.sequenceBarLabel.textContent = `Buffer: ${count}/30`;
}

// ─── TTS ──────────────────────────────────────────────────────────────────────
let _lastSpokenWord = '';
function speakText(text) {
  if (!state.speechSynth || text === _lastSpokenWord) return;
  _lastSpokenWord = text;
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 0.9; utt.pitch = 1; utt.volume = 1;
  state.speechSynth.speak(utt);
}

// ─── Events ───────────────────────────────────────────────────────────────────
function bindEvents() {
  dom.startBtn.addEventListener('click', () => {
    state.isRunning = true;
    state.isPaused  = false;
    state.startTime = Date.now();
    dom.startBtn.disabled = true;
    dom.pauseBtn.disabled = false;
    dom.stopBtn.disabled  = false;
    connectWS();
  });

  dom.pauseBtn.addEventListener('click', () => {
    state.isPaused = !state.isPaused;
    dom.pauseBtn.innerHTML = state.isPaused
      ? '<span class="btn-icon">▶</span><span class="btn-label">Resume</span>'
      : '<span class="btn-icon">⏸</span><span class="btn-label">Pause</span>';
    showToast('info', state.isPaused ? '⏸ Paused' : '▶ Resumed', 1500);
  });

  dom.stopBtn.addEventListener('click', () => {
    stopAll();
    dom.video.style.display = 'none';
    dom.startBtn.disabled = false;
    dom.pauseBtn.disabled = true;
    dom.stopBtn.disabled  = true;
    dom.scanOverlay.classList.remove('active');
    dom.scanLabel.textContent = 'READY';
    dom.fpsValue.textContent  = '--';
    dom.landmarkIndicator.classList.remove('detected');
    dom.landmarkText.textContent = 'No hands detected';
    dom.pauseBtn.innerHTML = '<span class="btn-icon">⏸</span><span class="btn-label">Pause</span>';
    setConnectionStatus('disconnected', 'Disconnected');
    showToast('info', '⏹ Detection stopped');
  });

  dom.flipBtn.addEventListener('click', () => {
    state.isFlipped = !state.isFlipped;
    dom.video.style.transform = state.isFlipped ? 'none' : 'scaleX(-1)';
    dom.canvas.style.transform = state.isFlipped ? 'none' : 'scaleX(-1)';
    showToast('info', `↔ Camera flipped`, 1500);
  });

  dom.ttsBtn.addEventListener('click', () => {
    state.ttsEnabled    = !state.ttsEnabled;
    state.config.autoTts = state.ttsEnabled;
    dom.ttsBtn.style.background = state.ttsEnabled ? 'rgba(0,245,212,0.15)' : '';
    showToast('info', state.ttsEnabled ? '🔊 TTS ON' : '🔇 TTS OFF', 1500);
  });

  dom.copyBtn.addEventListener('click', () => {
    const chips = dom.sentenceDisplay.querySelectorAll('.word-chip');
    const text = Array.from(chips).map(c => c.textContent.replace(/[^\w\s]/g, '').trim()).join(' ');
    if (text) navigator.clipboard.writeText(text).then(() => showToast('success', '📋 Copied!', 2000));
    else showToast('info', 'Nothing to copy', 1500);
  });

  dom.clearBtn.addEventListener('click', () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN)
      state.ws.send(JSON.stringify({ type: 'reset_sentence' }));
    updateSentenceDisplay([]);
    showToast('info', '🗑 Cleared', 1500);
  });

  dom.clearHistoryBtn.addEventListener('click', () => {
    dom.historyList.innerHTML = '<div class="empty-state">No detections yet</div>';
  });

  dom.settingsBtn.addEventListener('click', () => dom.settingsModal.classList.add('open'));
  dom.closeSettings.addEventListener('click', () => dom.settingsModal.classList.remove('open'));
  dom.settingsModal.addEventListener('click', e => {
    if (e.target === dom.settingsModal) dom.settingsModal.classList.remove('open');
  });

  dom.thresholdSlider.addEventListener('input', () => {
    dom.thresholdValue.textContent = parseFloat(dom.thresholdSlider.value).toFixed(2);
  });

  dom.saveSettings.addEventListener('click', () => {
    state.config.backendUrl    = dom.backendUrl.value.trim().replace(/\/$/, '');
    state.config.threshold     = parseFloat(dom.thresholdSlider.value);
    state.config.fpsLimit      = parseInt(dom.fpsLimit.value);
    state.config.showAnnotated = dom.showAnnotated.checked;
    state.config.autoTts       = dom.autoTts.checked;
    state.config.showLandmarks = dom.showLandmarks.checked;
    saveConfig();
    dom.settingsModal.classList.remove('open');
    showToast('success', '⚙️ Saved!', 2000);
  });

  dom.resetSettings.addEventListener('click', () => {
    state.config = { ...DEFAULT_CONFIG };
    applyConfigToForm();
    saveConfig();
    showToast('info', '↺ Reset', 1500);
  });

  dom.helpBtn.addEventListener('click', () =>
    showToast('info', '🤟 Point hands at camera — hold gesture for ~1s', 5000));

  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && !e.repeat) { e.preventDefault(); if (state.isRunning) dom.pauseBtn.click(); }
    if (e.code === 'KeyF') dom.flipBtn.click();
    if (e.code === 'KeyC' && e.ctrlKey) dom.copyBtn.click();
    if (e.code === 'Escape') dom.settingsModal.classList.remove('open');
  });
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function setConnectionStatus(type, text) {
  dom.statusPill.className = `status-pill ${type}`;
  if (dom.statusText) dom.statusText.textContent = text;
}

function showToast(type, message, duration = 3000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span class="toast-msg">${message}</span>`;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function generateSessionId() {
  return 'ss_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now();
}

document.addEventListener('DOMContentLoaded', init);
