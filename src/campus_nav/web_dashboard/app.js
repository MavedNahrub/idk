/**
 * SAYE Autonomous Buggy — Live Lane Perception & Navigation Dashboard
 * Real-Time Video Streaming, Telemetry Gauges, Waveform Graphs & Mission Control
 */

(function () {
  'use strict';

  // --- State ---
  const state = {
    currentFeed: 'lane', // 'lane', 'birdseye', 'split', 'raw'
    connected: false,
    speed: 0.0,
    speedKmh: 0.0,
    steerAngleDeg: 0.0,
    steerAngleRad: 0.0,
    orientErrDeg: 0.0,
    edgeDist: 2.4,
    edgeWheel: 'rr',
    edgeClearance: 2.4,
    mode: 'FORWARD',
    isAutoCorrecting: false,
    gateCurrent: 0,
    gateTotal: 1367,
    gatePercent: 0,
    checkpointTitle: 'Awaiting Mission Request...',
    checkpointDist: '-- m',
    checkpointEta: '-- s',
    checkpointTurn: 'straight',
    history: {
      timestamps: [],
      speed: [],
      steer: [],
      orient: [],
    },
    maxHistoryPoints: 120, // 30s at ~4Hz record
  };

  // --- DOM Elements ---
  const el = {
    primaryStreamImg: document.getElementById('primaryStreamImg'),
    singleStreamWrapper: document.getElementById('singleStreamWrapper'),
    splitStreamWrapper: document.getElementById('splitStreamWrapper'),
    splitStreamLeft: document.getElementById('splitStreamLeft'),
    splitStreamRight: document.getElementById('splitStreamRight'),
    streamLoader: document.getElementById('streamLoader'),
    hudModeLabel: document.getElementById('hudModeLabel'),
    hudSpeedReadout: document.getElementById('hudSpeedReadout'),
    hudSteerReadout: document.getElementById('hudSteerReadout'),
    hudRadiusValue: document.getElementById('hudRadiusValue'),
    hudRecoveryTag: document.getElementById('hudRecoveryTag'),
    systemStateChip: document.getElementById('systemStateChip'),
    systemStateText: document.getElementById('systemStateText'),
    clockDisplay: document.getElementById('clockDisplay'),
    telemetryCanvas: document.getElementById('telemetryCanvas'),
    // Gauges
    valSpeed: document.getElementById('valSpeed'),
    valSpeedKmh: document.getElementById('valSpeedKmh'),
    speedBar: document.getElementById('speedBar'),
    valSteer: document.getElementById('valSteer'),
    valSteerDir: document.getElementById('valSteerDir'),
    steerBar: document.getElementById('steerBar'),
    valOrient: document.getElementById('valOrient'),
    orientFill: document.getElementById('orientFill'),
    orientNeedle: document.getElementById('orientNeedle'),
    // Radar
    valFL: document.getElementById('valFL'),
    valFR: document.getElementById('valFR'),
    valRL: document.getElementById('valRL'),
    valRR: document.getElementById('valRR'),
    podFL: document.getElementById('podFL'),
    podFR: document.getElementById('podFR'),
    podRL: document.getElementById('podRL'),
    podRR: document.getElementById('podRR'),
    criticalWheelName: document.getElementById('criticalWheelName'),
    criticalWheelClearance: document.getElementById('criticalWheelClearance'),
    edgeSafetyBadge: document.getElementById('edgeSafetyBadge'),
    // Nav
    gateReadout: document.getElementById('gateReadout'),
    routeProgressFill: document.getElementById('routeProgressFill'),
    checkpointTitle: document.getElementById('checkpointTitle'),
    checkpointDist: document.getElementById('checkpointDist'),
    checkpointEta: document.getElementById('checkpointEta'),
    turnGlyph: document.getElementById('turnGlyph'),
    // Controls
    selectStartStop: document.getElementById('selectStartStop'),
    selectGoalStop: document.getElementById('selectGoalStop'),
    btnLaunchMission: document.getElementById('btnLaunchMission'),
    btnEstop: document.getElementById('btnEstop'),
    btnSnapshot: document.getElementById('btnSnapshot'),
    btnFullscreen: document.getElementById('btnFullscreen'),
    footerConnection: document.getElementById('footerConnection'),
  };

  const canvasCtx = el.telemetryCanvas ? el.telemetryCanvas.getContext('2d') : null;

  // --- Clock Updater ---
  function updateClock() {
    const now = new Date();
    el.clockDisplay.textContent = now.toTimeString().split(' ')[0];
  }
  setInterval(updateClock, 1000);
  updateClock();

  // --- Feed Switching ---
  function setFeed(feedName) {
    state.currentFeed = feedName;
    document.querySelectorAll('.tab-pill').forEach(pill => {
      const isSelected = pill.getAttribute('data-feed') === feedName;
      pill.classList.toggle('active', isSelected);
      pill.setAttribute('aria-selected', isSelected);
    });

    if (feedName === 'split') {
      el.singleStreamWrapper.style.display = 'none';
      el.splitStreamWrapper.style.display = 'grid';
      el.splitStreamLeft.src = '/stream/lane?t=' + Date.now();
      el.splitStreamRight.src = '/stream/birdseye?t=' + Date.now();
    } else {
      el.splitStreamWrapper.style.display = 'none';
      el.singleStreamWrapper.style.display = 'block';
      el.primaryStreamImg.src = `/stream/${feedName}?t=` + Date.now();
    }
  }

  document.querySelectorAll('.tab-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const feed = pill.getAttribute('data-feed');
      setFeed(feed);
    });
  });

  // --- Snapshot Capture ---
  el.btnSnapshot.addEventListener('click', () => {
    const activeImg = (state.currentFeed === 'split') ? el.splitStreamLeft : el.primaryStreamImg;
    if (!activeImg) return;
    try {
      const c = document.createElement('canvas');
      c.width = activeImg.naturalWidth || 640;
      c.height = activeImg.naturalHeight || 360;
      const ctx = c.getContext('2d');
      ctx.drawImage(activeImg, 0, 0);
      const a = document.createElement('a');
      a.download = `saye_lane_capture_${Date.now()}.png`;
      a.href = c.toDataURL('image/png');
      a.click();
    } catch (e) {
      console.warn('Direct snapshot download fallback:', e);
      window.open(activeImg.src, '_blank');
    }
  });

  // --- Fullscreen Toggle ---
  el.btnFullscreen.addEventListener('click', () => {
    const stage = document.getElementById('streamStage');
    if (!document.fullscreenElement) {
      stage.requestFullscreen().catch(err => console.warn(err));
    } else {
      document.exitFullscreen();
    }
  });

  // --- Mission Presets ---
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const start = btn.getAttribute('data-start');
      const goal = btn.getAttribute('data-goal');
      if (el.selectStartStop) el.selectStartStop.value = start;
      if (el.selectGoalStop) el.selectGoalStop.value = goal;
      dispatchMission(start, goal);
    });
  });

  // --- Launch Mission Action ---
  if (el.btnLaunchMission) {
    el.btnLaunchMission.addEventListener('click', () => {
      const start = el.selectStartStop.value;
      const goal = el.selectGoalStop.value;
      dispatchMission(start, goal);
    });
  }

  // --- Emergency Stop Action ---
  let isEStopped = false;
  if (el.btnEstop) {
    el.btnEstop.addEventListener('click', () => {
      isEStopped = !isEStopped;
      fetch('/api/estop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stop: isEStopped }),
      }).catch(err => console.warn(err));

      el.btnEstop.textContent = isEStopped ? 'RESUME' : 'ESTOP';
      el.btnEstop.style.background = isEStopped ? '#10b981' : 'rgba(239, 68, 68, 0.2)';
    });
  }

  function dispatchMission(start, goal) {
    const btn = el.btnLaunchMission;
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span>Launching Route...</span>';
    }

    fetch('/api/mission', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start, goal }),
    })
      .then(res => res.json())
      .then(data => {
        console.log('Mission dispatched successfully:', data);
        if (btn) {
          btn.innerHTML = '<span>&check; Route Dispatched!</span>';
          setTimeout(() => {
            btn.disabled = false;
            btn.innerHTML = '<span class="icon-rocket"></span><span>Dispatch Autonomous Mission</span>';
          }, 2000);
        }
      })
      .catch(err => {
        console.error('Mission dispatch error:', err);
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<span>Failed &bull; Retry</span>';
        }
      });
  }

  // --- Telemetry Updates ---
  function applyTelemetry(data) {
    if (!data) return;

    state.speed = Number(data.speed || 0.0);
    state.speedKmh = Number(data.speed_kmh || (state.speed * 3.6));
    state.steerAngleDeg = Number(data.steer_angle_deg || 0.0);
    state.steerAngleRad = Number(data.steer_angle_rad || 0.0);
    state.orientErrDeg = Number(data.orientation_err_deg || 0.0);
    state.edgeDist = Number(data.edge_dist || 0.0);
    state.edgeWheel = String(data.edge_wheel || 'center');
    state.edgeClearance = Number(data.edge_clearance || Math.max(0.1, 3.0 - state.edgeDist));
    state.mode = String(data.mode || 'FORWARD');
    state.isAutoCorrecting = Boolean(data.is_autocorrecting || state.mode.includes('RECOVERY'));
    state.gateCurrent = Number(data.gate_current || 0);
    state.gateTotal = Number(data.gate_total || 1367);
    state.gatePercent = Number(data.gate_percent || 0);

    // Populate stops if received once
    if (data.stops && Array.isArray(data.stops) && el.selectStartStop.options.length <= 8) {
      populateStops(data.stops);
    }

    // 1. Digital Readouts
    el.valSpeed.textContent = state.speed.toFixed(2);
    el.valSpeedKmh.textContent = state.speedKmh.toFixed(1) + ' km/h';
    el.valSteer.textContent = (state.steerAngleDeg >= 0 ? '+' : '') + state.steerAngleDeg.toFixed(1) + '°';
    el.valSteerDir.textContent = Math.abs(state.steerAngleDeg) < 1.0 ? 'NEUTRAL' : (state.steerAngleDeg > 0 ? 'LEFT' : 'RIGHT');
    el.valOrient.textContent = (state.orientErrDeg >= 0 ? '+' : '') + state.orientErrDeg.toFixed(1) + '°';

    // 2. HUD Chips on Video Stream
    el.hudSpeedReadout.textContent = state.speed.toFixed(2) + ' m/s';
    el.hudSteerReadout.textContent = state.steerAngleDeg.toFixed(1) + '°';
    if (data.dynamic_radius) {
      el.hudRadiusValue.textContent = (data.dynamic_radius > 50) ? 'STRAIGHT' : (data.dynamic_radius.toFixed(1) + 'm');
    }

    // 3. System State Badges
    if (state.isAutoCorrecting) {
      el.systemStateChip.className = 'status-chip recovery';
      el.systemStateText.textContent = 'AUTO-CORRECTION REVERSE';
      el.hudRecoveryTag.style.display = 'inline-flex';
      el.hudModeLabel.textContent = 'RECOVERY REVERSE';
    } else if (Math.abs(state.steerAngleDeg) > 12.0) {
      el.systemStateChip.className = 'status-chip active';
      el.systemStateText.textContent = 'DYNAMIC CORNERING';
      el.hudRecoveryTag.style.display = 'none';
      el.hudModeLabel.textContent = 'DYNAMIC CURVE';
    } else {
      el.systemStateChip.className = 'status-chip active';
      el.systemStateText.textContent = 'AUTONOMOUS LANE CRUISE';
      el.hudRecoveryTag.style.display = 'none';
      el.hudModeLabel.textContent = 'LANE LOCK: KEEP-LEFT';
    }

    // 4. Circular SVG Gauge Bars
    // Speed Bar: max 3.0 m/s -> circumference = 2 * PI * 48 = 301.6
    const circumference = 301.6;
    const speedRatio = Math.max(0, Math.min(1.0, state.speed / 3.0));
    el.speedBar.style.strokeDashoffset = circumference * (1.0 - speedRatio);

    // Steer Bar: -28.6° to +28.6°
    const steerRatio = Math.max(0, Math.min(1.0, (state.steerAngleDeg + 28.6) / 57.2));
    el.steerBar.style.strokeDashoffset = circumference * (1.0 - steerRatio);

    // 5. Bipolar Lane Orientation Meter
    // Needle position: -90° -> 0%, 0° -> 50%, +90° -> 100%
    const clampedErr = Math.max(-90, Math.min(90, state.orientErrDeg));
    const needlePct = 50 + (clampedErr / 90) * 50;
    el.orientNeedle.style.left = needlePct.toFixed(1) + '%';
    if (clampedErr >= 0) {
      el.orientFill.style.left = '50%';
      el.orientFill.style.width = ((clampedErr / 90) * 50).toFixed(1) + '%';
      el.orientFill.style.background = (clampedErr > 50) ? 'var(--rose-danger)' : ((clampedErr > 30) ? 'var(--amber-warn)' : 'var(--cyan-primary)');
    } else {
      const w = (Math.abs(clampedErr) / 90) * 50;
      el.orientFill.style.left = (50 - w).toFixed(1) + '%';
      el.orientFill.style.width = w.toFixed(1) + '%';
      el.orientFill.style.background = (Math.abs(clampedErr) > 50) ? 'var(--rose-danger)' : ((Math.abs(clampedErr) > 30) ? 'var(--amber-warn)' : 'var(--cyan-primary)');
    }

    // 6. 4-Wheel Clearance Radar Pods
    const curbLimit = 3.0; // half-width
    const dEdge = Math.max(0.1, curbLimit - state.edgeDist);
    el.criticalWheelName.textContent = state.edgeWheel.toUpperCase();
    el.criticalWheelClearance.textContent = dEdge.toFixed(2) + ' m';

    // Update wheel pods
    const pods = [el.podFL, el.podFR, el.podRL, el.podRR];
    const vals = [el.valFL, el.valFR, el.valRL, el.valRR];
    const names = ['fl', 'fr', 'rl', 'rr'];

    for (let i = 0; i < 4; i++) {
      let wClearance = dEdge;
      if (names[i] !== state.edgeWheel) {
        wClearance = Math.min(2.8, dEdge + 0.7);
      }
      vals[i].textContent = wClearance.toFixed(2) + 'm';

      if (wClearance < 0.6) {
        vals[i].className = 'pod-val danger';
      } else if (wClearance < 0.9) {
        vals[i].className = 'pod-val warning';
      } else {
        vals[i].className = 'pod-val';
      }
    }

    if (dEdge < 0.6) {
      el.edgeSafetyBadge.textContent = 'CRITICAL CURB WARNING';
      el.edgeSafetyBadge.className = 'status-indicator-badge danger';
      el.criticalWheelClearance.className = 'stat-val danger';
    } else if (dEdge < 0.9) {
      el.edgeSafetyBadge.textContent = 'EDGE BIAS ACTIVE';
      el.edgeSafetyBadge.className = 'status-indicator-badge warning';
      el.criticalWheelClearance.className = 'stat-val warning';
    } else {
      el.edgeSafetyBadge.textContent = 'SAFE CLEARANCE (>2.0m)';
      el.edgeSafetyBadge.className = 'status-indicator-badge safe';
      el.criticalWheelClearance.className = 'stat-val safe';
    }

    // 7. Route & Gate Navigation
    el.gateReadout.textContent = `Gate ${state.gateCurrent} / ${state.gateTotal} (${state.gatePercent.toFixed(0)}%)`;
    el.routeProgressFill.style.width = Math.min(100, Math.max(0, state.gatePercent)) + '%';

    if (data.checkpoint_str) {
      el.checkpointTitle.textContent = data.checkpoint_str;
      if (data.checkpoint_str.includes('LEFT')) {
        el.turnGlyph.innerHTML = '&larr;';
      } else if (data.checkpoint_str.includes('RIGHT')) {
        el.turnGlyph.innerHTML = '&rarr;';
      } else if (data.checkpoint_str.includes('ARRIVED')) {
        el.turnGlyph.innerHTML = '&check;';
      } else {
        el.turnGlyph.innerHTML = '&uarr;';
      }
    }
    if (data.checkpoint_dist) el.checkpointDist.textContent = `${data.checkpoint_dist}m ahead`;
    if (data.checkpoint_eta) el.checkpointEta.textContent = `ETA: ${data.checkpoint_eta}s`;

    // 8. Push to Rolling History for Chart
    recordHistoryPoint(state.speed, state.steerAngleDeg, state.orientErrDeg);
  }

  function populateStops(stops) {
    if (!el.selectStartStop || !el.selectGoalStop) return;
    el.selectStartStop.innerHTML = '';
    el.selectGoalStop.innerHTML = '';
    stops.forEach(s => {
      const opt1 = document.createElement('option');
      opt1.value = s;
      opt1.textContent = s;
      if (s === 'Ravi Back') opt1.selected = true;
      el.selectStartStop.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = s;
      opt2.textContent = s;
      if (s === 'Chemistry Block') opt2.selected = true;
      el.selectGoalStop.appendChild(opt2);
    });
  }

  // --- Rolling Telemetry Chart Rendering ---
  function recordHistoryPoint(speed, steer, orient) {
    const h = state.history;
    h.speed.push(speed);
    h.steer.push(steer);
    h.orient.push(orient);
    if (h.speed.length > state.maxHistoryPoints) {
      h.speed.shift();
      h.steer.shift();
      h.orient.shift();
    }
  }

  function renderWaveformChart() {
    if (!canvasCtx || !el.telemetryCanvas) return;
    const canvas = el.telemetryCanvas;
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvasCtx;

    ctx.clearRect(0, 0, w, h);

    // Background Grid
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    for (let x = 60; x < w; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Zero / Center Baseline
    const midY = h / 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    const pts = state.history.speed.length;
    if (pts < 2) return;

    const dx = w / (state.maxHistoryPoints - 1);
    const startX = w - (pts - 1) * dx;

    // Helper to draw series
    function drawSeries(series, color, scale, offsetY) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < pts; i++) {
        const x = startX + i * dx;
        const y = offsetY - series[i] * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 1. Speed Curve (0 to 3.0 m/s, maps from bottom h-10 to top 20)
    // Scale: (h - 40) / 3.0
    const speedScale = (h - 40) / 3.0;
    drawSeries(state.history.speed, '#00f2fe', speedScale, h - 20);

    // 2. Steer Angle Curve (-30° to +30°, centered on midY)
    // Scale: (h/2 - 20) / 30
    const steerScale = (midY - 20) / 30.0;
    drawSeries(state.history.steer, '#10b981', steerScale, midY);

    // 3. Orientation Error Curve (-90° to +90°, centered on midY)
    const orientScale = (midY - 20) / 90.0;
    drawSeries(state.history.orient, '#f59e0b', orientScale, midY);
  }

  setInterval(renderWaveformChart, 100);

  // --- Telemetry Polling & SSE Transport ---
  function connectSSE() {
    if (!!window.EventSource) {
      const source = new EventSource('/api/stream_telemetry');
      source.onmessage = function (event) {
        try {
          const data = JSON.parse(event.data);
          state.connected = true;
          el.footerConnection.textContent = 'Telemetry Connected via SSE';
          el.footerConnection.style.color = 'var(--emerald-safe)';
          applyTelemetry(data);
        } catch (e) {
          console.warn('SSE parse error:', e);
        }
      };
      source.onerror = function () {
        console.warn('SSE disconnected, falling back to 20Hz polling...');
        source.close();
        startPolling();
      };
    } else {
      startPolling();
    }
  }

  function startPolling() {
    setInterval(() => {
      fetch('/api/telemetry')
        .then(res => res.json())
        .then(data => {
          state.connected = true;
          el.footerConnection.textContent = 'Telemetry Polling: 20Hz OK';
          el.footerConnection.style.color = 'var(--emerald-safe)';
          applyTelemetry(data);
        })
        .catch(() => {
          state.connected = false;
          el.footerConnection.textContent = 'Awaiting ROS Bridge connection...';
          el.footerConnection.style.color = 'var(--amber-warn)';
        });
    }, 100);
  }

  // Initialize
  connectSSE();
  console.log('SAYE Autonomous Dashboard Online.');
})();
