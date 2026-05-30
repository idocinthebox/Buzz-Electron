/**
 * renderer/transcribe.js — Transcribe page logic for the unified Buzz app.
 *
 * Reuses the single WebSocket owned by renderer.js via window.BuzzBus:
 *   - BuzzBus.send(obj)           — send a command
 *   - BuzzBus._onEvent = fn       — claim transcription events (return true)
 *   - BuzzBus.onReady = fn        — called once when the socket opens
 *   - BuzzBus.appendLog / showToast
 *
 * Backend commands used: import_files, get_tasks, start_transcription,
 * cancel_transcription, get_segments, export_transcript, delete_task.
 */

'use strict';

(function () {
  const $ = (id) => document.getElementById(id);
  const bus = window.BuzzBus;

  // ── DOM ──
  const modelTypeEl = $('tx-model-type');
  const modelSizeEl = $('tx-model-size');
  const languageEl  = $('tx-language');
  const taskEl      = $('tx-task');
  const wordTimings = $('tx-word-timings');

  const btnImport = $('tx-btn-import');
  const btnStart  = $('tx-btn-start');
  const btnCancel = $('tx-btn-cancel');

  const statusDot  = $('tx-status-dot');
  const statusText = $('tx-status-text');
  const progWrap   = $('tx-progress-wrapper');
  const progFill   = $('tx-progress-fill');
  const progLabel  = $('tx-progress-label');

  const emptyState = $('tx-empty-state');
  const table      = $('tx-table');
  const tableBody  = $('tx-table-body');

  const viewer        = $('tx-viewer');
  const viewerTitle   = $('tx-viewer-title');
  const segmentsBody  = $('tx-segments-body');
  const exportFormat  = $('tx-export-format');
  const btnExport     = $('tx-btn-export');
  const btnCloseViewer= $('tx-btn-close-viewer');

  // ── State ──
  let tasks = [];                 // task dicts from backend
  let modelsData = [];            // reuse model list shape from renamer
  let runningId = null;           // currently transcribing task id
  let queue = [];                 // ids waiting to be transcribed (sequential)
  let openTaskId = null;          // viewer's task

  const WHISPER_CPP = 'Whisper.cpp';

  // ── Helpers ──
  function setStatus(label, cls) {
    statusText.textContent = label;
    statusDot.className = `status-dot ${cls || ''}`;
  }
  function baseName(p) { return p ? p.replace(/\\/g, '/').split('/').pop() : ''; }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function fmtTs(ms) {
    const total = Math.floor(ms / 1000);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const msPart = String(Math.floor(ms % 1000)).padStart(3, '0');
    const hms = (h > 0 ? `${h}:` : '') +
      `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${hms}.${msPart}`;
  }

  const STATUS_LABEL = {
    queued: 'Queued', in_progress: 'Working', completed: 'Completed',
    failed: 'Failed', canceled: 'Canceled',
  };

  // ── Model dropdowns (mirror renamer's populateModelDropdowns) ──
  function populateModels(models) {
    modelsData = models;
    modelTypeEl.innerHTML = '';
    models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.type; opt.textContent = m.type;
      modelTypeEl.appendChild(opt);
    });
    const cpp = [...modelTypeEl.options].find(o => o.value === WHISPER_CPP);
    if (cpp) modelTypeEl.value = WHISPER_CPP;
    updateSizes();
  }
  function updateSizes() {
    const model = modelsData.find(m => m.type === modelTypeEl.value);
    modelSizeEl.innerHTML = '';
    if (model && model.sizes.length) {
      model.sizes.forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.size;
        opt.textContent = s.downloaded ? `${s.label} ✓` : s.label;
        modelSizeEl.appendChild(opt);
      });
      const base = [...modelSizeEl.options].find(o => o.value === 'base');
      if (base) modelSizeEl.value = 'base';
    }
  }
  modelTypeEl.addEventListener('change', updateSizes);

  function populateLanguages(langs) {
    languageEl.innerHTML = '<option value="">Detect automatically</option>';
    langs.forEach((l) => {
      const opt = document.createElement('option');
      opt.value = l.code; opt.textContent = `${l.name} (${l.code})`;
      if (l.code === 'en') opt.selected = true;
      languageEl.appendChild(opt);
    });
  }

  function currentConfig() {
    return {
      model_type: modelTypeEl.value,
      model_size: modelSizeEl.value,
      language: languageEl.value,
      task: taskEl.value,
      word_level_timings: wordTimings.checked,
    };
  }

  // ── Task table ──
  function renderTable() {
    tableBody.innerHTML = '';
    if (!tasks.length) {
      table.style.display = 'none';
      emptyState.style.display = 'flex';
      return;
    }
    emptyState.style.display = 'none';
    table.style.display = 'table';
    tasks.forEach((t) => tableBody.appendChild(renderRow(t)));
    updateButtons();
  }

  function renderRow(t) {
    const tr = document.createElement('tr');
    tr.dataset.id = t.id;
    const pct = Math.round((t.progress || 0) * 100);
    const statusCls = `tx-badge tx-${t.status}`;
    tr.innerHTML =
      `<td><span class="${statusCls}">${STATUS_LABEL[t.status] || t.status}</span></td>` +
      `<td class="tx-file" title="${esc(t.file || '')}">${esc(t.name || baseName(t.file))}</td>` +
      `<td>${esc(t.model_type || '')}${t.model_size ? ' / ' + esc(t.model_size) : ''}</td>` +
      `<td><div class="mini-progress"><div class="mini-progress-fill" style="width:${t.status === 'completed' ? 100 : pct}%"></div></div></td>` +
      `<td class="tx-row-actions">` +
        (t.status === 'completed' ? `<button class="row-act" data-act="open">Open</button>` : '') +
        `<button class="row-act row-act-del" data-act="delete">✕</button>` +
      `</td>`;

    tr.querySelectorAll('.row-act').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (btn.dataset.act === 'open') openViewer(t.id);
        else if (btn.dataset.act === 'delete') bus.send({ cmd: 'delete_task', id: t.id });
      });
    });
    tr.addEventListener('dblclick', () => {
      if (t.status === 'completed') openViewer(t.id);
    });
    return tr;
  }

  function updateRow(t) {
    const idx = tasks.findIndex(x => x.id === t.id);
    if (idx >= 0) tasks[idx] = { ...tasks[idx], ...t };
    const tr = tableBody.querySelector(`tr[data-id="${t.id}"]`);
    if (tr) tr.replaceWith(renderRow(tasks[idx]));
    else renderTable();
  }

  function updateButtons() {
    const hasQueued = tasks.some(t => t.status === 'queued');
    btnStart.disabled = !hasQueued || runningId !== null;
    btnCancel.disabled = runningId === null;
  }

  // ── Sequential transcription runner ──
  function startQueued() {
    queue = tasks.filter(t => t.status === 'queued').map(t => t.id);
    runNext();
  }
  function runNext() {
    if (runningId !== null) return;
    const next = queue.shift();
    if (next === undefined) {
      setStatus('Ready', 'connected');
      progWrap.style.display = 'none';
      updateButtons();
      return;
    }
    runningId = next;
    progWrap.style.display = 'flex';
    progFill.style.width = '0%';
    progLabel.textContent = '';
    setStatus('Transcribing…', 'running');
    bus.send({ cmd: 'start_transcription', id: next });
    updateButtons();
  }

  // ── Viewer ──
  function openViewer(id) {
    openTaskId = id;
    const t = tasks.find(x => x.id === id);
    viewerTitle.textContent = t ? (t.name || baseName(t.file)) : 'Transcript';
    segmentsBody.innerHTML = '<tr><td colspan="3" class="muted">Loading…</td></tr>';
    viewer.style.display = 'flex';
    bus.send({ cmd: 'get_segments', id });
  }
  function renderSegments(segments) {
    segmentsBody.innerHTML = '';
    if (!segments.length) {
      segmentsBody.innerHTML = '<tr><td colspan="3" class="muted">No segments.</td></tr>';
      return;
    }
    segments.forEach((s) => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td class="seg-time">${fmtTs(s.start)}</td>` +
        `<td class="seg-time">${fmtTs(s.end)}</td>` +
        `<td class="seg-text" contenteditable="true" data-seg="${s.id}">${esc(s.text)}</td>`;
      const cell = tr.querySelector('.seg-text');
      cell.addEventListener('blur', () => {
        bus.send({ cmd: 'update_segment', id: openTaskId,
                   segment_id: s.id, text: cell.textContent });
      });
      segmentsBody.appendChild(tr);
    });
  }
  btnCloseViewer.addEventListener('click', () => {
    viewer.style.display = 'none';
    openTaskId = null;
  });
  btnExport.addEventListener('click', async () => {
    if (openTaskId == null) return;
    const fmt = exportFormat.value;
    const t = tasks.find(x => x.id === openTaskId);
    const suggested = (t ? (t.name || baseName(t.file)).replace(/\.[^.]+$/, '') : 'transcript') + '.' + fmt;
    const outPath = await window.electronAPI.saveFile(suggested);
    if (!outPath) return;
    bus.send({ cmd: 'export_transcript', id: openTaskId, format: fmt, output_path: outPath });
  });

  // ── Buttons ──
  btnImport.addEventListener('click', async () => {
    const files = await window.electronAPI.openFiles();
    if (!files || !files.length) return;
    bus.send({ cmd: 'import_files', files, config: currentConfig() });
  });
  btnStart.addEventListener('click', startQueued);
  btnCancel.addEventListener('click', () => {
    if (runningId !== null) bus.send({ cmd: 'cancel_transcription', id: runningId });
  });

  // ── Event handling: claim transcription events from the shared socket ──
  bus._onEvent = function (msg) {
    switch (msg.event) {
      case 'files_imported':
        tasks = mergeTasks(msg.tasks);
        renderTable();
        bus.appendLog(`Imported ${msg.tasks.length} file(s) for transcription.`, 'info');
        return true;
      case 'tasks':
        tasks = msg.tasks;
        renderTable();
        return true;
      case 'task_started':
        runningId = idNum(msg.id);
        updateRow({ id: runningId, status: 'in_progress' });
        return true;
      case 'task_completed':
        updateRow({ id: idNum(msg.id), status: 'completed', progress: 1 });
        bus.appendLog(`Transcription complete (${msg.segment_count} segments).`, 'info');
        runningId = null;
        runNext();
        return true;
      case 'task_canceled':
        updateRow({ id: idNum(msg.id), status: 'canceled' });
        runningId = null;
        runNext();
        return true;
      case 'task_error':
        updateRow({ id: idNum(msg.id), status: 'failed', error: msg.message });
        bus.showToast(`Transcription failed: ${msg.message}`, 'error');
        runningId = null;
        runNext();
        return true;
      case 'segments':
        if (idEq(msg.id, openTaskId)) renderSegments(msg.segments);
        return true;
      case 'segment_updated':
        return true;
      case 'export_done':
        bus.showToast(`Exported to ${msg.path}`, 'success');
        bus.appendLog(`Exported transcript: ${msg.path}`, 'info');
        return true;
      case 'task_deleted':
        tasks = tasks.filter(t => !idEq(msg.id, t.id));
        renderTable();
        return true;
      default:
        return false; // not ours — let the renamer handle it
    }
  };

  // ids: backend sends string uuids; tasks store the same string. Keep loose.
  function idNum(x) { return x; }
  function idEq(a, b) { return String(a) === String(b); }
  function mergeTasks(incoming) {
    const map = new Map(tasks.map(t => [String(t.id), t]));
    incoming.forEach(t => map.set(String(t.id), t));
    return [...map.values()];
  }

  // ── On socket ready: load models, languages, and existing tasks ──
  bus.onReady = function () {
    bus.send({ cmd: 'get_tasks' });
    // models/languages are already requested by renderer.js; mirror them in too
  };

  // Mirror the renamer's model/language data into our dropdowns by also
  // listening for those events (renderer.js handles them for its own page;
  // we piggyback here without claiming them).
  const origOnEvent = bus._onEvent;
  bus._onEvent = function (msg) {
    if (msg.event === 'models') { populateModels(msg.models); /* fallthrough */ }
    if (msg.event === 'languages') { populateLanguages(msg.languages); }
    return origOnEvent(msg); // returns false for models/languages → renamer also updates
  };

  // initial empty render
  renderTable();
})();
