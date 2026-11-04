import '@fontsource-variable/atkinson-hyperlegible-next';
import './style.css';
import { bodyFrame, extractFeatures } from './features';
import { Library, newId, normaliseLabel, SampleStore, type SampleKind } from './library';
import { BLEND_LABELS, drawObservation, drawSkeleton } from './overlay';
import { Recognizer, type Candidate, type Mode, type Snapshot } from './recognizer';
import { Recorder } from './recorder';
import { loadSettings, recognizerConfig, saveSettings } from './settings';
import { BLEND_KEYS, DropoutFiller, toSkeleton, type Skeleton } from './skeleton';
import { Tracker, type Backend } from './tracker';
import { Transcriber } from './transcriber';
import { VOCAB } from './vocab';
import { ALL_LETTERS } from './fingerspell';
import type { SegState } from './segmenter';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const settings = loadSettings();
const transcriber = new Transcriber();
let library = new Library(null);
const recognizer = new Recognizer(
  transcriber,
  () => library.templates,
  () => library.taughtLetters(),
  recognizerConfig(settings),
);
const recorder = new Recorder();
recorder.holdMs = settings.holdMs;
const filler = new DropoutFiller();

let tracker: Tracker | null = null;
let trackerJob: Promise<void> | null = null;
let stream: MediaStream | null = null;
let running = false;
let lastVideoTime = -1;
const frameTimes: number[] = [];

const video = $<HTMLVideoElement>('video');
const overlay = $<HTMLCanvasElement>('overlay');

// ---------------------------------------------------------------------------
// Status

let statusBase = 'Camera off';
function setStatus(text: string): void {
  statusBase = text;
  $('status').textContent = text;
}

function fps(): number {
  const now = performance.now();
  while (frameTimes.length && frameTimes[0] < now - 1000) frameTimes.shift();
  return frameTimes.length;
}

function trackerStatus(): string {
  if (!tracker) return statusBase;
  const model = tracker.backend === 'holistic' ? 'Combined model' : 'Separate models';
  return `${model} on ${tracker.delegate === 'GPU' ? 'the GPU' : 'the CPU'}, ${fps()} frames a second`;
}

// ---------------------------------------------------------------------------
// Camera and tracking

async function startCamera(): Promise<void> {
  const err = $('start-error');
  err.textContent = '';
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    err.textContent = 'The camera only works on a secure page. Open this app from https:// or from localhost.';
    return;
  }
  const btn = $<HTMLButtonElement>('start');
  btn.disabled = true;
  setStatus('Asking for camera access');
  try {
    await openStream();
  } catch (e) {
    btn.disabled = false;
    setStatus('Camera off');
    err.textContent = cameraError(e);
    return;
  }
  $('stage-empty').hidden = true;
  void populateCameras();
  try {
    await ensureTracker();
  } catch {
    return;
  }
  if (!running) {
    running = true;
    scheduleFrame();
  }
}

function cameraError(e: unknown): string {
  const name = e instanceof DOMException ? e.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Camera access is blocked. Allow the camera for this site in your browser settings, then press Start camera again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found. Connect one and try again.';
  if (name === 'NotReadableError') return 'The camera is being used by another app. Close that app and try again.';
  return `The camera could not start (${e instanceof Error ? e.message : String(e)}).`;
}

async function openStream(): Promise<void> {
  stream?.getTracks().forEach((t) => t.stop());
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const wanted: MediaStreamConstraints = {
    audio: false,
    video: settings.deviceId ? { ...size, deviceId: { exact: settings.deviceId } } : { ...size, facingMode: 'user' },
  };
  try {
    stream = await navigator.mediaDevices.getUserMedia(wanted);
  } catch (e) {
    if (!settings.deviceId) throw e;
    // The saved camera is gone; fall back to the default one.
    settings.deviceId = null;
    saveSettings(settings);
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...size, facingMode: 'user' } });
  }
  stream.getVideoTracks()[0]?.addEventListener('ended', () => {
    running = false;
    $('stage-empty').hidden = false;
    $<HTMLButtonElement>('start').disabled = false;
    $('start-error').textContent = 'The camera stopped. Press Start camera to reconnect.';
    setStatus('Camera off');
  });
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
  $('stage').style.setProperty('--ar', `${video.videoWidth / video.videoHeight}`);
}

async function populateCameras(): Promise<void> {
  const select = $<HTMLSelectElement>('set-camera');
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = stream?.getVideoTracks()[0]?.getSettings().deviceId ?? settings.deviceId ?? '';
  select.replaceChildren(
    ...devices.map((d, i) => {
      const o = document.createElement('option');
      o.value = d.deviceId;
      o.textContent = d.label || `Camera ${i + 1}`;
      o.selected = d.deviceId === current;
      return o;
    }),
  );
}

async function ensureTracker(): Promise<void> {
  while (trackerJob) await trackerJob;
  if (tracker && tracker.backend === settings.backend) return;
  trackerJob = (async () => {
    tracker?.close();
    tracker = null;
    const describe = (b: Backend) => (b === 'holistic' ? 'the combined model' : 'the separate models');
    try {
      tracker = await Tracker.create(settings.backend, setStatus);
    } catch (e) {
      console.error(e);
      const other: Backend = settings.backend === 'holistic' ? 'separate' : 'holistic';
      setStatus(`${describe(settings.backend)} would not load; trying ${describe(other)}`);
      try {
        tracker = await Tracker.create(other, setStatus);
        settings.backend = other;
        saveSettings(settings);
        syncSettingsForm();
      } catch (e2) {
        console.error(e2);
        setStatus('The tracking models could not load. Check your connection and reload the page.');
        throw e2;
      }
    }
    filler.reset();
    recognizer.reset();
  })();
  try {
    await trackerJob;
  } finally {
    trackerJob = null;
  }
}

function scheduleFrame(): void {
  if (!running) return;
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

function onFrame(): void {
  if (!running) return;
  try {
    if (tracker && !trackerJob && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      processFrame();
      frameErrors = 0;
    }
  } catch (e) {
    console.error(e);
    if (++frameErrors === 5) void recoverTracking(e);
  }
  scheduleFrame();
}

let frameErrors = 0;
let switchedBackend = false;

/** If tracking keeps failing, try the other tracking option once before giving up. */
async function recoverTracking(e: unknown): Promise<void> {
  const reason = e instanceof Error ? e.message.split('\n')[0] : String(e);
  tracker?.close();
  tracker = null;
  if (switchedBackend) {
    setStatus(`Tracking stopped working (${reason}). Reload the page to try again.`);
    return;
  }
  switchedBackend = true;
  settings.backend = settings.backend === 'holistic' ? 'separate' : 'holistic';
  saveSettings(settings);
  syncSettingsForm();
  setStatus('Tracking failed; trying the other tracking option');
  try {
    await ensureTracker();
    frameErrors = 0;
  } catch {
    // ensureTracker has already explained the failure in the status line.
  }
}

let lastUiAt = 0;
let lastSnap: Snapshot | null = null;

function processFrame(): void {
  const now = performance.now();
  const raw = tracker!.detect(video, now);
  frameTimes.push(now);
  const skel = filler.apply(toSkeleton(raw, settings.dominant));
  const feat = extractFeatures(skel);

  const recording = recorder.phase === 'countdown' || recorder.phase === 'capturing';
  let seg: SegState;
  if (recording) {
    recorder.feed(skel, feat);
    seg = recorder.segState;
    if (recorder.phase === 'countdown') $('countdown').textContent = String(Math.ceil(recorder.countdownLeft / 1000));
  } else {
    lastSnap = recognizer.push(skel, feat);
    seg = lastSnap.seg;
  }

  const bf = bodyFrame(skel);
  const restY = recognizer.segmenter.opts.restY;
  drawObservation(overlay, raw, {
    dominant: settings.dominant,
    showFace: settings.showFace,
    mirror: settings.mirrorView,
    restLineY: bf.source === 'guess' ? null : bf.oy + restY * bf.scale,
  });
  renderRail(seg);

  if (now - lastUiAt > 90) {
    lastUiAt = now;
    if (!recording) renderNow(lastSnap, now);
    renderFace(skel);
    $('status').textContent = trackerStatus();
    if (settings.showDebug) renderDebug(seg, skel, bf.source);
  }
}

// ---------------------------------------------------------------------------
// Live display

function renderRail(seg: SegState): void {
  const rail = $('rail');
  rail.dataset.phase = seg.phase;
  const on = seg.phase === 'still' ? 'hold' : seg.phase;
  for (const st of rail.querySelectorAll<HTMLElement>('.station')) st.classList.toggle('on', st.dataset.phase === on);
  const ring = rail.querySelector<SVGCircleElement>('.ring')!;
  ring.style.strokeDashoffset = String(100 - Math.round(seg.holdProgress * 100));
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

function describe(c: Candidate): string {
  const what = c.kind === 'letter' ? 'Letter' : 'Sign';
  const from = c.source === 'built-in' ? 'built-in alphabet' : 'your examples';
  return `${what} from ${from}, ${pct(c.score)}`;
}

function renderCandidates(list: Candidate[]): void {
  $('cands').replaceChildren(
    ...list.slice(0, 3).map((c) => {
      const li = document.createElement('li');
      li.className = c.kind;
      const name = document.createElement('span');
      name.textContent = c.label;
      const bar = document.createElement('span');
      bar.className = 'bar';
      const fill = document.createElement('span');
      fill.style.width = pct(Math.min(1, c.score));
      bar.append(fill);
      const p = document.createElement('span');
      p.className = 'pct';
      p.textContent = pct(c.score);
      li.append(name, bar, p);
      return li;
    }),
  );
}

function setNow(label: string, cls: string, meta: string): void {
  const el = $('now-label');
  el.textContent = label;
  el.hidden = !label;
  const size = label.length > 10 ? ' longer' : label.length > 5 ? ' long' : '';
  el.className = `now-label ${cls}${size}`;
  $('now-meta').textContent = meta;
}

function renderNow(snap: Snapshot | null, now: number): void {
  if (!snap) return;
  if (snap.guess) {
    const g = snap.guess;
    setNow(g.label, `${g.kind}${snap.guessAccepted ? '' : ' tentative'}`, snap.guessAccepted ? describe(g) : `Unsure (${pct(g.score)})`);
    renderCandidates(snap.guessCandidates);
    return;
  }
  const d = snap.last;
  if (d && now - d.t < 4000) {
    if (d.accepted) setNow(d.accepted.label, d.accepted.kind, describe(d.accepted));
    else setNow('', '', d.via === 'motion' ? 'That movement did not match a taught sign' : 'That handshape did not match a letter');
    renderCandidates(d.candidates);
    return;
  }
  const phase = snap.seg.phase;
  setNow('', '', phase === 'rest' ? (snap.seg.active ? 'Hands up' : 'Waiting for hands') : 'Watching');
  renderCandidates([]);
}

function renderFace(skel: Skeleton): void {
  const line = $('face-line');
  if (!settings.useFace) {
    line.textContent = '';
    return;
  }
  if (!skel.face || !skel.blend) {
    line.textContent = 'Face not in view';
    return;
  }
  const active = BLEND_KEYS.filter((_, i) => skel.blend![i] > 0.4).map((k) => BLEND_LABELS[k]);
  line.replaceChildren('Face: ');
  if (!active.length) line.append('neutral');
  active.forEach((a, i) => {
    const s = document.createElement('strong');
    s.textContent = a;
    line.append(...(i ? [', ', s] : [s]));
  });
}

function renderDebug(seg: SegState, skel: Skeleton, frameSource: string): void {
  $('debug').textContent = [
    `state ${seg.phase}   motion ${seg.motion.toFixed(2)}   speed ${seg.speed.toFixed(2)} sw/s   handshape change ${seg.shapeRate.toFixed(2)}/s`,
    `hands: dominant ${skel.dom ? (skel.dom.world ? 'yes (3D)' : 'yes') : 'no'}, other ${skel.non ? 'yes' : 'no'}   face ${skel.face ? 'yes' : 'no'}   body ${skel.pose ? 'yes' : 'no'}   body frame from ${frameSource}`,
    `taught examples ${library.samples.length}   threshold ${recognizer.config.signThreshold.toFixed(2)} (moving) / ${recognizer.config.staticThreshold.toFixed(2)} (held)`,
  ].join('\n');
}

// ---------------------------------------------------------------------------
// Transcript

function renderTranscript(): void {
  const box = $('transcript');
  if (!transcriber.items.length) {
    const p = document.createElement('p');
    p.className = 'placeholder';
    p.textContent = 'Recognised signs appear here. Fingerspelled words are underlined; faded words may still change.';
    box.replaceChildren(p);
    return;
  }
  box.replaceChildren(
    ...transcriber.lines().map((words) => {
      const p = document.createElement('p');
      for (const w of words) {
        const s = document.createElement('span');
        s.className = `w ${w.kind}${w.provisional ? ' provisional' : ''}`;
        s.textContent = w.text;
        p.append(s);
      }
      if (!words.length) p.append('\u00a0');
      return p;
    }),
  );
  box.scrollTop = box.scrollHeight;
}

async function copyText(): Promise<void> {
  const text = transcriber.text();
  const btn = $<HTMLButtonElement>('copy');
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copied';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => (btn.textContent = 'Copy'), 1500);
}

function initTranscript(): void {
  transcriber.onChange(renderTranscript);
  $('undo').addEventListener('click', () => transcriber.undo());
  $('space').addEventListener('click', () => transcriber.space());
  $('newline').addEventListener('click', () => transcriber.newline());
  $('copy').addEventListener('click', () => void copyText());
  $('clear').addEventListener('click', () => {
    if (transcriber.items.length && confirm('Clear the whole transcript?')) transcriber.clear();
  });
  document.addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (t.closest('input, select, textarea, [contenteditable="true"]')) return;
    const onButton = !!t.closest('button, [role="tab"], [role="radio"], label');
    if (e.key === 'Backspace') {
      transcriber.undo();
      e.preventDefault();
    } else if (!onButton && e.key === ' ') {
      transcriber.space();
      e.preventDefault();
    } else if (!onButton && e.key === 'Enter') {
      transcriber.newline();
      e.preventDefault();
    }
  });
  renderTranscript();
}

// ---------------------------------------------------------------------------
// Mode and tabs

function initMode(): void {
  const buttons = document.querySelectorAll<HTMLButtonElement>('.mode button');
  const sync = () => buttons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === settings.mode)));
  buttons.forEach((b) =>
    b.addEventListener('click', () => {
      settings.mode = b.dataset.mode as Mode;
      saveSettings(settings);
      recognizer.configure(recognizerConfig(settings));
      sync();
    }),
  );
  sync();
}

function selectTab(id: string): void {
  for (const tab of document.querySelectorAll<HTMLButtonElement>('[role="tab"]')) {
    const on = tab.id === id;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    $(tab.getAttribute('aria-controls')!).hidden = !on;
  }
}

function initTabs(): void {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTab(tab.id));
    tab.addEventListener('keydown', (e) => {
      const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!step) return;
      const next = tabs[(i + step + tabs.length) % tabs.length];
      selectTab(next.id);
      next.focus();
    });
  });
  selectTab('tab-teach');
}

// ---------------------------------------------------------------------------
// Teach

let stopReplay: (() => void) | null = null;

function replay(canvas: HTMLCanvasElement, frames: Skeleton[]): () => void {
  let raf = 0;
  const t0 = frames[0].t;
  const span = frames[frames.length - 1].t - t0;
  const start = performance.now();
  const tick = () => {
    const elapsed = (performance.now() - start) % (span + 700);
    const t = t0 + Math.min(elapsed, span);
    let i = 0;
    while (i < frames.length - 1 && frames[i + 1].t <= t) i++;
    drawSkeleton(canvas, frames[i], settings.mirrorView);
    raf = requestAnimationFrame(tick);
  };
  tick();
  return () => cancelAnimationFrame(raf);
}

function teachMessage(text: string, isError = false): void {
  const el = $('teach-msg');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

function selectedKind(): SampleKind {
  return (document.querySelector<HTMLInputElement>('input[name="kind"]:checked')?.value ?? 'sign') as SampleKind;
}

function renderTeach(): void {
  const phase = recorder.phase;
  const recording = phase === 'countdown' || phase === 'capturing';
  $('teach-form').hidden = phase !== 'idle';
  $('teach-help').hidden = phase === 'review';
  $('capture').hidden = !recording;
  $('review').hidden = phase !== 'review';
  $('countdown').hidden = phase !== 'countdown';
  $('rec-badge').hidden = phase !== 'capturing';
  $('stop').textContent = phase === 'countdown' ? 'Cancel' : 'Stop recording';

  if (phase === 'countdown') {
    $('capture-text').textContent = `Get into the starting position for ${recorder.label}.`;
    teachMessage('');
  } else if (phase === 'capturing') {
    $('capture-text').textContent =
      recorder.kind === 'letter'
        ? `Hold ${recorder.label} still.`
        : `Sign ${recorder.label} now, then hold the last position still.`;
  } else if (phase === 'idle' && recorder.error) {
    teachMessage(recorder.error, true);
  }

  stopReplay?.();
  stopReplay = null;
  if (phase === 'review' && recorder.result) {
    const r = recorder.result;
    stopReplay = replay($<HTMLCanvasElement>('review-canvas'), r.frames);
    $('review-title').textContent = r.kind === 'letter' ? `Letter ${r.label}` : r.label;
    const secs = (r.durationMs / 1000).toFixed(1);
    $('review-info').textContent = r.dynamic
      ? `Movement, ${secs} seconds${r.parts > 1 ? `, in ${r.parts} parts` : ''}.`
      : 'Held still, no movement.';
    const similar = recognizer.compareRecording(r.feats, r.dynamic, r.kind);
    const existing = library.summary().find((s) => s.label === r.label && s.kind === r.kind)?.count ?? 0;
    const clash = similar.find((c) => c.label !== r.label && c.score >= 0.5);
    const el = $('review-similar');
    el.classList.toggle('warn', !!clash);
    if (clash) {
      el.textContent = `This looks a lot like ${clash.label} (${pct(clash.score)}). If both are kept they may be confused; check they differ in handshape, place or movement.`;
    } else if (existing) {
      const self = similar.find((c) => c.label === r.label);
      el.textContent = self
        ? `${pct(self.score)} like your other ${existing === 1 ? 'example' : `${existing} examples`} of ${r.label}.`
        : `You have ${existing} other ${existing === 1 ? 'example' : 'examples'} of ${r.label}.`;
    } else {
      el.textContent = `This will be the first example of ${r.label}.`;
    }
  }
}

function startRecording(kind: SampleKind, label: string): void {
  if (!running) {
    teachMessage('Start the camera first.', true);
    return;
  }
  recorder.holdMs = settings.holdMs;
  recorder.start(kind, label, performance.now());
}

function initTeach(): void {
  const list = $('vocab');
  list.replaceChildren(
    ...[...ALL_LETTERS, ...VOCAB].map((v) => {
      const o = document.createElement('option');
      o.value = v;
      return o;
    }),
  );
  recorder.onChange = renderTeach;

  const input = $<HTMLInputElement>('teach-label');
  input.addEventListener('input', () => {
    // A single letter is most likely a fingerspelling letter.
    const v = normaliseLabel(input.value);
    if (/^[A-Z]$/.test(v)) {
      const letter = document.querySelector<HTMLInputElement>('input[name="kind"][value="letter"]')!;
      letter.checked = true;
    }
  });

  $('teach-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const label = normaliseLabel(input.value);
    const kind = selectedKind();
    if (!label) return;
    if (kind === 'letter' && !/^[A-Z0-9]$/.test(label)) {
      teachMessage('A letter is a single character, such as A. Choose Word sign for anything longer.', true);
      return;
    }
    input.value = label;
    startRecording(kind, label);
  });
  $('stop').addEventListener('click', () => {
    if (recorder.phase === 'countdown') recorder.cancel();
    else recorder.stop();
  });
  $('keep').addEventListener('click', async () => {
    const r = recorder.result;
    if (!r) return;
    try {
      await library.add({
        id: newId(),
        label: r.label,
        kind: r.kind,
        createdAt: Date.now(),
        frames: r.frames,
        dynamic: r.dynamic,
        parts: r.parts,
      });
      const count = library.summary().find((s) => s.label === r.label && s.kind === r.kind)?.count ?? 1;
      recorder.cancel();
      teachMessage(`Saved. ${r.label} has ${count} ${count === 1 ? 'example' : 'examples'}.`);
    } catch (e) {
      teachMessage(`The example could not be saved (${e instanceof Error ? e.message : String(e)}).`, true);
    }
    recognizer.reset();
  });
  $('again').addEventListener('click', () => {
    const r = recorder.result;
    if (r) startRecording(r.kind, r.label);
  });
  $('discard').addEventListener('click', () => {
    recorder.cancel();
    teachMessage('Discarded.');
    recognizer.reset();
  });
  renderTeach();
}

// ---------------------------------------------------------------------------
// Library

function libMessage(text: string): void {
  $('lib-msg').textContent = text;
}

function renderLibrary(): void {
  const items = library.summary();
  $('lib-empty').hidden = items.length > 0;
  $('lib-list').replaceChildren(
    ...items.map((item) => {
      const li = document.createElement('li');
      li.className = `lib-item ${item.kind}`;
      const name = document.createElement('span');
      name.className = 'lib-name';
      name.textContent = item.label;
      const detail = document.createElement('span');
      detail.className = 'lib-detail';
      const kind = item.kind === 'letter' ? 'Letter' : 'Word sign';
      detail.textContent = `${kind}, ${item.dynamic ? 'moving' : 'held'}, ${item.count} ${item.count === 1 ? 'example' : 'examples'}`;

      const actions = document.createElement('span');
      actions.className = 'lib-actions';
      const add = document.createElement('button');
      add.type = 'button';
      add.textContent = 'Add example';
      add.addEventListener('click', () => {
        selectTab('tab-teach');
        $<HTMLInputElement>('teach-label').value = item.label;
        document.querySelector<HTMLInputElement>(`input[name="kind"][value="${item.kind}"]`)!.checked = true;
        $('record').focus();
      });
      const play = document.createElement('button');
      play.type = 'button';
      play.textContent = 'Play';
      let stop: (() => void) | null = null;
      play.addEventListener('click', () => {
        const existing = li.querySelector('canvas');
        if (existing) {
          stop?.();
          existing.remove();
          play.textContent = 'Play';
          return;
        }
        const sample = library.samples.filter((s) => s.label === item.label && s.kind === item.kind).at(-1);
        if (!sample || !sample.frames.length) return;
        const canvas = document.createElement('canvas');
        li.append(canvas);
        stop = replay(canvas, sample.frames);
        play.textContent = 'Hide';
      });
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'danger';
      del.textContent = 'Delete';
      del.addEventListener('click', async () => {
        if (!confirm(`Delete all ${item.count} ${item.count === 1 ? 'example' : 'examples'} of ${item.label}?`)) return;
        stop?.();
        await library.removeLabel(item.label, item.kind);
        libMessage(`Deleted ${item.label}.`);
      });
      actions.append(add, play, del);
      li.append(name, actions, detail);
      return li;
    }),
  );
}

function initLibraryPanel(): void {
  $('export').addEventListener('click', () => {
    if (!library.samples.length) {
      libMessage('There is nothing to export yet.');
      return;
    }
    const blob = new Blob([library.exportJson()], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `sgsl-signs-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    libMessage(`Exported ${library.samples.length} examples.`);
  });
  const file = $<HTMLInputElement>('import');
  file.addEventListener('change', async () => {
    const f = file.files?.[0];
    file.value = '';
    if (!f) return;
    try {
      const added = await library.importJson(await f.text());
      libMessage(added ? `Imported ${added} new ${added === 1 ? 'example' : 'examples'}.` : 'Every example in that file is already here.');
    } catch (e) {
      libMessage(e instanceof SyntaxError ? 'That file is not valid JSON.' : e instanceof Error ? e.message : String(e));
    }
  });
}

// ---------------------------------------------------------------------------
// Settings

function syncSettingsForm(): void {
  const radio = (name: string, value: string) => {
    const el = document.querySelector<HTMLInputElement>(`input[name="${name}"][value="${value}"]`);
    if (el) el.checked = true;
  };
  radio('dominant', settings.dominant);
  radio('backend', settings.backend);
  $<HTMLInputElement>('set-sensitivity').value = String(settings.sensitivity);
  $<HTMLInputElement>('set-hold').value = String(settings.holdMs);
  $<HTMLInputElement>('set-face').checked = settings.useFace;
  $<HTMLInputElement>('set-body').checked = settings.useBody;
  $<HTMLInputElement>('set-alphabet').checked = settings.builtinAlphabet;
  $<HTMLInputElement>('set-showface').checked = settings.showFace;
  $<HTMLInputElement>('set-mirror').checked = settings.mirrorView;
  $<HTMLInputElement>('set-debug').checked = settings.showDebug;
  const s = settings.sensitivity;
  $('out-sensitivity').textContent = s < 0.85 ? 'Strict' : s > 1.15 ? 'Lenient' : 'Balanced';
  $('out-hold').textContent = `${(settings.holdMs / 1000).toFixed(2)} s`;
  $('stage').classList.toggle('mirrored', settings.mirrorView);
  $('debug').hidden = !settings.showDebug;
}

function initSettingsForm(): void {
  const form = $<HTMLFormElement>('settings-form');
  form.addEventListener('submit', (e) => e.preventDefault());
  form.addEventListener('change', async (e) => {
    const t = e.target as HTMLInputElement | HTMLSelectElement;
    const before = { ...settings };
    if (t.name === 'dominant') settings.dominant = t.value as 'right' | 'left';
    if (t.name === 'backend') settings.backend = t.value as Backend;
    if (t.id === 'set-camera') settings.deviceId = t.value || null;
    settings.useFace = $<HTMLInputElement>('set-face').checked;
    settings.useBody = $<HTMLInputElement>('set-body').checked;
    settings.builtinAlphabet = $<HTMLInputElement>('set-alphabet').checked;
    settings.showFace = $<HTMLInputElement>('set-showface').checked;
    settings.mirrorView = $<HTMLInputElement>('set-mirror').checked;
    settings.showDebug = $<HTMLInputElement>('set-debug').checked;
    saveSettings(settings);
    syncSettingsForm();
    recognizer.configure(recognizerConfig(settings));

    if (settings.dominant !== before.dominant) {
      filler.reset();
      recognizer.reset();
    }
    if (settings.deviceId !== before.deviceId && running) {
      try {
        await openStream();
      } catch (err) {
        setStatus(cameraError(err));
      }
    }
    if (settings.backend !== before.backend && running) {
      try {
        await ensureTracker();
      } catch {
        // Status already explains the failure.
      }
    }
  });
  form.addEventListener('input', (e) => {
    const t = e.target as HTMLInputElement;
    if (t.id === 'set-sensitivity') settings.sensitivity = Number(t.value);
    else if (t.id === 'set-hold') settings.holdMs = Number(t.value);
    else return;
    saveSettings(settings);
    syncSettingsForm();
    recognizer.configure(recognizerConfig(settings));
  });
  syncSettingsForm();
}

// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  initMode();
  initTabs();
  initSettingsForm();
  initTranscript();
  initTeach();
  initLibraryPanel();
  $('start').addEventListener('click', () => void startCamera());

  try {
    library = new Library(await SampleStore.open());
  } catch (e) {
    console.warn('IndexedDB unavailable', e);
    libMessage('This browser is not letting the app save examples, so they will be lost when the page closes. Export them to keep them.');
  }
  library.onChange(renderLibrary);
  try {
    await library.load();
  } catch (e) {
    console.error(e);
    libMessage('Saved examples could not be read.');
  }
  renderLibrary();
}

void boot();
