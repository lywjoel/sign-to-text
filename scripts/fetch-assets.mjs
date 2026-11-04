// Copies the MediaPipe WASM runtime out of node_modules and downloads the
// landmark models into public/, so the app runs without third-party CDNs.
// Safe to re-run: existing files are skipped. If a download fails the app
// falls back to loading that asset from Google's model storage at runtime.
import { copyFile, mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const quiet = process.argv.includes('--quiet');
const log = (...args) => {
  if (!quiet) console.log(...args);
};

const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';
const MODELS = {
  'holistic_landmarker.task': `${MODEL_BASE}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`,
  'hand_landmarker.task': `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
  'pose_landmarker_lite.task': `${MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`,
  'face_landmarker.task': `${MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
};

const WASM_FILES = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
];

async function exists(path) {
  try {
    return (await stat(path)).size > 0;
  } catch {
    return false;
  }
}

async function copyWasm() {
  const src = join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
  const dest = join(root, 'public/mediapipe/wasm');
  await mkdir(dest, { recursive: true });
  for (const file of WASM_FILES) {
    const to = join(dest, file);
    if (await exists(to)) continue;
    await copyFile(join(src, file), to);
    log(`copied ${file}`);
  }
}

async function download(name, url) {
  const dest = join(root, 'public/models', name);
  if (await exists(dest)) return;
  await mkdir(dirname(dest), { recursive: true });
  console.log(`downloading ${name} ...`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const tmp = `${dest}.part`;
  await writeFile(tmp, new Uint8Array(await res.arrayBuffer()));
  await rename(tmp, dest);
  log(`saved public/models/${name}`);
}

await copyWasm();
for (const [name, url] of Object.entries(MODELS)) {
  try {
    await download(name, url);
  } catch (err) {
    await rm(join(root, 'public/models', `${name}.part`), { force: true });
    console.warn(`could not download ${name} (${err.message}); the app will load it from ${MODEL_BASE} instead.`);
  }
}
