import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  HolisticLandmarker,
  PoseLandmarker,
  type Category,
  type Classifications,
} from '@mediapipe/tasks-vision';
import { POSE, type LM, type RawHand, type RawObs } from './skeleton';

type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

export type Backend = 'holistic' | 'separate';

const MEDIAPIPE_VERSION = '1.2.0';
const CDN_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';
const REMOTE_MODELS = {
  holistic: `${MODEL_BASE}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`,
  hand: `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
  pose: `${MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`,
  face: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
};
const LOCAL_MODELS = {
  holistic: 'models/holistic_landmarker.task',
  hand: 'models/hand_landmarker.task',
  pose: 'models/pose_landmarker_lite.task',
  face: 'models/face_landmarker.task',
};

function localUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, location.href).href;
}

/** True if a bundled asset is present (dev servers answer missing files with index.html). */
async function hasLocal(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { method: 'HEAD' });
    return res.ok && !(res.headers.get('content-type') ?? '').includes('text/html');
  } catch {
    return false;
  }
}

async function modelUrl(name: keyof typeof LOCAL_MODELS): Promise<string> {
  const local = localUrl(LOCAL_MODELS[name]);
  return (await hasLocal(local)) ? local : REMOTE_MODELS[name];
}

let filesetPromise: Promise<WasmFileset> | null = null;
function fileset(): Promise<WasmFileset> {
  filesetPromise ??= (async () => {
    const local = localUrl('mediapipe/wasm');
    const base = (await hasLocal(`${local}/vision_wasm_internal.js`)) ? local : CDN_WASM;
    return FilesetResolver.forVisionTasks(base);
  })();
  return filesetPromise;
}

interface VideoTask {
  close(): void;
  detectForVideo(frame: HTMLCanvasElement, ts: number): unknown;
}

async function withGpuFallback<T extends VideoTask>(
  make: (delegate: 'GPU' | 'CPU') => Promise<T>,
): Promise<{ task: T; delegate: 'GPU' | 'CPU' }> {
  let lastError: unknown;
  for (const delegate of ['GPU', 'CPU'] as const) {
    let task: T | null = null;
    try {
      task = await make(delegate);
      // Some graphs only fail when they first run (e.g. an op the GPU backend lacks), so run one frame now.
      warm(task);
      return { task, delegate };
    } catch (err) {
      console.warn(`Could not run on the ${delegate}`, err);
      task?.close();
      lastError = err;
    }
  }
  throw lastError;
}

let warmUpCanvas: HTMLCanvasElement | null = null;
function blankFrame(): HTMLCanvasElement {
  if (!warmUpCanvas) {
    warmUpCanvas = document.createElement('canvas');
    warmUpCanvas.width = 64;
    warmUpCanvas.height = 64;
    warmUpCanvas.getContext('2d')!.fillRect(0, 0, 64, 64);
  }
  return warmUpCanvas;
}

let warmUpTs = 0;
function warm(task: VideoTask): void {
  warmUpTs = Math.max(warmUpTs + 1, Math.round(performance.now()));
  task.detectForVideo(blankFrame(), warmUpTs);
}

function blendMap(c: Classifications | undefined): Map<string, number> | null {
  if (!c) return null;
  return new Map(c.categories.map((cat: Category) => [cat.categoryName, cat.score]));
}

interface Detected {
  landmarks: LM[];
  world: LM[] | null;
  /** The signer's side according to the model, if known. */
  side: 'left' | 'right' | null;
}

/**
 * Decides which detected hand is the signer's left and right. When the body
 * pose is known each hand goes to the nearer pose wrist, which works whichever
 * way a model labels hands. Otherwise the model's own labels are used.
 */
function assignHands(hands: Detected[], pose: LM[] | null): { left: RawHand | null; right: RawHand | null } {
  const raw = (h: Detected): RawHand => ({ landmarks: h.landmarks, world: h.world });
  const lw = pose?.[POSE.LEFT_WRIST];
  const rw = pose?.[POSE.RIGHT_WRIST];
  if (lw && rw && hands.length) {
    const d = (h: Detected, w: LM) => Math.hypot(h.landmarks[0].x - w.x, h.landmarks[0].y - w.y);
    if (hands.length >= 2) {
      const [a, b] = hands;
      return d(a, lw) + d(b, rw) <= d(a, rw) + d(b, lw)
        ? { left: raw(a), right: raw(b) }
        : { left: raw(b), right: raw(a) };
    }
    const h = hands[0];
    // Only trust the pose if it clearly prefers one wrist; otherwise use the label.
    const dl = d(h, lw);
    const dr = d(h, rw);
    if (Math.abs(dl - dr) > 0.05 || !h.side) return dl <= dr ? { left: raw(h), right: null } : { left: null, right: raw(h) };
  }
  let left: RawHand | null = null;
  let right: RawHand | null = null;
  for (const h of hands) {
    if (h.side === 'left' && !left) left = raw(h);
    else if (h.side === 'right' && !right) right = raw(h);
    else if (!left) left = raw(h);
    else if (!right) right = raw(h);
  }
  return { left, right };
}

export class Tracker {
  private lastTs = -1;

  private constructor(
    readonly backend: Backend,
    readonly delegate: 'GPU' | 'CPU',
    private readonly holistic: HolisticLandmarker | null,
    private readonly hands: HandLandmarker | null,
    private readonly pose: PoseLandmarker | null,
    private readonly face: FaceLandmarker | null,
  ) {}

  static async create(backend: Backend, status: (msg: string) => void = () => {}): Promise<Tracker> {
    status('Loading the vision runtime');
    const vision = await fileset();

    if (backend === 'holistic') {
      status('Loading the combined model');
      const url = await modelUrl('holistic');
      const { task, delegate } = await withGpuFallback((d) =>
        HolisticLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: url, delegate: d },
          runningMode: 'VIDEO',
          outputFaceBlendshapes: true,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minHandLandmarksConfidence: 0.5,
        }),
      );
      return new Tracker('holistic', delegate, task, null, null, null);
    }

    status('Loading the hand model');
    const handUrl = await modelUrl('hand');
    const hands = await withGpuFallback((d) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: handUrl, delegate: d },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
    );
    status('Loading the body model');
    const poseUrl = await modelUrl('pose');
    const pose = await withGpuFallback((d) =>
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: poseUrl, delegate: d },
        runningMode: 'VIDEO',
        numPoses: 1,
      }),
    );
    status('Loading the face model');
    const faceUrl = await modelUrl('face');
    const face = await withGpuFallback((d) =>
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: faceUrl, delegate: d },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
      }),
    );
    const delegate = hands.delegate === 'GPU' || pose.delegate === 'GPU' || face.delegate === 'GPU' ? 'GPU' : 'CPU';
    return new Tracker('separate', delegate, null, hands.task, pose.task, face.task);
  }

  detect(video: HTMLVideoElement, now: number): RawObs {
    const ts = Math.max(Math.round(now), this.lastTs + 1, warmUpTs + 1);
    this.lastTs = ts;
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (this.holistic) {
      const r = this.holistic.detectForVideo(video, ts);
      const pose = r.poseLandmarks[0] ?? null;
      const detected: Detected[] = [];
      if (r.leftHandLandmarks[0]) {
        detected.push({ landmarks: r.leftHandLandmarks[0], world: r.leftHandWorldLandmarks[0] ?? null, side: 'left' });
      }
      if (r.rightHandLandmarks[0]) {
        detected.push({ landmarks: r.rightHandLandmarks[0], world: r.rightHandWorldLandmarks[0] ?? null, side: 'right' });
      }
      const { left, right } = assignHands(detected, pose);
      return {
        t: now,
        width,
        height,
        left,
        right,
        pose,
        face: r.faceLandmarks[0] ?? null,
        blendshapes: blendMap(r.faceBlendshapes[0]),
      };
    }

    const h = this.hands!.detectForVideo(video, ts);
    const p = this.pose!.detectForVideo(video, ts);
    const f = this.face!.detectForVideo(video, ts);
    const pose = p.landmarks[0] ?? null;
    // In a camera image (not mirrored) the model's Left/Right label is the signer's own hand;
    // this was checked against photographs of right hands. With a body pose the wrists decide instead.
    const detected: Detected[] = h.landmarks.map((lm, i) => {
      const label = h.handedness[i]?.[0]?.categoryName;
      return {
        landmarks: lm,
        world: h.worldLandmarks[i] ?? null,
        side: label === 'Left' ? 'left' : label === 'Right' ? 'right' : null,
      };
    });
    const { left, right } = assignHands(detected, pose);
    return {
      t: now,
      width,
      height,
      left,
      right,
      pose,
      face: f.faceLandmarks[0] ?? null,
      blendshapes: blendMap(f.faceBlendshapes[0]),
    };
  }

  close(): void {
    this.holistic?.close();
    this.hands?.close();
    this.pose?.close();
    this.face?.close();
  }
}
