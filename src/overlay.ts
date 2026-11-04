import { FaceLandmarker } from '@mediapipe/tasks-vision';
import { BLEND_KEYS, FACE, HAND_CONNECTIONS, POSE, type Dominant, type LM, type RawObs, type Skeleton } from './skeleton';

/** Overlay colours follow Singapore's MRT line colours. */
export const COLORS = {
  dominant: '#D42E12', // North–South line red
  other: '#009645', // East–West line green
  face: '#FA9E0D', // Circle line orange
  body: '#005EC4', // Downtown line blue
  spelled: '#9900AA', // North East line purple
};

const FACE_CONTOURS = [
  FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
  FaceLandmarker.FACE_LANDMARKS_LIPS,
  FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
  FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
  FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
  FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
];

const FACE_ANCHORS = [
  FACE.FOREHEAD,
  FACE.NOSE_TIP,
  FACE.CHIN,
  FACE.RIGHT_CHEEK,
  FACE.LEFT_CHEEK,
] as const;

const BODY_LINKS: [number, number][] = [
  [POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER],
  [POSE.LEFT_SHOULDER, POSE.LEFT_ELBOW],
  [POSE.LEFT_ELBOW, POSE.LEFT_WRIST],
  [POSE.RIGHT_SHOULDER, POSE.RIGHT_ELBOW],
  [POSE.RIGHT_ELBOW, POSE.RIGHT_WRIST],
  [POSE.LEFT_SHOULDER, POSE.LEFT_HIP],
  [POSE.RIGHT_SHOULDER, POSE.RIGHT_HIP],
];

export interface OverlayOptions {
  dominant: Dominant;
  showFace: boolean;
  /** Draw mirrored to match a mirrored (selfie) video. */
  mirror: boolean;
  /** Normalised image y of the rest line, if known. */
  restLineY: number | null;
}

function setupCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return ctx;
}

function drawHand(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  color: string,
  weight: number,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = weight;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(pts[a].x, pts[a].y);
    ctx.lineTo(pts[b].x, pts[b].y);
  }
  ctx.stroke();
  ctx.fillStyle = '#fff';
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, weight * 0.9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

/** Draws the live tracking result over the video. */
export function drawObservation(canvas: HTMLCanvasElement, obs: RawObs | null, opts: OverlayOptions): void {
  const ctx = setupCanvas(canvas);
  if (!obs) return;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  const px = (p: LM) => ({ x: (opts.mirror ? 1 - p.x : p.x) * W, y: p.y * H });

  if (opts.restLineY !== null && opts.restLineY < 1) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.setLineDash([6, 8]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, opts.restLineY * H);
    ctx.lineTo(W, opts.restLineY * H);
    ctx.stroke();
    ctx.restore();
  }

  if (obs.pose) {
    const P = obs.pose;
    ctx.strokeStyle = COLORS.body;
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (const [a, b] of BODY_LINKS) {
      if ((P[a].visibility ?? 1) < 0.5 || (P[b].visibility ?? 1) < 0.5) continue;
      const pa = px(P[a]);
      const pb = px(P[b]);
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (obs.face && opts.showFace) {
    const F = obs.face;
    ctx.strokeStyle = COLORS.face;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (const set of FACE_CONTOURS) {
      for (const c of set) {
        const a = px(F[c.start]);
        const b = px(F[c.end]);
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (obs.face) {
    const F = obs.face;
    const handTips = [obs.left, obs.right]
      .filter((h) => !!h)
      .flatMap((h) => [4, 8, 12, 16, 20].map((i) => h!.landmarks[i]));
    const faceW = Math.hypot(F[FACE.LEFT_EDGE].x - F[FACE.RIGHT_EDGE].x, F[FACE.LEFT_EDGE].y - F[FACE.RIGHT_EDGE].y);
    const mouth = { x: (F[FACE.UPPER_LIP].x + F[FACE.LOWER_LIP].x) / 2, y: (F[FACE.UPPER_LIP].y + F[FACE.LOWER_LIP].y) / 2, z: 0 };
    const anchors = [...FACE_ANCHORS.map((i) => F[i]), mouth];
    for (const a of anchors) {
      const near = handTips.some((t) => Math.hypot(t.x - a.x, t.y - a.y) < faceW * 0.35);
      const p = px(a);
      ctx.beginPath();
      ctx.arc(p.x, p.y, near ? 7 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.face;
      ctx.globalAlpha = near ? 0.95 : 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  const domRaw = opts.dominant === 'right' ? obs.right : obs.left;
  const otherRaw = opts.dominant === 'right' ? obs.left : obs.right;
  if (otherRaw) drawHand(ctx, otherRaw.landmarks.map(px), COLORS.other, 2.5);
  if (domRaw) drawHand(ctx, domRaw.landmarks.map(px), COLORS.dominant, 3.5);
}

/** Draws one stored skeleton frame (used to replay recordings), fitted inside the canvas. */
export function drawSkeleton(canvas: HTMLCanvasElement, s: Skeleton, mirror: boolean): void {
  const ctx = setupCanvas(canvas);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.fillStyle = '#1B2733';
  ctx.fillRect(0, 0, W, H);
  // Skeleton x runs 0..aspect and y 0..1; keep that shape rather than stretching it.
  const k = Math.min(W / s.aspect, H);
  const ox = (W - s.aspect * k) / 2;
  const oy = (H - k) / 2;
  const px = (x: number, y: number) => ({ x: ox + (mirror ? s.aspect - x : x) * k, y: oy + y * k });

  if (s.pose) {
    const p = s.pose;
    ctx.strokeStyle = COLORS.body;
    ctx.lineWidth = 3;
    ctx.beginPath();
    const links: [typeof p.shD, typeof p.shD][] = [
      [p.shD, p.shN],
      [p.shD, p.elD],
      [p.elD, p.wrD],
      [p.shN, p.elN],
      [p.elN, p.wrN],
    ];
    for (const [a, b] of links) {
      if (a[2] < 0.5 || b[2] < 0.5) continue;
      const pa = px(a[0], a[1]);
      const pb = px(b[0], b[1]);
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
    }
    ctx.stroke();
  }
  if (s.face) {
    const f = s.face;
    ctx.fillStyle = COLORS.face;
    for (const q of [f.forehead, f.eyeD, f.eyeN, f.nose, f.mouth, f.chin, f.cheekD, f.cheekN]) {
      const p = px(q[0], q[1]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    const a = px(f.edgeD[0], f.edgeD[1]);
    const b = px(f.edgeN[0], f.edgeN[1]);
    const c = px(f.forehead[0], f.forehead[1]);
    const d = px(f.chin[0], f.chin[1]);
    ctx.strokeStyle = COLORS.face;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.ellipse((a.x + b.x) / 2, (c.y + d.y) / 2, Math.abs(b.x - a.x) / 2, Math.abs(d.y - c.y) / 2 + 6, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  const handPx = (img: number[]) => Array.from({ length: 21 }, (_, i) => px(img[i * 3], img[i * 3 + 1]));
  if (s.non) drawHand(ctx, handPx(s.non.img), COLORS.other, 2);
  if (s.dom) drawHand(ctx, handPx(s.dom.img), COLORS.dominant, 2.5);
}

/** Facial expression names for display, matching BLEND_KEYS. */
export const BLEND_LABELS: Record<(typeof BLEND_KEYS)[number], string> = {
  browInnerUp: 'brows raised (inner)',
  browOuterUp: 'brows raised (outer)',
  browDown: 'brows lowered',
  eyeWide: 'eyes wide',
  eyeSquint: 'eyes narrowed',
  eyeBlink: 'eyes closed',
  cheekPuff: 'cheeks puffed',
  jawOpen: 'mouth open',
  mouthFunnel: 'mouth rounded',
  mouthPucker: 'lips pursed',
  mouthSmile: 'smile',
  mouthFrown: 'frown',
};
