import { describeHand, handPoints, resolveChirality, type Chirality, type HandDesc } from './handshape';
import { HAND, type HandPts, type P2, type Skeleton } from './skeleton';

/**
 * Per-frame feature vector used for matching signs.
 *
 * Every dimension is divided by a typical variation ("sigma") when it is
 * extracted, so a difference of 1 in any dimension means "one typical
 * variation apart". Each group's distance is the RMS over its dimensions.
 */

export type GroupName =
  | 'domShape' | 'domOri' | 'domLoc' | 'domCon'
  | 'nonShape' | 'nonOri' | 'nonLoc' | 'nonCon'
  | 'rel' | 'face' | 'head' | 'body';

const SIZES: [GroupName, number][] = [
  ['domShape', 25], ['domOri', 6], ['domLoc', 5], ['domCon', 10],
  ['nonShape', 25], ['nonOri', 6], ['nonLoc', 5], ['nonCon', 10],
  ['rel', 4], ['face', 12], ['head', 5], ['body', 5],
];

export const GROUPS = {} as Record<GroupName, [number, number]>;
let offset = 0;
for (const [name, size] of SIZES) {
  GROUPS[name] = [offset, offset + size];
  offset += size;
}
export const FEATURE_DIM = offset;

export type Weights = Record<GroupName, number>;

export const SIGN_WEIGHTS: Weights = {
  domShape: 1, domOri: 0.6, domLoc: 0.8, domCon: 0.5,
  nonShape: 0.6, nonOri: 0.4, nonLoc: 0.6, nonCon: 0.3,
  rel: 0.5, face: 0.35, head: 0.25, body: 0.2,
};

/** Fingerspelled letters only depend on the dominant handshape and its orientation. */
export const LETTER_WEIGHTS: Weights = {
  domShape: 1, domOri: 0.5, domLoc: 0, domCon: 0,
  nonShape: 0, nonOri: 0, nonLoc: 0, nonCon: 0,
  rel: 0, face: 0, head: 0, body: 0,
};

export function signWeights(opts: { useFace: boolean; useBody: boolean }): Weights {
  return {
    ...SIGN_WEIGHTS,
    face: opts.useFace ? SIGN_WEIGHTS.face : 0,
    head: opts.useFace ? SIGN_WEIGHTS.head : 0,
    domCon: opts.useFace ? SIGN_WEIGHTS.domCon : SIGN_WEIGHTS.domCon * 0.4,
    nonCon: opts.useFace ? SIGN_WEIGHTS.nonCon : SIGN_WEIGHTS.nonCon * 0.4,
    body: opts.useBody ? SIGN_WEIGHTS.body : 0,
  };
}

/** The part of a frame that matching needs; this is what templates store. */
export interface CoreFrame {
  t: number;
  v: Float32Array;
  dom: boolean;
  non: boolean;
  face: boolean;
  pose: boolean;
}

export interface FeatureFrame extends CoreFrame {
  /** Hand centres in body units (shoulder widths; x towards the dominant side, y down). */
  domC: [number, number] | null;
  nonC: [number, number] | null;
  /** Dominant index and pinky fingertips in body units, for letter trajectories (J, Z). */
  domIndexTip: [number, number] | null;
  domPinkyTip: [number, number] | null;
  /** Normalised image y of each hand centre, to notice hands dropping out of frame. */
  domImgY: number;
  nonImgY: number;
  /** Descriptor of the dominant hand, reused by the fingerspelling rules. */
  domDesc: HandDesc | null;
}

export interface BodyFrame {
  /** Origin (between the shoulders) in aspect-corrected image coordinates. */
  ox: number;
  oy: number;
  /** Shoulder width in image units. */
  scale: number;
  /** How the frame was estimated. */
  source: 'pose' | 'face' | 'guess';
}

export function bodyFrame(s: Skeleton): BodyFrame {
  const p = s.pose;
  if (p && p.shD[2] > 0.5 && p.shN[2] > 0.5) {
    const w = Math.hypot(p.shD[0] - p.shN[0], p.shD[1] - p.shN[1]);
    if (w > 0.05) {
      return { ox: (p.shD[0] + p.shN[0]) / 2, oy: (p.shD[1] + p.shN[1]) / 2, scale: w, source: 'pose' };
    }
  }
  const f = s.face;
  if (f) {
    const fw = Math.hypot(f.edgeD[0] - f.edgeN[0], f.edgeD[1] - f.edgeN[1]);
    if (fw > 0.02) {
      return { ox: f.nose[0], oy: f.chin[1] + 0.8 * fw, scale: 2.6 * fw, source: 'face' };
    }
  }
  return { ox: s.aspect / 2, oy: 0.75, scale: 0.4, source: 'guess' };
}

/** Image point → body units: x positive towards the dominant side, y positive downwards. */
export function toBody(bf: BodyFrame, x: number, y: number): [number, number] {
  return [(bf.ox - x) / bf.scale, (y - bf.oy) / bf.scale];
}

const SIG_ANGLE = 0.4;
const SIG_LOCAL = 0.3;
const SIG_DIR = 0.5;
const SIG_LOC = 0.3;
const SIG_PROX = 0.35;
const PROX_RANGE = 0.12;
const SIG_BLEND = 0.25;

const clip = (x: number, lim: number) => (x > lim ? lim : x < -lim ? -lim : x);
const proximity = (d: number) => Math.exp(-d / PROX_RANGE);

function handCenterImg(img: number[]): P2 {
  const ids = [HAND.WRIST, HAND.INDEX_MCP, HAND.MIDDLE_MCP, HAND.PINKY_MCP];
  let x = 0;
  let y = 0;
  for (const i of ids) {
    x += img[i * 3];
    y += img[i * 3 + 1];
  }
  return [x / ids.length, y / ids.length];
}

/** Hand points used when measuring contact with the face/body. */
function probes(img: number[]): P2[] {
  const out: P2[] = [handCenterImg(img)];
  for (const i of [HAND.THUMB_TIP, HAND.INDEX_TIP, HAND.MIDDLE_TIP, HAND.RING_TIP, HAND.PINKY_TIP]) {
    out.push([img[i * 3], img[i * 3 + 1]]);
  }
  return out;
}

function minDist(points: P2[], anchor: P2 | null): number {
  if (!anchor) return Infinity;
  let best = Infinity;
  for (const p of points) best = Math.min(best, Math.hypot(p[0] - anchor[0], p[1] - anchor[1]));
  return best;
}

interface HandFeatures {
  desc: HandDesc;
  center: [number, number];
  centerImgY: number;
  probes: P2[];
}

function writeHand(
  v: Float32Array,
  h: HandPts,
  chirality: Chirality,
  isDom: boolean,
  s: Skeleton,
  bf: BodyFrame,
): HandFeatures {
  const [s0] = GROUPS[isDom ? 'domShape' : 'nonShape'];
  const [o0] = GROUPS[isDom ? 'domOri' : 'nonOri'];
  const [l0] = GROUPS[isDom ? 'domLoc' : 'nonLoc'];
  const [c0] = GROUPS[isDom ? 'domCon' : 'nonCon'];

  const points = handPoints(h);
  const desc = describeHand(points, resolveChirality(points, chirality));
  let k = s0;
  for (const f of desc.flex) v[k++] = f / SIG_ANGLE;
  for (const sp of desc.spread) v[k++] = sp / SIG_ANGLE;
  const tt = desc.local[HAND.THUMB_TIP];
  v[k++] = clip(tt.x, 3) / SIG_LOCAL;
  v[k++] = clip(tt.y, 3) / SIG_LOCAL;
  v[k++] = clip(tt.z, 3) / SIG_LOCAL;
  for (const d of desc.thumbTo) v[k++] = Math.min(d, 2) / SIG_LOCAL;

  k = o0;
  v[k++] = desc.normal.x / SIG_DIR;
  v[k++] = desc.normal.y / SIG_DIR;
  v[k++] = desc.normal.z / SIG_DIR;
  v[k++] = desc.up.x / SIG_DIR;
  v[k++] = desc.up.y / SIG_DIR;
  v[k++] = desc.up.z / SIG_DIR;

  const cImg = handCenterImg(h.img);
  const center = toBody(bf, cImg[0], cImg[1]);
  const tip = toBody(bf, h.img[HAND.INDEX_TIP * 3], h.img[HAND.INDEX_TIP * 3 + 1]);
  const palmImg = Math.hypot(
    h.img[HAND.MIDDLE_MCP * 3] - h.img[0],
    h.img[HAND.MIDDLE_MCP * 3 + 1] - h.img[1],
  );
  k = l0;
  v[k++] = center[0] / SIG_LOC;
  v[k++] = center[1] / SIG_LOC;
  v[k++] = tip[0] / SIG_LOC;
  v[k++] = tip[1] / SIG_LOC;
  v[k++] = Math.log(Math.max(palmImg, 1e-3) / bf.scale) / 0.35;

  const pr = probes(h.img);
  const f = s.face;
  const p = s.pose;
  const chest: P2 = [bf.ox, bf.oy + 0.45 * bf.scale];
  const vis = (q: [number, number, number] | undefined): P2 | null => (q && q[2] > 0.5 ? [q[0], q[1]] : null);
  const anchors: (P2 | null)[] = [
    f?.forehead ?? null,
    null, // eyes: handled below (closest eye)
    f?.nose ?? null,
    f?.mouth ?? null,
    f?.chin ?? null,
    f ? (isDom ? f.cheekD : f.cheekN) : null,
    f ? (isDom ? f.cheekN : f.cheekD) : null,
    p ? vis(isDom ? p.earD : p.earN) : null,
    chest,
    p ? vis(isDom ? p.shN : p.shD) : null,
  ];
  k = c0;
  for (let i = 0; i < anchors.length; i++) {
    let d: number;
    if (i === 1) d = f ? Math.min(minDist(pr, f.eyeD), minDist(pr, f.eyeN)) : Infinity;
    else d = minDist(pr, anchors[i]);
    v[k++] = (Number.isFinite(d) ? proximity(d / bf.scale) : 0) / SIG_PROX;
  }

  return { desc, center, centerImgY: cImg[1], probes: pr };
}

export function extractFeatures(s: Skeleton): FeatureFrame {
  const v = new Float32Array(FEATURE_DIM);
  const bf = bodyFrame(s);

  const dom = s.dom ? writeHand(v, s.dom, 'right', true, s, bf) : null;
  const non = s.non ? writeHand(v, s.non, 'left', false, s, bf) : null;

  if (dom && non) {
    let k = GROUPS.rel[0];
    v[k++] = (non.center[0] - dom.center[0]) / SIG_LOC;
    v[k++] = (non.center[1] - dom.center[1]) / SIG_LOC;
    let d = Infinity;
    for (const a of dom.probes) d = Math.min(d, minDist(non.probes, a));
    v[k++] = proximity(d / bf.scale) / SIG_PROX;
    const nd = dom.desc.normal;
    const nn = non.desc.normal;
    v[k++] = (nd.x * nn.x + nd.y * nn.y + nd.z * nn.z) / SIG_DIR;
  }

  const f = s.face;
  if (f && s.blend) {
    let k = GROUPS.face[0];
    for (const b of s.blend) v[k++] = b / SIG_BLEND;
  }
  if (f) {
    let k = GROUPS.head[0];
    const roll = Math.atan2(f.eyeN[1] - f.eyeD[1], f.eyeN[0] - f.eyeD[0]);
    const faceW = Math.max(1e-3, Math.hypot(f.edgeN[0] - f.edgeD[0], f.edgeN[1] - f.edgeD[1]));
    const yaw = (f.nose[0] - (f.edgeD[0] + f.edgeN[0]) / 2) / faceW;
    const eyesY = (f.eyeD[1] + f.eyeN[1]) / 2;
    const pitch = (f.nose[1] - eyesY) / Math.max(1e-3, f.chin[1] - eyesY);
    const nose = toBody(bf, f.nose[0], f.nose[1]);
    v[k++] = roll / 0.15;
    v[k++] = yaw / 0.1;
    v[k++] = pitch / 0.08;
    v[k++] = nose[0] / 0.15;
    v[k++] = nose[1] / 0.15;
  }

  const p = s.pose;
  const posePresent = !!p && p.shD[2] > 0.5 && p.shN[2] > 0.5;
  if (p && posePresent) {
    let k = GROUPS.body[0];
    const eD = toBody(bf, p.elD[0], p.elD[1]);
    const eN = toBody(bf, p.elN[0], p.elN[1]);
    v[k++] = eD[0] / 0.35;
    v[k++] = eD[1] / 0.35;
    v[k++] = eN[0] / 0.35;
    v[k++] = eN[1] / 0.35;
    v[k++] = Math.atan2(p.shN[1] - p.shD[1], p.shN[0] - p.shD[0]) / 0.1;
  }

  const bodyPt = (h: HandPts, i: number) => toBody(bf, h.img[i * 3], h.img[i * 3 + 1]);

  return {
    t: s.t,
    v,
    dom: !!dom,
    non: !!non,
    face: !!f,
    pose: posePresent,
    domC: dom ? dom.center : null,
    nonC: non ? non.center : null,
    domIndexTip: s.dom ? bodyPt(s.dom, HAND.INDEX_TIP) : null,
    domPinkyTip: s.dom ? bodyPt(s.dom, HAND.PINKY_TIP) : null,
    domImgY: dom ? dom.centerImgY : NaN,
    nonImgY: non ? non.centerImgY : NaN,
    domDesc: dom ? dom.desc : null,
  };
}

// ---------------------------------------------------------------------------
// Distance

const MISSING = 2.0;

const HAND_GROUPS = {
  dom: ['domShape', 'domOri', 'domLoc', 'domCon'] as GroupName[],
  non: ['nonShape', 'nonOri', 'nonLoc', 'nonCon'] as GroupName[],
};

function groupSq(a: Float32Array, b: Float32Array, g: GroupName): number {
  const [s, e] = GROUPS[g];
  let sum = 0;
  for (let i = s; i < e; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum / (e - s);
}

/**
 * Weighted distance between two frames: the square root of the weighted sum of
 * squared group distances, so a difference in any one group (handshape, place,
 * orientation, face…) counts in full rather than being averaged away.
 * Roughly 0.4–0.7 for two performances of the same sign, above 1 for different ones.
 */
export function frameDistance(a: CoreFrame, b: CoreFrame, w: Weights): number {
  let sum = 0;
  let compared = false;
  for (const side of ['dom', 'non'] as const) {
    const groups = HAND_GROUPS[side];
    if (a[side] && b[side]) {
      for (const g of groups) {
        const wg = w[g];
        if (wg <= 0) continue;
        sum += wg * groupSq(a.v, b.v, g);
        compared = true;
      }
    } else if (a[side] !== b[side]) {
      let wsum = 0;
      for (const g of groups) wsum += w[g];
      if (wsum > 0) {
        sum += wsum * MISSING * MISSING;
        compared = true;
      }
    }
  }
  if (w.rel > 0 && a.dom && a.non && b.dom && b.non) {
    sum += w.rel * groupSq(a.v, b.v, 'rel');
  }
  if (a.face && b.face) {
    if (w.face > 0) sum += w.face * groupSq(a.v, b.v, 'face');
    if (w.head > 0) sum += w.head * groupSq(a.v, b.v, 'head');
  }
  if (w.body > 0 && a.pose && b.pose) {
    sum += w.body * groupSq(a.v, b.v, 'body');
  }
  // Nothing comparable (e.g. a letter template against a frame without that hand) counts as far apart.
  return compared ? Math.sqrt(sum) : MISSING;
}

/** Distance travelled by the hands (body units) over a run of frames; the larger of the two hands. */
export function travel(frames: CoreFrame[], from = 0, to = frames.length - 1): number {
  const [d0] = GROUPS.domLoc;
  const [n0] = GROUPS.nonLoc;
  let best = 0;
  for (const [key, off] of [['dom', d0], ['non', n0]] as const) {
    let len = 0;
    for (let i = Math.max(1, from + 1); i <= to; i++) {
      const a = frames[i - 1];
      const b = frames[i];
      if (!a[key] || !b[key]) continue;
      len += Math.hypot(b.v[off] - a.v[off], b.v[off + 1] - a.v[off + 1]) * SIG_LOC;
    }
    best = Math.max(best, len);
  }
  return best;
}

/** Average of several frames, e.g. over a held handshape. Presence flags go by majority. */
export function meanFrame(frames: CoreFrame[]): CoreFrame {
  const n = frames.length;
  const v = new Float32Array(FEATURE_DIM);
  const majority = (key: 'dom' | 'non' | 'face' | 'pose') => frames.filter((f) => f[key]).length * 2 > n;
  const out: CoreFrame = {
    t: frames[n - 1]?.t ?? 0,
    v,
    dom: majority('dom'),
    non: majority('non'),
    face: majority('face'),
    pose: majority('pose'),
  };
  const need: Record<GroupName, (f: CoreFrame) => boolean> = {
    domShape: (f) => f.dom, domOri: (f) => f.dom, domLoc: (f) => f.dom, domCon: (f) => f.dom,
    nonShape: (f) => f.non, nonOri: (f) => f.non, nonLoc: (f) => f.non, nonCon: (f) => f.non,
    rel: (f) => f.dom && f.non, face: (f) => f.face, head: (f) => f.face, body: (f) => f.pose,
  };
  for (const g of Object.keys(GROUPS) as GroupName[]) {
    const [s, e] = GROUPS[g];
    const use = frames.filter(need[g]);
    if (!use.length) continue;
    for (const f of use) for (let i = s; i < e; i++) v[i] += f.v[i];
    for (let i = s; i < e; i++) v[i] /= use.length;
  }
  return out;
}

/** Resamples a sequence to a fixed rate so recordings made at different frame rates line up. */
export function resample<T extends CoreFrame>(frames: T[], hz = 15): CoreFrame[] {
  if (frames.length < 2) return frames.map(core);
  const step = 1000 / hz;
  const t0 = frames[0].t;
  const t1 = frames[frames.length - 1].t;
  const out: CoreFrame[] = [];
  let j = 0;
  for (let t = t0; t <= t1 + 1e-6; t += step) {
    while (j < frames.length - 2 && frames[j + 1].t < t) j++;
    const a = frames[j];
    const b = frames[j + 1];
    const span = b.t - a.t;
    const u = span > 0 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 0;
    if (a.dom === b.dom && a.non === b.non && a.face === b.face && a.pose === b.pose) {
      const v = new Float32Array(FEATURE_DIM);
      for (let i = 0; i < FEATURE_DIM; i++) v[i] = a.v[i] + (b.v[i] - a.v[i]) * u;
      out.push({ t, v, dom: a.dom, non: a.non, face: a.face, pose: a.pose });
    } else {
      out.push({ ...core(u < 0.5 ? a : b), t });
    }
  }
  return out;
}

export function core(f: CoreFrame): CoreFrame {
  return { t: f.t, v: f.v, dom: f.dom, non: f.non, face: f.face, pose: f.pose };
}

/** Root-mean-square difference of the dominant handshape block, for detecting handshape changes. */
export function shapeChange(a: CoreFrame, b: CoreFrame, side: 'dom' | 'non'): number {
  return Math.sqrt(groupSq(a.v, b.v, side === 'dom' ? 'domShape' : 'nonShape'));
}
