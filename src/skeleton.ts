/**
 * Landmark data in a canonical, dominant-hand-relative form.
 *
 * The tracker reports hands as the signer's left/right. For recognition we
 * want "dominant" and "non-dominant" instead, so a left-handed signer's frames
 * are mirrored to look like a right-handed signer's. After canonicalisation the
 * dominant side is always the signer's right, i.e. the left side of the raw
 * (unmirrored) camera image.
 *
 * Image coordinates are "aspect-corrected": x is multiplied by width/height so
 * x and y share the same unit (one image height).
 */

export type Dominant = 'right' | 'left';

/** [x, y] in aspect-corrected image units. */
export type P2 = [number, number];
/** [x, y, visibility] in aspect-corrected image units. */
export type PV = [number, number, number];

export interface HandPts {
  /** 21 × (x, y, z), aspect-corrected image coordinates. */
  img: number[];
  /** 21 × (x, y, z) in metres around the hand centre, or null if unavailable. */
  world: number[] | null;
}

export interface PosePts {
  nose: PV;
  shD: PV;
  shN: PV;
  elD: PV;
  elN: PV;
  wrD: PV;
  wrN: PV;
  earD: PV;
  earN: PV;
}

export interface FacePts {
  forehead: P2;
  nose: P2;
  mouth: P2;
  chin: P2;
  eyeD: P2;
  eyeN: P2;
  cheekD: P2;
  cheekN: P2;
  edgeD: P2;
  edgeN: P2;
}

export interface Skeleton {
  /** Milliseconds. */
  t: number;
  /** Source frame width / height. */
  aspect: number;
  dom: HandPts | null;
  non: HandPts | null;
  pose: PosePts | null;
  face: FacePts | null;
  /** Values for BLEND_KEYS, 0..1, or null if no face. */
  blend: number[] | null;
}

/** Minimal landmark shape shared by MediaPipe's NormalizedLandmark and Landmark. */
export interface LM {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface RawHand {
  landmarks: LM[];
  world: LM[] | null;
}

/** One tracker observation, in raw camera-image orientation. */
export interface RawObs {
  t: number;
  width: number;
  height: number;
  /** The signer's left hand. */
  left: RawHand | null;
  /** The signer's right hand. */
  right: RawHand | null;
  pose: LM[] | null;
  face: LM[] | null;
  blendshapes: Map<string, number> | null;
}

export const HAND = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

/** MediaPipe pose indices (signer's perspective). */
export const POSE = {
  NOSE: 0,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
} as const;

/** Face-mesh indices used as body-location anchors (signer's perspective). */
export const FACE = {
  FOREHEAD: 151,
  NOSE_TIP: 1,
  UPPER_LIP: 13,
  LOWER_LIP: 14,
  CHIN: 152,
  RIGHT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 133,
  LEFT_EYE_OUTER: 263,
  LEFT_EYE_INNER: 362,
  RIGHT_CHEEK: 205,
  LEFT_CHEEK: 425,
  RIGHT_EDGE: 234,
  LEFT_EDGE: 454,
} as const;

/** Facial-expression channels, each the mean of the left/right blendshapes where paired. */
export const BLEND_KEYS = [
  'browInnerUp',
  'browOuterUp',
  'browDown',
  'eyeWide',
  'eyeSquint',
  'eyeBlink',
  'cheekPuff',
  'jawOpen',
  'mouthFunnel',
  'mouthPucker',
  'mouthSmile',
  'mouthFrown',
] as const;

function blendValue(map: Map<string, number>, key: string): number {
  const direct = map.get(key);
  if (direct !== undefined) return direct;
  const l = map.get(`${key}Left`) ?? 0;
  const r = map.get(`${key}Right`) ?? 0;
  return (l + r) / 2;
}

/** Sign of the correlation between two coordinate series; used to align world axes with image axes. */
function axisSign(a: LM[], b: LM[], axis: 'x' | 'y'): number {
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < a.length; i++) {
    ma += a[i][axis];
    mb += b[i][axis];
  }
  ma /= a.length;
  mb /= b.length;
  let c = 0;
  for (let i = 0; i < a.length; i++) c += (a[i][axis] - ma) * (b[i][axis] - mb);
  return c < 0 ? -1 : 1;
}

function canonHand(h: RawHand | null, aspect: number, mirror: boolean): HandPts | null {
  if (!h || h.landmarks.length < 21) return null;
  const img: number[] = new Array(63);
  for (let i = 0; i < 21; i++) {
    const p = h.landmarks[i];
    img[i * 3] = mirror ? aspect - p.x * aspect : p.x * aspect;
    img[i * 3 + 1] = p.y;
    img[i * 3 + 2] = p.z * aspect;
  }
  let world: number[] | null = null;
  if (h.world && h.world.length >= 21) {
    // World landmarks should share the image's x/y directions; guard against a flip.
    const sx = axisSign(h.landmarks, h.world, 'x') * (mirror ? -1 : 1);
    const sy = axisSign(h.landmarks, h.world, 'y');
    world = new Array(63);
    for (let i = 0; i < 21; i++) {
      const p = h.world[i];
      world[i * 3] = p.x * sx;
      world[i * 3 + 1] = p.y * sy;
      world[i * 3 + 2] = p.z;
    }
  }
  return { img, world };
}

export function toSkeleton(raw: RawObs, dominant: Dominant): Skeleton {
  const aspect = raw.width > 0 && raw.height > 0 ? raw.width / raw.height : 4 / 3;
  const mirror = dominant === 'left';
  const fx = (x: number) => (mirror ? aspect - x * aspect : x * aspect);
  const p2 = (lm: LM): P2 => [fx(lm.x), lm.y];
  const pv = (lm: LM): PV => [fx(lm.x), lm.y, lm.visibility ?? 1];
  const avg = (a: LM, b: LM): LM => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });

  const domRaw = mirror ? raw.left : raw.right;
  const nonRaw = mirror ? raw.right : raw.left;

  let pose: PosePts | null = null;
  if (raw.pose && raw.pose.length >= 25) {
    const P = raw.pose;
    const pick = (right: number, left: number): [LM, LM] => (mirror ? [P[left], P[right]] : [P[right], P[left]]);
    const [shD, shN] = pick(POSE.RIGHT_SHOULDER, POSE.LEFT_SHOULDER);
    const [elD, elN] = pick(POSE.RIGHT_ELBOW, POSE.LEFT_ELBOW);
    const [wrD, wrN] = pick(POSE.RIGHT_WRIST, POSE.LEFT_WRIST);
    const [earD, earN] = pick(POSE.RIGHT_EAR, POSE.LEFT_EAR);
    pose = {
      nose: pv(P[POSE.NOSE]),
      shD: pv(shD),
      shN: pv(shN),
      elD: pv(elD),
      elN: pv(elN),
      wrD: pv(wrD),
      wrN: pv(wrN),
      earD: pv(earD),
      earN: pv(earN),
    };
  }

  let face: FacePts | null = null;
  if (raw.face && raw.face.length >= 468) {
    const F = raw.face;
    const eyeR = p2(avg(F[FACE.RIGHT_EYE_OUTER], F[FACE.RIGHT_EYE_INNER]));
    const eyeL = p2(avg(F[FACE.LEFT_EYE_OUTER], F[FACE.LEFT_EYE_INNER]));
    const cheekR = p2(F[FACE.RIGHT_CHEEK]);
    const cheekL = p2(F[FACE.LEFT_CHEEK]);
    const edgeR = p2(F[FACE.RIGHT_EDGE]);
    const edgeL = p2(F[FACE.LEFT_EDGE]);
    face = {
      forehead: p2(F[FACE.FOREHEAD]),
      nose: p2(F[FACE.NOSE_TIP]),
      mouth: p2(avg(F[FACE.UPPER_LIP], F[FACE.LOWER_LIP])),
      chin: p2(F[FACE.CHIN]),
      eyeD: mirror ? eyeL : eyeR,
      eyeN: mirror ? eyeR : eyeL,
      cheekD: mirror ? cheekL : cheekR,
      cheekN: mirror ? cheekR : cheekL,
      edgeD: mirror ? edgeL : edgeR,
      edgeN: mirror ? edgeR : edgeL,
    };
  }

  const blend = raw.blendshapes && face ? BLEND_KEYS.map((k) => blendValue(raw.blendshapes!, k)) : null;

  return {
    t: raw.t,
    aspect,
    dom: canonHand(domRaw, aspect, mirror),
    non: canonHand(nonRaw, aspect, mirror),
    pose,
    face,
    blend,
  };
}

/**
 * Bridges short tracking dropouts: if a hand vanishes for less than `maxGapMs`
 * the last observation is reused, which stops one-frame misses from splitting
 * a sign in two.
 */
export class DropoutFiller {
  private lastDom: { t: number; h: HandPts } | null = null;
  private lastNon: { t: number; h: HandPts } | null = null;

  constructor(private readonly maxGapMs = 120) {}

  apply(s: Skeleton): Skeleton {
    let { dom, non } = s;
    if (dom) this.lastDom = { t: s.t, h: dom };
    else if (this.lastDom && s.t - this.lastDom.t <= this.maxGapMs) dom = this.lastDom.h;
    if (non) this.lastNon = { t: s.t, h: non };
    else if (this.lastNon && s.t - this.lastNon.t <= this.maxGapMs) non = this.lastNon.h;
    return dom === s.dom && non === s.non ? s : { ...s, dom, non };
  }

  reset(): void {
    this.lastDom = null;
    this.lastNon = null;
  }
}

/** Point i of a flat 21×3 array. */
export function pt(arr: number[], i: number): { x: number; y: number; z: number } {
  return { x: arr[i * 3], y: arr[i * 3 + 1], z: arr[i * 3 + 2] };
}
