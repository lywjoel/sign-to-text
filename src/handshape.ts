import { angle, cross, dist, dot, normalize, reject, sub, type Vec3 } from './geometry';
import { HAND, pt, type HandPts } from './skeleton';

export type Chirality = 'right' | 'left';

/** Orientation-independent description of a hand pose. */
export interface HandDesc {
  /** Wrist → middle-finger knuckle length, in source units. */
  palm: number;
  /** Unit vectors in source coordinates: towards the fingers, towards the index side, out of the palm. */
  up: Vec3;
  side: Vec3;
  normal: Vec3;
  /**
   * Joint flexion in radians (0 = straight):
   * [thumbAbduction, thumbMcp, thumbIp, indexMcp, indexPip, indexDip, middle…, ring…, pinky…]
   */
  flex: number[];
  /** Angle between neighbouring fingers in the palm plane: [index-middle, middle-ring, ring-pinky]. */
  spread: number[];
  /** All 21 points in the hand's own frame (s = index side, u = fingers, n = palm side), in palm units. */
  local: Vec3[];
  /** Thumb-tip distance to the index, middle, ring and pinky tips, in palm units. */
  thumbTo: number[];
  /**
   * How far each finger reaches: knuckle-to-tip distance over the finger's
   * length (thumb, index, middle, ring, pinky). About 1 when straight, under
   * 0.6 when curled. Robust where joint angles are not, e.g. a hand seen side-on.
   */
  reach: number[];
}

const FINGERS = [
  [HAND.INDEX_MCP, HAND.INDEX_PIP, HAND.INDEX_DIP, HAND.INDEX_TIP],
  [HAND.MIDDLE_MCP, HAND.MIDDLE_PIP, HAND.MIDDLE_DIP, HAND.MIDDLE_TIP],
  [HAND.RING_MCP, HAND.RING_PIP, HAND.RING_DIP, HAND.RING_TIP],
  [HAND.PINKY_MCP, HAND.PINKY_PIP, HAND.PINKY_DIP, HAND.PINKY_TIP],
] as const;

/** Points of a hand, preferring metric world coordinates when present. */
export function handPoints(h: HandPts): Vec3[] {
  const src = h.world ?? h.img;
  const out: Vec3[] = new Array(21);
  for (let i = 0; i < 21; i++) out[i] = pt(src, i);
  return out;
}

export function imagePoints(h: HandPts): Vec3[] {
  const out: Vec3[] = new Array(21);
  for (let i = 0; i < 21; i++) out[i] = pt(h.img, i);
  return out;
}

/**
 * Builds the hand descriptor. `chirality` says whether the points form a right
 * or a left hand, which decides which way the palm normal points.
 */
export function describeHand(p: Vec3[], chirality: Chirality): HandDesc {
  const wrist = p[HAND.WRIST];
  const up = normalize(sub(p[HAND.MIDDLE_MCP], wrist));
  const side = normalize(reject(sub(p[HAND.INDEX_MCP], p[HAND.PINKY_MCP]), up));
  const normal = normalize(chirality === 'right' ? cross(side, up) : cross(up, side));
  const palm = Math.max(1e-6, dist(p[HAND.MIDDLE_MCP], wrist));

  const local = p.map((q) => {
    const d = sub(q, wrist);
    return { x: dot(d, side) / palm, y: dot(d, up) / palm, z: dot(d, normal) / palm };
  });

  const flex: number[] = [];
  flex.push(angle(sub(p[HAND.THUMB_MCP], p[HAND.THUMB_CMC]), sub(p[HAND.INDEX_MCP], wrist)));
  flex.push(angle(sub(p[HAND.THUMB_MCP], p[HAND.THUMB_CMC]), sub(p[HAND.THUMB_IP], p[HAND.THUMB_MCP])));
  flex.push(angle(sub(p[HAND.THUMB_IP], p[HAND.THUMB_MCP]), sub(p[HAND.THUMB_TIP], p[HAND.THUMB_IP])));

  const proximal: Vec3[] = [];
  for (const [m, pip, dip, tip] of FINGERS) {
    const prox = sub(p[pip], p[m]);
    proximal.push(prox);
    // MCP flexion: how far the first segment tips out of the palm plane towards the palm side.
    const dir = normalize(prox);
    const dn = dot(dir, normal);
    const inPlane = reject(dir, normal);
    const along = Math.hypot(inPlane.x, inPlane.y, inPlane.z) * (dot(inPlane, up) >= 0 ? 1 : -1);
    flex.push(Math.atan2(dn, along));
    flex.push(angle(prox, sub(p[dip], p[pip])));
    flex.push(angle(sub(p[dip], p[pip]), sub(p[tip], p[dip])));
  }

  const spread: number[] = [];
  for (let i = 0; i < 3; i++) {
    const a = reject(proximal[i], normal);
    const b = reject(proximal[i + 1], normal);
    const la = Math.hypot(a.x, a.y, a.z);
    const lb = Math.hypot(b.x, b.y, b.z);
    const usable = la > 0.3 * Math.hypot(proximal[i].x, proximal[i].y, proximal[i].z) &&
      lb > 0.3 * Math.hypot(proximal[i + 1].x, proximal[i + 1].y, proximal[i + 1].z);
    spread.push(usable ? angle(a, b) : 0);
  }

  const thumbTip = p[HAND.THUMB_TIP];
  const thumbTo = [HAND.INDEX_TIP, HAND.MIDDLE_TIP, HAND.RING_TIP, HAND.PINKY_TIP].map(
    (i) => dist(thumbTip, p[i]) / palm,
  );

  const chainReach = (ids: readonly number[]) => {
    let len = 0;
    for (let i = 1; i < ids.length; i++) len += dist(p[ids[i - 1]], p[ids[i]]);
    return len > 1e-9 ? dist(p[ids[0]], p[ids[ids.length - 1]]) / len : 1;
  };
  const reach = [chainReach([HAND.THUMB_MCP, HAND.THUMB_IP, HAND.THUMB_TIP]), ...FINGERS.map(chainReach)];

  return { palm, up, side, normal, flex, spread, local, thumbTo, reach };
}

/**
 * Evidence for which hand the points form, read from the shape itself:
 * positive means a right hand, negative a left hand. The base of the thumb
 * sits in front of the palm, and bent fingers fold towards the palm, so both
 * show which side the palm is on. Near zero (a flat hand seen edge-on) means
 * the shape alone can't tell.
 */
export function chiralityEvidence(p: Vec3[]): number {
  const wrist = p[HAND.WRIST];
  const up = normalize(sub(p[HAND.MIDDLE_MCP], wrist));
  const side = normalize(reject(sub(p[HAND.INDEX_MCP], p[HAND.PINKY_MCP]), up));
  const nRight = cross(side, up);
  const palm = Math.max(1e-6, dist(p[HAND.MIDDLE_MCP], wrist));
  const toPalm = (a: Vec3, b: Vec3) => dot(nRight, sub(b, a)) / palm;
  // Nearly straight fingers bend either way within tracking error, so only clear bends count.
  const bend = (x: number) => Math.max(-0.6, Math.min(0.6, Math.sign(x) * Math.max(0, Math.abs(x) - 0.12)));
  let e = toPalm(wrist, p[HAND.THUMB_MCP]) + 0.5 * toPalm(wrist, p[HAND.THUMB_IP]);
  for (const [m, , , tip] of FINGERS) e += bend(toPalm(p[m], p[tip]));
  return e;
}

/**
 * The hand to describe the points as: `assumed` unless the 3D shape clearly
 * says the tracker read it the other way round.
 */
export function resolveChirality(points: Vec3[], assumed: Chirality): Chirality {
  const e = chiralityEvidence(points);
  return e > 0.6 ? 'right' : e < -0.6 ? 'left' : assumed;
}

/** Mean of several hands' points (used to steady a held handshape). */
export function averagePoints(sets: Vec3[][]): Vec3[] {
  const n = sets.length;
  const out: Vec3[] = [];
  for (let i = 0; i < 21; i++) {
    let x = 0;
    let y = 0;
    let z = 0;
    for (const s of sets) {
      x += s[i].x;
      y += s[i].y;
      z += s[i].z;
    }
    out.push({ x: x / n, y: y / n, z: z / n });
  }
  return out;
}
