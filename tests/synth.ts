import type { Vec3 } from '../src/geometry';
import type { HandPts } from '../src/skeleton';

/**
 * A simple kinematic hand for tests. Poses are given in the hand's own frame
 * (s = towards the index side, u = towards the fingers, n = out of the palm),
 * in palm units (wrist → middle knuckle = 1), then rotated into camera space
 * (x right, y down, z away from the camera).
 */

export interface FingerPose {
  mcp: number;
  pip: number;
  dip: number;
  /** Sideways rotation at the knuckle, positive towards the index side. */
  abd?: number;
}

export interface HandPose {
  fingers: [FingerPose, FingerPose, FingerPose, FingerPose];
  /** Thumb tip target in the hand frame. */
  thumb: [number, number, number];
}

const MCPS: [number, number, number][] = [
  [0.28, 0.96, 0],
  [0, 1, 0],
  [-0.24, 0.94, 0],
  [-0.45, 0.83, 0],
];
const FAN = [0.12, 0, -0.1, -0.22];
const LENGTHS = [
  [0.42, 0.25, 0.2],
  [0.46, 0.29, 0.21],
  [0.43, 0.27, 0.2],
  [0.34, 0.2, 0.18],
];
const CMC: [number, number, number] = [0.25, 0.2, 0.08];

export function localHand(pose: HandPose): Vec3[] {
  const pts: Vec3[] = new Array(21);
  pts[0] = { x: 0, y: 0, z: 0 };
  const [tx, ty, tz] = pose.thumb;
  const lerp = (k: number, bump: number): Vec3 => ({
    x: CMC[0] + (tx - CMC[0]) * k + bump,
    y: CMC[1] + (ty - CMC[1]) * k,
    z: CMC[2] + (tz - CMC[2]) * k,
  });
  pts[1] = { x: CMC[0], y: CMC[1], z: CMC[2] };
  pts[2] = lerp(0.42, 0.04);
  pts[3] = lerp(0.72, 0.02);
  pts[4] = { x: tx, y: ty, z: tz };

  for (let f = 0; f < 4; f++) {
    const fp = pose.fingers[f];
    const theta = FAN[f] + (fp.abd ?? 0);
    const d0 = { x: Math.sin(theta), y: Math.cos(theta) };
    let p: Vec3 = { x: MCPS[f][0], y: MCPS[f][1], z: MCPS[f][2] };
    const base = 5 + f * 4;
    pts[base] = p;
    let phi = 0;
    const flex = [fp.mcp, fp.pip, fp.dip];
    for (let k = 0; k < 3; k++) {
      phi += flex[k];
      const len = LENGTHS[f][k];
      p = {
        x: p.x + len * d0.x * Math.cos(phi),
        y: p.y + len * d0.y * Math.cos(phi),
        z: p.z + len * Math.sin(phi),
      };
      pts[base + k + 1] = p;
    }
  }
  return pts;
}

/** Orientation: where the hand's s, u and n axes point in camera space. Must be a proper rotation. */
export interface Orientation {
  s: Vec3;
  u: Vec3;
  n: Vec3;
}

export const UPRIGHT: Orientation = {
  s: { x: 1, y: 0, z: 0 },
  u: { x: 0, y: -1, z: 0 },
  n: { x: 0, y: 0, z: -1 },
};

/** Fingers pointing across the body (camera +x), palm towards the signer. */
export const SIDEWAYS: Orientation = {
  s: { x: 0, y: -1, z: 0 },
  u: { x: 1, y: 0, z: 0 },
  n: { x: 0, y: 0, z: 1 },
};

/** Fingers pointing down and towards the camera. */
export const DOWNWARD: Orientation = (() => {
  const u = { x: 0, y: 0.9, z: -0.44 };
  const l = Math.hypot(u.x, u.y, u.z);
  const uu = { x: u.x / l, y: u.y / l, z: u.z / l };
  const s = { x: 1, y: 0, z: 0 };
  const n = { x: s.y * uu.z - s.z * uu.y, y: s.z * uu.x - s.x * uu.z, z: s.x * uu.y - s.y * uu.x };
  return { s, u: uu, n };
})();

/** Fingers up, palm facing the signer's left (camera +x): how C and O are shown. */
export const PALM_SIDE: Orientation = {
  s: { x: 0, y: 0, z: 1 },
  u: { x: 0, y: -1, z: 0 },
  n: { x: 1, y: 0, z: 0 },
};

/** Rotates an orientation about the camera axes by small angles (radians). */
export function tilt(o: Orientation, rx: number, ry: number, rz: number): Orientation {
  const rot = (v: Vec3): Vec3 => {
    let { x, y, z } = v;
    [y, z] = [y * Math.cos(rx) - z * Math.sin(rx), y * Math.sin(rx) + z * Math.cos(rx)];
    [x, z] = [x * Math.cos(ry) + z * Math.sin(ry), -x * Math.sin(ry) + z * Math.cos(ry)];
    [x, y] = [x * Math.cos(rz) - y * Math.sin(rz), x * Math.sin(rz) + y * Math.cos(rz)];
    return { x, y, z };
  };
  return { s: rot(o.s), u: rot(o.u), n: rot(o.n) };
}

export function toCamera(local: Vec3[], o: Orientation, size = 0.09, origin: Vec3 = { x: 0, y: 0, z: 0 }): Vec3[] {
  return local.map((p) => ({
    x: origin.x + size * (p.x * o.s.x + p.y * o.u.x + p.z * o.n.x),
    y: origin.y + size * (p.x * o.s.y + p.y * o.u.y + p.z * o.n.y),
    z: origin.z + size * (p.x * o.s.z + p.y * o.u.z + p.z * o.n.z),
  }));
}

/** Mirror image of a right hand: a left hand in the same pose. */
export function mirror(points: Vec3[]): Vec3[] {
  return points.map((p) => ({ x: -p.x, y: p.y, z: p.z }));
}

/** HandPts with image coordinates centred at (cx, cy) (aspect-corrected units) plus world coordinates. */
export function handPts(world: Vec3[], cx: number, cy: number, imgScale = 1): HandPts {
  const img: number[] = [];
  const w: number[] = [];
  for (const p of world) {
    img.push(cx + p.x * imgScale, cy + p.y * imgScale, p.z * imgScale);
    w.push(p.x, p.y, p.z);
  }
  return { img, world: w };
}

const FIST: FingerPose = { mcp: 1.5, pip: 1.75, dip: 0.7 };
const STRAIGHT: FingerPose = { mcp: 0.05, pip: 0.05, dip: 0.05 };
const ROUND: FingerPose = { mcp: 0.8, pip: 1.0, dip: 0.5 };

export const LETTER_POSES: Record<string, { pose: HandPose; orient: Orientation }> = {
  // Thumb up beside the fist, its tip level with or above the index's middle joint.
  A: { pose: { fingers: [FIST, FIST, FIST, FIST], thumb: [0.48, 1.15, 0.22] }, orient: UPRIGHT },
  B: {
    pose: {
      fingers: [{ ...STRAIGHT, abd: -0.1 }, STRAIGHT, { ...STRAIGHT, abd: 0.08 }, { ...STRAIGHT, abd: 0.18 }],
      thumb: [0.0, 0.55, 0.18],
    },
    orient: UPRIGHT,
  },
  C: {
    pose: {
      fingers: [
        { mcp: 0.4, pip: 0.9, dip: 0.6 },
        { mcp: 0.4, pip: 0.9, dip: 0.6 },
        { mcp: 0.4, pip: 0.9, dip: 0.6 },
        { mcp: 0.4, pip: 0.9, dip: 0.6 },
      ],
      thumb: [1.1, 0.9, 0.4],
    },
    orient: PALM_SIDE,
  },
  D: { pose: { fingers: [STRAIGHT, ROUND, ROUND, ROUND], thumb: [0.08, 1.08, 0.74] }, orient: UPRIGHT },
  // Fingers bent at the middle joints with their tips resting on the thumb, which is folded across below them.
  E: {
    pose: {
      fingers: [
        { mcp: 0.2, pip: 1.2, dip: 0.6 },
        { mcp: 0.2, pip: 1.2, dip: 0.6 },
        { mcp: 0.25, pip: 1.2, dip: 0.6 },
        { mcp: 0.3, pip: 1.2, dip: 0.6 },
      ],
      thumb: [0.12, 1.1, 0.45],
    },
    orient: UPRIGHT,
  },
  F: {
    pose: {
      fingers: [ROUND, STRAIGHT, { ...STRAIGHT, abd: -0.1 }, { ...STRAIGHT, abd: -0.15 }],
      thumb: [0.25, 1.0, 0.68],
    },
    orient: UPRIGHT,
  },
  G: { pose: { fingers: [STRAIGHT, FIST, FIST, FIST], thumb: [0.4, 1.15, 0.2] }, orient: SIDEWAYS },
  H: {
    pose: { fingers: [{ ...STRAIGHT, abd: -0.1 }, STRAIGHT, FIST, FIST], thumb: [-0.1, 0.75, 0.55] },
    orient: SIDEWAYS,
  },
  I: { pose: { fingers: [FIST, FIST, FIST, STRAIGHT], thumb: [0.1, 0.7, 0.6] }, orient: UPRIGHT },
  K: {
    pose: { fingers: [STRAIGHT, { mcp: 0.6, pip: 0.1, dip: 0.05 }, FIST, FIST], thumb: [0.12, 1.3, 0.25] },
    orient: UPRIGHT,
  },
  L: { pose: { fingers: [STRAIGHT, FIST, FIST, FIST], thumb: [1.05, 0.45, 0.05] }, orient: UPRIGHT },
  M: { pose: { fingers: [FIST, FIST, FIST, FIST], thumb: [-0.36, 0.8, 0.3] }, orient: UPRIGHT },
  N: { pose: { fingers: [FIST, FIST, FIST, FIST], thumb: [-0.12, 0.85, 0.32] }, orient: UPRIGHT },
  O: { pose: { fingers: [ROUND, ROUND, ROUND, ROUND], thumb: [0.17, 1.02, 0.72] }, orient: PALM_SIDE },
  P: {
    pose: { fingers: [STRAIGHT, { mcp: 0.6, pip: 0.1, dip: 0.05 }, FIST, FIST], thumb: [0.12, 1.3, 0.25] },
    orient: DOWNWARD,
  },
  Q: { pose: { fingers: [STRAIGHT, FIST, FIST, FIST], thumb: [0.4, 1.15, 0.2] }, orient: DOWNWARD },
  R: {
    pose: { fingers: [{ ...STRAIGHT, abd: -0.5 }, { ...STRAIGHT, abd: 0.1 }, FIST, FIST], thumb: [-0.1, 0.75, 0.55] },
    orient: UPRIGHT,
  },
  S: { pose: { fingers: [FIST, FIST, FIST, FIST], thumb: [0.1, 0.7, 0.62] }, orient: UPRIGHT },
  T: { pose: { fingers: [FIST, FIST, FIST, FIST], thumb: [0.15, 0.9, 0.33] }, orient: UPRIGHT },
  U: {
    pose: { fingers: [{ ...STRAIGHT, abd: -0.1 }, STRAIGHT, FIST, FIST], thumb: [-0.1, 0.75, 0.55] },
    orient: UPRIGHT,
  },
  V: {
    pose: { fingers: [{ ...STRAIGHT, abd: 0.15 }, { ...STRAIGHT, abd: -0.2 }, FIST, FIST], thumb: [-0.1, 0.75, 0.55] },
    orient: UPRIGHT,
  },
  W: {
    pose: {
      fingers: [{ ...STRAIGHT, abd: 0.15 }, STRAIGHT, { ...STRAIGHT, abd: -0.2 }, FIST],
      thumb: [-0.3, 0.6, 0.3],
    },
    orient: UPRIGHT,
  },
  X: {
    pose: { fingers: [{ mcp: 0.3, pip: 1.5, dip: 1.0 }, FIST, FIST, FIST], thumb: [0.05, 0.75, 0.55] },
    orient: UPRIGHT,
  },
  Y: { pose: { fingers: [FIST, FIST, FIST, STRAIGHT], thumb: [1.0, 0.5, 0.05] }, orient: UPRIGHT },
};

export const OPEN_HAND: HandPose = {
  fingers: [
    { ...STRAIGHT, abd: 0.2 },
    STRAIGHT,
    { ...STRAIGHT, abd: -0.2 },
    { ...STRAIGHT, abd: -0.35 },
  ],
  thumb: [1.0, 0.55, 0.1],
};

export const RELAXED_HAND: HandPose = {
  fingers: [
    { mcp: 0.4, pip: 0.5, dip: 0.2 },
    { mcp: 0.45, pip: 0.6, dip: 0.25 },
    { mcp: 0.5, pip: 0.65, dip: 0.3 },
    { mcp: 0.55, pip: 0.7, dip: 0.3 },
  ],
  thumb: [0.55, 0.75, 0.3],
};

/** Deterministic pseudo-random numbers for repeatable noise. */
export function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

export function jitter(points: Vec3[], amount: number, rand: () => number): Vec3[] {
  return points.map((p) => ({
    x: p.x + (rand() - 0.5) * 2 * amount,
    y: p.y + (rand() - 0.5) * 2 * amount,
    z: p.z + (rand() - 0.5) * 2 * amount,
  }));
}
