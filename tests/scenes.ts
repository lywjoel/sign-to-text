import { localHand, LETTER_POSES, mirror, toCamera, handPts, type HandPose, type Orientation } from './synth';
import type { FacePts, HandPts, PosePts, Skeleton } from '../src/skeleton';
import { BLEND_KEYS } from '../src/skeleton';

/**
 * Synthetic signer for end-to-end tests: a fixed body and face, with hands
 * placed in body units (shoulder widths; x towards the dominant side, y down
 * from the shoulder line).
 */

export const ASPECT = 16 / 9;
export const BODY = { ox: 0.89, oy: 0.75, sw: 0.38 };
export const FPS = 30;

export function bodyToImg(X: number, Y: number): [number, number] {
  return [BODY.ox - X * BODY.sw, BODY.oy + Y * BODY.sw];
}

const POSE: PosePts = {
  nose: [0.89, 0.42, 1],
  shD: [BODY.ox - BODY.sw / 2, BODY.oy, 1],
  shN: [BODY.ox + BODY.sw / 2, BODY.oy, 1],
  elD: [0.6, 1.0, 0.3],
  elN: [1.18, 1.0, 0.3],
  wrD: [0.6, 1.1, 0.2],
  wrN: [1.18, 1.1, 0.2],
  earD: [0.81, 0.4, 0.9],
  earN: [0.97, 0.4, 0.9],
};

const FACE: FacePts = {
  forehead: [0.89, 0.3],
  nose: [0.89, 0.42],
  mouth: [0.89, 0.47],
  chin: [0.89, 0.52],
  eyeD: [0.85, 0.38],
  eyeN: [0.93, 0.38],
  cheekD: [0.84, 0.45],
  cheekN: [0.94, 0.45],
  edgeD: [0.81, 0.42],
  edgeN: [0.97, 0.42],
};

export interface HandSpec {
  pose: HandPose;
  orient: Orientation;
  /** Wrist position in body units. */
  at: [number, number];
}

export function letterHand(letter: string, at: [number, number]): HandSpec {
  const { pose, orient } = LETTER_POSES[letter];
  return { pose, orient, at };
}

function placeHand(spec: HandSpec, left: boolean): HandPts {
  let world = toCamera(localHand(spec.pose), spec.orient);
  if (left) world = mirror(world);
  const [x, y] = bodyToImg(spec.at[0], spec.at[1]);
  return handPts(world, x, y);
}

export function skeleton(t: number, dom: HandSpec | null, non: HandSpec | null = null, blend?: Partial<Record<(typeof BLEND_KEYS)[number], number>>): Skeleton {
  return {
    t,
    aspect: ASPECT,
    dom: dom ? placeHand(dom, false) : null,
    non: non ? placeHand(non, true) : null,
    pose: POSE,
    face: FACE,
    blend: BLEND_KEYS.map((k) => blend?.[k] ?? 0),
  };
}

/** A clip of `ms` milliseconds; `at(u)` gives the scene at progress u in [0, 1]. */
export interface Clip {
  ms: number;
  at: (u: number) => { dom: HandSpec | null; non?: HandSpec | null };
}

export function hold(ms: number, dom: HandSpec | null, non: HandSpec | null = null): Clip {
  return { ms, at: () => ({ dom, non }) };
}

export function rest(ms: number): Clip {
  return hold(ms, null, null);
}

export function move(ms: number, from: HandSpec, path: (u: number) => [number, number]): Clip {
  return { ms, at: (u) => ({ dom: { ...from, at: path(u) } }) };
}

function lerpPose(a: HandPose, b: HandPose, u: number): HandPose {
  const l = (x: number, y: number) => x + (y - x) * u;
  return {
    fingers: a.fingers.map((f, i) => {
      const g = b.fingers[i];
      return { mcp: l(f.mcp, g.mcp), pip: l(f.pip, g.pip), dip: l(f.dip, g.dip), abd: l(f.abd ?? 0, g.abd ?? 0) };
    }) as HandPose['fingers'],
    thumb: [l(a.thumb[0], b.thumb[0]), l(a.thumb[1], b.thumb[1]), l(a.thumb[2], b.thumb[2])],
  };
}

function lerpOrient(a: Orientation, b: Orientation, u: number): Orientation {
  const l = (p: Orientation['s'], q: Orientation['s']) => ({ x: p.x + (q.x - p.x) * u, y: p.y + (q.y - p.y) * u, z: p.z + (q.z - p.z) * u });
  const norm = (v: Orientation['s']) => {
    const n = Math.hypot(v.x, v.y, v.z) || 1;
    return { x: v.x / n, y: v.y / n, z: v.z / n };
  };
  const s = norm(l(a.s, b.s));
  let w = l(a.u, b.u);
  const d = s.x * w.x + s.y * w.y + s.z * w.z;
  w = norm({ x: w.x - d * s.x, y: w.y - d * s.y, z: w.z - d * s.z });
  const n = { x: s.y * w.z - s.z * w.y, y: s.z * w.x - s.x * w.z, z: s.x * w.y - s.y * w.x };
  return { s, u: w, n };
}

/** Smoothly changes handshape (and orientation) from one hand to another in place. */
export function morph(ms: number, from: HandSpec, to: HandSpec): Clip {
  return {
    ms,
    at: (u) => ({
      dom: {
        pose: lerpPose(from.pose, to.pose, u),
        orient: lerpOrient(from.orient, to.orient, u),
        at: [from.at[0] + (to.at[0] - from.at[0]) * u, from.at[1] + (to.at[1] - from.at[1]) * u],
      },
    }),
  };
}

export function render(clips: Clip[], t0 = 0): Skeleton[] {
  const out: Skeleton[] = [];
  let t = t0;
  const step = 1000 / FPS;
  for (const c of clips) {
    const n = Math.max(1, Math.round(c.ms / step));
    for (let i = 0; i < n; i++) {
      const s = c.at(n === 1 ? 1 : i / (n - 1));
      out.push(skeleton(t, s.dom, s.non ?? null));
      t += step;
    }
  }
  return out;
}

export const lerp2 = (a: [number, number], b: [number, number]) => (u: number): [number, number] => [
  a[0] + (b[0] - a[0]) * u,
  a[1] + (b[1] - a[1]) * u,
];

/**
 * Adds tracking-like noise: independent jitter on every landmark (world hand
 * points in metres, image points in image units) plus a slow drift.
 */
export function noisy(frames: Skeleton[], rand: () => number, worldMm = 3, img = 0.003): Skeleton[] {
  const j = (a: number) => (rand() - 0.5) * 2 * a;
  let driftX = 0;
  let driftY = 0;
  const hand = (h: Skeleton['dom']) =>
    h && {
      img: h.img.map((v, i) => v + j(img) + (i % 3 === 0 ? driftX : i % 3 === 1 ? driftY : 0)),
      world: h.world ? h.world.map((v) => v + j(worldMm / 1000)) : null,
    };
  return frames.map((s) => {
    driftX = driftX * 0.95 + j(0.001);
    driftY = driftY * 0.95 + j(0.001);
    const p2 = (q: [number, number]): [number, number] => [q[0] + j(img), q[1] + j(img)];
    const pv = (q: [number, number, number]): [number, number, number] => [q[0] + j(img), q[1] + j(img), q[2]];
    return {
      ...s,
      dom: hand(s.dom),
      non: hand(s.non),
      face: s.face && {
        forehead: p2(s.face.forehead), nose: p2(s.face.nose), mouth: p2(s.face.mouth), chin: p2(s.face.chin),
        eyeD: p2(s.face.eyeD), eyeN: p2(s.face.eyeN), cheekD: p2(s.face.cheekD), cheekN: p2(s.face.cheekN),
        edgeD: p2(s.face.edgeD), edgeN: p2(s.face.edgeN),
      },
      pose: s.pose && {
        nose: pv(s.pose.nose), shD: pv(s.pose.shD), shN: pv(s.pose.shN), elD: pv(s.pose.elD), elN: pv(s.pose.elN),
        wrD: pv(s.pose.wrD), wrN: pv(s.pose.wrN), earD: pv(s.pose.earD), earN: pv(s.pose.earN),
      },
      blend: s.blend && s.blend.map((b) => Math.min(1, Math.max(0, b + j(0.05)))),
    };
  });
}
