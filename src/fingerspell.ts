import { angle, band, normalize, ramp, sub, type Vec3 } from './geometry';
import { describeHand, resolveChirality, type Chirality, type HandDesc } from './handshape';
import { HAND } from './skeleton';
import type { FeatureFrame } from './features';

/**
 * Built-in recognition of the SgSL one-handed manual alphabet (the same
 * handshapes as the ASL alphabet). Each letter is a set of soft conditions on
 * how far each finger reaches, where the thumb sits relative to the fingers,
 * and which way the hand points. The score is the conditions' weighted
 * geometric mean, so every condition has to hold reasonably well.
 *
 * Thresholds were checked against MediaPipe landmarks from photographs of
 * fingerspelled letters. M, N and T hide the thumb under the fingers, where
 * tracking can only guess, so those are unreliable; letters recorded in Teach
 * mode take priority over the rules and fix this for a given signer.
 */

export const STATIC_LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');
export const ALL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

type Cond = [value: number, weight: number];
type Finger = 1 | 2 | 3 | 4;

export interface LetterScore {
  letter: string;
  score: number;
}

export interface HandGeometry {
  desc: HandDesc;
  /** Index knuckle → tip direction, in the source (camera-aligned) coordinates. */
  indexDir: Vec3;
}

/**
 * Describes a hand for the letter rules. `assumed` is which hand it should be;
 * when the 3D shape clearly says otherwise (the tracker read it the other way
 * round) the shape wins, so finger bends and thumb placement stay meaningful.
 */
export function handGeometry(points: Vec3[], assumed: Chirality): HandGeometry {
  return {
    desc: describeHand(points, resolveChirality(points, assumed)),
    indexDir: normalize(sub(points[HAND.INDEX_TIP], points[HAND.INDEX_MCP])),
  };
}

/**
 * Weighted geometric mean of the conditions, pulled down further by the worst
 * important condition so that one clearly violated requirement rules a letter out.
 */
function combine(conds: Cond[]): number {
  let acc = 0;
  let sw = 0;
  let worst = 1;
  for (const [v, w] of conds) {
    const c = Math.max(v, 1e-3);
    acc += w * Math.log(c);
    sw += w;
    if (w >= 0.5 && c < worst) worst = c;
  }
  return sw > 0 ? Math.exp(acc / sw) * Math.pow(worst, 0.3) : 0;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const MCP_OF = { 1: HAND.INDEX_MCP, 2: HAND.MIDDLE_MCP, 3: HAND.RING_MCP, 4: HAND.PINKY_MCP } as const;
const PIP_OF = { 1: HAND.INDEX_PIP, 2: HAND.MIDDLE_PIP, 3: HAND.RING_PIP, 4: HAND.PINKY_PIP } as const;
const TIP_OF = { 1: HAND.INDEX_TIP, 2: HAND.MIDDLE_TIP, 3: HAND.RING_TIP, 4: HAND.PINKY_TIP } as const;

/** Measurements shared by the letter definitions. All lengths are in palm units. */
function measure(g: HandGeometry) {
  const d = g.desc;
  const L = d.local;
  const reach = (f: Finger) => d.reach[f];
  /** How far the fingertip is past the knuckle row, along the palm. Folded fingers are at or below it. */
  const tipAbove = (f: Finger) => L[TIP_OF[f]].y - L[MCP_OF[f]].y;
  /** How far the middle joint stands past the knuckle row: high for a hooked X finger, low in a fist. */
  const pipAbove = (f: Finger) => L[PIP_OF[f]].y - L[MCP_OF[f]].y;
  /** How far the fingertip is in front of the palm. */
  const tipFront = (f: Finger) => L[TIP_OF[f]].z;
  const tipGap = (a: Finger, b: Finger) => {
    const p = L[TIP_OF[a]];
    const q = L[TIP_OF[b]];
    return Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
  };

  const straight = (f: Finger) => ramp(reach(f), 0.86, 0.93);
  const folded = (f: Finger) =>
    Math.min(ramp(tipAbove(f), 0.32, 0.12), ramp(reach(f), 0.92, 0.82), ramp(pipAbove(f), 0.42, 0.32));
  /** Bent but not folded away: the C and O curve, with the fingertip up or out in front. */
  const curved = (f: Finger) =>
    Math.min(band(reach(f), 0.5, 0.62, 0.9, 0.95), Math.max(ramp(tipAbove(f), 0.0, 0.15), ramp(tipFront(f), 0.4, 0.55)));

  const tt = L[HAND.THUMB_TIP];
  const pipI = L[HAND.INDEX_PIP];
  const pipM = L[HAND.MIDDLE_PIP];
  const mcpI = L[HAND.INDEX_MCP];
  const tipI = L[HAND.INDEX_TIP];
  const pipN = (pipI.z + pipM.z) / 2;
  // How many of index/middle/ring lie on the index side of the thumb tip:
  // ~0 for A, ~1 for T, ~2 for N, ~3 for M.
  const covered = [pipI.x, pipM.x, L[HAND.RING_PIP].x].reduce((c, s) => c + sigmoid((s - tt.x) / 0.05), 0);
  const thumbOut = ramp(tt.x - mcpI.x, 0.35, 0.55);
  const notOut = ramp(tt.x - mcpI.x, 0.45, 0.3);
  /** Thumb tip lower than the middle finger's middle joint (resting on the folded fingers). */
  const thumbLow = ramp(pipM.y - tt.y, -0.05, 0.12);

  const idx = g.indexDir;
  const indexUp = ramp(-idx.y, 0.3, 0.7);
  const handUp = ramp(-d.up.y, 0.2, 0.6);
  const indexDown = ramp(idx.y, 0.15, 0.5);
  const indexSide = Math.min(ramp(Math.abs(idx.x), 0.5, 0.8), ramp(Math.abs(idx.y), 0.65, 0.4));

  return {
    d, L, reach, tipAbove, pipAbove, tipFront, tipGap, straight, folded, curved, tt, pipI, pipM, mcpI, tipI, pipN,
    covered, thumbOut, notOut, thumbLow, idx, indexUp, handUp, indexDown, indexSide,
  };
}

type M = ReturnType<typeof measure>;
interface Def {
  shape: (m: M) => Cond[];
  orient: (m: M) => Cond[];
}

const fist = (m: M, fingers: Finger[], w = 1): Cond[] => fingers.map((f) => [m.folded(f), w]);
const straight = (m: M, fingers: Finger[], w = 1): Cond[] => fingers.map((f) => [m.straight(f), w]);
const distTo = (a: Vec3, b: Vec3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

const DEFS: Record<string, Def> = {
  A: {
    shape: (m) => [
      ...fist(m, [1, 2, 3, 4]),
      [ramp(m.tt.x - m.pipI.x, 0.0, 0.12), 1.5],
      [m.notOut, 1],
      [ramp(m.tt.y - m.pipI.y, -0.15, 0.05), 1],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  B: {
    shape: (m) => [
      // Fully straight: a relaxed, slightly curled open hand is not B.
      ...([1, 2, 3, 4] as Finger[]).map((f): Cond => [ramp(m.reach(f), 0.93, 0.97), 1]),
      [ramp(m.tipGap(1, 2), 0.5, 0.35), 0.6],
      [ramp(m.tipGap(2, 3), 0.5, 0.35), 0.4],
      [m.notOut, 1],
      [ramp(m.mcpI.x - m.tt.x, -0.2, 0.0), 0.7],
      [m.thumbLow, 0.7],
    ],
    orient: (m) => [[m.handUp, 0.7]],
  },
  C: {
    shape: (m) => [
      [m.curved(1), 1.2],
      [m.curved(2), 1.2],
      [m.curved(3), 0.6],
      [m.curved(4), 0.6],
      [band(m.d.thumbTo[0], 0.3, 0.5, 1.6, 2.0), 1.5],
      [ramp(m.tt.x - m.mcpI.x, 0.3, 0.6), 0.7],
    ],
    orient: () => [],
  },
  D: {
    shape: (m) => [
      [m.straight(1), 1.5],
      [ramp(m.reach(2), 0.9, 0.75), 1],
      [ramp(m.reach(3), 0.9, 0.75), 1],
      [ramp(m.reach(4), 0.9, 0.75), 1],
      [ramp(m.d.thumbTo[1], 0.6, 0.3), 1],
    ],
    orient: (m) => [[m.indexUp, 0.7]],
  },
  E: {
    shape: (m) => [
      [ramp(m.reach(1), 0.92, 0.82), 1],
      [ramp(m.reach(2), 0.92, 0.82), 1],
      [ramp(m.reach(3), 0.95, 0.85), 0.5],
      [ramp(m.tipAbove(1), 0.0, 0.15), 0.7],
      [ramp(m.tipAbove(2), 0.0, 0.15), 0.7],
      [ramp(m.pipI.x - m.tt.x, 0.0, 0.15), 0.7],
      [ramp(m.tipI.y - m.tt.y, 0.0, 0.12), 1],
      [ramp(m.d.thumbTo[0], 0.6, 0.35), 1],
      [ramp(m.d.thumbTo[1], 0.7, 0.45), 0.7],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  F: {
    shape: (m) => [
      [ramp(m.d.thumbTo[0], 0.45, 0.25), 1.5],
      [ramp(m.reach(1), 0.95, 0.88), 1],
      ...straight(m, [2, 3, 4]),
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  G: {
    shape: (m) => [[m.straight(1), 1.5], ...fist(m, [2, 3, 4]), [m.notOut, 0.5]],
    orient: (m) => [[m.indexSide, 1.5]],
  },
  H: {
    shape: (m) => [
      [m.straight(1), 1.5],
      [m.straight(2), 1.5],
      [ramp(m.tipGap(1, 2), 0.5, 0.35), 0.7],
      ...fist(m, [3, 4]),
      [m.thumbLow, 0.5],
    ],
    orient: (m) => [[m.indexSide, 1.5]],
  },
  I: {
    shape: (m) => [[m.straight(4), 1.5], ...fist(m, [1, 2, 3]), [m.notOut, 1]],
    orient: (m) => [[m.handUp, 0.5]],
  },
  K: {
    shape: (m) => [
      [m.straight(1), 1.5],
      [ramp(m.reach(2), 0.82, 0.92), 1],
      [ramp(m.tipGap(1, 2), 0.4, 0.55), 0.7],
      ...fist(m, [3, 4]),
      [ramp(distTo(m.tt, m.pipM), 0.55, 0.3), 1],
    ],
    orient: (m) => [[ramp(-m.idx.y, 0.2, 0.6), 0.7]],
  },
  L: {
    shape: (m) => [[m.straight(1), 1.5], ...fist(m, [2, 3, 4]), [m.thumbOut, 1.5]],
    orient: (m) => [[m.indexUp, 0.5]],
  },
  M: {
    shape: (m) => [
      ...fist(m, [1, 2, 3, 4]),
      [ramp(m.covered, 2.3, 2.75), 1.5],
      [ramp(m.pipN - m.tt.z, -0.1, 0.05), 0.7],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  N: {
    shape: (m) => [
      ...fist(m, [1, 2, 3, 4]),
      [band(m.covered, 1.4, 1.8, 2.25, 2.65), 1.5],
      [ramp(m.pipN - m.tt.z, -0.1, 0.05), 0.7],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  O: {
    shape: (m) => [
      [m.curved(1), 1],
      [m.curved(2), 1],
      [ramp(m.d.thumbTo[0], 0.45, 0.25), 1.5],
      [ramp(m.d.thumbTo[1], 0.55, 0.3), 1],
      [ramp(0.2 - Math.abs(m.tipI.y - m.tt.y), 0, 0.1), 0.7],
    ],
    orient: () => [],
  },
  P: {
    shape: (m) => [
      [m.straight(1), 1],
      [ramp(m.reach(2), 0.8, 0.9), 0.7],
      ...fist(m, [3, 4]),
      [ramp(distTo(m.tt, m.pipM), 0.6, 0.35), 0.7],
    ],
    orient: (m) => [[m.indexDown, 1.5]],
  },
  Q: {
    shape: (m) => [[m.straight(1), 1], ...fist(m, [2, 3, 4]), [ramp(m.tt.x - m.mcpI.x, 0.7, 0.4), 0.5]],
    orient: (m) => [[ramp(m.idx.y, 0.3, 0.6), 1.5]],
  },
  R: {
    shape: (m) => [
      [m.straight(1), 1],
      [m.straight(2), 1],
      [ramp(m.L[HAND.MIDDLE_TIP].x - m.tipI.x, 0.0, 0.08), 1.5],
      ...fist(m, [3, 4]),
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  S: {
    shape: (m) => [
      ...fist(m, [1, 2, 3, 4]),
      [ramp(m.tt.z - m.pipN, -0.03, 0.1), 1.5],
      [ramp(m.pipI.x - m.tt.x, -0.02, 0.1), 1],
      [ramp(m.pipI.y - m.tt.y, -0.08, 0.05), 0.7],
      // Fingertips tucked into the palm, not meeting the thumb out in front (O).
      [ramp(m.tipFront(1), 0.6, 0.45), 0.7],
      [ramp(m.d.thumbTo[0], 0.2, 0.35), 0.5],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  T: {
    shape: (m) => [
      ...fist(m, [2, 3, 4]),
      [ramp(m.tipAbove(1), 0.35, 0.15), 1],
      [band(m.covered, 0.35, 0.75, 1.3, 1.7), 1.5],
      [ramp(m.pipN - m.tt.z, -0.1, 0.05), 0.7],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  U: {
    shape: (m) => [
      [m.straight(1), 1.2],
      [m.straight(2), 1.2],
      [ramp(m.tipGap(1, 2), 0.5, 0.35), 1],
      [ramp(m.tipI.x - m.L[HAND.MIDDLE_TIP].x, -0.02, 0.06), 0.7],
      ...fist(m, [3, 4]),
      [m.thumbLow, 0.7],
    ],
    orient: (m) => [[ramp(-m.idx.y, 0.4, 0.75), 1]],
  },
  V: {
    shape: (m) => [
      [m.straight(1), 1.2],
      [m.straight(2), 1.2],
      [ramp(m.tipGap(1, 2), 0.4, 0.55), 1.5],
      [ramp(m.tipI.x - m.L[HAND.MIDDLE_TIP].x, -0.02, 0.06), 0.7],
      ...fist(m, [3, 4]),
      [m.thumbLow, 0.7],
      // K differs by the thumb resting against the middle finger.
      [ramp(distTo(m.tt, m.pipM), 0.2, 0.35), 0.7],
    ],
    orient: (m) => [[m.handUp, 0.7]],
  },
  W: {
    shape: (m) => [
      ...straight(m, [1, 2, 3]),
      [m.folded(4), 1],
      [ramp(m.tipGap(1, 2), 0.3, 0.45), 0.5],
      [ramp(m.tipGap(2, 3), 0.3, 0.45), 0.5],
      [ramp(m.d.thumbTo[3], 0.8, 0.45), 0.5],
    ],
    orient: (m) => [[m.handUp, 0.5]],
  },
  X: {
    shape: (m) => [
      [band(m.reach(1), 0.45, 0.55, 0.8, 0.88), 1.5],
      [ramp(m.tipAbove(1), 0.0, 0.2), 1],
      [ramp(m.pipAbove(1), 0.28, 0.38), 1],
      [ramp(m.d.thumbTo[0], 0.25, 0.45), 0.7],
      ...fist(m, [2, 3, 4]),
    ],
    orient: (m) => [[m.handUp, 0.3]],
  },
  Y: {
    shape: (m) => [[m.straight(4), 1.5], ...fist(m, [1, 2, 3]), [m.thumbOut, 1.5]],
    orient: () => [],
  },
};

function scoreLetter(letter: string, g: HandGeometry, withOrientation: boolean): number {
  const def = DEFS[letter];
  const m = measure(g);
  const conds = def.shape(m);
  if (withOrientation) conds.push(...def.orient(m));
  return combine(conds);
}

/** Individual condition values for one letter (for debugging and tests). */
export function explainLetter(letter: string, g: HandGeometry): { shape: number[]; orient: number[] } {
  const m = measure(g);
  return { shape: DEFS[letter].shape(m).map((c) => c[0]), orient: DEFS[letter].orient(m).map((c) => c[0]) };
}

/** Scores all static letters, best first. */
export function scoreLetters(g: HandGeometry, exclude: ReadonlySet<string> = new Set()): LetterScore[] {
  const out: LetterScore[] = [];
  for (const letter of STATIC_LETTERS) {
    if (exclude.has(letter)) continue;
    out.push({ letter, score: scoreLetter(letter, g, true) });
  }
  return out.sort((a, b) => b.score - a.score);
}

export const RULE_MIN_SCORE = 0.55;
export const RULE_MIN_MARGIN = 0.04;

/** The best letter if it is confident and clearly ahead of the runner-up. */
export function pickLetter(scores: LetterScore[], minScore = RULE_MIN_SCORE): LetterScore | null {
  const [best, second] = scores;
  if (!best || best.score < minScore) return null;
  if (second && best.score - second.score < RULE_MIN_MARGIN) return null;
  return best;
}

// ---------------------------------------------------------------------------
// Moving letters

export interface MotionLetter {
  letter: 'J' | 'Z';
  score: number;
}

function pathLength(pts: [number, number][]): number {
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return s;
}

/** Number of horizontal direction reversals larger than `minSwing` (body units). */
function horizontalReversals(xs: number[], minSwing: number): number {
  let reversals = 0;
  let dir = 0;
  let extreme = xs[0];
  for (const x of xs) {
    if (dir >= 0 && x > extreme) extreme = x;
    else if (dir <= 0 && x < extreme) extreme = x;
    if (dir >= 0 && extreme - x > minSwing) {
      if (dir > 0) reversals++;
      dir = -1;
      extreme = x;
    } else if (dir <= 0 && x - extreme > minSwing) {
      if (dir < 0) reversals++;
      dir = 1;
      extreme = x;
    }
  }
  return reversals;
}

/**
 * Detects J (I-hand, little finger draws a hook downwards) and Z (index draws a Z)
 * in a stretch of movement.
 */
export function detectMotionLetter(frames: FeatureFrame[]): MotionLetter | null {
  const usable = frames.filter((f) => f.domDesc && f.domIndexTip && f.domPinkyTip);
  if (usable.length < 5) return null;

  let iShape = 0;
  let indexShape = 0;
  for (const f of usable) {
    const g: HandGeometry = { desc: f.domDesc!, indexDir: { x: 0, y: -1, z: 0 } };
    if (scoreLetter('I', g, false) > 0.5) iShape++;
    const m = measure(g);
    const pointOnly = combine([[m.straight(1), 1.5], ...fist(m, [2, 3, 4])]);
    if (pointOnly > 0.5) indexShape++;
  }
  const n = usable.length;

  if (iShape / n >= 0.6) {
    const tips = usable.map((f) => f.domPinkyTip!);
    const len = pathLength(tips);
    const drop = tips[tips.length - 1][1] - tips[0][1];
    const first = usable[0].domDesc!.normal;
    const last = usable[n - 1].domDesc!.normal;
    const twist = angle(first, last);
    const score = Math.min(ramp(len, 0.08, 0.2), Math.max(ramp(drop, 0.02, 0.1), ramp(twist, 0.4, 0.9)));
    if (score > 0.5) return { letter: 'J', score };
  }

  if (indexShape / n >= 0.6) {
    const tips = usable.map((f) => f.domIndexTip!);
    const len = pathLength(tips);
    const reversals = horizontalReversals(tips.map((p) => p[0]), 0.06);
    const drop = tips[tips.length - 1][1] - tips[0][1];
    const score = Math.min(ramp(len, 0.25, 0.45), ramp(reversals, 1, 2), ramp(drop, 0.0, 0.08));
    if (score > 0.5) return { letter: 'Z', score };
  }
  return null;
}
