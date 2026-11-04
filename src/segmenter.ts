import { GROUPS, type FeatureFrame } from './features';

export interface SegmenterOptions {
  /** Stillness needed before a pause counts as a hold, ms. */
  holdMs: number;
  /** Time without active hands before declaring rest, ms. */
  restMs: number;
  /** A hand centre lower than this (body units below the shoulder line) is resting. */
  restY: number;
  /** Hand speed (shoulder widths / s) that counts as one unit of motion. */
  speedThreshold: number;
  /** Handshape change rate (feature units / s) that counts as one unit of motion. */
  shapeThreshold: number;
  /** Look-back window for measuring motion, ms. */
  windowMs: number;
  /** Very long motions are split so matching stays bounded, ms. */
  maxMotionMs: number;
}

export const DEFAULT_SEGMENTER: SegmenterOptions = {
  holdMs: 320,
  restMs: 220,
  restY: 1.7,
  speedThreshold: 0.25,
  shapeThreshold: 2.5,
  windowMs: 180,
  maxMotionMs: 6000,
};

export type Phase = 'rest' | 'moving' | 'still' | 'hold';

export type SegEvent =
  | { type: 'active'; t: number }
  /** A stretch of movement between pauses, rests or both. */
  | { type: 'motion'; t0: number; t1: number }
  /** A pause has lasted long enough to count as a hold. Stillness began at t0. */
  | { type: 'hold'; t0: number; t: number }
  | { type: 'holdEnd'; t0: number; t1: number }
  | { type: 'rest'; t: number; lastActive: number };

export interface SegState {
  phase: Phase;
  /** Motion score: below 1 is still, above ~1.3 is moving. */
  motion: number;
  speed: number;
  shapeRate: number;
  /** 0..1 progress towards confirming a hold. */
  holdProgress: number;
  active: boolean;
}

const STILL = 1;
const MOVE = 1.3;

export function isActiveHand(c: [number, number] | null, imgY: number, restY: number): boolean {
  return c !== null && c[1] < restY && !(imgY > 0.97);
}

export class Segmenter {
  opts: SegmenterOptions;
  state: SegState = { phase: 'rest', motion: 0, speed: 0, shapeRate: 0, holdProgress: 0, active: false };

  private hist: Smoothed[] = [];
  private activeRun = 0;
  private motionStart = 0;
  private stillStart = 0;
  private holdStart = 0;
  private inactiveSince: number | null = null;
  private lastActive = 0;

  constructor(opts: Partial<SegmenterOptions> = {}) {
    this.opts = { ...DEFAULT_SEGMENTER, ...opts };
  }

  reset(): void {
    this.hist = [];
    this.activeRun = 0;
    this.inactiveSince = null;
    this.state = { phase: 'rest', motion: 0, speed: 0, shapeRate: 0, holdProgress: 0, active: false };
  }

  get phase(): Phase {
    return this.state.phase;
  }

  update(f: FeatureFrame): SegEvent[] {
    const o = this.opts;
    const events: SegEvent[] = [];
    const now = f.t;

    this.hist.push(this.smooth(f));
    while (this.hist.length > 2 && this.hist[1].t <= now - o.windowMs) this.hist.shift();

    const active =
      isActiveHand(f.domC, f.domImgY, o.restY) || isActiveHand(f.nonC, f.nonImgY, o.restY);
    const { score, speed, shapeRate } = this.measure(f);
    this.state.motion = score;
    this.state.speed = speed;
    this.state.shapeRate = shapeRate;
    this.state.active = active;

    if (this.state.phase === 'rest') {
      this.state.holdProgress = 0;
      if (!active) {
        this.activeRun = 0;
        return events;
      }
      this.activeRun++;
      if (this.activeRun === 1) this.motionStart = now;
      if (this.activeRun >= 2) {
        this.state.phase = 'moving';
        this.inactiveSince = null;
        this.lastActive = now;
        events.push({ type: 'active', t: this.motionStart });
      }
      return events;
    }

    if (!active) {
      if (this.inactiveSince === null) this.inactiveSince = now;
      if (now - this.inactiveSince >= o.restMs) {
        if (this.state.phase === 'hold') {
          events.push({ type: 'holdEnd', t0: this.holdStart, t1: this.lastActive });
        } else {
          events.push({ type: 'motion', t0: this.motionStart, t1: this.lastActive });
        }
        events.push({ type: 'rest', t: now, lastActive: this.lastActive });
        this.state.phase = 'rest';
        this.state.holdProgress = 0;
        this.activeRun = 0;
      }
      return events;
    }
    this.inactiveSince = null;
    this.lastActive = now;

    const lag = o.windowMs / 2;
    switch (this.state.phase) {
      case 'moving':
        if (score < STILL) {
          this.state.phase = 'still';
          this.stillStart = Math.max(this.motionStart, now - lag);
        } else if (now - this.motionStart > o.maxMotionMs) {
          events.push({ type: 'motion', t0: this.motionStart, t1: now });
          this.motionStart = now;
        }
        break;
      case 'still':
        if (score > MOVE) {
          this.state.phase = 'moving';
        } else if (now - this.stillStart >= o.holdMs) {
          events.push({ type: 'motion', t0: this.motionStart, t1: this.stillStart });
          events.push({ type: 'hold', t0: this.stillStart, t: now });
          this.holdStart = this.stillStart;
          this.state.phase = 'hold';
        }
        break;
      case 'hold':
        if (score > MOVE) {
          const t1 = Math.max(this.holdStart, now - lag);
          events.push({ type: 'holdEnd', t0: this.holdStart, t1 });
          this.motionStart = t1;
          this.state.phase = 'moving';
        }
        break;
    }

    this.state.holdProgress =
      this.state.phase === 'hold' ? 1 : this.state.phase === 'still' ? Math.min(1, (now - this.stillStart) / o.holdMs) : 0;
    return events;
  }

  /** Closes whatever is in progress, e.g. at the end of a recording. */
  finish(t: number): SegEvent[] {
    const events: SegEvent[] = [];
    if (this.state.phase === 'moving') events.push({ type: 'motion', t0: this.motionStart, t1: t });
    else if (this.state.phase === 'still') events.push({ type: 'motion', t0: this.motionStart, t1: this.stillStart });
    else if (this.state.phase === 'hold') events.push({ type: 'holdEnd', t0: this.holdStart, t1: t });
    this.state.phase = 'rest';
    this.activeRun = 0;
    return events;
  }

  private measure(f: FeatureFrame): { score: number; speed: number; shapeRate: number } {
    const ref = this.hist[0];
    const cur = this.hist[this.hist.length - 1];
    if (!ref || ref === cur) return { score: 0, speed: 0, shapeRate: 0 };
    const dt = Math.max(1e-3, (cur.t - ref.t) / 1000);
    let speed = 0;
    let shapeRate = 0;
    let appeared = false;
    for (const side of ['dom', 'non'] as const) {
      const c = cur[side];
      const r = ref[side];
      if (c && r) {
        speed = Math.max(speed, Math.hypot(c.c[0] - r.c[0], c.c[1] - r.c[1]) / dt);
        let sq = 0;
        for (let i = 0; i < c.shape.length; i++) sq += (c.shape[i] - r.shape[i]) ** 2;
        shapeRate = Math.max(shapeRate, Math.sqrt(sq / c.shape.length) / dt);
      } else if (c || r) {
        // A hand entering or leaving the signing space counts as movement;
        // one flickering at the bottom edge does not.
        const present = c ? f : ref.frame;
        const pc = side === 'dom' ? present.domC : present.nonC;
        const py = side === 'dom' ? present.domImgY : present.nonImgY;
        if (isActiveHand(pc, py, this.opts.restY)) appeared = true;
      }
    }
    const o = this.opts;
    let score = Math.max(speed / o.speedThreshold, shapeRate / o.shapeThreshold);
    if (appeared) score = Math.max(score, 2);
    return { score, speed, shapeRate };
  }

  /** Light temporal smoothing of hand position and handshape, so tracking jitter doesn't read as movement. */
  private smooth(f: FeatureFrame): Smoothed {
    const prev = this.hist[this.hist.length - 1];
    const dt = prev ? Math.max(0, f.t - prev.t) : 0;
    const a = prev ? 1 - Math.exp(-dt / SMOOTH_MS) : 1;
    const side = (key: 'dom' | 'non'): SmoothHand | null => {
      const c = key === 'dom' ? f.domC : f.nonC;
      if (!c) return null;
      const [s0, s1] = GROUPS[key === 'dom' ? 'domShape' : 'nonShape'];
      const shape = f.v.slice(s0, s1);
      const p = prev?.[key];
      if (!p) return { c: [c[0], c[1]], shape };
      for (let i = 0; i < shape.length; i++) shape[i] = p.shape[i] + a * (shape[i] - p.shape[i]);
      return { c: [p.c[0] + a * (c[0] - p.c[0]), p.c[1] + a * (c[1] - p.c[1])], shape };
    };
    return { t: f.t, frame: f, dom: side('dom'), non: side('non') };
  }
}

interface SmoothHand {
  c: [number, number];
  shape: Float32Array;
}

interface Smoothed {
  t: number;
  frame: FeatureFrame;
  dom: SmoothHand | null;
  non: SmoothHand | null;
}

const SMOOTH_MS = 50;
