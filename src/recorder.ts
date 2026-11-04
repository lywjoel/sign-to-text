import type { FeatureFrame } from './features';
import { analyseMotion, type SampleKind } from './library';
import { Segmenter } from './segmenter';
import type { Skeleton } from './skeleton';

export type RecorderPhase = 'idle' | 'countdown' | 'capturing' | 'review';

export interface Recording {
  kind: SampleKind;
  label: string;
  frames: Skeleton[];
  feats: FeatureFrame[];
  dynamic: boolean;
  parts: number;
  durationMs: number;
}

export interface RecFrame {
  t: number;
  /** Segmenter motion score (below 1 still, above 1.3 moving). */
  motion: number;
  /** A hand is in the signing space. */
  active: boolean;
  /** Vertical position of the active hand (body units, down positive), or NaN. */
  y: number;
}

const COUNTDOWN_MS = 3000;
const MAX_MS = 6000;
const END_STILL_MS = 700;
const STATIC_LETTER_MS = 700;
const STATIC_SIGN_MS = 1100;
const PAD_MS = 250;
const STATIC_KEEP_MS = 900;

/**
 * Picks the part of a take that is the sign itself: drops the hands rising
 * into position at the start and dropping to rest at the end, keeping a short
 * pad of the start and end holds. Returns inclusive indices, or null if no
 * hand was seen.
 */
export function trimRecording(frames: RecFrame[]): { start: number; end: number; moving: boolean } | null {
  const n = frames.length;
  let a0 = -1;
  let a1 = -1;
  for (let i = 0; i < n; i++) {
    if (frames[i].active) {
      if (a0 < 0) a0 = i;
      a1 = i;
    }
  }
  if (a0 < 0) return null;

  const vy = frames.map((_, i) => {
    const a = frames[Math.max(a0, i - 2)];
    const b = frames[Math.min(a1, i + 2)];
    const dt = (b.t - a.t) / 1000;
    return dt > 0 && Number.isFinite(a.y) && Number.isFinite(b.y) ? (b.y - a.y) / dt : 0;
  });
  const indexAtOrAfter = (t: number) => {
    for (let i = a0; i <= a1; i++) if (frames[i].t >= t) return i;
    return a1;
  };
  const indexAtOrBefore = (t: number) => {
    for (let i = a1; i >= a0; i--) if (frames[i].t <= t) return i;
    return a0;
  };
  const staticResult = () => ({ start: indexAtOrAfter(frames[a1].t - STATIC_KEEP_MS), end: a1, moving: false });

  // If the hands came up from rest during the take, skip that entrance.
  let from = a0;
  const startedAtRest = a0 > 0 && frames[a0].t - frames[0].t >= 100;
  if (startedAtRest) {
    let i = indexAtOrAfter(frames[a0].t + 150);
    while (i < a1 && frames[i].motion >= 1.3) i++;
    from = i;
  }

  let firstMove = -1;
  for (let i = from; i <= a1; i++) {
    if (frames[i].motion > 1.3) {
      firstMove = i;
      break;
    }
  }
  if (firstMove < 0) return staticResult();

  let lastMove = firstMove;
  for (let i = a1; i >= firstMove; i--) {
    if (frames[i].motion > 1.0) {
      lastMove = i;
      break;
    }
  }
  // If the take ended with the hands dropping to rest, cut the drop.
  const endedAtRest = a1 < n - 1;
  if (endedAtRest) {
    while (lastMove > firstMove && vy[lastMove] > 0.6) lastMove--;
  }
  if (lastMove <= firstMove) return staticResult();

  const start = Math.max(from, indexAtOrAfter(frames[firstMove].t - PAD_MS));
  const end = Math.min(a1, indexAtOrBefore(frames[lastMove].t + PAD_MS));
  return { start, end, moving: true };
}

export class Recorder {
  phase: RecorderPhase = 'idle';
  kind: SampleKind = 'sign';
  label = '';
  result: Recording | null = null;
  error: string | null = null;
  countdownLeft = 0;
  elapsed = 0;
  holdMs = 320;
  onChange: (() => void) | null = null;

  private countdownEnd = 0;
  private startAt = 0;
  private take: { skel: Skeleton; feat: FeatureFrame; rec: RecFrame }[] = [];
  private seg = new Segmenter();
  private moved = false;
  private moveRun = 0;
  private stillSince: number | null = null;
  private restSeen = false;

  start(kind: SampleKind, label: string, now: number): void {
    this.kind = kind;
    this.label = label;
    this.result = null;
    this.error = null;
    this.phase = 'countdown';
    this.countdownEnd = now + COUNTDOWN_MS;
    this.countdownLeft = COUNTDOWN_MS;
    this.changed();
  }

  /** Abandons a countdown or capture, or dismisses a finished take. */
  cancel(): void {
    this.phase = 'idle';
    this.take = [];
    this.result = null;
    this.error = null;
    this.changed();
  }

  /** Ends the capture now (the "Stop" button). */
  stop(): void {
    if (this.phase === 'capturing') this.finish();
  }

  feed(skel: Skeleton, feat: FeatureFrame): void {
    const now = feat.t;
    if (this.phase === 'countdown') {
      this.countdownLeft = Math.max(0, this.countdownEnd - now);
      if (now < this.countdownEnd) return;
      this.phase = 'capturing';
      this.startAt = now;
      this.take = [];
      this.seg = new Segmenter({ holdMs: this.holdMs });
      this.moved = false;
      this.moveRun = 0;
      this.stillSince = null;
      this.restSeen = false;
      this.changed();
    }
    if (this.phase !== 'capturing') return;

    const events = this.seg.update(feat);
    const st = this.seg.state;
    const c = feat.domC ?? feat.nonC;
    this.take.push({ skel, feat, rec: { t: now, motion: st.motion, active: st.active, y: c ? c[1] : NaN } });

    if (st.active && st.motion > 1.3) {
      if (++this.moveRun >= 3) this.moved = true;
    } else {
      this.moveRun = 0;
    }
    // Hysteresis: small wobbles between 1 and 1.3 neither start nor end stillness.
    if (!st.active || st.motion > 1.3) this.stillSince = null;
    else if (st.motion < 1) this.stillSince ??= now;
    if (events.some((e) => e.type === 'rest')) this.restSeen = true;

    this.elapsed = now - this.startAt;
    const stillFor = this.stillSince === null ? 0 : now - this.stillSince;
    const staticMs = this.kind === 'letter' ? STATIC_LETTER_MS : STATIC_SIGN_MS;
    if (
      this.elapsed >= MAX_MS ||
      (this.moved && this.restSeen) ||
      (this.moved && stillFor >= END_STILL_MS) ||
      (!this.moved && stillFor >= staticMs && this.elapsed >= 600)
    ) {
      this.finish();
    }
  }

  private finish(): void {
    const cut = trimRecording(this.take.map((x) => x.rec));
    if (!cut || cut.end - cut.start < 4) {
      this.error = 'No hands were seen. Keep both hands in view and try again.';
      this.phase = 'idle';
      this.take = [];
      this.changed();
      return;
    }
    const part = this.take.slice(cut.start, cut.end + 1);
    const feats = part.map((x) => x.feat);
    const motion = cut.moving ? analyseMotion(feats, this.holdMs) : { dynamic: false, parts: 1 };
    this.result = {
      kind: this.kind,
      label: this.label,
      frames: part.map((x) => x.skel),
      feats,
      dynamic: motion.dynamic,
      parts: motion.parts,
      durationMs: feats[feats.length - 1].t - feats[0].t,
    };
    this.take = [];
    this.phase = 'review';
    this.changed();
  }

  private changed(): void {
    this.onChange?.();
  }

  /** Motion state of the take in progress (for the status rail). */
  get segState() {
    return this.seg.state;
  }
}
