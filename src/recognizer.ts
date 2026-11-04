import { subsequenceDtw } from './dtw';
import {
  frameDistance,
  LETTER_WEIGHTS,
  meanFrame,
  resample,
  shapeChange,
  signWeights,
  travel,
  type CoreFrame,
  type FeatureFrame,
  type Weights,
} from './features';
import { detectMotionLetter, handGeometry, pickLetter, scoreLetters } from './fingerspell';
import { averagePoints, handPoints, imagePoints } from './handshape';
import { MATCH_HZ, type SampleKind, type Template } from './library';
import { isActiveHand, Segmenter, type SegEvent, type SegState } from './segmenter';
import type { HandPts, Skeleton } from './skeleton';
import type { Transcriber } from './transcriber';

export type Mode = 'auto' | 'fingerspell' | 'signs';

export interface RecognizerConfig {
  mode: Mode;
  /** Mean frame distance (DTW) at or below which a moving sign is accepted. */
  signThreshold: number;
  /** Distance at or below which a taught letter or held sign is accepted. */
  staticThreshold: number;
  /** Minimum score for a built-in alphabet letter. */
  ruleMinScore: number;
  builtinAlphabet: boolean;
  useFace: boolean;
  useBody: boolean;
  holdMs: number;
  /** A rest longer than this starts a new line, ms. */
  sentenceMs: number;
}

export const DEFAULT_CONFIG: RecognizerConfig = {
  mode: 'auto',
  signThreshold: 1.2,
  staticThreshold: 0.9,
  ruleMinScore: 0.55,
  builtinAlphabet: true,
  useFace: true,
  useBody: true,
  holdMs: 320,
  sentenceMs: 4000,
};

export interface Candidate {
  label: string;
  kind: SampleKind;
  /** 0..1. Taught samples score 0.5 exactly at the acceptance threshold. */
  score: number;
  /** Matching distance, for taught samples. */
  cost?: number;
  source: 'taught' | 'built-in';
}

export interface Decision {
  t: number;
  via: 'hold' | 'motion';
  accepted: Candidate | null;
  candidates: Candidate[];
}

export interface Snapshot {
  seg: SegState;
  /** Best guess for the handshape being held right now. */
  guess: Candidate | null;
  /** Whether the guess would be accepted if the hold were confirmed now. */
  guessAccepted: boolean;
  guessCandidates: Candidate[];
  last: Decision | null;
}

interface Entry {
  skel: Skeleton;
  feat: FeatureFrame;
}

interface SegRecord {
  type: 'hold' | 'motion' | 'rest';
  t0: number;
  t1: number;
  tokens: number[];
  /** First movement after the hands came up from rest (it starts with the hands rising into place). */
  first?: boolean;
}

const BUFFER_MS = 12000;
const PRE_PAD_MS = 250;
const POST_PAD_MS = 300;
const GUESS_EVERY_MS = 150;
const MAX_PARTS = 3;
/** Below this much hand travel (body units) and handshape change, a movement is ignored. */
const MIN_TRAVEL = 0.1;
const MIN_SHAPE_CHANGE = 1.0;
const SUPPRESSED = '\u0000suppressed';

/** Moving letters (J, Z, or taught ones) depend on the dominant hand only. */
const MOTION_LETTER_WEIGHTS: Weights = { ...LETTER_WEIGHTS, domLoc: 0.6 };

export function costToScore(cost: number, threshold: number): number {
  const r = cost / Math.max(threshold, 1e-6);
  return Math.exp(-Math.LN2 * r * r);
}

const labelKey = (t: { kind: SampleKind; label: string }) => `${t.kind}:${t.label}`;

/**
 * DTW cost of a moving template against a query, with an extra penalty when
 * the matched stretch moves much more or much less than the template does
 * (e.g. raising the hands into position is not the end of a sign).
 */
export function matchMoving(t: Template, query: CoreFrame[], w: Weights): { cost: number; start: number; end: number } {
  const m = subsequenceDtw(t.seq, query, (a, b) => frameDistance(a, b, w), { stepPenalty: 0.2 });
  if (!Number.isFinite(m.cost)) return m;
  const moved = travel(query, m.start, m.end);
  const ratio = (moved + 0.05) / (t.travel + 0.05);
  const excess = Math.max(0, Math.abs(Math.log2(ratio)) - 0.6);
  return { ...m, cost: m.cost * (1 + 0.6 * excess) };
}

export class Recognizer {
  config: RecognizerConfig;
  readonly segmenter: Segmenter;
  guess: Candidate | null = null;
  guessAccepted = false;
  guessCandidates: Candidate[] = [];
  last: Decision | null = null;
  onDecision: ((d: Decision) => void) | null = null;

  private buffer: Entry[] = [];
  private records: SegRecord[] = [];
  private suppressNextHold = false;
  private afterActive = false;
  /**
   * What the current hold was read as. A hold whose reading was suppressed
   * (the end of a sign) starts as SUPPRESSED and adopts its first reading silently.
   */
  private holdLabel: string | null = null;
  private pending: { label: string; since: number } | null = null;
  private restAt: number | null = null;
  private newlineDone = true;
  private lastGuessAt = -Infinity;

  constructor(
    private readonly transcriber: Transcriber,
    private readonly templates: () => Template[],
    private readonly taughtLetters: () => Set<string>,
    config: Partial<RecognizerConfig> = {},
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.segmenter = new Segmenter({ holdMs: this.config.holdMs });
  }

  configure(c: Partial<RecognizerConfig>): void {
    Object.assign(this.config, c);
    this.segmenter.opts.holdMs = this.config.holdMs;
  }

  reset(): void {
    this.buffer = [];
    this.records = [];
    this.segmenter.reset();
    this.suppressNextHold = false;
    this.guess = null;
  }

  push(skel: Skeleton, feat: FeatureFrame): Snapshot {
    const now = feat.t;
    this.buffer.push({ skel, feat });
    while (this.buffer.length && this.buffer[0].feat.t < now - BUFFER_MS) this.buffer.shift();
    while (this.records.length && this.records[0].t1 < now - BUFFER_MS) this.records.shift();

    for (const e of this.segmenter.update(feat)) this.handle(e, now);

    if (this.segmenter.phase === 'rest' && this.restAt !== null && !this.newlineDone && now - this.restAt >= this.config.sentenceMs) {
      this.transcriber.newline();
      this.newlineDone = true;
    }
    this.updateGuess(now);
    this.watchHold(now);
    this.transcriber.settle(now);
    return {
      seg: { ...this.segmenter.state },
      guess: this.guess,
      guessAccepted: this.guessAccepted,
      guessCandidates: this.guessCandidates,
      last: this.last,
    };
  }

  private handle(e: SegEvent, now: number): void {
    switch (e.type) {
      case 'active':
        this.restAt = null;
        this.afterActive = true;
        break;
      case 'motion': {
        const rec: SegRecord = { type: 'motion', t0: e.t0, t1: e.t1, tokens: [], first: this.afterActive };
        this.afterActive = false;
        this.records.push(rec);
        this.onMotion(rec, now);
        break;
      }
      case 'hold': {
        const rec: SegRecord = { type: 'hold', t0: e.t0, t1: e.t, tokens: [] };
        this.records.push(rec);
        this.pending = null;
        if (this.suppressNextHold) {
          this.suppressNextHold = false;
          this.holdLabel = SUPPRESSED;
        } else {
          this.holdLabel = this.onHold(rec, now);
        }
        break;
      }
      case 'holdEnd': {
        const rec = this.lastRecord('hold');
        if (rec) rec.t1 = e.t1;
        this.holdLabel = null;
        this.pending = null;
        break;
      }
      case 'rest':
        this.records.push({ type: 'rest', t0: e.t, t1: e.t, tokens: [] });
        this.suppressNextHold = false;
        this.restAt = e.t;
        this.newlineDone = false;
        this.transcriber.space();
        this.transcriber.settleAll();
        break;
    }
  }

  private lastRecord(type: SegRecord['type']): SegRecord | null {
    for (let i = this.records.length - 1; i >= 0; i--) if (this.records[i].type === type) return this.records[i];
    return null;
  }

  private range(t0: number, t1: number): Entry[] {
    return this.buffer.filter((e) => e.feat.t >= t0 && e.feat.t <= t1);
  }

  private get allowLetters(): boolean {
    return this.config.mode !== 'signs';
  }

  private get allowSigns(): boolean {
    return this.config.mode !== 'fingerspell';
  }

  private emit(d: Decision): void {
    this.last = d;
    this.onDecision?.(d);
  }

  // -------------------------------------------------------------------------
  // Per-sign thresholds
  //
  // How far a match may be from a sign's examples depends on how consistently
  // the signer made those examples, so each label with two or more examples
  // gets a threshold scaled from the typical distance between them.

  private intra = new Map<string, number>();
  /** Labels whose examples are mostly moving (true) or mostly held (false); the minority type is ignored. */
  private moving = new Map<string, boolean>();
  private intraKey = '';

  private ensureIntra(): void {
    const all = this.templates();
    const c = this.config;
    const key = `${all.length}|${all[0]?.sampleId}|${all.at(-1)?.sampleId}|${c.useFace}|${c.useBody}`;
    if (key === this.intraKey) return;
    this.intraKey = key;
    this.intra.clear();
    this.moving.clear();
    const groups = new Map<string, Template[]>();
    for (const t of all) {
      const k = labelKey(t);
      groups.set(k, [...(groups.get(k) ?? []), t]);
    }
    const signW = signWeights(c);
    for (const [k, ts] of groups) {
      const dynamicCount = ts.filter((t) => t.dynamic).length;
      if (dynamicCount * 2 !== ts.length) this.moving.set(k, dynamicCount * 2 > ts.length);
      const recent = ts.slice(-6);
      if (recent.length < 2) continue;
      const costs: number[] = [];
      for (const a of recent) {
        for (const b of recent) {
          if (a === b || a.dynamic !== b.dynamic) continue;
          if (a.dynamic) costs.push(matchMoving(a, b.seq, a.kind === 'letter' ? MOTION_LETTER_WEIGHTS : signW).cost);
          else costs.push(frameDistance(a.mean, b.mean, a.kind === 'letter' ? LETTER_WEIGHTS : signW));
        }
      }
      if (!costs.length) continue;
      costs.sort((x, y) => x - y);
      this.intra.set(k, costs[Math.floor(costs.length / 2)]);
    }
  }

  /** Acceptance threshold for a template's label. */
  thresholdFor(t: Template): number {
    this.ensureIntra();
    const c = this.config;
    const base = !t.dynamic && t.kind === 'letter' ? c.staticThreshold : c.signThreshold;
    const intra = this.intra.get(labelKey(t));
    if (intra === undefined) return base;
    const sensitivity = c.signThreshold / DEFAULT_CONFIG.signThreshold;
    return Math.min(1.6 * base, Math.max(0.7 * base, 1.8 * intra * sensitivity));
  }

  /** Templates of the given type, leaving out examples that disagree with the rest of their label. */
  private usable(dynamic: boolean): Template[] {
    this.ensureIntra();
    return this.templates().filter((t) => {
      if (t.dynamic !== dynamic) return false;
      const mostly = this.moving.get(labelKey(t));
      return mostly === undefined || mostly === dynamic;
    });
  }

  // -------------------------------------------------------------------------
  // Held handshapes: letters and static signs

  private onHold(rec: SegRecord, now: number): string | null {
    const { candidates, accepted } = this.classifyStatic(this.range(rec.t0, now));
    if (!candidates.length) return null;
    if (accepted) {
      const id = this.transcriber.add({
        kind: accepted.kind,
        text: accepted.label,
        t0: rec.t0,
        t1: now,
        provisional: this.config.mode === 'auto',
      });
      rec.tokens.push(id);
    }
    this.emit({ t: now, via: 'hold', accepted, candidates });
    return accepted?.label ?? null;
  }

  /**
   * Some letters differ so little (M/N, U/V) that changing between them in
   * place doesn't register as movement. While holding, a different confident
   * reading that stays put for a pause length counts as a new letter.
   */
  private watchHold(now: number): void {
    if (this.segmenter.phase !== 'hold') return;
    const g = this.guessAccepted ? this.guess : null;
    if (!g) {
      this.pending = null;
      return;
    }
    if (this.holdLabel === SUPPRESSED) {
      this.holdLabel = g.label;
      return;
    }
    if (g.label === this.holdLabel) {
      this.pending = null;
      return;
    }
    if (!this.pending || this.pending.label !== g.label) {
      this.pending = { label: g.label, since: now };
      return;
    }
    if (now - this.pending.since < this.config.holdMs) return;
    const rec: SegRecord = { type: 'hold', t0: this.pending.since, t1: now, tokens: [] };
    this.records.push(rec);
    rec.tokens.push(
      this.transcriber.add({ kind: g.kind, text: g.label, t0: rec.t0, t1: now, provisional: this.config.mode === 'auto' }),
    );
    this.emit({ t: now, via: 'hold', accepted: g, candidates: this.guessCandidates });
    this.holdLabel = g.label;
    this.pending = null;
  }

  classifyStatic(entries: Entry[]): { candidates: Candidate[]; accepted: Candidate | null } {
    const c = this.config;
    const feats = entries.map((e) => e.feat).filter((f) => f.dom || f.non);
    if (feats.length < 3) return { candidates: [], accepted: null };
    const mean = meanFrame(feats);
    const signW = signWeights(c);

    const best = new Map<string, Candidate>();
    for (const t of this.usable(false)) {
      if (t.kind === 'letter' ? !this.allowLetters : !this.allowSigns) continue;
      const cost = frameDistance(mean, t.mean, t.kind === 'letter' ? LETTER_WEIGHTS : signW);
      const key = labelKey(t);
      const prev = best.get(key);
      if (!prev || cost < prev.cost!) {
        best.set(key, { label: t.label, kind: t.kind, cost, score: costToScore(cost, this.thresholdFor(t)), source: 'taught' });
      }
    }
    const taught = [...best.values()].sort((a, b) => b.score - a.score);
    let accepted: Candidate | null = taught[0] && taught[0].score >= 0.5 ? taught[0] : null;

    const candidates = [...taught];
    if (this.allowLetters && c.builtinAlphabet) {
      const rule = this.ruleLetter(entries);
      if (rule) {
        candidates.push(...rule.top);
        if (!accepted && rule.picked) accepted = rule.picked;
      }
    }
    candidates.sort((a, b) => b.score - a.score);
    return { candidates: candidates.slice(0, 4), accepted };
  }

  /** Built-in alphabet on the held hand. */
  private ruleLetter(entries: Entry[]): { top: Candidate[]; picked: Candidate | null } | null {
    const skels = entries.map((e) => e.skel);
    const feats = entries.map((e) => e.feat);
    const domCount = skels.filter((s) => s.dom).length;
    const nonCount = skels.filter((s) => s.non).length;
    if (Math.max(domCount, nonCount) < 3) return null;

    // Fingerspelling is one-handed: in auto mode, two raised hands close together mean a sign.
    if (this.config.mode === 'auto' && domCount >= 3 && nonCount >= 3) {
      const last = feats[feats.length - 1];
      const restY = this.segmenter.opts.restY;
      if (
        isActiveHand(last.domC, last.domImgY, restY) &&
        isActiveHand(last.nonC, last.nonImgY, restY) &&
        last.nonC &&
        last.domC &&
        last.nonC[1] < 0.9 &&
        Math.hypot(last.nonC[0] - last.domC[0], last.nonC[1] - last.domC[1]) < 0.9
      ) {
        return null;
      }
    }

    // Normally the dominant hand spells; fall back to the other one if it is the only hand up.
    const useDom = domCount >= nonCount;
    const hands = skels.map((s) => (useDom ? s.dom : s.non)).filter((h): h is HandPts => !!h);
    const world = hands.every((h) => h.world);
    const pts = averagePoints(hands.map((h) => (world ? handPoints(h) : imagePoints(h))));
    const scores = scoreLetters(handGeometry(pts, useDom ? 'right' : 'left'), this.taughtLetters());
    const top: Candidate[] = scores.slice(0, 3).map((s) => ({
      label: s.letter,
      kind: 'letter',
      score: s.score,
      source: 'built-in',
    }));
    const picked = pickLetter(scores, this.config.ruleMinScore);
    return { top, picked: picked ? top.find((t) => t.label === picked.letter) ?? null : null };
  }

  // -------------------------------------------------------------------------
  // Movements: dynamic signs, moving letters

  private onMotion(rec: SegRecord, now: number): void {
    const c = this.config;
    // A "movement" in which the hands neither travel nor change shape (e.g. a hand
    // simply appearing in place) holds no moving sign.
    if (this.isTrivial(rec)) return;
    const dyn = this.usable(true).filter((t) => (t.kind === 'sign' ? this.allowSigns : this.allowLetters));
    let candidates: Candidate[] = [];
    let accepted: Candidate | null = null;
    let windowStart = rec.t0;

    if (dyn.length) {
      const motions: SegRecord[] = [];
      for (let i = this.records.length - 1; i >= 0 && motions.length < MAX_PARTS; i--) {
        const r = this.records[i];
        if (r.type === 'rest') break;
        if (r.type === 'motion') motions.unshift(r);
      }
      const signW = signWeights(c);
      const best = new Map<string, { cand: Candidate; t0: number }>();
      for (let p = 1; p <= motions.length; p++) {
        const group = dyn.filter((t) => Math.min(t.parts, MAX_PARTS) === p);
        if (!group.length) continue;
        const firstPart = motions[motions.length - p];
        let t0 = firstPart.t0 - PRE_PAD_MS;
        if (firstPart.first) {
          // Leave out the hands rising into position; if that was all this movement was, there is no sign in it.
          t0 = this.riseEnd(firstPart);
          if (p === 1 && t0 >= firstPart.t1 - 100) continue;
        }
        const t1 = Math.min(now, rec.t1 + POST_PAD_MS);
        const query = resample(this.range(t0, t1).map((e) => e.feat), MATCH_HZ);
        if (query.length < 3) continue;
        for (const t of group) {
          const w = t.kind === 'letter' ? MOTION_LETTER_WEIGHTS : signW;
          const { cost } = matchMoving(t, query, w);
          const key = labelKey(t);
          const prev = best.get(key);
          if (!prev || cost < prev.cand.cost!) {
            best.set(key, {
              cand: { label: t.label, kind: t.kind, cost, score: costToScore(cost, this.thresholdFor(t)), source: 'taught' },
              t0,
            });
          }
        }
      }
      const ranked = [...best.values()].sort((a, b) => b.cand.score - a.cand.score);
      candidates = ranked.slice(0, 4).map((r) => r.cand);
      if (ranked[0] && ranked[0].cand.score >= 0.5) {
        accepted = ranked[0].cand;
        windowStart = ranked[0].t0;
      }
    }

    if (accepted) {
      // Anything recognised inside this movement (e.g. its starting handshape read as a letter) is withdrawn.
      for (const r of this.records) {
        if (r.type !== 'rest' && r !== rec && r.t1 >= windowStart && r.t0 <= rec.t1) this.transcriber.withdraw(r.tokens);
      }
      rec.tokens.push(
        this.transcriber.add({
          kind: accepted.kind,
          text: accepted.label,
          t0: windowStart,
          t1: rec.t1,
          provisional: c.mode === 'auto',
        }),
      );
      this.suppressNextHold = true;
      this.emit({ t: now, via: 'motion', accepted, candidates });
      return;
    }

    if (this.allowLetters && c.builtinAlphabet) {
      const found = detectMotionLetter(this.range(rec.t0 - 150, rec.t1 + 100).map((e) => e.feat));
      if (found && !this.taughtLetters().has(found.letter)) {
        const startHold = this.records.find((r) => r.type === 'hold' && r.t1 >= rec.t0 - 400 && r.t1 <= rec.t0 + 50);
        if (startHold) this.transcriber.withdraw(startHold.tokens);
        const cand: Candidate = { label: found.letter, kind: 'letter', score: found.score, source: 'built-in' };
        rec.tokens.push(
          this.transcriber.add({ kind: 'letter', text: found.letter, t0: rec.t0, t1: rec.t1, provisional: c.mode === 'auto' }),
        );
        this.suppressNextHold = true;
        this.emit({ t: now, via: 'motion', accepted: cand, candidates: [cand, ...candidates].slice(0, 4) });
        return;
      }
    }

    if (candidates.length) this.emit({ t: now, via: 'motion', accepted: null, candidates });
  }

  // -------------------------------------------------------------------------

  private isTrivial(rec: SegRecord): boolean {
    const start = rec.first ? this.riseEnd(rec) : rec.t0;
    const frames = resample(this.range(start, rec.t1).map((e) => e.feat), MATCH_HZ);
    if (frames.length < 2) return true;
    if (travel(frames) >= MIN_TRAVEL) return false;
    const head = frames.slice(0, 3);
    const tail = frames.slice(-3);
    for (const side of ['dom', 'non'] as const) {
      if (!head.every((f) => f[side]) || !tail.every((f) => f[side])) continue;
      if (shapeChange(meanFrame(head), meanFrame(tail), side) >= MIN_SHAPE_CHANGE) return false;
    }
    return true;
  }

  /** End of the initial upward movement of a motion that started from rest. */
  private riseEnd(rec: SegRecord): number {
    const frames = this.range(rec.t0, rec.t1).map((e) => e.feat);
    const y = (f: FeatureFrame) => (f.domC ? f.domC[1] : f.nonC ? f.nonC[1] : NaN);
    let i = 0;
    while (i + 3 < frames.length) {
      const a = frames[i];
      const b = frames[i + 3];
      const vy = (y(b) - y(a)) / Math.max(1e-3, (b.t - a.t) / 1000);
      if (!(vy < -0.5)) break;
      i++;
    }
    return frames[i]?.t ?? rec.t0;
  }

  private updateGuess(now: number): void {
    const phase = this.segmenter.phase;
    if (phase !== 'still' && phase !== 'hold') {
      this.guess = null;
      this.guessAccepted = false;
      this.guessCandidates = [];
      return;
    }
    if (now - this.lastGuessAt < GUESS_EVERY_MS) return;
    this.lastGuessAt = now;
    const { candidates, accepted } = this.classifyStatic(this.range(now - 300, now));
    this.guess = accepted ?? candidates[0] ?? null;
    this.guessAccepted = !!accepted;
    this.guessCandidates = candidates;
  }

  /** Matches a finished recording against the library, best first (for the Teach review). */
  compareRecording(feats: FeatureFrame[], dynamic: boolean, kind: SampleKind): Candidate[] {
    const best = new Map<string, Candidate>();
    const signW = signWeights(this.config);
    const query = dynamic ? resample(feats, MATCH_HZ) : [];
    const mean = dynamic ? null : meanFrame(feats);
    for (const t of this.templates()) {
      if (t.dynamic !== dynamic) continue;
      const cost = dynamic
        ? matchMoving(t, query, t.kind === 'letter' ? MOTION_LETTER_WEIGHTS : signW).cost
        : frameDistance(mean!, t.mean, kind === 'letter' && t.kind === 'letter' ? LETTER_WEIGHTS : signW);
      const key = labelKey(t);
      const prev = best.get(key);
      if (!prev || cost < prev.cost!) {
        best.set(key, { label: t.label, kind: t.kind, cost, score: costToScore(cost, this.thresholdFor(t)), source: 'taught' });
      }
    }
    return [...best.values()].sort((a, b) => b.score - a.score).slice(0, 3);
  }
}
