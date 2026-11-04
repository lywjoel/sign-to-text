import { describe, expect, it } from 'vitest';
import { extractFeatures } from '../src/features';
import { Library, newId } from '../src/library';
import { Recognizer } from '../src/recognizer';
import { Recorder } from '../src/recorder';
import { Transcriber } from '../src/transcriber';
import type { Skeleton } from '../src/skeleton';
import { hold, letterHand, morph, move, noisy, render, rest, type Clip, type HandSpec } from './scenes';
import { rng } from './synth';

/**
 * Signs performed with landmark jitter and natural variation in place, size
 * and speed. Checks that taught signs survive the noise and that movements
 * nobody taught are still turned away.
 */

type Path = (u: number) => [number, number];

interface SignDef {
  label: string;
  hand: (at: [number, number]) => HandSpec;
  path: Path;
  ms: number;
}

const B = (at: [number, number]) => letterHand('B', at);
const D = (at: [number, number]) => letterHand('D', at);
const A = (at: [number, number]) => letterHand('A', at);
const Y = (at: [number, number]) => letterHand('Y', at);

const SIGNS: SignDef[] = [
  { label: 'WAVE', hand: B, ms: 900, path: (u) => [0.6 - 0.4 * Math.sin(u * Math.PI * 2), -0.6] },
  { label: 'CIRCLE', hand: D, ms: 1000, path: (u) => [0.3 + 0.25 * Math.cos(u * Math.PI * 2), 0.4 + 0.25 * Math.sin(u * Math.PI * 2)] },
  { label: 'HELLO', hand: B, ms: 500, path: (u) => [0.05 + 0.55 * u, -0.75 + 0.1 * u] },
  { label: 'PHONE', hand: Y, ms: 700, path: (u) => [0.35 - 0.1 * u, -0.45 + 0.15 * Math.sin(u * Math.PI)] },
  { label: 'DROP', hand: A, ms: 600, path: (u) => [0.2, -0.2 + 0.7 * u] },
];

const UNTAUGHT: SignDef[] = [
  { label: 'SWIPE', hand: A, ms: 700, path: (u) => [-0.3 + 1.0 * u, 0.6 - 1.1 * u] },
  { label: 'BECKON', hand: D, ms: 800, path: (u) => [0.4 - 0.15 * Math.sin(u * Math.PI * 2), 0.1] },
  { label: 'LIFT', hand: B, ms: 600, path: (u) => [0.7, 0.6 - 0.9 * u] },
];

function vary(def: SignDef, rand: () => number): { clips: Clip[]; start: HandSpec } {
  const dx = (rand() - 0.5) * 0.16;
  const dy = (rand() - 0.5) * 0.16;
  const size = 0.85 + rand() * 0.3;
  const ms = def.ms * (0.8 + rand() * 0.45);
  const p0 = def.path(0);
  const path: Path = (u) => {
    const p = def.path(u);
    return [p0[0] + (p[0] - p0[0]) * size + dx, p0[1] + (p[1] - p0[1]) * size + dy];
  };
  const start = def.hand(path(0));
  return { start, clips: [move(ms, start, path), hold(0, def.hand(path(1)))] };
}

async function teachNoisy(lib: Library, def: SignDef, rand: () => number): Promise<void> {
  const { start, clips } = vary(def, rand);
  const rec = new Recorder();
  rec.start('sign', def.label, 0);
  const frames = noisy(render([{ ...hold(0, start), ms: 3300 }, clips[0], hold(1500, clips[1].at(1).dom)]), rand);
  for (const s of frames) {
    rec.feed(s, extractFeatures(s));
    if (rec.phase === 'review') break;
  }
  expect(rec.phase, `${def.label}: ${rec.error}`).toBe('review');
  const r = rec.result!;
  await lib.add({ id: newId(), label: def.label, kind: 'sign', createdAt: 0, frames: r.frames, dynamic: r.dynamic, parts: r.parts });
}

function perform(recog: Recognizer, def: SignDef, rand: () => number, t0: number): number {
  const { start, clips } = vary(def, rand);
  const frames: Skeleton[] = noisy(
    render([rest(400), hold(500, start), clips[0], hold(600, clips[1].at(1).dom), rest(500)], t0),
    rand,
  );
  for (const s of frames) recog.push(s, extractFeatures(s));
  return frames.at(-1)!.t + 33;
}

describe('Fingerspelling under noise', () => {
  const WORDS = ['MRT', 'BUS', 'CAB', 'YES', 'LOVE', 'HI', 'FIX', 'WAG', 'SUM', 'VAN', 'NUT', 'TEA'];

  it.each([3, 11])('spells words with natural transitions (seed %i)', (seed) => {
    const rand = rng(seed);
    const tr = new Transcriber();
    const recog = new Recognizer(tr, () => [], () => new Set());
    let t = 0;
    let right = 0;
    const misses: string[] = [];
    for (const word of WORDS) {
      tr.clear();
      const at: [number, number] = [0.55 + (rand() - 0.5) * 0.2, -0.3 + (rand() - 0.5) * 0.2];
      const clips: Clip[] = [rest(400)];
      let prev: HandSpec | null = null;
      for (const letter of word) {
        const h = letterHand(letter, at);
        if (prev) clips.push(morph(150 + rand() * 120, prev, h));
        clips.push(hold(480 + rand() * 250, h));
        prev = h;
      }
      clips.push(rest(500));
      const frames = noisy(render(clips, t), rand, 2, 0.002);
      for (const s of frames) recog.push(s, extractFeatures(s));
      t = frames.at(-1)!.t + 33;
      if (tr.text() === word) right++;
      else misses.push(`${word}→${tr.text()}`);
    }
    console.log(`spelled ${right}/${WORDS.length}`, misses.join(' '));
    expect(right / WORDS.length).toBeGreaterThanOrEqual(0.75);
  });
});

describe('Recognition under noise and variation', () => {
  it.each([42, 7, 1234, 99])('recognises taught signs and rejects untaught movements (seed %i)', async (seed) => {
    const rand = rng(seed);
    const lib = new Library(null);
    for (const def of SIGNS) for (let i = 0; i < 3; i++) await teachNoisy(lib, def, rand);

    const tr = new Transcriber();
    const recog = new Recognizer(tr, () => lib.templates, () => lib.taughtLetters(), { builtinAlphabet: false });
    let t = 0;
    let correct = 0;
    let wrong = 0;
    const trials = 6;
    for (const def of SIGNS) {
      for (let i = 0; i < trials; i++) {
        tr.clear();
        t = perform(recog, def, rand, t);
        const out = tr.text();
        if (out === def.label) correct++;
        else if (out) {
          wrong++;
          console.log(`expected ${def.label}, got "${out}"`);
        }
      }
    }
    let falseAlarms = 0;
    for (const def of UNTAUGHT) {
      for (let i = 0; i < trials; i++) {
        tr.clear();
        t = perform(recog, def, rand, t);
        if (tr.text()) falseAlarms++;
      }
    }
    const total = SIGNS.length * trials;
    const negatives = UNTAUGHT.length * trials;
    console.log(`correct ${correct}/${total}, wrong ${wrong}, false alarms ${falseAlarms}/${negatives}`);
    expect(correct / total).toBeGreaterThanOrEqual(0.85);
    expect(wrong).toBeLessThanOrEqual(1);
    expect(falseAlarms / negatives).toBeLessThanOrEqual(0.2);
  });
});
