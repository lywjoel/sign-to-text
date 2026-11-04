import { describe, expect, it } from 'vitest';
import { describeHand } from '../src/handshape';
import { handGeometry, pickLetter, scoreLetters, STATIC_LETTERS } from '../src/fingerspell';
import {
  LETTER_POSES,
  OPEN_HAND,
  RELAXED_HAND,
  UPRIGHT,
  jitter,
  localHand,
  mirror,
  rng,
  tilt,
  toCamera,
} from './synth';

function classify(letter: string, opts: { tilt?: [number, number, number]; noise?: number; seed?: number; left?: boolean } = {}) {
  const { pose, orient } = LETTER_POSES[letter];
  const o = opts.tilt ? tilt(orient, ...opts.tilt) : orient;
  let pts = toCamera(localHand(pose), o);
  if (opts.noise) pts = jitter(pts, opts.noise, rng(opts.seed ?? 1));
  if (opts.left) pts = mirror(pts);
  return scoreLetters(handGeometry(pts, opts.left ? 'left' : 'right'));
}

describe('hand descriptor', () => {
  it('recovers the palm frame of an upright, palm-forward right hand', () => {
    const pts = toCamera(localHand(LETTER_POSES.B.pose), UPRIGHT);
    const d = describeHand(pts, 'right');
    expect(d.up.y).toBeLessThan(-0.95);
    expect(d.side.x).toBeGreaterThan(0.95);
    expect(d.normal.z).toBeLessThan(-0.95);
    for (let f = 1; f <= 4; f++) expect(d.flex[f * 3]).toBeLessThan(0.15);
  });

  it('reports the same handshape for a mirrored (left) hand', () => {
    const pts = toCamera(localHand(LETTER_POSES.K.pose), UPRIGHT);
    const r = describeHand(pts, 'right');
    const l = describeHand(mirror(pts), 'left');
    for (let i = 0; i < r.flex.length; i++) expect(l.flex[i]).toBeCloseTo(r.flex[i], 6);
    expect(l.local[4].x).toBeCloseTo(r.local[4].x, 6);
    expect(l.normal.z).toBeCloseTo(r.normal.z, 6);
    expect(l.normal.x).toBeCloseTo(-r.normal.x, 6);
  });

  it('measures knuckle flexion of a fist', () => {
    const pts = toCamera(localHand(LETTER_POSES.S.pose), UPRIGHT);
    const d = describeHand(pts, 'right');
    expect(d.flex[3]).toBeCloseTo(1.5, 1);
    expect(d.flex[4]).toBeCloseTo(1.75, 1);
  });
});

describe('fingerspelling rules', () => {
  it.each(STATIC_LETTERS)('recognises a canonical %s', (letter) => {
    const scores = classify(letter);
    const picked = pickLetter(scores);
    expect(picked?.letter, JSON.stringify(scores.slice(0, 3))).toBe(letter);
  });

  it.each(STATIC_LETTERS)('recognises %s when tilted and noisy', (letter) => {
    const scores = classify(letter, { tilt: [0.15, -0.2, 0.12], noise: 0.002, seed: letter.charCodeAt(0) });
    expect(scores[0].letter, JSON.stringify(scores.slice(0, 3))).toBe(letter);
  });

  it.each(STATIC_LETTERS)('recognises %s on a left hand', (letter) => {
    const scores = classify(letter, { left: true });
    expect(scores[0].letter, JSON.stringify(scores.slice(0, 3))).toBe(letter);
  });

  it('does not spell anything for an open or relaxed hand', () => {
    for (const pose of [OPEN_HAND, RELAXED_HAND]) {
      const pts = toCamera(localHand(pose), UPRIGHT);
      const scores = scoreLetters(handGeometry(pts, 'right'));
      expect(pickLetter(scores), JSON.stringify(scores.slice(0, 3))).toBeNull();
    }
  });
});
