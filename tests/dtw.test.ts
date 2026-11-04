import { describe, expect, it } from 'vitest';
import { subsequenceDtw } from '../src/dtw';

const d = (a: number, b: number) => Math.abs(a - b);
const wave = (n: number, phase = 0) => Array.from({ length: n }, (_, i) => Math.sin((i / (n - 1)) * Math.PI * 2 + phase));

describe('subsequenceDtw', () => {
  it('finds a template embedded in a longer query', () => {
    const template = [0, 1, 2, 3, 2, 1, 0];
    const query = [5, 5, 5, ...template, 5, 5];
    const m = subsequenceDtw(template, query, d, { stepPenalty: 0 });
    expect(m.cost).toBe(0);
    expect(m.start).toBe(3);
    expect(m.end).toBe(9);
  });

  it('tolerates the same pattern performed more slowly', () => {
    const template = wave(20);
    const slow = wave(30);
    const other = wave(20, Math.PI);
    const same = subsequenceDtw(template, [0, 0, ...slow, 0], d);
    const diff = subsequenceDtw(template, [0, 0, ...other, 0], d);
    expect(same.cost).toBeLessThan(0.2);
    expect(diff.cost).toBeGreaterThan(same.cost * 3);
  });

  it('penalises squeezing a long template into a few query frames', () => {
    const template = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const squeezed = subsequenceDtw(template, [0, 9], d, { stepPenalty: 0 });
    const fair = subsequenceDtw(template, template, d, { stepPenalty: 0 });
    expect(fair.cost).toBe(0);
    expect(squeezed.cost).toBeGreaterThan(1);
  });

  it('returns Infinity for empty input', () => {
    expect(subsequenceDtw([], [1], d).cost).toBe(Infinity);
  });
});
