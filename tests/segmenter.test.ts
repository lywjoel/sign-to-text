import { describe, expect, it } from 'vitest';
import { extractFeatures } from '../src/features';
import { Segmenter, type SegEvent } from '../src/segmenter';
import { hold, lerp2, letterHand, move, render, rest } from './scenes';

function run(clips: Parameters<typeof render>[0]): SegEvent[] {
  const seg = new Segmenter();
  const events: SegEvent[] = [];
  for (const s of render(clips)) events.push(...seg.update(extractFeatures(s)));
  return events;
}

describe('Segmenter', () => {
  it('splits rest, movement and holds', () => {
    const B = letterHand('B', [0.5, 0]);
    const events = run([
      rest(400),
      hold(600, B),
      move(500, B, lerp2([0.5, 0], [0.1, -0.4])),
      hold(600, { ...B, at: [0.1, -0.4] }),
      rest(500),
    ]);
    expect(events.map((e) => e.type)).toEqual(['active', 'motion', 'hold', 'holdEnd', 'motion', 'hold', 'holdEnd', 'rest']);
    const motion = events.filter((e) => e.type === 'motion')[1] as Extract<SegEvent, { type: 'motion' }>;
    // The movement ran from 1000 ms to 1500 ms.
    expect(motion.t0).toBeGreaterThan(900);
    expect(motion.t0).toBeLessThan(1100);
    expect(motion.t1).toBeGreaterThan(1400);
    expect(motion.t1).toBeLessThan(1650);
  });

  it('treats a change of handshape in place as movement', () => {
    const events = run([
      rest(300),
      hold(600, letterHand('A', [0.5, 0])),
      hold(600, letterHand('V', [0.5, 0])),
      rest(400),
    ]);
    expect(events.filter((e) => e.type === 'hold').length).toBe(2);
  });

  it('counts hands below the rest line as resting', () => {
    const events = run([hold(500, letterHand('B', [0.5, 0])), hold(600, letterHand('B', [0.5, 2.2]))]);
    expect(events.at(-1)?.type).toBe('rest');
  });
});
