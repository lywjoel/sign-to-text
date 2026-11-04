import { beforeEach, describe, expect, it } from 'vitest';
import { extractFeatures } from '../src/features';
import { Library, newId, type SampleKind } from '../src/library';
import { Recognizer, type RecognizerConfig } from '../src/recognizer';
import { Recorder } from '../src/recorder';
import { Transcriber } from '../src/transcriber';
import type { Skeleton } from '../src/skeleton';
import { hold, lerp2, letterHand, morph, move, render, rest, skeleton, type Clip, type HandSpec } from './scenes';

const B = (at: [number, number]) => letterHand('B', at);
const D = (at: [number, number]) => letterHand('D', at);
const A = (at: [number, number]) => letterHand('A', at);

function both(ms: number, dom: (u: number) => HandSpec, non: (u: number) => HandSpec): Clip {
  return { ms, at: (u) => ({ dom: dom(u), non: non(u) }) };
}

/** Teaches a sign through the real recorder: countdown in the start position, then the clips, then a still end. */
async function teach(lib: Library, label: string, kind: SampleKind, start: Clip, clips: Clip[]): Promise<void> {
  const rec = new Recorder();
  rec.start(kind, label, 0);
  const last = clips[clips.length - 1];
  const frames = render([
    { ...start, ms: 3300 },
    ...clips,
    { ms: 1500, at: () => last.at(1) },
  ]);
  for (const s of frames) {
    rec.feed(s, extractFeatures(s));
    if (rec.phase === 'review') break;
  }
  expect(rec.phase, rec.error ?? '').toBe('review');
  const r = rec.result!;
  await lib.add({ id: newId(), label, kind, createdAt: 0, frames: r.frames, dynamic: r.dynamic, parts: r.parts });
}

function play(recog: Recognizer, frames: Skeleton[]): void {
  for (const s of frames) recog.push(s, extractFeatures(s));
}

function setup(config: Partial<RecognizerConfig> = {}) {
  const lib = new Library(null);
  const tr = new Transcriber();
  const recog = new Recognizer(tr, () => lib.templates, () => lib.taughtLetters(), config);
  return { lib, tr, recog };
}

const wavePath = (y: number) => (u: number): [number, number] => [0.6 - 0.4 * Math.sin(u * Math.PI * 2), y];
const circlePath = (u: number): [number, number] => [0.3 + 0.25 * Math.cos(u * Math.PI * 2), 0.4 + 0.25 * Math.sin(u * Math.PI * 2)];

describe('Recognizer with taught signs', () => {
  let ctx: ReturnType<typeof setup>;

  beforeEach(async () => {
    ctx = setup();
    const { lib } = ctx;
    await teach(lib, 'WAVE', 'sign', hold(0, B([0.6, -0.6])), [move(900, B([0.6, -0.6]), wavePath(-0.6))]);
    await teach(lib, 'WAVE', 'sign', hold(0, B([0.62, -0.55])), [move(1000, B([0.62, -0.55]), wavePath(-0.55))]);
    await teach(lib, 'CIRCLE', 'sign', hold(0, D(circlePath(0))), [move(1000, D(circlePath(0)), circlePath)]);
    await teach(lib, 'HELLO', 'sign', hold(0, B([0.05, -0.75])), [
      move(500, B([0.05, -0.75]), lerp2([0.05, -0.75], [0.6, -0.65])),
    ]);
    await teach(lib, 'OPEN', 'sign', hold(0, B([0.12, 0.3])), [
      both(
        700,
        (u) => B([0.12 + 0.5 * u, 0.3]),
        (u) => B([-0.12 - 0.5 * u, 0.3]),
      ),
    ]);
  });

  it('stores what was taught', () => {
    const s = ctx.lib.summary();
    expect(s.map((x) => `${x.label}:${x.count}:${x.dynamic}`)).toEqual([
      'CIRCLE:1:true',
      'HELLO:1:true',
      'OPEN:1:true',
      'WAVE:2:true',
    ]);
  });

  it('recognises a slower, slightly shifted performance', () => {
    const { recog, tr } = ctx;
    play(recog, render([
      rest(500),
      move(300, B([0.6, 1.4]), lerp2([0.6, 1.4], [0.58, -0.58])),
      hold(500, B([0.58, -0.58])),
      move(1300, B([0.58, -0.58]), wavePath(-0.58)),
      hold(600, B([0.58, -0.58])),
      rest(600),
    ]));
    expect(tr.text()).toBe('WAVE');
  });

  it('tells two taught signs apart', () => {
    const { recog, tr } = ctx;
    play(recog, render([
      rest(400),
      hold(500, D(circlePath(0))),
      move(1100, D(circlePath(0)), circlePath),
      hold(600, D(circlePath(1))),
      rest(400),
      hold(500, B([0.6, -0.6])),
      move(900, B([0.6, -0.6]), wavePath(-0.6)),
      hold(600, B([0.6, -0.6])),
      rest(400),
    ]));
    expect(tr.text()).toBe('CIRCLE WAVE');
  });

  it('ignores movements that were never taught', () => {
    const { recog, tr } = ctx;
    recog.configure({ builtinAlphabet: false });
    play(recog, render([
      rest(400),
      hold(500, A([-0.3, 0.6])),
      move(700, A([-0.3, 0.6]), lerp2([-0.3, 0.6], [0.7, -0.5])),
      hold(600, A([0.7, -0.5])),
      rest(400),
    ]));
    expect(tr.text()).toBe('');
  });

  it('withdraws a letter read from the starting handshape of a sign', () => {
    const { recog, tr } = ctx;
    const frames = render([
      rest(400),
      hold(700, B([0.05, -0.75])),
    ]);
    play(recog, frames);
    expect(tr.text()).toBe('B');
    play(recog, render([
      move(500, B([0.05, -0.75]), lerp2([0.05, -0.75], [0.6, -0.65])),
      hold(600, B([0.6, -0.65])),
      rest(400),
    ], frames.at(-1)!.t + 33));
    expect(tr.text()).toBe('HELLO');
  });

  it('recognises a two-handed sign', () => {
    const { recog, tr } = ctx;
    play(recog, render([
      rest(400),
      both(500, () => B([0.12, 0.3]), () => B([-0.12, 0.3])),
      both(800, (u) => B([0.12 + 0.5 * u, 0.3]), (u) => B([-0.12 - 0.5 * u, 0.3])),
      both(600, () => B([0.62, 0.3]), () => B([-0.62, 0.3])),
      rest(400),
    ]));
    expect(tr.text()).toBe('OPEN');
  });

  it('only spells in fingerspelling mode and only signs in signs mode', () => {
    const { recog, tr } = ctx;
    const scene = (t0: number) => render([
      rest(400),
      hold(700, A([0.6, -0.3])),
      hold(700, B([0.6, -0.3])),
      rest(400),
      hold(500, B([0.6, -0.6])),
      move(900, B([0.6, -0.6]), wavePath(-0.6)),
      hold(600, B([0.6, -0.6])),
      rest(400),
    ], t0);
    recog.configure({ mode: 'fingerspell' });
    play(recog, scene(0));
    expect(tr.text()).not.toContain('WAVE');
    expect(tr.text().startsWith('AB')).toBe(true);
    tr.clear();
    recog.reset();
    recog.configure({ mode: 'signs' });
    play(recog, scene(10000));
    expect(tr.text()).toBe('WAVE');
  });
});

describe('Recognizer fingerspelling', () => {
  it('spells held letters into a word and starts a new word after a rest', () => {
    const { recog, tr } = setup();
    play(recog, render([
      rest(400),
      hold(700, letterHand('M', [0.6, -0.3])),
      hold(700, letterHand('R', [0.6, -0.3])),
      hold(700, letterHand('T', [0.6, -0.3])),
      rest(600),
      hold(700, letterHand('G', [0.6, -0.3])),
      hold(700, letterHand('O', [0.6, -0.3])),
      rest(600),
    ]));
    expect(tr.text()).toBe('MRT GO');
  });

  it('notices a change between near-identical letters made in place', () => {
    const { recog, tr } = setup();
    const at: [number, number] = [0.6, -0.3];
    play(recog, render([
      rest(400),
      hold(700, letterHand('M', at)),
      morph(150, letterHand('M', at), letterHand('N', at)),
      hold(800, letterHand('N', at)),
      rest(500),
    ]));
    expect(tr.text()).toBe('MN');
  });

  it('spells a double letter when the hand slides between them', () => {
    const { recog, tr } = setup();
    const L = (at: [number, number]) => letterHand('L', at);
    play(recog, render([
      rest(400),
      hold(700, L([0.6, -0.3])),
      move(180, L([0.6, -0.3]), lerp2([0.6, -0.3], [0.75, -0.3])),
      hold(700, L([0.75, -0.3])),
      rest(500),
    ]));
    expect(tr.text()).toBe('LL');
  });

  it('reads J from the little finger drawing a hook', () => {
    const { recog, tr } = setup();
    const I = (at: [number, number]) => letterHand('I', at);
    const jPath = (u: number): [number, number] => {
      // Down, then curving across towards the body.
      if (u < 0.6) return [0.6, -0.3 + (u / 0.6) * 0.25];
      const v = (u - 0.6) / 0.4;
      return [0.6 - 0.15 * Math.sin((v * Math.PI) / 2), -0.05 + 0.05 * Math.sin(v * Math.PI)];
    };
    play(recog, render([rest(400), hold(600, I([0.6, -0.3])), move(500, I([0.6, -0.3]), jPath), hold(600, I(jPath(1))), rest(500)]));
    expect(tr.text()).toBe('J');
  });

  it('prefers a taught letter over the built-in rule', async () => {
    const { recog, tr, lib } = setup();
    // Teach the open hand as "B" for this signer.
    const open = { ...letterHand('B', [0.6, -0.3]) };
    open.pose = { ...open.pose, thumb: [1.0, 0.55, 0.1] };
    await teach(lib, 'B', 'letter', hold(0, open), [hold(900, open)]);
    expect(lib.templates[0].dynamic).toBe(false);
    play(recog, render([rest(400), hold(700, open), rest(500)]));
    expect(tr.text()).toBe('B');
  });
});

describe('Recognizer multi-part signs', () => {
  it('matches a sign made of two movements with a pause between', async () => {
    const { recog, tr, lib } = setup();
    const point = D([0.3, 0.0]);
    const part1 = move(500, point, lerp2([0.3, 0.0], [0.1, -0.2]));
    const pause = hold(500, D([0.1, -0.2]));
    const part2 = move(500, A([0.1, -0.2]), lerp2([0.1, -0.2], [0.2, 0.4]));
    await teach(lib, 'YOU-CAN', 'sign', hold(0, point), [part1, pause, part2]);
    expect(lib.templates[0].parts).toBe(2);
    play(recog, render([rest(400), hold(500, point), part1, pause, part2, hold(600, A([0.2, 0.4])), rest(400)]));
    expect(tr.text()).toBe('YOU-CAN');
  });
});

it('does nothing without hands', () => {
  const { recog, tr } = setup();
  play(recog, Array.from({ length: 60 }, (_, i) => skeleton(i * 33, null)));
  expect(tr.text()).toBe('');
});
