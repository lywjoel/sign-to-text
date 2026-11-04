import { describe, expect, it } from 'vitest';
import { Transcriber } from '../src/transcriber';

const tok = (text: string, kind: 'letter' | 'sign' = 'letter', provisional = false, t = 0) => ({
  kind,
  text,
  t0: t,
  t1: t,
  provisional,
});

describe('Transcriber', () => {
  it('joins letters into a fingerspelled word until a break', () => {
    const tr = new Transcriber();
    for (const l of 'MRT') tr.add(tok(l));
    tr.space();
    tr.add(tok('GO', 'sign'));
    expect(tr.text()).toBe('MRT GO');
    expect(tr.lines()[0][0]).toMatchObject({ text: 'MRT', kind: 'spelled' });
  });

  it('separates a sign from the letters around it', () => {
    const tr = new Transcriber();
    tr.add(tok('A'));
    tr.add(tok('HELLO', 'sign'));
    tr.add(tok('B'));
    expect(tr.text()).toBe('A HELLO B');
  });

  it('withdraws provisional tokens only', () => {
    const tr = new Transcriber();
    const a = tr.add(tok('B', 'letter', true));
    const b = tr.add(tok('C', 'letter', false));
    expect(tr.withdraw([a, b])).toBe(1);
    expect(tr.text()).toBe('C');
  });

  it('settles provisional tokens after a delay', () => {
    const tr = new Transcriber();
    tr.add(tok('B', 'letter', true, 1000));
    tr.settle(2000);
    expect(tr.lines()[0][0].provisional).toBe(true);
    tr.settle(1000 + tr.settleMs);
    expect(tr.lines()[0][0].provisional).toBe(false);
  });

  it('puts new lines between sentences and ignores redundant breaks', () => {
    const tr = new Transcriber();
    tr.space();
    tr.newline();
    tr.add(tok('HI', 'sign'));
    tr.space();
    tr.space();
    tr.newline();
    tr.add(tok('BYE', 'sign'));
    expect(tr.text()).toBe('HI\nBYE');
  });

  it('undoes the last token and any breaks after it', () => {
    const tr = new Transcriber();
    tr.add(tok('A'));
    tr.add(tok('B'));
    tr.space();
    tr.undo();
    expect(tr.text()).toBe('A');
    tr.undo();
    tr.undo();
    expect(tr.text()).toBe('');
  });
});
