export type TokenKind = 'letter' | 'sign';

export interface Token {
  id: number;
  kind: TokenKind;
  text: string;
  t0: number;
  t1: number;
  /** Provisional tokens can still be withdrawn, e.g. when a hold turns out to be the start of a sign. */
  provisional: boolean;
}

export type Item = Token | { id: number; kind: 'space' | 'newline' };

export interface Word {
  text: string;
  /** Fingerspelled words are built from letters; signs are glosses. */
  kind: 'spelled' | 'sign';
  provisional: boolean;
}

/** The transcript as lines of words. */
export type Lines = Word[][];

/**
 * Holds recognised letters and signs. Consecutive letters form one
 * fingerspelled word until a rest, a sign or a manual space.
 */
export class Transcriber {
  items: Item[] = [];
  settleMs = 2500;
  private nextId = 1;
  private listeners = new Set<() => void>();

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  add(tok: Omit<Token, 'id'>): number {
    const id = this.nextId++;
    this.items.push({ ...tok, id });
    this.emit();
    return id;
  }

  /** Withdraws provisional tokens. Settled ones are kept. Returns how many were removed. */
  withdraw(ids: number[]): number {
    const drop = new Set(ids);
    const before = this.items.length;
    this.items = this.items.filter((it) => !(drop.has(it.id) && isToken(it) && it.provisional));
    const removed = before - this.items.length;
    if (removed) {
      this.tidy();
      this.emit();
    }
    return removed;
  }

  settle(now: number): void {
    let changed = false;
    for (const it of this.items) {
      if (isToken(it) && it.provisional && now - it.t1 >= this.settleMs) {
        it.provisional = false;
        changed = true;
      }
    }
    if (changed) this.emit();
  }

  settleAll(): void {
    let changed = false;
    for (const it of this.items) {
      if (isToken(it) && it.provisional) {
        it.provisional = false;
        changed = true;
      }
    }
    if (changed) this.emit();
  }

  /** Ends the current fingerspelled word. */
  space(): void {
    const last = this.items[this.items.length - 1];
    if (!last || last.kind === 'space' || last.kind === 'newline') return;
    this.items.push({ id: this.nextId++, kind: 'space' });
    this.emit();
  }

  newline(): void {
    const last = this.items[this.items.length - 1];
    if (!last || last.kind === 'newline') return;
    if (last.kind === 'space') this.items.pop();
    this.items.push({ id: this.nextId++, kind: 'newline' });
    this.emit();
  }

  /** Removes the last letter or sign (and any break after it). */
  undo(): void {
    while (this.items.length && !isToken(this.items[this.items.length - 1])) this.items.pop();
    this.items.pop();
    this.tidy();
    this.emit();
  }

  clear(): void {
    this.items = [];
    this.emit();
  }

  /** Drops breaks that no longer separate anything; of two adjacent breaks the stronger wins. */
  private tidy(): void {
    const out: Item[] = [];
    for (const it of this.items) {
      const prev = out[out.length - 1];
      if (isToken(it) || (prev && isToken(prev))) {
        out.push(it);
      } else if (prev && it.kind === 'newline') {
        out[out.length - 1] = it;
      }
    }
    this.items = out;
  }

  lines(): Lines {
    const lines: Lines = [[]];
    let spelled: Word | null = null;
    const flush = () => {
      if (spelled) lines[lines.length - 1].push(spelled);
      spelled = null;
    };
    for (const it of this.items) {
      if (it.kind === 'letter') {
        spelled ??= { text: '', kind: 'spelled', provisional: false };
        spelled.text += it.text;
        spelled.provisional ||= it.provisional;
      } else if (it.kind === 'sign') {
        flush();
        lines[lines.length - 1].push({ text: it.text, kind: 'sign', provisional: it.provisional });
      } else if (it.kind === 'space') {
        flush();
      } else {
        flush();
        lines.push([]);
      }
    }
    flush();
    return lines;
  }

  text(): string {
    return this.lines()
      .map((words) => words.map((w) => w.text).join(' '))
      .join('\n')
      .trim();
  }
}

export function isToken(it: Item): it is Token {
  return it.kind === 'letter' || it.kind === 'sign';
}
