import { extractFeatures, meanFrame, resample, travel, type CoreFrame, type FeatureFrame } from './features';
import { Segmenter } from './segmenter';
import type { Skeleton } from './skeleton';

export type SampleKind = 'sign' | 'letter';

/** One recorded example, stored as canonical skeleton frames so features can be re-derived later. */
export interface Sample {
  id: string;
  label: string;
  kind: SampleKind;
  createdAt: number;
  frames: Skeleton[];
  /** True if the sample contains movement (matched with DTW), false for a held pose. */
  dynamic: boolean;
  /** Number of movement stretches separated by pauses. */
  parts: number;
}

/** A sample prepared for matching. */
export interface Template {
  sampleId: string;
  label: string;
  kind: SampleKind;
  dynamic: boolean;
  parts: number;
  /** Feature frames resampled to MATCH_HZ, for dynamic matching. */
  seq: CoreFrame[];
  /** Mean feature frame, for static matching. */
  mean: CoreFrame;
  /** How far the hands travel during the sample, in body units. */
  travel: number;
}

export const MATCH_HZ = 15;

export function compile(sample: Sample): Template {
  const feats = sample.frames.map(extractFeatures);
  const seq = resample(feats, MATCH_HZ);
  return {
    sampleId: sample.id,
    label: sample.label,
    kind: sample.kind,
    dynamic: sample.dynamic,
    parts: sample.parts,
    seq,
    mean: meanFrame(feats),
    travel: travel(seq),
  };
}

/** Path length of the hand centres (body units) over a run of frames. */
export function pathLength(frames: FeatureFrame[]): number {
  let total = 0;
  for (const key of ['domC', 'nonC'] as const) {
    let len = 0;
    let prev: [number, number] | null = null;
    for (const f of frames) {
      const c = f[key];
      if (c && prev) len += Math.hypot(c[0] - prev[0], c[1] - prev[1]);
      prev = c;
    }
    total = Math.max(total, len);
  }
  return total;
}

/**
 * Works out whether a recording moves and how many separate movements it has,
 * using the same segmentation as live recognition.
 */
export function analyseMotion(feats: FeatureFrame[], holdMs: number): { dynamic: boolean; parts: number } {
  const seg = new Segmenter({ holdMs, restY: Infinity });
  let parts = 0;
  const count = (t0: number, t1: number) => {
    const span = feats.filter((f) => f.t >= t0 && f.t <= t1);
    if (pathLength(span) >= 0.12) parts++;
  };
  for (const f of feats) {
    for (const e of seg.update(f)) if (e.type === 'motion') count(e.t0, e.t1);
  }
  const last = feats[feats.length - 1];
  if (last) for (const e of seg.finish(last.t)) if (e.type === 'motion') count(e.t0, e.t1);
  return { dynamic: parts > 0, parts: Math.max(1, parts) };
}

// ---------------------------------------------------------------------------
// Persistence

const DB_NAME = 'sgsl-sign-to-text';
const STORE = 'samples';

function request<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

export class SampleStore {
  private constructor(private readonly db: IDBDatabase) {}

  static open(): Promise<SampleStore> {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE, { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(new SampleStore(req.result));
      req.onerror = () => reject(req.error);
    });
  }

  private tx(mode: IDBTransactionMode): IDBObjectStore {
    return this.db.transaction(STORE, mode).objectStore(STORE);
  }

  all(): Promise<Sample[]> {
    return request(this.tx('readonly').getAll() as IDBRequest<Sample[]>);
  }

  async put(samples: Sample[]): Promise<void> {
    const tx = this.db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    for (const s of samples) store.put(s);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async delete(ids: string[]): Promise<void> {
    const tx = this.db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    for (const id of ids) store.delete(id);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}

// ---------------------------------------------------------------------------
// In-memory library

export interface LabelSummary {
  label: string;
  kind: SampleKind;
  count: number;
  dynamic: boolean;
  sampleIds: string[];
}

const EXPORT_FORMAT = 'sgsl-sign-to-text-library';

export function normaliseLabel(label: string): string {
  return label.trim().replace(/\s+/g, '-').toUpperCase();
}

export function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function roundDeep<T>(value: T): T {
  return JSON.parse(
    JSON.stringify(value, (_k, v) => (typeof v === 'number' && !Number.isInteger(v) ? Math.round(v * 1e4) / 1e4 : v)),
  );
}

export class Library {
  samples: Sample[] = [];
  templates: Template[] = [];
  private listeners = new Set<() => void>();

  constructor(private readonly store: SampleStore | null) {}

  async load(): Promise<void> {
    if (!this.store) return;
    this.samples = await this.store.all();
    this.samples.sort((a, b) => a.createdAt - b.createdAt);
    this.templates = this.samples.map(compile);
    this.emit();
  }

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  async add(sample: Sample): Promise<void> {
    await this.store?.put([sample]);
    this.samples.push(sample);
    this.templates.push(compile(sample));
    this.emit();
  }

  async remove(ids: string[]): Promise<void> {
    const drop = new Set(ids);
    await this.store?.delete(ids);
    this.samples = this.samples.filter((s) => !drop.has(s.id));
    this.templates = this.templates.filter((t) => !drop.has(t.sampleId));
    this.emit();
  }

  async removeLabel(label: string, kind: SampleKind): Promise<void> {
    await this.remove(this.samples.filter((s) => s.label === label && s.kind === kind).map((s) => s.id));
  }

  summary(): LabelSummary[] {
    const map = new Map<string, LabelSummary>();
    for (const s of this.samples) {
      const key = `${s.kind}:${s.label}`;
      let entry = map.get(key);
      if (!entry) {
        entry = { label: s.label, kind: s.kind, count: 0, dynamic: false, sampleIds: [] };
        map.set(key, entry);
      }
      entry.count++;
      entry.dynamic ||= s.dynamic;
      entry.sampleIds.push(s.id);
    }
    return [...map.values()].sort((a, b) =>
      a.kind === b.kind ? a.label.localeCompare(b.label) : a.kind === 'letter' ? -1 : 1,
    );
  }

  /** Letters the signer has taught; the built-in rule for these is switched off. */
  taughtLetters(): Set<string> {
    return new Set(this.samples.filter((s) => s.kind === 'letter').map((s) => s.label));
  }

  exportJson(): string {
    return JSON.stringify({ format: EXPORT_FORMAT, version: 1, exportedAt: new Date().toISOString(), samples: roundDeep(this.samples) });
  }

  /** Adds samples from an export. Returns how many were new. */
  async importJson(text: string): Promise<number> {
    const data = JSON.parse(text) as { format?: string; samples?: Sample[] };
    if (data.format !== EXPORT_FORMAT || !Array.isArray(data.samples)) {
      throw new Error('This file is not a sign library export.');
    }
    const known = new Set(this.samples.map((s) => s.id));
    const fresh = data.samples.filter(
      (s) => s && typeof s.id === 'string' && typeof s.label === 'string' && Array.isArray(s.frames) && !known.has(s.id),
    );
    if (!fresh.length) return 0;
    await this.store?.put(fresh);
    this.samples.push(...fresh);
    this.templates.push(...fresh.map(compile));
    this.emit();
    return fresh.length;
  }
}
