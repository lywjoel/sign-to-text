import type { Mode, RecognizerConfig } from './recognizer';
import type { Dominant } from './skeleton';
import type { Backend } from './tracker';

export interface Settings {
  mode: Mode;
  dominant: Dominant;
  backend: Backend;
  deviceId: string | null;
  /** 0.5 (strict) .. 1.5 (lenient); scales every acceptance threshold. */
  sensitivity: number;
  /** How long a pause has to last to count as a hold, ms. */
  holdMs: number;
  useFace: boolean;
  useBody: boolean;
  builtinAlphabet: boolean;
  showFace: boolean;
  mirrorView: boolean;
  showDebug: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  mode: 'auto',
  dominant: 'right',
  backend: 'separate',
  deviceId: null,
  sensitivity: 1,
  holdMs: 320,
  useFace: true,
  useBody: true,
  builtinAlphabet: true,
  showFace: true,
  mirrorView: true,
  showDebug: false,
};

const KEY = 'sgsl-sign-to-text.settings';

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    // Unreadable settings fall back to defaults.
  }
  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // Private browsing can refuse storage; settings then last for the session only.
  }
}

export function recognizerConfig(s: Settings): Partial<RecognizerConfig> {
  const k = s.sensitivity;
  return {
    mode: s.mode,
    signThreshold: 1.2 * k,
    staticThreshold: 0.9 * k,
    ruleMinScore: Math.min(0.8, Math.max(0.35, 0.55 + (1 - k) * 0.3)),
    builtinAlphabet: s.builtinAlphabet,
    useFace: s.useFace,
    useBody: s.useBody,
    holdMs: s.holdMs,
  };
}
