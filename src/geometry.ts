export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const scale = (a: Vec3, k: number): Vec3 => ({ x: a.x * k, y: a.y * k, z: a.z * k });
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const norm = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
export const dist = (a: Vec3, b: Vec3): number => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

export function normalize(a: Vec3): Vec3 {
  const n = norm(a);
  return n > 1e-9 ? scale(a, 1 / n) : { x: 0, y: 0, z: 0 };
}

/** Unsigned angle between two vectors, in radians (0..π). */
export function angle(a: Vec3, b: Vec3): number {
  const na = norm(a);
  const nb = norm(b);
  if (na < 1e-9 || nb < 1e-9) return 0;
  return Math.acos(clamp(dot(a, b) / (na * nb), -1, 1));
}

/** Removes the component of `a` along unit vector `n`. */
export function reject(a: Vec3, n: Vec3): Vec3 {
  return sub(a, scale(n, dot(a, n)));
}

export const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);

/** 0 at `from`, 1 at `to`, linear in between. Works in either direction. */
export function ramp(x: number, from: number, to: number): number {
  if (from === to) return x >= to ? 1 : 0;
  return clamp((x - from) / (to - from), 0, 1);
}

/** Trapezoid membership: 0 below a, rises to 1 at b, stays 1 until c, falls to 0 at d. */
export function band(x: number, a: number, b: number, c: number, d: number): number {
  if (x <= a || x >= d) return 0;
  if (x < b) return (x - a) / (b - a);
  if (x <= c) return 1;
  return (d - x) / (d - c);
}
