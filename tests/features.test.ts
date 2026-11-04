import { describe, expect, it } from 'vitest';
import { extractFeatures, frameDistance, GROUPS, meanFrame, resample, SIGN_WEIGHTS, FEATURE_DIM } from '../src/features';
import { toSkeleton, FACE, POSE, type LM, type RawObs } from '../src/skeleton';
import { letterHand, skeleton } from './scenes';
import { LETTER_POSES, localHand, mirror, toCamera } from './synth';

function rawSigner(mirrored: boolean): RawObs {
  // A right-handed signer; `mirrored` gives the same scene seen in a mirror,
  // i.e. a left-handed signer doing the mirror-image sign.
  const fx = (x: number) => (mirrored ? 1 - x : x);
  const lm = (x: number, y: number, z = 0): LM => ({ x: fx(x), y, z, visibility: 1 });
  const pose: LM[] = Array.from({ length: 33 }, () => lm(0.5, 0.9));
  const side = (right: number, left: number) => (mirrored ? [left, right] : [right, left]);
  const set = (idxR: number, idxL: number, xr: number, xl: number, y: number) => {
    const [a, b] = side(idxR, idxL);
    pose[a] = lm(xr, y);
    pose[b] = lm(xl, y);
  };
  pose[POSE.NOSE] = lm(0.5, 0.42);
  set(POSE.RIGHT_SHOULDER, POSE.LEFT_SHOULDER, 0.39, 0.61, 0.75);
  set(POSE.RIGHT_ELBOW, POSE.LEFT_ELBOW, 0.33, 0.67, 1.0);
  set(POSE.RIGHT_WRIST, POSE.LEFT_WRIST, 0.36, 0.66, 0.6);
  set(POSE.RIGHT_EAR, POSE.LEFT_EAR, 0.45, 0.55, 0.4);

  const face: LM[] = Array.from({ length: 478 }, () => lm(0.5, 0.42));
  const fset = (idxR: number, idxL: number, xr: number, xl: number, y: number) => {
    const [a, b] = mirrored ? [idxL, idxR] : [idxR, idxL];
    face[a] = lm(xr, y);
    face[b] = lm(xl, y);
  };
  face[FACE.FOREHEAD] = lm(0.5, 0.3);
  face[FACE.NOSE_TIP] = lm(0.5, 0.42);
  face[FACE.UPPER_LIP] = lm(0.5, 0.46);
  face[FACE.LOWER_LIP] = lm(0.5, 0.48);
  face[FACE.CHIN] = lm(0.5, 0.52);
  fset(FACE.RIGHT_EYE_OUTER, FACE.LEFT_EYE_OUTER, 0.46, 0.54, 0.38);
  fset(FACE.RIGHT_EYE_INNER, FACE.LEFT_EYE_INNER, 0.48, 0.52, 0.38);
  fset(FACE.RIGHT_CHEEK, FACE.LEFT_CHEEK, 0.46, 0.54, 0.45);
  fset(FACE.RIGHT_EDGE, FACE.LEFT_EDGE, 0.43, 0.57, 0.42);

  // Dominant (right) hand making K near the right shoulder; the other hand making B lower down.
  const toLm = (pts: { x: number; y: number; z: number }[], cx: number, cy: number) =>
    pts.map((p) => ({ x: fx(cx + p.x), y: cy + p.y, z: p.z, visibility: 1 }));
  const worldLm = (pts: { x: number; y: number; z: number }[]) =>
    pts.map((p) => ({ x: mirrored ? -p.x : p.x, y: p.y, z: p.z, visibility: 1 }));
  const k = toCamera(localHand(LETTER_POSES.K.pose), LETTER_POSES.K.orient);
  const b = mirror(toCamera(localHand(LETTER_POSES.B.pose), LETTER_POSES.B.orient));
  const right = { landmarks: toLm(k, 0.36, 0.6), world: worldLm(k) };
  const left = { landmarks: toLm(b, 0.62, 0.8), world: worldLm(b) };

  return {
    t: 0,
    width: 1000,
    height: 1000,
    left: mirrored ? right : left,
    right: mirrored ? left : right,
    pose,
    face,
    blendshapes: new Map([['jawOpen', 0.7], ['browInnerUp', 0.2]]),
  };
}

describe('features', () => {
  it('gives a left-handed signer the same features as the mirror-image right-handed signer', () => {
    const right = extractFeatures(toSkeleton(rawSigner(false), 'right'));
    const left = extractFeatures(toSkeleton(rawSigner(true), 'left'));
    expect(left.dom && left.non && right.dom && right.non).toBe(true);
    for (let i = 0; i < FEATURE_DIM; i++) expect(left.v[i]).toBeCloseTo(right.v[i], 4);
    expect(frameDistance(left, right, SIGN_WEIGHTS)).toBeLessThan(1e-4);
  });

  it('puts the dominant hand on the dominant side of the body', () => {
    const f = extractFeatures(skeleton(0, letterHand('B', [0.6, -0.2])));
    expect(f.domC![0]).toBeGreaterThan(0.4);
    expect(f.domC![1]).toBeLessThan(0);
  });

  it('is zero for identical frames and large when a hand is missing', () => {
    const a = extractFeatures(skeleton(0, letterHand('B', [0.5, 0])));
    const b = extractFeatures(skeleton(0, null));
    expect(frameDistance(a, a, SIGN_WEIGHTS)).toBe(0);
    expect(frameDistance(a, b, SIGN_WEIGHTS)).toBeGreaterThan(1);
  });

  it('separates different handshapes at the same place', () => {
    const a = extractFeatures(skeleton(0, letterHand('B', [0.5, 0])));
    const b = extractFeatures(skeleton(0, letterHand('A', [0.5, 0])));
    const near = extractFeatures(skeleton(0, letterHand('B', [0.52, 0.02])));
    expect(frameDistance(a, b, SIGN_WEIGHTS)).toBeGreaterThan(0.8);
    expect(frameDistance(a, near, SIGN_WEIGHTS)).toBeLessThan(0.2);
  });

  it('notices a hand touching the face', () => {
    const far = extractFeatures(skeleton(0, letterHand('B', [0.8, 0.3])));
    // Wrist low enough that the fingertips reach the chin.
    const chin = extractFeatures(skeleton(0, letterHand('B', [0.0, -0.25])));
    const [c0] = GROUPS.domCon;
    const chinProximity = 4; // forehead, eyes, nose, mouth, *chin*
    expect(chin.v[c0 + chinProximity]).toBeGreaterThan(far.v[c0 + chinProximity]);
  });

  it('averages and resamples frames', () => {
    const frames = [0, 33, 66, 100].map((t, i) => extractFeatures(skeleton(t, letterHand('B', [0.5 + i * 0.1, 0]))));
    const mean = meanFrame(frames);
    expect(mean.dom).toBe(true);
    const r = resample(frames, 15);
    expect(r.length).toBe(2);
    expect(r[0].t).toBe(0);
  });
});
