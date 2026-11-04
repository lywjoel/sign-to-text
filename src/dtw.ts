export interface DtwMatch {
  /** Mean per-step cost along the best path (after the span penalty). */
  cost: number;
  /** First and last query index covered by the match. */
  start: number;
  end: number;
}

export interface DtwOptions {
  /** Extra cost for each non-diagonal step, which discourages extreme warping. */
  stepPenalty?: number;
  /** Matched query length / template length outside this band is penalised. */
  minRatio?: number;
  maxRatio?: number;
}

/**
 * Subsequence dynamic time warping: the whole template must be matched, but it
 * may line up with any stretch of the query. This tolerates extra movement
 * before and after a sign (raising the hands, moving into the next sign).
 */
export function subsequenceDtw<T>(
  template: T[],
  query: T[],
  distance: (a: T, b: T) => number,
  opts: DtwOptions = {},
): DtwMatch {
  const m = template.length;
  const n = query.length;
  if (m === 0 || n === 0) return { cost: Infinity, start: 0, end: 0 };
  const pen = opts.stepPenalty ?? 0.15;
  const minRatio = opts.minRatio ?? 0.55;
  const maxRatio = opts.maxRatio ?? 1.8;

  let prevCost = new Float64Array(n);
  let prevLen = new Int32Array(n);
  let prevStart = new Int32Array(n);
  let curCost = new Float64Array(n);
  let curLen = new Int32Array(n);
  let curStart = new Int32Array(n);

  for (let j = 0; j < n; j++) {
    prevCost[j] = distance(template[0], query[j]);
    prevLen[j] = 1;
    prevStart[j] = j;
  }

  for (let i = 1; i < m; i++) {
    const ti = template[i];
    for (let j = 0; j < n; j++) {
      const c = distance(ti, query[j]);
      // Vertical step: this template frame shares query frame j with the previous one.
      let best = prevCost[j] + pen;
      let len = prevLen[j];
      let start = prevStart[j];
      let bestNorm = best / (len + 1);
      if (j > 0) {
        const diag = prevCost[j - 1];
        const diagNorm = diag / (prevLen[j - 1] + 1);
        if (diagNorm < bestNorm) {
          best = diag;
          bestNorm = diagNorm;
          len = prevLen[j - 1];
          start = prevStart[j - 1];
        }
        // Horizontal step: the template frame is held across several query frames.
        const horiz = curCost[j - 1] + pen;
        const horizNorm = horiz / (curLen[j - 1] + 1);
        if (horizNorm < bestNorm) {
          best = horiz;
          bestNorm = horizNorm;
          len = curLen[j - 1];
          start = curStart[j - 1];
        }
      }
      curCost[j] = best + c;
      curLen[j] = len + 1;
      curStart[j] = start;
    }
    [prevCost, curCost] = [curCost, prevCost];
    [prevLen, curLen] = [curLen, prevLen];
    [prevStart, curStart] = [curStart, prevStart];
  }

  let result: DtwMatch = { cost: Infinity, start: 0, end: 0 };
  for (let j = 0; j < n; j++) {
    const span = j - prevStart[j] + 1;
    const ratio = span / m;
    const excess = Math.max(0, Math.log2(minRatio / ratio), Math.log2(ratio / maxRatio));
    const cost = (prevCost[j] / prevLen[j]) * (1 + excess);
    if (cost < result.cost) result = { cost, start: prevStart[j], end: j };
  }
  return result;
}
