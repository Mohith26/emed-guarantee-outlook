// Calibration metrics for the back-test: PIT coverage of central intervals, reliability of
// guarantee-probabilities across a ladder of targets, and Brier score vs a plain "on track" flag.
import { quantile } from './model.js';

export const PERSIST_TARGETS = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88];

export function pit(samples, x) {
  let below = 0;
  for (const v of samples) if (v < x) below++; else if (v === x) below += 0.5;
  return below / samples.length;
}

export function evaluateBacktest(rows) {
  const cov = { persist: { c50: 0, c80: 0 }, loss: { c50: 0, c80: 0 } };
  const pitHist = { persist: new Array(10).fill(0), loss: new Array(10).fill(0) };
  const events = [];
  for (const r of rows) {
    for (const key of ['persist', 'loss']) {
      const s = r[key], x = key === 'persist' ? r.realized.persistence : r.realized.meanLoss;
      if (x >= quantile(s, 0.25) && x <= quantile(s, 0.75)) cov[key].c50++;
      if (x >= quantile(s, 0.1) && x <= quantile(s, 0.9)) cov[key].c80++;
      pitHist[key][Math.min(9, Math.floor(pit(s, x) * 10))]++;
    }
    for (const tgt of PERSIST_TARGETS) {
      let p = 0; for (const v of r.persist) if (v >= tgt) p++;
      p /= r.persist.length;
      events.push({ p, naive: r.todayActive >= tgt ? 1 : 0, y: r.realized.persistence >= tgt ? 1 : 0 });
    }
  }
  const K = rows.length;
  for (const key of ['persist', 'loss']) { cov[key].c50 /= K; cov[key].c80 /= K; }
  const brier = { forecast: 0, naive: 0, climatology: 0 };
  const base = events.reduce((s, e) => s + e.y, 0) / events.length;
  for (const e of events) { brier.forecast += (e.p - e.y) ** 2; brier.naive += (e.naive - e.y) ** 2; brier.climatology += (base - e.y) ** 2; }
  for (const k in brier) brier[k] /= events.length;
  const bins = Array.from({ length: 5 }, (_, i) => ({ lo: i / 5, hi: (i + 1) / 5, n: 0, sumP: 0, sumY: 0 }));
  for (const e of events) { const b = bins[Math.min(4, Math.floor(e.p * 5))]; b.n++; b.sumP += e.p; b.sumY += e.y; }
  const reliability = bins.filter((b) => b.n > 0).map((b) => ({ lo: b.lo, hi: b.hi, n: b.n, meanP: b.sumP / b.n, hitRate: b.sumY / b.n }));
  const onTrackFlags = events.filter((e) => e.naive === 1);
  const onTrack = { flagged: onTrackFlags.length, missed: onTrackFlags.filter((e) => e.y === 0).length };
  return { coverage: cov, pitHist, brier, reliability, onTrack, events: events.length, baseRate: base };
}

// Split-conformal width calibration. Score = |realized - median| / (80% half-width).
// The finite-sample-corrected (1 - alpha) quantile of the calibration scores becomes a
// multiplier on the spread of the simulated samples around their median.
export function conformalScale(rows, key, alpha = 0.2) {
  const scores = rows.map((r) => {
    const s = r[key], x = key === 'persist' ? r.realized.persistence : r.realized.meanLoss;
    const med = quantile(s, 0.5), hw = (quantile(s, 0.9) - quantile(s, 0.1)) / 2;
    return Math.abs(x - med) / Math.max(1e-9, hw);
  }).sort((a, b) => a - b);
  const n = scores.length;
  const k = Math.min(n, Math.ceil((n + 1) * (1 - alpha)));
  return scores[k - 1];
}

export function rescale(samples, c) {
  const med = quantile(samples, 0.5);
  return Array.from(samples, (v) => med + c * (v - med));
}
