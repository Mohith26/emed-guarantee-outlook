// Back-test: forecast 60 independent synthetic employer cohorts at the same cut-off and
// compare with their hidden realized 12-month outcomes. Cohorts 1-30 calibrate the
// conformal width multipliers; cohorts 31-60 are held out for every reported metric.
import fs from 'node:fs';
import { generateCohort, realizedOutcome } from '../src/generate.js';
import { fitAll, simulate, todaySnapshot } from '../src/model.js';
import { evaluateBacktest, conformalScale, rescale } from '../src/calibration.js';
const K = +(process.argv[2] || 60), DRAWS = +(process.argv[3] || 800);
const rows = [];
for (let k = 0; k < K; k++) {
  const c = generateCohort({ seed: 9000 + k, n: 300 + ((k * 37) % 220), cutoff: '2026-11-10' });
  const fit = fitAll(c);
  const sim = simulate(c, fit, { draws: DRAWS, seed: 100 + k });
  rows.push({ seed: 9000 + k, n: c.n, todayActive: todaySnapshot(c).activeShare, realized: realizedOutcome(c), persist: Array.from(sim.persist), loss: Array.from(sim.meanLoss) });
  if (k % 10 === 9) console.error('cohort', k + 1);
}
const half = Math.floor(K / 2);
const calib = rows.slice(0, half), test = rows.slice(half);
const scale = { persist: conformalScale(calib, 'persist'), loss: conformalScale(calib, 'loss') };
const raw = evaluateBacktest(test);
const conf = evaluateBacktest(test.map((r) => ({ ...r, persist: rescale(r.persist, scale.persist), loss: rescale(r.loss, scale.loss) })));
const out = { generated: 'node scripts/backtest.mjs ' + K + ' ' + DRAWS, cohorts: K, calibrationCohorts: half, testCohorts: K - half, draws: DRAWS, scale, raw: { coverage: raw.coverage, brier: raw.brier }, ...conf };
fs.writeFileSync(new URL('../data/backtest.json', import.meta.url), JSON.stringify(out, null, 1));
console.log(JSON.stringify({ scale, rawCoverage: raw.coverage, confCoverage: conf.coverage, brier: conf.brier, onTrack: conf.onTrack, reliability: conf.reliability }, null, 1));
