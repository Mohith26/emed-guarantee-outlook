import test from 'node:test';
import assert from 'node:assert/strict';
import { generateCohort, realizedOutcome, TRUE_BETA, MEDS } from '../src/generate.js';
import { fitAll, fitTrajectories, personWeeks, fitLogistic, simulate, summarize, quantile, todaySnapshot } from '../src/model.js';
import { conformalScale, rescale, pit, evaluateBacktest } from '../src/calibration.js';
import { makeRng } from '../src/rng.js';
import { TERMS, MAIN_COHORT } from '../src/terms.js';

test('generator is deterministic and censors at the cut-off', () => {
  const a = generateCohort(MAIN_COHORT), b = generateCohort(MAIN_COHORT);
  assert.deepEqual(a.members.map((m) => m.y), b.members.map((m) => m.y));
  for (const m of a.members) {
    assert.equal(m.y.length, m.weeksObserved + 1);
    assert.ok(m.weeksObserved >= 19 && m.weeksObserved < 52);
    if (m.dropWeek !== null) assert.ok(m.dropWeek <= m.weeksObserved);
  }
});

test('quantile matches linear interpolation', () => {
  assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(quantile([5, 1, 3], 0), 1);
  assert.equal(quantile([5, 1, 3], 1), 5);
});

test('drop-off hazard model recovers the true coefficients on a large cohort', () => {
  const c = generateCohort({ n: 6000, seed: 11, employerShift: { A: 0, hazard: 0 }, cutoff: '2027-03-01' });
  const { X, Y } = personWeeks(c);
  const f = fitLogistic(X, Y);
  for (let i = 0; i < TRUE_BETA.length; i++) assert.ok(Math.abs(f.beta[i] - TRUE_BETA[i]) < 3 * f.se[i] + 0.05, `beta[${i}] ${f.beta[i]} vs ${TRUE_BETA[i]}`);
});

test('trajectory model recovers medication plateaus and noise', () => {
  const c = generateCohort({ n: 1500, seed: 12, employerShift: { A: 0, hazard: 0 } });
  const { prior } = fitTrajectories(c);
  for (const m of MEDS) assert.ok(Math.abs(prior.A[m.key] - m.A) < 1.0, `${m.key} ${prior.A[m.key]} vs ${m.A}`);
  assert.ok(Math.abs(prior.sigma - 0.75) < 0.05);
  assert.ok(Math.abs(Math.exp(prior.muLogTau) - 14) < 1.5);
});

test('forecast for the demo cohort brackets the hidden realized outcome', () => {
  const c = generateCohort(MAIN_COHORT);
  const fit = fitAll(c);
  const sim = simulate(c, fit, { draws: 600, seed: 3 });
  const real = realizedOutcome(c);
  assert.ok(real.persistence >= quantile(sim.persist, 0.05) && real.persistence <= quantile(sim.persist, 0.95));
  assert.ok(real.meanLoss >= quantile(sim.meanLoss, 0.02) && real.meanLoss <= quantile(sim.meanLoss, 0.98));
  // the plain "on program today" count is above the 12-month forecast because dropout continues
  assert.ok(todaySnapshot(c).activeShare > quantile(sim.persist, 0.9));
});

test('re-engagement lever raises adherence odds with common random numbers', () => {
  const c = generateCohort(MAIN_COHORT);
  const fit = fitAll(c);
  const s0 = summarize(simulate(c, fit, { draws: 500, seed: 5 }), TERMS);
  const s1 = summarize(simulate(c, fit, { draws: 500, seed: 5, reengage: 0.5 }), TERMS);
  assert.ok(s1.pPersist >= s0.pPersist);
  assert.ok(s1.persist.p50 > s0.persist.p50);
});

test('summary probabilities are monotone in the target and credit is probability-weighted', () => {
  const sim = { draws: 4, persist: [0.7, 0.8, 0.85, 0.9], meanLoss: [12, 13, 14, 15] };
  const a = summarize(sim, { ...TERMS, persistTarget: 0.8, lossTarget: 13.5 });
  assert.equal(a.pPersist, 0.75); assert.equal(a.pLoss, 0.5); assert.equal(a.pBoth, 0.5);
  assert.equal(a.expectedCredit, 0.25 * TERMS.atRiskPersist + 0.5 * TERMS.atRiskLoss);
  const b = summarize(sim, { ...TERMS, persistTarget: 0.86, lossTarget: 13.5 });
  assert.ok(b.pPersist <= a.pPersist);
});

test('split-conformal scale recovers an under-dispersed forecast', () => {
  const rng = makeRng(1);
  const rows = Array.from({ length: 400 }, () => {
    const s = Array.from({ length: 400 }, () => rng.normal(0, 1));
    return { persist: s, loss: s, realized: { persistence: rng.normal(0, 1.5), meanLoss: rng.normal(0, 1) } };
  });
  const c = conformalScale(rows, 'persist');
  assert.ok(Math.abs(c - 1.5) < 0.25, 'scale ' + c);
  const fixed = evaluateBacktest(rows.map((r) => ({ ...r, todayActive: 0, persist: rescale(r.persist, c) })));
  assert.ok(Math.abs(fixed.coverage.persist.c80 - 0.8) < 0.06);
  assert.equal(pit([1, 2, 3, 4], 2.5), 0.5);
});
