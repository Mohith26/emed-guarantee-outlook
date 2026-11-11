// Seeded synthetic employer cohort for an employer-sponsored GLP-1 program.
// Every member is simulated for the full 52 weeks; the "observed" view is censored at
// the data cut-off so the forecast only sees what a real dashboard would have.
import { makeRng, logistic } from './rng.js';
import { featuresAt, dot } from './features.js';

export const MEDS = [
  { key: 'zep', label: 'Zepbound (tirzepatide)', share: 0.45, A: 17.0 },
  { key: 'weg', label: 'Wegovy pen (semaglutide)', share: 0.40, A: 13.5 },
  { key: 'oral', label: 'Wegovy pill (oral semaglutide)', share: 0.15, A: 12.0 },
];
export const TRUE_BETA = [-6.5, 0.55, 1.1, 0.95, 0.5];
const DAY = 86400000;

export function isTitrationWeek(t) { return t >= 1 && t <= 17 && (t - 1) % 4 === 0; }
export function trajectory(A, tau, t) { return A * (1 - Math.exp(-t / tau)); }

export function generateCohort(opts = {}) {
  const {
    seed = 20261110,
    n = 412,
    enrollStart = '2026-02-02',
    enrollWeeks = 22,
    cutoff = '2026-11-10',
    employer = 'Calder Freight',
    employerShift = null, // {A, hazard}
  } = opts;
  const rng = makeRng(seed);
  const shift = employerShift || { A: rng.normal(0, 1.1), hazard: rng.normal(0, 0.3) };
  const beta = TRUE_BETA.slice();
  beta[0] += shift.hazard;
  const start = Date.parse(enrollStart + 'T00:00:00Z');
  const cut = Date.parse(cutoff + 'T00:00:00Z');
  const members = [];
  for (let i = 0; i < n; i++) {
    // more enrollment in the first open-enrollment weeks
    const wk = Math.min(enrollWeeks, Math.floor(enrollWeeks * Math.pow(rng.uniform(), 1.35)));
    const enroll = start + wk * 7 * DAY;
    const med = rng.pick(MEDS, MEDS.map((m) => m.share));
    const A = Math.max(1.5, rng.normal(med.A + shift.A, 4.8));
    const tau = Math.exp(rng.normal(Math.log(14), 0.3));
    const w0 = Math.exp(rng.normal(Math.log(226), 0.16));
    const pCheck = rng.beta(10, 1.6);
    const noise = 0.75;
    const y = [0], checkin = [1], se = [0];
    let dropWeek = null;
    for (let t = 1; t <= 52; t++) {
      // side-effect report (can be logged any time in the app)
      const pSE = isTitrationWeek(t) ? 0.07 : 0.018;
      se[t] = rng.bernoulli(pSE) ? (rng.bernoulli(0.38) ? 3 : 2) : 0;
      if (dropWeek === null) {
        const x = featuresAt(t, y, checkin, se);
        if (rng.bernoulli(logistic(dot(beta, x)))) dropWeek = t;
      }
      if (dropWeek !== null) { y[t] = null; checkin[t] = 0; se[t] = 0; continue; }
      const p = se[t] >= 2 ? pCheck * 0.8 : pCheck;
      if (rng.bernoulli(p)) { checkin[t] = 1; y[t] = trajectory(A, tau, t) + rng.normal(0, noise); }
      else { checkin[t] = 0; y[t] = null; }
    }
    const weeksObserved = Math.max(0, Math.min(52, Math.floor((cut - enroll) / (7 * DAY))));
    const y52 = dropWeek === null ? trajectory(A, tau, 52) + rng.normal(0, noise) : null;
    members.push({
      id: 'M' + String(1001 + i),
      enrollDate: new Date(enroll).toISOString().slice(0, 10),
      med: med.key,
      w0: Math.round(w0 * 10) / 10,
      weeksObserved,
      y: y.slice(0, weeksObserved + 1),
      checkin: checkin.slice(0, weeksObserved + 1),
      se: se.slice(0, weeksObserved + 1),
      dropWeek: dropWeek !== null && dropWeek <= weeksObserved ? dropWeek : null,
      truth: { A, tau, pCheck, dropWeek, active52: dropWeek === null, y52 },
    });
  }
  return { employer, cutoff, enrollStart, n, seed, shift, members };
}

// Realized guarantee metrics from the hidden truth (used only for back-testing and tests).
export function realizedOutcome(cohort) {
  const ms = cohort.members;
  const act = ms.filter((m) => m.truth.active52);
  const persistence = act.length / ms.length;
  const meanLoss = act.reduce((s, m) => s + m.truth.y52, 0) / Math.max(1, act.length);
  return { persistence, meanLoss, nActive52: act.length };
}
