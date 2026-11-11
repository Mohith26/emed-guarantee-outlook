// Guarantee Outlook core: hierarchical weight-trajectory model + discrete-time dropout
// hazard + Monte Carlo simulation of each member's remaining weeks to their 12-month mark.
import { makeRng, logistic } from './rng.js';
import { featuresAt, dot } from './features.js';
import { MEDS, isTitrationWeek, trajectory } from './generate.js';

export const GRID_A = Array.from({ length: 81 }, (_, i) => i * 0.5); // 0..40 % body weight
export const GRID_TAU = Array.from({ length: 24 }, (_, i) => Math.exp(Math.log(4) + i * (Math.log(60) - Math.log(4)) / 23));
const NG = GRID_A.length * GRID_TAU.length;

// ---------- 1. Trajectory model: L(t) = A (1 - exp(-t / tau)), empirical-Bayes grid posterior ----------
export function fitTrajectories(cohort, { iters = 6 } = {}) {
  const ms = cohort.members;
  const prior = { A: {}, sdA: 5, muLogTau: Math.log(15), sdLogTau: 0.45, sigma: 1.0 };
  for (const m of MEDS) prior.A[m.key] = 13;
  // precompute basis 1 - exp(-t/tau) for t = 0..52
  const basis = GRID_TAU.map((tau) => Array.from({ length: 53 }, (_, t) => 1 - Math.exp(-t / tau)));
  let post = null;
  for (let it = 0; it < iters; it++) {
    post = ms.map((m) => memberPosterior(m, prior, basis));
    // M-step (normal priors, EM with posterior moments)
    for (const med of MEDS) {
      const ps = post.filter((p, i) => ms[i].med === med.key);
      if (ps.length) prior.A[med.key] = ps.reduce((s, p) => s + p.meanA, 0) / ps.length;
    }
    let vA = 0, mT = 0, vT = 0;
    post.forEach((p, i) => { const mu = prior.A[ms[i].med]; vA += p.varA + (p.meanA - mu) ** 2; mT += p.meanLogTau; });
    mT /= post.length;
    post.forEach((p) => { vT += p.varLogTau + (p.meanLogTau - mT) ** 2; });
    prior.sdA = Math.sqrt(vA / post.length);
    prior.muLogTau = mT;
    prior.sdLogTau = Math.max(0.05, Math.sqrt(vT / post.length));
    // residual noise from posterior-expected squared residuals
    let ss = 0, nn = 0;
    post.forEach((p) => { ss += p.expSSR; nn += p.nObs; });
    prior.sigma = Math.max(0.2, Math.sqrt(ss / Math.max(1, nn)));
  }
  return { prior, post, basis };
}

function memberPosterior(m, prior, basis) {
  const obsT = [], obsY = [];
  for (let t = 1; t < m.y.length; t++) if (m.y[t] !== null && m.y[t] !== undefined) { obsT.push(t); obsY.push(m.y[t]); }
  const muA = prior.A[m.med], s2 = prior.sigma * prior.sigma;
  const logw = new Float64Array(NG);
  let mx = -Infinity, k = 0;
  for (let j = 0; j < GRID_TAU.length; j++) {
    const b = basis[j];
    const lt = Math.log(GRID_TAU[j]);
    const lpT = -0.5 * ((lt - prior.muLogTau) / prior.sdLogTau) ** 2;
    for (let i = 0; i < GRID_A.length; i++, k++) {
      const A = GRID_A[i];
      let ssr = 0;
      for (let o = 0; o < obsT.length; o++) { const r = obsY[o] - A * b[obsT[o]]; ssr += r * r; }
      const v = -0.5 * ssr / s2 - 0.5 * ((A - muA) / prior.sdA) ** 2 + lpT;
      logw[k] = v;
      if (v > mx) mx = v;
    }
  }
  const w = new Float64Array(NG);
  let tot = 0;
  for (let q = 0; q < NG; q++) { w[q] = Math.exp(logw[q] - mx); tot += w[q]; }
  let meanA = 0, meanLT = 0, eA2 = 0, eLT2 = 0, expSSR = 0;
  k = 0;
  for (let j = 0; j < GRID_TAU.length; j++) {
    const lt = Math.log(GRID_TAU[j]);
    for (let i = 0; i < GRID_A.length; i++, k++) {
      const p = w[k] / tot; w[k] = p;
      const A = GRID_A[i];
      meanA += p * A; eA2 += p * A * A; meanLT += p * lt; eLT2 += p * lt * lt;
      if (p > 1e-9) { let ssr = 0; for (let o = 0; o < obsT.length; o++) { const r = obsY[o] - A * basis[j][obsT[o]]; ssr += r * r; } expSSR += p * ssr; }
    }
  }
  const cdf = new Float64Array(NG);
  let c = 0;
  for (let q = 0; q < NG; q++) { c += w[q]; cdf[q] = c; }
  return { cdf, meanA, varA: Math.max(0, eA2 - meanA * meanA), meanLogTau: meanLT, varLogTau: Math.max(0, eLT2 - meanLT * meanLT), expSSR, nObs: obsT.length };
}

export function samplePosterior(p, u) {
  let lo = 0, hi = NG - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (p.cdf[mid] < u) lo = mid + 1; else hi = mid; }
  const j = Math.floor(lo / GRID_A.length), i = lo % GRID_A.length;
  return { A: GRID_A[i], tau: GRID_TAU[j] };
}

// ---------- 2. Dropout hazard: discrete-time logistic regression, fit by IRLS ----------
export function personWeeks(cohort) {
  const X = [], Y = [];
  for (const m of cohort.members) {
    const last = m.dropWeek !== null ? m.dropWeek : m.weeksObserved;
    for (let t = 1; t <= last; t++) {
      X.push(featuresAt(t, m.y, m.checkin, m.se));
      Y.push(m.dropWeek === t ? 1 : 0);
    }
  }
  return { X, Y };
}

export function fitLogistic(X, Y, { ridge = 1e-3, iters = 30 } = {}) {
  const d = X[0].length;
  let beta = new Array(d).fill(0);
  beta[0] = Math.log((Y.reduce((s, v) => s + v, 0) + 0.5) / (Y.length + 1));
  let H = null;
  for (let it = 0; it < iters; it++) {
    const g = new Array(d).fill(0);
    H = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let r = 0; r < X.length; r++) {
      const x = X[r], p = logistic(dot(beta, x)), wv = p * (1 - p);
      for (let a = 0; a < d; a++) { g[a] += (Y[r] - p) * x[a]; for (let b = 0; b < d; b++) H[a][b] += wv * x[a] * x[b]; }
    }
    for (let a = 1; a < d; a++) { g[a] -= ridge * beta[a]; H[a][a] += ridge; }
    const step = solve(H, g);
    let mx = 0;
    for (let a = 0; a < d; a++) { beta[a] += step[a]; mx = Math.max(mx, Math.abs(step[a])); }
    if (mx < 1e-8) break;
  }
  const cov = invert(H);
  return { beta, cov, chol: cholesky(cov), se: cov.map((row, i) => Math.sqrt(Math.max(0, row[i]))), n: X.length, events: Y.reduce((s, v) => s + v, 0) };
}

export function cholesky(S) {
  const n = S.length, L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let v = S[i][j];
    for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(v, 1e-12)) : v / L[j][j];
  }
  return L;
}

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}
function invert(A) {
  const n = A.length;
  return Array.from({ length: n }, (_, j) => solve(A, Array.from({ length: n }, (_, i) => (i === j ? 1 : 0)))).map((col, j, cols) => cols.map((c) => c[j]));
}

// ---------- 3. Engagement processes for the simulator ----------
export function fitEngagement(cohort) {
  // side-effect report rates (moderate/severe) on titration vs maintenance weeks, active weeks only
  let tN = 0, tE = 0, mN = 0, mE = 0;
  const rates = [];
  for (const m of cohort.members) {
    const last = m.dropWeek !== null ? m.dropWeek - 1 : m.weeksObserved;
    let n = 0, c = 0;
    for (let t = 1; t <= last; t++) {
      if (isTitrationWeek(t)) { tN++; if (m.se[t] >= 2) tE++; } else { mN++; if (m.se[t] >= 2) mE++; }
      n++; c += m.checkin[t];
    }
    rates.push({ n, c });
  }
  // beta prior for weekly check-in probability by method of moments on members with >= 8 weeks
  const ps = rates.filter((r) => r.n >= 8).map((r) => r.c / r.n);
  const mu = ps.reduce((s, v) => s + v, 0) / ps.length;
  const v = ps.reduce((s, x) => s + (x - mu) ** 2, 0) / ps.length;
  const k = Math.max(2, mu * (1 - mu) / Math.max(1e-4, v) - 1);
  return { pSETitration: tE / Math.max(1, tN), pSEMaint: mE / Math.max(1, mN), checkPrior: { a: mu * k, b: (1 - mu) * k }, rates };
}

export function fitAll(cohort) {
  const traj = fitTrajectories(cohort);
  const { X, Y } = personWeeks(cohort);
  const hazard = fitLogistic(X, Y);
  const engage = fitEngagement(cohort);
  return { traj, hazard, engage };
}

// ---------- 4. Monte Carlo forecast ----------
// scenario.reengage in [0,1): outreach cuts the dropout odds of members who missed >= 2 of their last 4 check-ins
export function simulate(cohort, fit, { draws = 1500, seed = 7, reengage = 0 } = {}) {
  const rng = makeRng(seed);
  const ms = cohort.members, N = ms.length;
  const { traj, hazard, engage } = fit;
  const sigma = traj.prior.sigma;
  const d0 = hazard.beta.length;
  const logOddsCut = reengage > 0 ? Math.log(1 - reengage) : 0;
  const persist = new Float64Array(draws), meanLoss = new Float64Array(draws);
  // per program week: share active and mean loss among active (observed check-ins where observed, simulated otherwise)
  const weekActive = Array.from({ length: 53 }, () => new Float64Array(draws));
  const weekLoss = Array.from({ length: 53 }, () => new Float64Array(draws));
  const segKeys = segmentKeys(ms);
  const seg = {};
  for (const key of segKeys.all) seg[key] = { act: new Float64Array(draws), loss: new Float64Array(draws), n: segKeys.count[key] };
  // observed parts are fixed across draws: precompute
  const obsActive = new Float64Array(53), obsLossSum = new Float64Array(53), obsLossN = new Float64Array(53);
  const reached = new Float64Array(53);
  for (const m of ms) {
    for (let t = 1; t <= 52; t++) {
      if (t > m.weeksObserved && (m.dropWeek === null)) continue; // future of still-active members: simulated
      if (t > m.weeksObserved && m.dropWeek !== null) { reached[t]++; continue; } // already dropped: known inactive
      reached[t]++;
      const active = m.dropWeek === null || t < m.dropWeek;
      if (active) { obsActive[t]++; if (m.y[t] !== null && m.y[t] !== undefined) { obsLossSum[t] += m.y[t]; obsLossN[t]++; } }
    }
  }
  const yBuf = new Array(53), cBuf = new Array(53), sBuf = new Array(53);
  for (let d = 0; d < draws; d++) {
    // parameter uncertainty: one hazard-coefficient draw per simulated future
    const z = Array.from({ length: d0 }, () => rng.normal());
    const beta = hazard.beta.map((b, i) => b + hazard.chol[i].reduce((s, l, k) => s + l * z[k], 0));
    let nAct = 0, sumLoss = 0;
    const wAct = new Float64Array(53), wSum = new Float64Array(53), wN = new Float64Array(53);
    for (const key of segKeys.all) { seg[key]._a = 0; seg[key]._l = 0; }
    for (let i = 0; i < N; i++) {
      const m = ms[i];
      if (m.dropWeek !== null) continue;
      const th = samplePosterior(traj.post[i], rng.uniform());
      const r = engage.rates[i];
      const pc = rng.beta(engage.checkPrior.a + r.c, engage.checkPrior.b + (r.n - r.c));
      const W = m.weeksObserved;
      for (let t = 0; t <= W; t++) { yBuf[t] = m.y[t]; cBuf[t] = m.checkin[t]; sBuf[t] = m.se[t]; }
      let alive = true, y52 = 0;
      for (let t = W + 1; t <= 52; t++) {
        sBuf[t] = rng.uniform() < (isTitrationWeek(t) ? engage.pSETitration : engage.pSEMaint) ? 2 : 0;
        const x = featuresAt(t, yBuf, cBuf, sBuf);
        let z = dot(beta, x);
        if (logOddsCut !== 0 && x[1] >= 2) z += logOddsCut;
        if (rng.uniform() < logistic(z)) { alive = false; break; }
        const val = trajectory(th.A, th.tau, t) + rng.normal(0, sigma);
        const p = sBuf[t] >= 2 ? pc * 0.8 : pc;
        if (rng.uniform() < p) { cBuf[t] = 1; yBuf[t] = val; } else { cBuf[t] = 0; yBuf[t] = null; }
        wAct[t]++; wSum[t] += val; wN[t]++;
        if (t === 52) y52 = val;
      }
      if (alive) {
        nAct++; sumLoss += y52;
        for (const key of m._segs) { seg[key]._a++; seg[key]._l += y52; }
      }
    }
    persist[d] = nAct / N;
    meanLoss[d] = sumLoss / Math.max(1, nAct);
    for (let t = 1; t <= 52; t++) {
      weekActive[t][d] = (obsActive[t] + wAct[t]) / N;
      const n = obsLossN[t] + wN[t];
      weekLoss[t][d] = n ? (obsLossSum[t] + wSum[t]) / n : NaN;
    }
    for (const key of segKeys.all) { seg[key].act[d] = seg[key]._a / seg[key].n; seg[key].loss[d] = seg[key]._l / Math.max(1, seg[key]._a); }
  }
  return { draws, persist, meanLoss, weekActive, weekLoss, seg, segLabels: segKeys.labels, reached };
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function segmentKeys(ms) {
  const count = {}, labels = {};
  for (const m of ms) {
    const mo = +m.enrollDate.slice(5, 7);
    const wave = 'wave:' + mo;
    const med = 'med:' + m.med;
    m._segs = [wave, med];
    for (const k of m._segs) count[k] = (count[k] || 0) + 1;
    labels[wave] = 'Enrolled ' + MONTHS[mo - 1] + ' 2026';
    labels[med] = MEDS.find((x) => x.key === m.med).label;
  }
  const all = Object.keys(count).sort((a, b) => (a < b ? -1 : 1));
  return { all, count, labels };
}

// ---------- 5. Summaries ----------
export function quantile(arr, q) {
  const a = Array.from(arr).filter((v) => !Number.isNaN(v)).sort((x, y) => x - y);
  if (!a.length) return NaN;
  const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

export function summarize(sim, terms) {
  const { lossTarget, persistTarget, atRiskLoss, atRiskPersist } = terms;
  let pL = 0, pP = 0, pB = 0;
  for (let d = 0; d < sim.draws; d++) {
    const a = sim.meanLoss[d] >= lossTarget, b = sim.persist[d] >= persistTarget;
    if (a) pL++; if (b) pP++; if (a && b) pB++;
  }
  pL /= sim.draws; pP /= sim.draws; pB /= sim.draws;
  const corr = pearson(sim.meanLoss, sim.persist);
  return {
    pLoss: pL, pPersist: pP, pBoth: pB, corr,
    loss: { p10: quantile(sim.meanLoss, 0.1), p50: quantile(sim.meanLoss, 0.5), p90: quantile(sim.meanLoss, 0.9) },
    persist: { p10: quantile(sim.persist, 0.1), p50: quantile(sim.persist, 0.5), p90: quantile(sim.persist, 0.9) },
    expectedCredit: (1 - pL) * atRiskLoss + (1 - pP) * atRiskPersist,
  };
}

function pearson(a, b) {
  const n = a.length; let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}

// What a plain dashboard shows today: share of enrolled members still active, average logged loss among active members
export function todaySnapshot(cohort) {
  const ms = cohort.members;
  const active = ms.filter((m) => m.dropWeek === null);
  let s = 0, n = 0, lapsed = 0, plateau = 0;
  for (const m of active) {
    for (let t = m.y.length - 1; t >= 1; t--) if (m.y[t] !== null && m.y[t] !== undefined) { s += m.y[t]; n++; break; }
    const x = featuresAt(m.weeksObserved + 1, m.y, m.checkin, m.se);
    if (x[1] >= 2) lapsed++;
    if (x[3] === 1) plateau++;
  }
  const wk = ms.map((m) => m.weeksObserved).sort((a, b) => a - b);
  return { enrolled: ms.length, active: active.length, activeShare: active.length / ms.length, meanLatestLoss: s / Math.max(1, n), lapsed, plateau, medianWeek: wk[Math.floor(wk.length / 2)], minWeek: wk[0], maxWeek: wk[wk.length - 1] };
}

// Naive projection a spreadsheet would make: carry today's active share and today's
// average loss forward with the program's typical 12-month curve shape, no further dropout.
export function naiveProjection(cohort, fit) {
  const snap = todaySnapshot(cohort);
  const tau = Math.exp(fit.traj.prior.muLogTau);
  let s = 0, n = 0;
  for (const m of cohort.members) {
    if (m.dropWeek !== null) continue;
    let last = null, lt = 0;
    for (let t = m.y.length - 1; t >= 1; t--) if (m.y[t] !== null && m.y[t] !== undefined) { last = m.y[t]; lt = t; break; }
    if (last === null) continue;
    s += last * (1 - Math.exp(-52 / tau)) / (1 - Math.exp(-lt / tau)); n++;
  }
  return { persistence: snap.activeShare, meanLoss: s / Math.max(1, n) };
}
