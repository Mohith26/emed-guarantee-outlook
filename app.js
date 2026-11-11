import { generateCohort, MEDS } from './src/generate.js';
import { fitAll, simulate, summarize, todaySnapshot, quantile } from './src/model.js';
import { rescale } from './src/calibration.js';
import { TERMS, MAIN_COHORT } from './src/terms.js';
import { FEATURE_NAMES } from './src/features.js';

const DRAWS = 1200, SEED = 7;
const $ = (id) => document.getElementById(id);
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const pp = (v, d = 1) => v.toFixed(d) + '%';
const usd = (v) => '$' + Math.round(v).toLocaleString('en-US');
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtDate = (iso) => { const d = new Date(iso + 'T00:00:00Z'); return MON[d.getUTCMonth()] + ' ' + d.getUTCDate() + ', ' + d.getUTCFullYear(); };
const addDays = (iso, n) => new Date(Date.parse(iso + 'T00:00:00Z') + n * 86400000).toISOString().slice(0, 10);

const state = { terms: { ...TERMS }, metric: 'persist', reengage: 0, scale: { persist: 1, loss: 1 } };
let cohort, fit, snap, calib, baseSim, scenSim;
const simCache = new Map();

function scaled(sim) {
  const c = state.scale;
  return {
    ...sim,
    persist: rescale(sim.persist, c.persist),
    meanLoss: rescale(sim.meanLoss, c.loss),
    weekActive: sim.weekActive.map((a) => rescale(a, c.persist)),
    weekLoss: sim.weekLoss.map((a) => rescale(a, c.loss)),
    seg: Object.fromEntries(Object.entries(sim.seg).map(([k, s]) => [k, { ...s, act: rescale(s.act, c.persist), loss: rescale(s.loss, c.loss) }])),
  };
}
function runSim(r) {
  if (!simCache.has(r)) simCache.set(r, scaled(simulate(cohort, fit, { draws: DRAWS, seed: SEED, reengage: r })));
  return simCache.get(r);
}

async function init() {
  try { calib = await (await fetch('data/backtest.json')).json(); state.scale = { persist: calib.scale.persist, loss: calib.scale.loss }; } catch (e) { calib = null; }
  cohort = generateCohort(MAIN_COHORT);
  fit = fitAll(cohort);
  snap = todaySnapshot(cohort);
  baseSim = runSim(0);
  scenSim = baseSim;
  const enr = cohort.members.map((m) => m.enrollDate).sort();
  $('orgName').textContent = cohort.employer;
  $('cohortLine').textContent = `2026 program cohort · ${cohort.n} members enrolled ${fmtDate(enr[0])} – ${fmtDate(enr[enr.length - 1])} · measured at each member's 12-month mark (${fmtDate(addDays(enr[0], 364))} – ${fmtDate(addDays(enr[enr.length - 1], 364))})`;
  $('cutoffChip').textContent = 'Data through ' + fmtDate(cohort.cutoff);
  const toGo = cohort.members.filter((m) => m.dropWeek === null).reduce((s, m) => s + (52 - m.weeksObserved), 0) / snap.active;
  $('infoText').textContent = `Each member is measured at their own 12-month mark, and none of your ${snap.active} active members has reached it yet (on average ${toGo.toFixed(1)} weeks to go). This page fits a weight-trajectory model and a drop-off model to your cohort's check-ins, simulates the rest of the program year ${DRAWS.toLocaleString('en-US')} times, and reports how often each guarantee is met.`;
  renderTerms();
  renderAll();
  renderHazard();
  renderCalibration();
  wire();
}

function renderAll() {
  const s = summarize(baseSim, state.terms);
  renderGauges(s);
  renderChart();
  renderLever(s);
  renderSegments();
}

function ringSVG(p) {
  const r = 51, c = 2 * Math.PI * r;
  const col = p >= 0.75 ? 'var(--good)' : p >= 0.4 ? 'var(--sev-warn)' : 'var(--sev-bad)';
  return `<svg width="120" height="120" viewBox="0 0 120 120"><circle cx="60" cy="60" r="${r}" fill="none" stroke="#ececef" stroke-width="9"/>
  <circle cx="60" cy="60" r="${r}" fill="none" stroke="${col}" stroke-width="9" stroke-linecap="round" stroke-dasharray="${(c * Math.max(0.005, p)).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 60 60)"/></svg>`;
}
function badge(p) {
  if (p >= 0.75) return '<span class="badge ok"><span class="ms small">check_circle</span>Likely</span>';
  if (p >= 0.4) return '<span class="badge warn"><span class="ms small">error</span>At risk</span>';
  return '<span class="badge bad"><span class="ms small">warning</span>Unlikely</span>';
}

function renderGauges(s) {
  const t = state.terms;
  const todayOk = snap.activeShare >= t.persistTarget;
  $('cardPersist').innerHTML = `<div class="g-top"><span class="g-title">Adherence guarantee</span>${badge(s.pPersist)}</div>
  <div class="g-body"><div class="ring">${ringSVG(s.pPersist)}<div class="ring-c"><div class="ring-v">${pct(s.pPersist)}</div><div class="ring-l">chance of<br>meeting it</div></div></div>
  <div class="kv">
    <div class="kv-row"><span>Target</span><b>≥ ${pct(t.persistTarget)} on program</b></div>
    <div class="kv-row"><span>Today</span><b>${pct(snap.activeShare, 1)}<small>${todayOk ? 'above target by today\u2019s count' : 'below target today'}</small></b></div>
    <div class="kv-row"><span>Month 12</span><b>${pct(s.persist.p50, 1)}<small>80% range ${pct(s.persist.p10, 1)} – ${pct(s.persist.p90, 1)}</small></b></div>
  </div></div>
  <div class="row-btn"><span title="Missed 2+ of their last 4 weekly check-ins">Behind on check-ins <span class="muted">2+ of last 4</span></span><b>${snap.lapsed} members <span class="ms small">chevron_right</span></b></div>`;
  $('cardLoss').innerHTML = `<div class="g-top"><span class="g-title">Weight-loss guarantee</span>${badge(s.pLoss)}</div>
  <div class="g-body"><div class="ring">${ringSVG(s.pLoss)}<div class="ring-c"><div class="ring-v">${pct(s.pLoss)}</div><div class="ring-l">chance of<br>meeting it</div></div></div>
  <div class="kv">
    <div class="kv-row"><span>Target</span><b>≥ ${pp(t.lossTarget)} avg loss</b></div>
    <div class="kv-row"><span>Today</span><b>${pp(snap.meanLatestLoss)}<small>latest check-in, active members</small></b></div>
    <div class="kv-row"><span>Month 12</span><b>${pp(s.loss.p50)}<small>80% range ${pp(s.loss.p10)} – ${pp(s.loss.p90)}</small></b></div>
  </div></div>
  <div class="row-btn"><span title="Under 0.6 points of extra loss across about 6 weeks">On a weight plateau <span class="muted">6 wks</span></span><b>${snap.plateau} members <span class="ms small">chevron_right</span></b></div>`;
  $('cardCredit').innerHTML = `<div class="g-top"><span class="g-title">Expected fee credit</span><span class="badge ${s.pBoth >= 0.75 ? 'ok' : s.pBoth >= 0.4 ? 'warn' : 'bad'}">Both met: ${pct(s.pBoth)}</span></div>
  <div class="credit-big">${usd(s.expectedCredit)}</div>
  <div class="credit-sub">Probability-weighted credit owed to ${cohort.employer} if a guarantee is missed</div>
  <div class="row-btn"><span>Adherence missed <span class="muted">${pct(1 - s.pPersist)} × ${usd(t.atRiskPersist)}</span></span><b>${usd((1 - s.pPersist) * t.atRiskPersist)}</b></div>
  <div class="row-btn"><span>Weight loss missed <span class="muted">${pct(1 - s.pLoss)} × ${usd(t.atRiskLoss)}</span></span><b>${usd((1 - s.pLoss) * t.atRiskLoss)}</b></div>
  <div class="row-btn"><span>Outcomes move together</span><b title="Correlation across simulated futures">r = ${s.corr.toFixed(2)}</b></div>`;
}

function renderChart() {
  const sim = baseSim, metric = state.metric, t = state.terms;
  const el = $('chart');
  const W = el.clientWidth || 760, H = 300, m = { l: 48, r: 16, t: 14, b: 34 };
  const arr = metric === 'persist' ? sim.weekActive : sim.weekLoss;
  const pts = [];
  for (let w = 1; w <= 52; w++) {
    const a = arr[w];
    pts.push({ w, q10: quantile(a, 0.1), q25: quantile(a, 0.25), q50: quantile(a, 0.5), q75: quantile(a, 0.75), q90: quantile(a, 0.9) });
  }
  let ymin, ymax, fmt, target;
  if (metric === 'persist') { ymin = 0.7; ymax = 1.0; fmt = (v) => Math.round(v * 100) + '%'; target = t.persistTarget; }
  else { ymin = 0; ymax = 16; fmt = (v) => v.toFixed(0) + '%'; target = t.lossTarget; }
  const x = (w) => m.l + (w - 1) / 51 * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (Math.min(ymax, Math.max(ymin, v)) - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  const area = (lo, hi) => 'M' + pts.map((p) => `${x(p.w).toFixed(1)},${y(p[hi]).toFixed(1)}`).join('L') + 'L' + pts.slice().reverse().map((p) => `${x(p.w).toFixed(1)},${y(p[lo]).toFixed(1)}`).join('L') + 'Z';
  const line = 'M' + pts.map((p) => `${x(p.w).toFixed(1)},${y(p.q50).toFixed(1)}`).join('L');
  const ticks = [];
  const step = metric === 'persist' ? 0.05 : 4;
  for (let v = ymin; v <= ymax + 1e-9; v += step) ticks.push(v);
  const xt = [1, 13, 26, 39, 52];
  const nowW = snap.medianWeek;
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  ticks.forEach((v) => { svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#eeeff1"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="#7d828a">${fmt(v)}</text>`; });
  xt.forEach((w) => { svg += `<text x="${x(w)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="#7d828a">Wk ${w}</text>`; });
  svg += `<rect x="${x(nowW)}" y="${m.t}" width="${x(52) - x(nowW)}" height="${H - m.t - m.b}" fill="#f7f9fd"/>`;
  svg += `<path d="${area('q10', 'q90')}" fill="#057fff" fill-opacity=".12"/><path d="${area('q25', 'q75')}" fill="#057fff" fill-opacity=".22"/>`;
  svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(target)}" y2="${y(target)}" stroke="#c8412f" stroke-width="1.5" stroke-dasharray="6 5"/>`;
  svg += `<text x="${m.l + 6}" y="${y(target) - 6}" font-size="11" fill="#c8412f" font-weight="500">Guarantee ${fmt(target)}</text>`;
  if (metric === 'persist') {
    svg += `<line x1="${x(nowW)}" x2="${W - m.r}" y1="${y(snap.activeShare)}" y2="${y(snap.activeShare)}" stroke="#7d828a" stroke-width="1.5" stroke-dasharray="2 4"/>`;
    svg += `<text x="${x(nowW) + 6}" y="${y(snap.activeShare) - 6}" font-size="11" fill="#494e54">Today: ${pct(snap.activeShare, 1)} still on program</text>`;
  }
  svg += `<path d="${line}" fill="none" stroke="#111214" stroke-width="2"/>`;
  svg += `<line x1="${x(nowW)}" x2="${x(nowW)}" y1="${m.t}" y2="${H - m.b}" stroke="#494e54" stroke-width="1"/>`;
  svg += `<text x="${x(nowW) + 6}" y="${m.t + 12}" font-size="11" fill="#494e54" font-weight="500">Today · median member at week ${nowW}</text>`;
  const end = pts[51];
  svg += `<circle cx="${x(52)}" cy="${y(end.q50)}" r="4" fill="#111214"/>`;
  svg += `<rect id="hover" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/><line id="hl" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="#111214" stroke-opacity=".25" visibility="hidden"/></svg>`;
  el.innerHTML = svg;
  $('chartSub').textContent = metric === 'persist'
    ? 'Share of all enrolled members still on program, by program week'
    : 'Average % body-weight loss among members still on program, by program week';
  $('legend').innerHTML = `<span><i class="sw line" style="background:#111214"></i>Median (logged check-ins, then simulated)</span><span><i class="sw" style="background:rgba(5,127,255,.3)"></i>50% range</span><span><i class="sw" style="background:rgba(5,127,255,.13)"></i>80% range</span><span><i class="sw dash"></i>Guarantee target</span>${metric === 'persist' ? '<span><i class="sw dot"></i>Today\u2019s count carried forward</span>' : ''}`;
  const hov = el.querySelector('#hover'), hl = el.querySelector('#hl'), tip = $('tip');
  hov.addEventListener('mousemove', (e) => {
    const r = hov.getBoundingClientRect();
    const w = Math.max(1, Math.min(52, Math.round(1 + (e.clientX - r.left) / r.width * 51)));
    const p = pts[w - 1];
    hl.setAttribute('x1', x(w)); hl.setAttribute('x2', x(w)); hl.setAttribute('visibility', 'visible');
    const f = metric === 'persist' ? (v) => pct(v, 1) : (v) => pp(v);
    tip.innerHTML = `<b>Week ${w}</b><br>Median ${f(p.q50)}<br>50%: ${f(p.q25)} – ${f(p.q75)}<br>80%: ${f(p.q10)} – ${f(p.q90)}`;
    tip.hidden = false; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY - 10) + 'px';
  });
  hov.addEventListener('mouseleave', () => { tip.hidden = true; hl.setAttribute('visibility', 'hidden'); });
}

function renderHazard() {
  const h = fit.hazard;
  const rows = [1, 2, 3, 4].map((i) => ({ name: FEATURE_NAMES[i], or: Math.exp(h.beta[i]), lo: Math.exp(h.beta[i] - 1.96 * h.se[i]), hi: Math.exp(h.beta[i] + 1.96 * h.se[i]) }));
  const unit = ['per missed check-in', 'reported in app', 'under 0.6 pts in 6 wks', 'vs later weeks'];
  const lx = (v) => Math.max(0, Math.min(1, (Math.log(v) + Math.log(8)) / (2 * Math.log(8))));
  $('hazSub').textContent = `Discrete-time drop-off model on ${h.n.toLocaleString('en-US')} member-weeks and ${h.events} drop-offs`;
  $('hazard').innerHTML = `<table class="htable"><thead><tr><th>Signal in the weeks before</th><th style="text-align:right;white-space:nowrap">Odds ×</th><th style="text-align:right">95% CI</th><th></th></tr></thead><tbody>${rows.map((r, i) => `<tr><td>${r.name[0].toUpperCase() + r.name.slice(1)}<br><span class="muted" style="font-size:11.5px">${unit[i]}</span></td><td class="num">${r.or.toFixed(2)}</td><td class="ci">${r.lo.toFixed(2)} – ${r.hi.toFixed(2)}</td><td><div class="orbar"><i style="left:${(lx(Math.min(r.lo, 1)) * 100).toFixed(1)}%;width:${((lx(r.hi) - lx(Math.min(r.lo, 1))) * 100).toFixed(1)}%"></i></div></td></tr>`).join('')}</tbody></table>
  <div class="foot-note">Baseline weekly drop-off: ${pct(1 / (1 + Math.exp(-h.beta[0])), 2)} for an engaged member after week 12. The simulator draws these coefficients from their sampling distribution, so model uncertainty widens the ranges above.</div>`;
}

function renderLever(base) {
  scenSim = runSim(state.reengage);
  const s = summarize(scenSim, state.terms);
  const d = (a, b) => { const v = (a - b) * 100; const cls = Math.abs(v) < 0.5 ? 'flat' : v > 0 ? 'up' : 'down'; return `<span class="d ${cls}">${cls === 'flat' ? 'no change' : (v > 0 ? '+' : '') + v.toFixed(0) + ' pts'}</span>`; };
  const dc = s.expectedCredit - base.expectedCredit;
  $('reVal').textContent = Math.round(state.reengage * 100) + '%';
  $('leverOut').innerHTML = `<div class="lo-cell"><div class="l">Adherence guarantee</div><div class="v">${pct(s.pPersist)}</div>${d(s.pPersist, base.pPersist)}</div>
  <div class="lo-cell"><div class="l">Weight-loss guarantee</div><div class="v">${pct(s.pLoss)}</div>${d(s.pLoss, base.pLoss)}</div>
  <div class="lo-cell"><div class="l">Expected fee credit</div><div class="v">${usd(s.expectedCredit)}</div><span class="d ${Math.abs(dc) < 1 ? 'flat' : dc < 0 ? 'up' : 'down'}">${Math.abs(dc) < 1 ? 'no change' : (dc < 0 ? '−' : '+') + usd(Math.abs(dc))}</span></div>`;
  renderSegments();
}

function renderSegments() {
  const sim = scenSim;
  const groups = cohort.members.reduce((acc, m) => { for (const k of m._segs) { (acc[k] = acc[k] || { n: 0, act: 0 }); acc[k].n++; if (m.dropWeek === null) acc[k].act++; } return acc; }, {});
  const keys = Object.keys(sim.seg).sort((a, b) => (a.startsWith('wave') === b.startsWith('wave') ? (a.startsWith('wave') ? +a.split(':')[1] - +b.split(':')[1] : MEDS.findIndex((x) => 'med:' + x.key === a) - MEDS.findIndex((x) => 'med:' + x.key === b)) : a.startsWith('wave') ? -1 : 1));
  const rows = keys.map((k) => {
    const g = groups[k], s = sim.seg[k];
    if (g.n < 11) return `<tr class="sup"><td>${sim.segLabels[k]}</td><td class="r">&lt; 11</td><td class="r" colspan="3">Hidden to protect member privacy</td></tr>`;
    return `<tr><td>${sim.segLabels[k]}</td><td class="r">${g.n}</td><td class="r">${pct(g.act / g.n, 0)}</td><td class="r">${pct(quantile(s.act, 0.5), 0)} <small>${pct(quantile(s.act, 0.1), 0)}–${pct(quantile(s.act, 0.9), 0)}</small></td><td class="r">${pp(quantile(s.loss, 0.5))}</td></tr>`;
  });
  $('segments').innerHTML = `<table class="stable"><thead><tr><th>Segment</th><th class="r">Members</th><th class="r">On program today</th><th class="r">At month 12 (80% range)</th><th class="r">Avg loss at month 12</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

function renderTerms() {
  const t = state.terms;
  const items = [
    ['persistTarget', 'Adherence target', 'Share of enrolled members on program at month 12', (t.persistTarget * 100).toFixed(0), '%'],
    ['lossTarget', 'Weight-loss target', 'Average % loss at month 12, members on program', t.lossTarget.toFixed(1), '%'],
    ['atRiskPersist', 'Credit if adherence missed', 'Fee credit to employer', t.atRiskPersist, '$'],
    ['atRiskLoss', 'Credit if weight loss missed', 'Fee credit to employer', t.atRiskLoss, '$'],
  ];
  $('terms').innerHTML = items.map(([k, l, s, v, u]) => `<div class="term"><div class="tl">${l}<small>${s}</small></div><div><input data-k="${k}" data-u="${u}" value="${v}" inputmode="decimal" aria-label="${l}"></div></div>`).join('') +
    `<div class="note"><span class="ms small fill">info</span><span>Targets and credits here are illustrative inputs, not eMed contract terms. Probabilities update without re-simulating.</span></div>` + howItWorks();
}

function howItWorks() {
  const pr = fit.traj.prior;
  const medA = MEDS.map((m) => `${m.label.split(' (')[0]} ${pr.A[m.key].toFixed(1)}%`).join(', ');
  const rows = [
    ['monitor_weight', 'Weight trajectory per member', `Loss = A(1 − e^(−t/τ)); empirical-Bayes priors by medication (plateau ${medA}; τ ≈ ${Math.exp(pr.muLogTau).toFixed(0)} wks), noise ${pr.sigma.toFixed(2)} pts`],
    ['event_busy', 'Drop-off model', `Weekly logistic hazard on check-in, side-effect and plateau signals (${fit.hazard.events} drop-offs so far)`],
    ['casino', 'Simulation', `${DRAWS.toLocaleString('en-US')} futures: posterior draw per member, weekly check-ins, side effects and drop-off to week 52`],
    ['tune', 'Calibrated ranges', calib ? `Split conformal on ${calib.calibrationCohorts} back-test cohorts widens ranges × ${calib.scale.persist.toFixed(2)} (adherence) and × ${calib.scale.loss.toFixed(2)} (weight loss)` : 'Back-test file not found'],
  ];
  return `<div class="how"><h3>How the forecast works</h3>${rows.map(([ic, t, d]) => `<div class="how-row"><span class="how-ic"><span class="ms">${ic}</span></span><div><div class="how-t">${t}</div><div class="how-s">${d}</div></div></div>`).join('')}</div>`;
}

function renderCalibration() {
  if (!calib) { $('reliab').innerHTML = '<div class="loading">Back-test file not found.</div>'; return; }
  const W = 420, H = 300, m = { l: 44, r: 12, t: 12, b: 40 };
  const x = (v) => m.l + v * (W - m.l - m.r), y = (v) => m.t + (1 - v) * (H - m.t - m.b);
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for (let v = 0; v <= 1.0001; v += 0.25) svg += `<line x1="${x(0)}" x2="${x(1)}" y1="${y(v)}" y2="${y(v)}" stroke="#eeeff1"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="#7d828a">${Math.round(v * 100)}%</text><text x="${x(v)}" y="${H - 22}" text-anchor="middle" font-size="11" fill="#7d828a">${Math.round(v * 100)}%</text>`;
  svg += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#c4c7cc" stroke-dasharray="4 4"/>`;
  const rel = calib.reliability;
  svg += `<path d="M${rel.map((b) => `${x(b.meanP).toFixed(1)},${y(b.hitRate).toFixed(1)}`).join('L')}" fill="none" stroke="#057fff" stroke-width="2"/>`;
  rel.forEach((b) => { svg += `<circle cx="${x(b.meanP)}" cy="${y(b.hitRate)}" r="${(3 + Math.sqrt(b.n) * 0.7).toFixed(1)}" fill="#057fff" fill-opacity=".85" stroke="#fff" stroke-width="1.5"><title>${b.n} forecasts · said ${pct(b.meanP)} · happened ${pct(b.hitRate)}</title></circle>`; });
  svg += `<text x="${(x(0) + x(1)) / 2}" y="${H - 4}" text-anchor="middle" font-size="11.5" fill="#494e54">Forecast chance of meeting the adherence target</text>`;
  svg += `<text transform="translate(12 ${(y(0) + y(1)) / 2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="#494e54">How often it was met</text></svg>`;
  $('reliab').innerHTML = svg;
  $('calibSub').textContent = `${calib.cohorts} simulated employer cohorts with known 12-month outcomes: ${calib.calibrationCohorts} tune the interval widths (split conformal), the other ${calib.testCohorts} are held out for every number below`;
  const c = calib.coverage;
  const missRate = calib.onTrack.missed / calib.onTrack.flagged;
  $('calibStats').innerHTML = `<div class="cs"><div class="l">80% ranges that contained the real adherence result</div><div class="v">${pct(c.persist.c80)}</div><div class="s">of ${calib.testCohorts} held-out cohorts (raw model: ${pct(calib.raw.coverage.persist.c80)})</div></div>
  <div class="cs"><div class="l">80% ranges that contained the real weight-loss result</div><div class="v">${pct(c.loss.c80)}</div><div class="s">raw model ${pct(calib.raw.coverage.loss.c80)}; widths × ${calib.scale.loss.toFixed(2)} after conformal step</div></div>
  <div class="cs"><div class="l">Brier score, chance of meeting adherence targets</div><div class="v">${calib.brier.forecast.toFixed(3)}</div><div class="s">vs ${calib.brier.naive.toFixed(3)} for an \u201con track today\u201d flag (lower is better)</div></div>
  <div class="cs"><div class="l">\u201cOn track today\u201d flags that ended in a miss</div><div class="v">${pct(missRate)}</div><div class="s">${calib.onTrack.missed} of ${calib.onTrack.flagged} target checks across held-out cohorts</div></div>`;
}

function wire() {
  document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
    state.metric = b.dataset.metric; renderChart();
  }));
  let timer = null;
  $('reengage').addEventListener('input', (e) => {
    state.reengage = +e.target.value / 100;
    $('reVal').textContent = e.target.value + '%';
    clearTimeout(timer);
    timer = setTimeout(() => renderLever(summarize(baseSim, state.terms)), 30);
  });
  $('terms').addEventListener('change', (e) => {
    const k = e.target.dataset.k; if (!k) return;
    let v = parseFloat(String(e.target.value).replace(/[^0-9.]/g, ''));
    if (!Number.isFinite(v)) { renderTerms(); return; }
    if (k === 'persistTarget') v = Math.min(99, Math.max(1, v)) / 100;
    if (k === 'lossTarget') v = Math.min(40, Math.max(0, v));
    state.terms[k] = v;
    renderAll();
  });
  $('dlBtn').addEventListener('click', () => {
    const s = summarize(baseSim, state.terms);
    const lines = [['metric', 'value'], ['employer', cohort.employer], ['data_through', cohort.cutoff], ['p_adherence_met', s.pPersist], ['p_weight_loss_met', s.pLoss], ['p_both_met', s.pBoth],
      ['adherence_month12_p10', s.persist.p10], ['adherence_month12_p50', s.persist.p50], ['adherence_month12_p90', s.persist.p90],
      ['loss_month12_p10', s.loss.p10], ['loss_month12_p50', s.loss.p50], ['loss_month12_p90', s.loss.p90], ['expected_credit_usd', Math.round(s.expectedCredit)]];
    const blob = new Blob([lines.map((r) => r.join(',')).join('\n')], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'guarantee-outlook.csv'; a.click();
  });
  window.addEventListener('resize', () => renderChart());
}

init();
