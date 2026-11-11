// Prints the numbers shown on the page for the demo cohort (same code path as app.js).
import fs from 'node:fs';
import { generateCohort, realizedOutcome } from '../src/generate.js';
import { fitAll, simulate, summarize, todaySnapshot } from '../src/model.js';
import { rescale } from '../src/calibration.js';
import { TERMS, MAIN_COHORT } from '../src/terms.js';
const bt = JSON.parse(fs.readFileSync(new URL('../data/backtest.json', import.meta.url)));
const c = generateCohort(MAIN_COHORT);
const fit = fitAll(c);
const scaled = (sim) => ({ ...sim, persist: rescale(sim.persist, bt.scale.persist), meanLoss: rescale(sim.meanLoss, bt.scale.loss) });
const s = summarize(scaled(simulate(c, fit, { draws: 1200, seed: 7 })), TERMS);
const snap = todaySnapshot(c);
console.log('today', JSON.stringify({ active: snap.active, activeShare: +snap.activeShare.toFixed(4), meanLatestLoss: +snap.meanLatestLoss.toFixed(2), lapsed: snap.lapsed, plateau: snap.plateau, medianWeek: snap.medianWeek }));
console.log('forecast', JSON.stringify({ pPersist: s.pPersist, pLoss: +s.pLoss.toFixed(3), pBoth: +s.pBoth.toFixed(3), persist: s.persist, loss: s.loss, expectedCredit: Math.round(s.expectedCredit), corr: +s.corr.toFixed(2) }));
console.log('hazard odds ratios', fit.hazard.beta.slice(1).map((b) => Math.exp(b).toFixed(2)).join(', '), `(${fit.hazard.n} member-weeks, ${fit.hazard.events} drop-offs)`);
for (const r of [0.2, 0.4, 0.6]) { const x = summarize(scaled(simulate(c, fit, { draws: 1200, seed: 7, reengage: r })), TERMS); console.log(`reengage ${r}: pPersist ${x.pPersist.toFixed(3)} pLoss ${x.pLoss.toFixed(3)} credit ${Math.round(x.expectedCredit)}`); }
console.log('hidden truth (never shown on the page)', JSON.stringify(realizedOutcome(c)));
