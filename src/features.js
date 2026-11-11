// Dropout-hazard features computed only from data a member has already generated
// (check-ins, logged weights, side-effect reports) strictly before week t.
// The same function is used by the synthetic generator, the hazard fit and the forecast
// simulator, so the forecast simulates exactly the signals the model was fit on.

export const FEATURE_NAMES = ['intercept', 'missed check-ins (last 4 wks)', 'moderate/severe side effect (last 3 wks)', 'weight plateau (last 6 wks)', 'first 12 weeks'];
export const PLATEAU_PTS = 0.6; // < 0.6 percentage points of extra loss across ~6 weeks

function meanObs(y, from, to) {
  let s = 0, n = 0;
  for (let k = Math.max(1, from); k <= to; k++) if (y[k] !== null && y[k] !== undefined) { s += y[k]; n++; }
  return n ? s / n : null;
}

// y: array of % weight lost (null when no check-in), checkin: array of 0/1, se: array of 0..3
export function featuresAt(t, y, checkin, se) {
  let missed = 0;
  for (let k = Math.max(1, t - 4); k <= t - 1; k++) if (!checkin[k]) missed++;
  let se3 = 0;
  for (let k = Math.max(1, t - 3); k <= t - 1; k++) if (se[k] >= 2) { se3 = 1; break; }
  let plateau = 0;
  if (t >= 16) {
    const recent = meanObs(y, t - 3, t - 1);
    const before = meanObs(y, t - 9, t - 7);
    if (recent !== null && before !== null && recent - before < PLATEAU_PTS) plateau = 1;
  }
  const early = t <= 12 ? 1 : 0;
  return [1, missed, se3, plateau, early];
}

export function dot(beta, x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += beta[i] * x[i];
  return s;
}
