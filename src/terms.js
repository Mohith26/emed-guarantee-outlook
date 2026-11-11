// Example contract terms (inputs, editable in the UI). Not eMed's real contract terms.
export const TERMS = {
  lossTarget: 13.5,        // average % body-weight loss at month 12 among members still on program
  persistTarget: 0.80,     // share of enrolled members still on program at their 12-month mark
  atRiskLoss: 60000,       // fee credit if the weight-loss guarantee is missed ($)
  atRiskPersist: 60000,    // fee credit if the adherence guarantee is missed ($)
};
export const MAIN_COHORT = { seed: 20261110, n: 412, employer: 'Calder Freight', cutoff: '2026-11-10', employerShift: { A: -0.6, hazard: -0.1 } };
