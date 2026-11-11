# Guarantee outlook for eMed (concept demo)

**Live demo:** https://mohith26.github.io/emed-guarantee-outlook/

eMed tells employers: "We guarantee employee weight loss and adherence. If we miss, you pay less." Each member is measured at their own 12-month mark, so for most of a plan year the guarantee is unresolved, and today's numbers are a poor guide to where it will land: members keep dropping off, and the ones who drop off are not random (missed check-ins, side effects, plateaus).

Guarantee outlook is a page for the employer dashboard (and eMed's own account team) that answers: **what is the chance each guarantee is met, how big is the expected fee credit, and what would move it?**

In the demo cohort (412 synthetic members at a fictional employer, data through Nov 10, 2026):

| | Today's dashboard | Forecast at the 12-month mark |
|---|---|---|
| Members still on program | 87.1% (above the 80% target) | 78.2%, 80% range 74.5% to 81.3%: **23% chance** of meeting 80% |
| Average weight loss, members on program | 12.3% (latest check-in) | 13.7%, range 13.5% to 13.8%: 93% chance of meeting 13.5% |
| Expected fee credit (example terms, $60k per guarantee) | n/a | $50,550 |

The generator's hidden truth for this cohort (never shown on the page) is 79.4% persistence, so the guarantee does in fact miss.

## How it works

1. **Weight trajectory per member.** % body-weight loss follows `A(1 - e^(-t/tau))`. Each member gets a grid posterior over (A, tau) (81 x 24 grid) with empirical-Bayes priors fit by EM: plateau prior by medication (Zepbound, Wegovy pen, Wegovy pill), a shared prior on log tau, and a pooled noise term.
2. **Drop-off model.** Discrete-time logistic hazard fit by IRLS on every observed member-week. Signals are computed only from data available before that week: missed check-ins in the last 4 weeks, a moderate/severe side-effect report in the last 3 weeks, a weight plateau (under 0.6 points of extra loss across about 6 weeks), and the first 12 weeks. The same feature function drives the generator, the fit and the simulator.
3. **Simulation.** 1,200 futures. In each, every active member draws (A, tau) from their posterior, a check-in rate from a beta posterior, side-effect reports at the cohort's titration/maintenance rates, and a set of hazard coefficients from their sampling distribution, then is simulated week by week to week 52. Probabilities, ranges and the expected credit are read off the simulated cohort metrics.
4. **Calibrated ranges.** `scripts/backtest.mjs` forecasts 60 independent synthetic employer cohorts at the same cut-off and compares with their hidden 12-month outcomes. Cohorts 1-30 set split-conformal width multipliers (x1.00 adherence, x1.19 weight loss); cohorts 31-60 are held out. On the held-out cohorts the 80% ranges contained the realized adherence result 87% of the time (raw model 93%) and the weight-loss result 90% (raw 77%). Across a ladder of adherence targets the forecast's Brier score is 0.057 versus 0.330 for an "on track today" flag, which ended in a miss in 99 of 275 checks (36%).
5. **Levers and privacy.** A re-engagement slider cuts drop-off odds for members behind on check-ins and re-simulates with common random numbers. Segment rows (enrollment month, medication) are aggregate only and suppress groups under 11 members.

Every number on the page is computed in the browser from the seeded generator by `src/`; the reliability card reads `data/backtest.json`, produced by the same code.

## Run it

```
node --test test/*.test.mjs        # 8 tests
node scripts/report.mjs            # numbers shown on the page
node scripts/backtest.mjs 60 800   # regenerates data/backtest.json (~2 min)
python3 -m http.server             # then open http://localhost:8000
```

Static HTML/CSS/JS, ES modules, no build step, no dependencies. Fonts and icons from Google Fonts (Poppins, Material Symbols).

## Honest limitations

- **Synthetic data.** The drop-off and weight processes are mine, and the model is well specified for them, so the back-test is a best case. Real check-in data would bring misspecification (pauses, dose holds, supply gaps, re-enrollment) that the conformal step would need to absorb with real past cohorts.
- **Contract terms are invented.** I don't know eMed's guarantee definitions (who is in the denominator, how missed 12-month weigh-ins are handled, how credits are computed). They are inputs on the page.
- The weight-loss guarantee here counts members still on program. That makes the two guarantees pull against each other a little (re-engaging slower responders helps adherence and slightly lowers average loss); the simulated correlation is shown.
- Plateau and drop-off rates use one shared population model; a real version should pool across employers hierarchically.
- No eMed employer-dashboard screenshots are public, so the page is built from the member app's 2026 components and emed.com's design tokens.

## Novelty

As of Nov 10, 2026 I found no eMed page, blog post, press release or app note describing a forecast of guarantee attainment; eMed's site describes the guarantee and an employer dashboard of weekly check-in data. Virta and Omada also advertise outcome or cost guarantees; I found no public mention of a probabilistic guarantee forecast from either.

Unofficial concept demo by Mohith Gajjela. Not affiliated with eMed. Synthetic data.
