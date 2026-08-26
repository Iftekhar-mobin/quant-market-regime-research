---
title: Quantitative Market Regime Research
emoji: 📉
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Walk-forward research console for market-regime trading studies
---

# Quantitative Market Regime Research

An interactive research console for testing whether **market regimes** — persistent
states of trend, volatility and structure — make systematic trading strategies more
robust out of sample.

This is a *study*, not a trading bot. Every number is produced by purged
walk-forward validation with an embargo, priced through an execution model with
realistic costs, and reported with a confidence interval and a correction for how
many configurations were tried.

**The honest headline: on the sample studied here, the answer is no.** The model
reaches 51% directional accuracy out of sample — statistically real — and still
loses money, because the edge is worth +4.28 bps per trade and the round trip
costs 5.00 bps.

## What you can do here

| Tab | What it does |
|---|---|
| **Overview** | The research question and the design decisions that determine the answer |
| **Data** | Price, ~80 causal features, return distribution, prediction target |
| **Regimes** | Fit k-means / GMM / rule-based detectors; measure persistence and stability |
| **Run a study** | Configure and launch a walk-forward experiment with a live log |
| **Results** | Economics, fold stability, regime breakdown, feature importance, significance |
| **Model comparison** | Stored studies side by side, regime arms paired against controls |
| **Signals** | Out-of-sample positions on the chart with model conviction over time |

Trimmed EURUSD, GBPUSD and GOLD H1 samples (15,000 bars each) ship with the Space,
so studies run immediately. A study takes roughly one to two minutes.

## Method, briefly

- **Causal features only.** Swing points are reported on the bar that *confirms*
  them, not the bar they occurred on.
- **Everything refitted inside each fold** — regime detector, scaler, imputer,
  feature selector, classifier.
- **An embargo** between train and test, because labels look forward.
- **Triple-barrier and meta-labelling** targets, not fixed-horizon return signs.
- **Next-bar-open fills**, costs on every position change, a minimum holding period.
- **Significance four ways**: stationary-bootstrap CI, probabilistic Sharpe,
  deflated Sharpe, cross-instrument holdout with a sign test.

## Links

- **Code and full methodology:** https://github.com/Iftekhar-mobin/quant-market-regime-research
- **Findings, including the 28-configuration ledger:** see `docs/findings.md` in the repo

---

Research code. Nothing here is investment advice, and past out-of-sample
performance is not a forecast of anything.
