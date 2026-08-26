# Findings

All results are out of sample, produced by purged walk-forward validation with
an embargo and priced through the execution model in
[methodology.md](methodology.md).

**Study setup.** EURUSD H1, 2018-01 to 2026-03. Six expanding folds, 12% of the
sample per test window, 48-bar embargo. Triple-barrier labels, 24-bar horizon,
symmetric 1.0 ATR barriers. Minimum holding period 24 bars. 2.5 bps charged on
every change in position (5.0 bps a round trip). Fills at the next bar's open.
~36,000 out-of-sample bars per arm.

**Summary.** There is a small, real directional signal in these features. It is
worth about **+4.3 bps per trade** and the round trip costs **5.0 bps**. Every
attempt to close that gap — 28 configurations, a structural change to the
learning problem, and four execution levers — failed to produce anything that
survived validation on instruments it was not tuned on.

---

## 1. Every learner finds the same small edge

| Learner | Regimes | Directional precision | Sharpe (OOS) | CAGR | Max drawdown |
|---|---|---|---|---|---|
| Random forest | none | 51.00% | **−0.37** | −3.2% | −26.9% |
| Random forest | k-means | 51.00% | −0.77 | −6.1% | −30.9% |
| LightGBM | k-means | 51.21% | −1.37 | −11.1% | −50.8% |
| XGBoost | k-means | 51.42% | −1.40 | −11.0% | −50.2% |
| LightGBM | none | 51.46% | −1.57 | −12.6% | −55.8% |
| Logistic regression | k-means | 50.73% | −1.66 | −12.8% | −57.1% |
| XGBoost | none | 51.50% | −1.71 | −13.6% | −57.5% |
| Logistic regression | none | 50.86% | −1.79 | −13.7% | −60.4% |

Directional precision spans **50.7% to 51.5%** across four model families — a
regularised linear model, a bagged ensemble, and two different gradient-boosting
libraries. The variation across folds within any single learner is larger than
the variation between learners.

A linear model extracts essentially all the signal these features contain. The
binding constraint is the feature set and the market, not the optimiser.

The Sharpe ranking is close to the reverse of model capacity: the most heavily
regularised learner (random forest, 50-sample minimum leaf) does best, and the
two boosting libraries do worst. With no real edge to find, what separates the
arms is how much turnover each one generates.

---

## 2. The signal is real. The spread is bigger.

Running the identical signal with costs switched off separates the two:

| | Total return | Per trade | Sharpe | Max drawdown |
|---|---|---|---|---|
| Reference | −17.1% | **−1.41 bps** | −0.37 | −26.9% |
| Zero transaction cost | +51.6% | **+4.28 bps** | **+0.95** | −9.7% |

Same model, same labels, same folds, same 1,206 trades. The only difference is
the toll booth, and the ~5.7 bps gap between the rows is the 5.0 bps round trip
charged 1,206 times, plus compounding.

So the signal is genuinely worth something. Out of sample, on bars the model had
never seen, it earns **+4.28 bps per trade — a Sharpe of 0.95 with a
single-digit drawdown**. It is simply smaller than the cost of trading it.

**Closing the gap needs roughly 17% more gross edge.** In hit-rate terms, using
the standard approximation for a symmetric bet:

```
edge per trade  ≈  (2p − 1) × E[|move over the holding period|]
cost per trade  =  2 × (spread + slippage)
```

At a per-bar volatility of 10.3 bps and a 24-bar hold, the typical absolute move
is roughly 10.3 × √24 ≈ 50 bps, putting the break-even hit rate near **55%**.
The models reach 51%.

The gap between "statistically better than chance" and "profitable after costs"
is the most under-reported fact in retail quantitative research. A
classification paper would call 51.5% on 36,000 samples a solid result — the
binomial standard error is 0.26%, so it sits more than five standard errors from
a coin flip. It is also a strategy that loses 1.4 bps every time it trades.

---

## 3. Execution assumptions move the result more than the model does

Identical window, learner, horizon, folds and seed. Only the named assumption
changes.

| Arm | Precision | Long / short labels | Trades | Sharpe | CAGR | Max drawdown |
|---|---|---|---|---|---|---|
| **Reference** (symmetric barriers, 24-bar hold) | 51.0% | 49.8 / 50.1 | 1,206 | −0.37 | −3.2% | −26.9% |
| No minimum holding period | 51.0% | 49.8 / 50.1 | 4,925 | **−5.51** | −31.4% | −89.0% |
| Asymmetric barriers (1.5 / 1.0 ATR) | **52.9%** | 39.5 / 60.1 | 888 | −0.83 | −5.1% | −32.3% |
| Both defects together | 52.9% | 39.5 / 60.1 | 2,565 | −4.64 | −17.3% | −67.3% |
| Reference with **zero transaction cost** | 51.0% | 49.8 / 50.1 | 1,206 | **+0.95** | +7.4% | −9.7% |

Two of these are not tuning choices; they are correctness issues.

**The minimum holding period.** A model trained on a 24-bar-ahead target holds
one opinion about the next 24 bars. Re-deciding every bar pays the spread up to
24 times to maintain what is economically a single position — worth **5.1 Sharpe
points** here. A backtest without this constraint is not measuring the signal;
it is measuring the spread.

**Barrier symmetry — the clearest exhibit in the study.** With a 1.5 ATR profit
target against a 1.0 ATR stop, the lower barrier is closer, so it is touched
more often and the label set tilts 60/40 short regardless of what the market
did.

The consequence is worth stating slowly: **directional precision rises from
51.0% to 52.9%, and the Sharpe ratio falls from −0.37 to −0.83.** A model that
leans short in a market that did not fall scores *better* on the classification
metric while losing more than twice as much money. The extra precision is
entirely the majority class being easier to guess.

Any pipeline that selects on validation accuracy would prefer the worse
configuration here, confidently, every time.

One asymmetry is worth noting so it is not misread: "both defects together"
(−4.64) scores better than "no minimum holding period" alone (−5.51). That is
not the asymmetric barriers helping — it is the short-tilted labels producing a
stickier signal, hence 2,565 trades instead of 4,925, hence less spread paid.
Two defects, one of which partially masks the other.

---

## 4. The regime effect is not stable enough to have a sign

This section previously reported that regime conditioning **hurt** in three of
four matched pairs. A later refactor of the runner — which changed nothing about
the regime logic, only how the feature and label indices are aligned, and
therefore moved the fold boundaries by a few bars — reversed it:

| Learner | Control (no regimes) | With k-means regimes | Change |
|---|---|---|---|
| Random forest | −0.37 | −0.77 | −0.41 |
| LightGBM | −1.57 | −1.37 | **+0.20** |
| XGBoost | −1.71 | −1.40 | **+0.32** |
| Logistic regression | −1.79 | −1.66 | **+0.13** |

Mean change **+0.06 Sharpe**; regimes now help in 3 of 4 pairs instead of
hurting in 3 of 4.

**Neither result is a finding. The instability is the finding.** An effect whose
sign flips when fold boundaries move by a handful of bars is noise. The honest
conclusion is that this study cannot resolve a regime effect of the size that
might plausibly exist — in either direction.

The regimes themselves are well behaved: k-means on the four descriptors finds
persistent, interpretable states with bar-to-bar persistence above 95% and mean
run lengths in the tens of bars. Buy-and-hold return and volatility differ
materially between them. They are real states.

They do not help because directional precision is roughly constant across them.
The states separate *market conditions*, which the feature set already encodes
directly. They do not separate the bars where the model is right from the bars
where it is wrong.

---

## 5. The improvement programme: 28 configurations

Full ledger: [`reports/improvement_ablation.csv`](../reports/improvement_ablation.csv).
Reproduce with `python scripts/run_improvement_study.py --fresh`.

Search instrument EURUSD H1 from 2020, 3 folds, random forest, no regimes.
Everything else was held out.

### 5.1 What was tried

**Execution levers** — session filter (trade only London/NY hours), top-20
feature selection, 10% volatility targeting, decision threshold 0.55, 48-bar
horizon.

**A structural change — meta-labelling.** Instead of asking the model for a
direction (hard, ~51%), a moving-average crossover proposes the direction and
timing, and the model is asked only *will this trade work?* Bars where the rule
proposes nothing are dropped from training, so the sample is not diluted by the
90% of bars where nothing was happening, and the model can veto a trade but
never invent one. Seven primary rules and combinations were tried.

**Tuning** — thresholds, four learners, regimes on top of meta, longer horizons.

**Slower timeframes** — H4 and D1.

**Consolidation** — 5-seed ensembling, 6 folds instead of 3, history from 2016,
and combinations.

### 5.2 What happened

| Arm | Sharpe | 95% CI | Deflated Sharpe | Trades |
|---|---|---|---|---|
| meta + top-20 + volatility target | **+0.28** | [−0.78, +1.33] | 0.000 | 92 |
| meta + top-20 | +0.13 | [−0.89, +1.22] | 0.000 | 92 |
| meta + session + top-20 | +0.12 | [−0.93, +1.19] | 0.000 | 90 |
| meta + top-20, 5 seeds | +0.07 | [−0.93, +1.16] | 0.000 | 93 |
| meta + k-means regimes | +0.03 | [−0.90, +1.08] | 0.000 | 96 |
| *…23 further arms, all negative…* | | | | |
| baseline | −0.26 | [−1.52, +1.14] | 0.000 | 447 |
| session filter alone | −1.10 | [−2.27, −0.05] | 0.000 | 407 |

**5 of 28 arms are positive.** If every arm had a true Sharpe of zero, about 14
would have landed above zero by chance. Only 5 did — the population of
configurations is centred *below* zero. These are not noisy draws around a
break-even strategy; they are draws from a losing one.

**No arm's 95% confidence interval excludes zero, and every deflated Sharpe is
0.000** — once the number of configurations tried is priced in, nothing here is
distinguishable from luck.

### 5.3 Three diagnostics that settle it

**Measurement choices flip the sign.** The same configuration, changed only in
how it is measured:

| meta + top-20, measured … | Sharpe |
|---|---|
| with 3 folds (as searched) | **+0.13** |
| with 6 folds | −0.13 |
| with 6 folds, history from 2016 | −0.26 |
| with 5 seeds, 6 folds, from 2016 | −0.27 |

A result that does not survive being measured more carefully will not survive
being traded.

**Volatility targeting is context-dependent.** It improved the meta arm
(+0.13 → +0.28) and worsened others. Sizing inversely to volatility raises
Sharpe only when returns are positive; applied to a losing signal it amplifies
the losses taken in calm periods.

**The primary rule dominates the meta result.** Swapping the moving-average
crossover for Donchian breakout (−0.75), an ADX-filtered trend (−1.16) or RSI
mean reversion (−0.71) destroys it. The learner is not rescuing a weak primary —
it is riding one specific one.

### 5.4 The holdout — and a warning about small ones

The winner (`meta + top-20 + volatility target`) was re-run on instruments the
search never touched.

**First attempt, two instruments:**

| Instrument | Sharpe |
|---|---|
| GBPUSD H1 | **+0.74** |
| GOLD H1 | **+0.85** |

Both positive, and *higher* than on the instrument it was tuned on — the
opposite of an overfitting signature. It was tempting to call that a result.

**Expanded to ten instruments:**

| Instrument | Sharpe | Instrument | Sharpe |
|---|---|---|---|
| GOLD | +0.85 | EURJPY | −0.35 |
| GBPUSD | +0.74 | AUDUSD | −0.55 |
| USDJPY | +0.58 | USDCHF | −0.60 |
| GBPJPY | +0.17 | EURGBP | −0.75 |
| USDCAD | +0.13 | CADCHF | −1.11 |

**5 of 10 positive. Median Sharpe −0.11, mean −0.09.**
**Sign test against a coin flip: p = 0.62. Cross-sectional t-test: t = −0.42,
p = 0.66.**

The configuration does not survive out of sample. It was the best of 28 draws on
EURUSD.

The two-instrument holdout was itself a small-sample illusion — it happened to
land on the two best instruments out of ten. **A holdout that is too small is not
a check; it is another lottery ticket.** That near-miss is the most instructive
thing in this document.

---

## 6. What would change the answer

1. **Features that are not technical.** Eighty transformations of OHLCV are
   eighty views of the same information. Order flow, positioning, rate
   differentials or news-derived features introduce genuinely new information.
   The present result bounds what price history alone contains, and that is the
   binding constraint.

2. **A cheaper way to trade.** The signal clears a 0.95 Sharpe at zero cost. At
   institutional spreads, or on an instrument with a tighter effective cost, the
   same signal could become viable without improving at all. That is a
   market-access question, not a modelling one.

3. **Regimes that condition on model reliability, not on the market.** The
   detectors describe the market's state. What the strategy needs is a state
   that separates the bars where the model is right from those where it is
   wrong — closer to meta-labelling than to clustering, and the meta arms are a
   first step in that direction.

4. **A cost model that varies.** Real spreads widen around news and in thin
   sessions, exactly when a volatility-driven model is most active. A constant
   cost biases towards optimism.

5. **Pre-registration.** Everything in §5 was chosen after looking at EURUSD.
   The disciplined version fixes the configuration first, then runs it once,
   across instruments, and reports whatever comes out.

---

## 7. What this study does not claim

- It does **not** show that machine learning cannot work in markets. It shows
  that ~80 technical features on H1 FX, at retail costs, do not clear the bar.
- It does **not** show that regimes are useless. It shows that this study cannot
  resolve their effect: the sign is not stable to incidental changes in fold
  boundaries.
- It does **not** show that meta-labelling does not work. It shows that this
  implementation, with this primary rule, on this instrument, did not survive a
  ten-instrument holdout.
- It does **not** report a tuned result as a discovery. The 28 arms are logged,
  the trial count is priced into the deflated Sharpe, and the winner failed its
  holdout.

A negative result reported in full is worth more than a positive result produced
by a pipeline that leaks — or by a search whose trial count went unmentioned.
The point of the framework is that it can tell the difference, and on this
occasion it did: it caught its own most promising result.
