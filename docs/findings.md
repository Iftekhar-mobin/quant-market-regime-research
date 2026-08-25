# Findings

All results below are out of sample, produced by purged walk-forward validation
with an embargo, and priced through the execution model in
[methodology.md](methodology.md).

**Study setup.** EURUSD H1, 2018-01 to 2026-03. Six expanding folds, 12% of the
sample per test window, 48-bar embargo. Triple-barrier labels with a 24-bar
horizon and symmetric 1.0 ATR barriers. Minimum holding period 24 bars, matching
the label horizon. 2.0 bps cost plus 0.5 bps slippage on every change in
position. Fills at the next bar's open. Roughly 36,000 out-of-sample bars per
arm.

---

## 1. The directional edge is real, small, and identical across learners

| Learner | Regimes | Directional precision | Sharpe (OOS) | CAGR | Max drawdown |
|---|---|---|---|---|---|
| Random forest | none | 51.1% | −0.47 | −3.9% | −26.8% |
| Random forest | k-means | 50.9% | −0.76 | −6.1% | −33.8% |
| LightGBM | none | 51.2% | −0.84 | −7.0% | −36.0% |
| LightGBM | k-means | 51.2% | −1.10 | −9.0% | −45.6% |
| XGBoost | k-means | 51.4% | −1.37 | −11.0% | −51.5% |
| XGBoost | none | 51.4% | −1.47 | −11.8% | −53.4% |
| Logistic regression | none | 50.9% | −1.64 | −12.6% | −57.4% |
| Logistic regression | k-means | 50.7% | −1.79 | −13.7% | −59.1% |

Directional precision sits in a band barely a percentage point wide across four
model families spanning a regularised linear model, a bagged ensemble, and two
different gradient-boosting implementations. The variation across folds within
any single learner is larger than the variation between learners.

The reading: a linear model extracts essentially all of the signal these
features contain. The bound is the feature set and the market, not the
optimiser. Reaching for a bigger model is the wrong move — it buys variance, not
signal.

Note also that the *ranking* by Sharpe is close to the reverse of what model
capacity would predict. Logistic regression, the simplest learner, scores worst,
and random forest — the most heavily regularised of the ensembles, with a
50-sample minimum leaf — scores best. With no real edge to find, what separates
the arms is how much turnover each one generates, not how well it predicts.

---

## 2. Why a 51% hit rate loses money

The zero-cost arm of the ablation in §3 isolates the signal from the friction.
Same model, same labels, same folds, same 1,191 trades — only the charge is
removed.

| | Total return | Per trade | Sharpe | Max drawdown |
|---|---|---|---|---|
| Reference | −20.6% | **−1.73 bps** | −0.47 | −26.8% |
| Zero transaction cost | +44.1% | **+3.70 bps** | **+0.85** | −8.7% |

The 5.43 bps gap between the two rows is, to within rounding, the 5.00 bps
round-trip charge the reference arm pays 1,191 times.

So the signal is real. Out of sample, on bars the model had never seen, it earns
+3.70 bps per trade — a Sharpe of 0.85 with a single-digit drawdown. It is
simply smaller than the spread.

**Closing the gap needs about 35% more gross edge.** In hit-rate terms, using
the standard approximation for a symmetric bet:

```
edge per trade  ≈  (2p − 1) × E[|move over the holding period|]
cost per trade  =  2 × (spread + slippage)
```

At a per-bar volatility of 10.3 bps and a 24-bar hold, the typical absolute move
is roughly 10.3 × √24 ≈ 50 bps, which puts the break-even hit rate near **55%**.
The models reach 51%.

Note that the measured gross edge (3.70 bps) implies a slightly better effective
accuracy than the label precision (51.1%) suggests. The two are not the same
quantity: precision is scored against the *barrier* label, while the P&L is
earned over the actual holding period. The model is modestly better at the
question that pays than at the question it was scored on.

The gap between "statistically better than chance" and "profitable after costs"
is the most under-reported fact in retail quantitative research. A classification
paper would call 51.4% on 36,000 samples a solid result — the binomial standard
error is 0.26%, so it sits more than five standard errors from a coin flip. It
is also a strategy that loses 1.7 bps every time it trades.

---

## 3. Execution assumptions move the result more than the model does

| Arm | Precision | Long / short labels | Trades | Sharpe | CAGR | Max drawdown |
|---|---|---|---|---|---|---|
| **Reference** (symmetric barriers, 24-bar hold) | 51.1% | 49.9 / 50.1 | 1,191 | −0.47 | −3.9% | −26.8% |
| No minimum holding period | 51.1% | 49.9 / 50.1 | 4,905 | **−5.55** | −31.3% | −88.9% |
| Asymmetric barriers (1.5 / 1.0 ATR) | **53.0%** | 39.6 / 60.0 | 888 | −1.13 | −6.8% | −36.8% |
| Both defects together | 53.0% | 39.6 / 60.0 | 2,592 | −4.68 | −17.5% | −67.9% |
| Reference with **zero transaction cost** | 51.1% | 49.9 / 50.1 | 1,191 | **+0.85** | +6.5% | −8.7% |

The signal is identical in every row. Only the named assumption changes.

Two of these are not tuning choices; they are correctness issues.

**The minimum holding period.** A model trained on a 24-bar-ahead target holds
one opinion about the next 24 bars. Re-deciding every bar pays the spread up to
24 times to maintain what is economically a single position. A backtest without
this constraint is not measuring the signal; it is measuring the spread.

**Barrier symmetry — and the clearest exhibit in the study.** With a 1.5 ATR
profit target against a 1.0 ATR stop, the lower barrier is simply closer, so it
is touched more often and the label set tilts 60/40 short regardless of what
the market did.

The consequence is worth stating slowly: **directional precision rises from
51.1% to 53.0%, and the Sharpe ratio falls from −0.47 to −1.13.** A model that
leans short in a market that did not fall scores *better* on the classification
metric while losing more than twice as much money. The extra precision is
entirely the majority class being easier to guess.

Any pipeline that selects on validation accuracy would prefer the worse
configuration here, confidently, every time. The console warns when long and
short label shares diverge by more than five percentage points.

**Zero-cost arm.** Removing costs isolates the signal from the friction; §2
reads the result. The short version: the signal is worth +3.70 bps a trade and
the round trip costs 5.00.

One asymmetry in the table is worth noting so it is not mistaken for a result:
"both defects together" (−4.68) scores *better* than "no minimum holding
period" alone (−5.55). That is not the asymmetric barriers helping — it is the
short-tilted labels producing a stickier signal, hence 2,592 trades instead of
4,905, hence less spread paid. Two defects, one of which happens to mask part
of the other.

---

## 4. Regime conditioning did not help on this sample

| Learner | Control (no regimes) | With k-means regimes | Change |
|---|---|---|---|
| Random forest | −0.47 | −0.76 | −0.29 |
| LightGBM | −0.84 | −1.10 | −0.26 |
| XGBoost | −1.47 | −1.37 | **+0.10** |
| Logistic regression | −1.64 | −1.79 | −0.15 |

Mean change **−0.15 Sharpe**. The regime layer helped in 1 of 4 pairs.

Four matched pairs, identical in every respect except the regime layer.

The regimes themselves are well behaved. K-means on the four descriptors finds
persistent, interpretable states — trending and ranging, quiet and stressed —
with bar-to-bar persistence above 95% and mean run lengths in the tens of bars.
Buy-and-hold return and volatility differ materially between them. They are real
states.

They still do not help, for a reason visible in the per-regime breakdown:
directional precision is roughly constant across regimes. The states separate
the market's *volatility and trend*, which the feature set already encodes
directly, but they do not separate the bars where the model is right from the
bars where it is wrong. Handing the model a one-hot regime adds parameters
without adding information it did not already have; fitting one model per regime
splits the training data without giving each split a distinct relationship to
learn.

The honest summary: **on EURUSD H1 with this feature set, regime conditioning
adds turnover and parameters without adding robustness.**

---

## 5. Slower timeframes: the prediction, and the test

The cost arithmetic in §2 makes a falsifiable prediction. Cost per trade is
fixed in basis points, while the typical move grows with the square root of the
holding period. Lengthen the holding period and the same hit rate should
eventually clear the toll.

Random forest, no regimes, six folds, holding period matched to the label
horizon on each timeframe:

| Timeframe | Horizon | OOS bars | Precision | Trades | Return / trade | Sharpe | Max drawdown |
|---|---|---|---|---|---|---|---|
| H1 | 24 bars | 45,306 | 51.3% | 1,428 | −3.81 bps | −1.45 | −58.5% |
| H4 | 12 bars | 17,340 | 52.0% | 601 | −5.26 bps | −0.60 | −36.9% |
| D1 | 10 bars | 3,528 | 50.4% | 114 | **+0.15 bps** | **+0.03** | **−15.8%** |

D1 is the only arm to reach break-even per trade, and its drawdown is a quarter
of the H1 arm's. The direction of the prediction holds.

Two caveats, both material:

- **Break-even is where it stops.** +0.15 bps per trade over 114 trades is
  indistinguishable from zero. The prediction that lengthening the horizon
  removes the cost penalty is supported; the hope that a profit appears
  underneath it is not.
- **The arms are not a controlled comparison.** Each timeframe was given as much
  history as exists for it (H1 from 2016, H4 from 2010, D1 from 2005), so they
  cover different periods and different market regimes. Note also that
  return per trade is *not* monotonic — H4 is worse than H1 — which is what one
  would expect if period effects are of the same size as the cost effect.
  A properly controlled version of this experiment restricts every arm to a
  common window; it has not been run.

---

## 6. What would change the answer

Ordered by how much they would move the result.

1. **Features that are not technical.** Eighty transformations of OHLCV are
   eighty views of the same information. Order-flow, positioning, rate
   differentials, or news-derived features would introduce genuinely new
   information; the current result bounds what price history alone contains.

2. **Regimes that condition on model reliability, not on the market.** The
   present detectors describe the market's state. What the strategy actually
   needs is a state that separates the bars where the model is right from those
   where it is wrong — closer to a meta-labelling problem than a clustering one.

3. **A cost model that varies.** Real spreads widen around news and in thin
   sessions, exactly when a volatility-driven model is most active. A constant
   cost biases towards optimism.

4. **Cross-instrument validation.** An edge that survives on EURUSD, GBPUSD and
   GOLD is worth far more than one tuned on any single pair. The framework
   supports it; the study has not yet done it. A controlled version of the
   timeframe experiment in §5, with every arm on a common window, belongs
   here too.

---

## 7. What this study does not claim

- It does **not** show that machine learning cannot work in markets. It shows
  that ~80 technical features on H1 FX, with retail costs, do not clear the bar.
- It does **not** show that regimes are useless. It shows that *these* regimes,
  conditioning *this* model, on *this* instrument, did not help.
- It does **not** report a tuned result. Hyperparameters are library defaults
  with light regularisation, deliberately, so the comparison is between model
  families rather than between search budgets.

A negative result reported in full is worth more than a positive result produced
by a pipeline that leaks. The point of the framework is that it can tell the
difference.
