# Methodology

This document states the assumptions the study rests on and the reasoning behind
each one. Every choice below changes the answer, so each is named rather than
buried in code.

---

## 1. Research question

> Do market regimes identified from price, volatility and technical structure
> make systematic strategies more robust out of sample?

"More robust" is defined in advance, so the question cannot be quietly rewritten
once the results are in. A regime-conditioned arm improves on its control if it
raises the out-of-sample Sharpe ratio **and** does so on more than half of the
walk-forward folds. Total return alone does not count: it is trivially inflated
by leverage, by trading more, and by a favourable sample.

The control arm is the identical pipeline with `regime.method = none`. Nothing
else differs — same features, same labels, same folds, same costs, same seed.

---

## 2. Data

Broker OHLCV exports, one CSV per symbol and timeframe. Normalisation happens
once, in `qmr.data.loader`:

- Timestamps are parsed from either epoch seconds or date strings.
- Duplicate timestamps keep the last observation — the repeated bar around
  daylight-saving changes is a re-send, not a second bar.
- Bars with `high < low`, non-positive prices or non-finite values are dropped
  as broker glitches.
- Spot FX without volume gets a constant, which leaves the volume features
  defined and flat, and therefore uninformative rather than wrong.

**Known limitation.** These are mid or bid prices from one broker. The true
spread varies with the session and widens sharply around news; the backtest
applies a constant cost instead. On liquid majors this understates cost during
thin hours.

---

## 3. Features

Roughly eighty features across seven blocks. Three rules govern all of them.

### 3.1 Causality

Every feature at bar *t* uses bars ≤ *t*. This is the single most important
property in the codebase and the reason the indicators are implemented directly
rather than imported.

The concrete case that matters: **swing points**. The conventional detection uses
`scipy.signal.argrelextrema`, which marks a pivot on the bar where it occurred.
That bar cannot be known to be a pivot until several bars later. A feature built
that way tells the model where the local top was *before the top had formed*, and
the resulting backtest is fiction. Here, `swing_points` reports a pivot on the
bar that confirms it, `right` bars afterwards.

The same reasoning applies to the Donchian channel, which is shifted one bar so
the current bar cannot set the channel it is then measured against.

### 3.2 Scale invariance

Features are ratios, z-scores or bounded oscillators — never raw prices. A model
fitted on EURUSD near 1.10 has to transfer to GOLD near 2400, and, more subtly, a
model given raw price levels can use the price as a proxy for the calendar date
and memorise the training period.

Distances are expressed in ATR units, so "far from the moving average" means the
same thing in a calm market and a violent one.

### 3.3 Stationarity by construction

Anything that trends without bound — on-balance volume, the
accumulation/distribution line, raw MACD — enters as a rolling z-score or a
slope. A tree splitting on a cumulative sum splits on time.

### 3.4 The regime descriptors

Four interpretable coordinates, used by the regime layer instead of the full
feature matrix:

| Descriptor | Definition | Reads as |
|---|---|---|
| `trend_strength` | (EMA20 − EMA100) / ATR14 | signed trend, in volatility units |
| `vol_percentile` | rank of 20-bar realised volatility over 500 bars | 0 = calmest, 1 = most stressed |
| `momentum_score` | (RSI14 − 50) / 50 | centred momentum, roughly [−1, 1] |
| `mean_reversion_score` | 0.5 − rolling Hurst exponent | above 0 mean-reverting, below 0 trending |

Clustering four interpretable axes yields states that can be *named*. Clustering
eighty features yields a partition nobody can defend.

---

## 4. Labels

Three schemes are available: `directional`, `triple_barrier` and `meta`.

### 4.1 The triple barrier

From each bar, a hypothetical position is followed forward until it touches an
upper barrier, a lower barrier, or the time limit. The label records which came
first. Both barriers are set in ATR units, so they adapt to the volatility
regime rather than applying one fixed pip distance to a market whose character
changes.

**Why not the sign of the forward return.** A fixed-horizon label calls a
+30 pip move a win even if the path first went 60 pips against the position — a
trade that would have been stopped out in reality. The triple barrier prices the
path in, which is the difference between labelling what happened and labelling
what could have been captured.

**Barriers must be symmetric unless the asymmetry is deliberate.** With a
1.5 ATR profit target against a 1.0 ATR stop, the lower barrier is simply closer
and gets hit more often; the label set tilts short, and the model inherits a
directional bias that has nothing to do with the market.

Measured, this costs more than it looks: the tilt raises directional precision
from 51.0% to 52.9% while taking the Sharpe ratio from −0.37 to −0.83. The
classification metric improves because the majority class is easier to guess,
and the strategy loses twice as much. The default is 1.0 / 1.0 for exactly this
reason, and the console warns when the label balance drifts by more than five
percentage points.

**Both barriers in one bar.** The intrabar path is unknown, so the adverse
outcome is assumed. That is conservative by design.

### 4.2 Meta-labelling

Direct direction prediction is hard, and the accuracy it reaches here (~51%)
does not cover the spread. Meta-labelling changes the question rather than
trying harder at the same one.

A **primary rule** — by default a moving-average crossover — proposes both the
direction and the timing. The learner is then asked only: *will this particular
trade work?* The label is the primary side when the trade would have won and 0
when it would have lost; bars where the rule proposes nothing are dropped.

Three reasons this is an easier problem:

1. The direction is no longer the learner's responsibility.
2. The training set contains only bars where a trade was actually proposed, so
   it is not diluted by the majority of bars where nothing was happening.
3. A filter that removes bad trades raises precision *and* cuts turnover — the
   two levers this study most needs.

At prediction time the learner may **veto** a trade the rule proposed but can
never invent one of its own:

```python
predicted = predicted.where(predicted == fold_primary, 0)
```

Because meta labels exist only on proposed bars while the backtest needs an
unbroken bar series, the runner keeps every bar, trains on the labelled subset
and scores the whole test window.

Reference: López de Prado, *Advances in Financial Machine Learning*, ch. 3.

### 4.3 Labels look forward; features must not

Every label at bar *t* is a function of bars after *t*. That is correct for a
target and fatal for a feature. Keeping the two apart is the job of the embargo.

---

## 5. Validation

### 5.1 Walk-forward, never k-fold

Shuffled cross-validation on a price series trains on Tuesday to predict Monday.
Every fold leaks, every score is optimistic, and the size of the optimism cannot
be recovered. Only forward splits are used: train on the past, test on the
future that follows it.

### 5.2 The embargo

Even a strictly forward split leaks when the target looks forward. The label on
the last training bar is a function of the next *h* bars — which are the first
bars of the test window. The fix is to purge `embargo_bars` from the tail of each
training window.

**The embargo must be at least the label horizon.** The console sets it to twice
the horizon by default.

### 5.3 Everything is refitted inside the fold

The regime detector, the feature scaler, the imputer, the **feature selector**
and the classifier are all fitted on the training window alone.

Two leaks this prevents, both common and both invisible in the metrics:

- **Scaling on the full sample.** The mean and variance of the test period are
  not knowable while the model is being fitted.
- **Clustering on the full sample.** Fitting the regime detector once on all the
  data and then evaluating "out of sample" within those clusters leaks the
  entire future into the regime assignment. This is the most seductive version of
  the mistake, because the resulting regime chart looks so convincing.

The Regimes tab in the console *does* fit on the whole window — it is for
describing the market, not scoring a strategy — and says so on screen.

---

## 6. Execution and costs

| Assumption | Value | Reasoning |
|---|---|---|
| Fill | open of the next bar | The close that produced the signal was not tradeable when the decision was made. |
| Cost | 2.0 bps per transaction | Typical retail spread on liquid majors. |
| Slippage | 0.5 bps per transaction | Charged alongside the spread. |
| Round trip | 5.0 bps | Entry plus exit. A long-to-short flip is two transactions and pays twice. |
| Minimum hold | = label horizon | See below. |
| Sizing | fixed fraction of capital | Keeps the comparison about signal quality, not leverage. |
| Volatility target | off by default | When enabled, positions are sized inversely to trailing volatility and the size is **fixed at entry**, so targeting does not generate rebalancing costs. |
| Session filter | off by default | When enabled, new entries are restricted to nominated hours. Spreads are widest in thin hours and a constant cost assumption understates that. |

**The minimum holding period is not a detail.** A model trained on a 24-bar-ahead
target expresses one opinion about the next 24 bars, not 24 independent
opinions. Acting on its output every bar pays the spread 24 times to hold what
is economically a single position. In the controlled ablation in
[findings.md](findings.md#3-execution-assumptions-move-the-result-more-than-the-model-does),
removing the minimum hold moved the out-of-sample Sharpe from −0.37 to −5.51 on
an identical signal — same model, same labels, same folds, 1,206 trades becoming
4,925. That is the cost of re-deciding, and nothing else.

Flipping from long to short pays the cost twice, because it is two transactions.

---

## 7. Evaluation

### 7.1 Why accuracy is nearly useless here

On a three-class directional target where flat is common, a model reaches
respectable accuracy by never taking a position. The metrics that carry
information are:

- **Directional precision** — of the bars where a position *was* taken, what
  fraction were right. This is the number that has to cover costs.
- **Matthews correlation** — stays honest under class imbalance.
- **The backtest** — the only place the economic value of a call appears.

### 7.2 The arithmetic that decides everything

For a symmetric directional bet:

```
expected edge per trade  ≈  (2p − 1) × E[|move over the holding period|]
cost per trade           =  2 × (spread + slippage)
```

At a per-bar volatility of 10.3 bps and a 24-bar hold, the typical absolute move
is about 50 bps, which puts the break-even hit rate near 55%. The models reach
51%. Measured directly from the zero-cost ablation arm, the signal is worth
+4.28 bps a trade against a 5.00 bps round trip: **real, and smaller than the
spread.**

This single calculation explains most published retail results, and it is why
this framework reports return per trade next to precision.

### 7.3 Significance

Three checks accompany every headline number:

- **Stationary bootstrap confidence interval.** Resamples in geometrically
  distributed blocks so the autocorrelation of the return stream survives.
  Resampling individual bars gives an interval far too narrow.
- **Probabilistic Sharpe ratio.** Corrects the standard error for skew and
  excess kurtosis. Financial returns have both, and ignoring them overstates
  significance in the flattering direction.
- **Deflated Sharpe ratio.** Discounts for the number of configurations tried.
  Searching forty variants and reporting the best produces an impressive Sharpe
  from pure noise.

### 7.4 Fold stability

A strategy that earns its entire Sharpe in one fold and loses money in the other
five has found one good year, not an edge. The share of profitable folds, and a
one-sided t-test on the fold means, are reported alongside the pooled number.

---

## 8. Searching for a result without fooling yourself

Once a study can be run in a minute, the binding constraint stops being compute
and becomes discipline. Twenty-eight configurations were tried in the
improvement programme ([findings.md](findings.md) §5), and every one of them is
a trial that inflates the best result.

Four defences are built in, in increasing order of strength:

1. **Every arm is logged**, including failures, to
   `reports/improvement_ablation.csv`. The trial count is auditable rather than
   implied by whatever the author chose to mention.
2. **The deflated Sharpe ratio** discounts the headline by the number of arms
   actually run.
3. **Re-measurement.** Running the winning configuration with more folds, more
   history or more seeds is not another search — it is the same hypothesis
   measured more carefully. A result that moves under that is noise. (The
   leading arm went from +0.13 with 3 folds to −0.27 with 6 folds, 5 seeds and a
   longer window.)
4. **A cross-instrument holdout with a sign test.** The winner is re-run on
   instruments the search never touched, and judged by a binomial test against a
   coin flip rather than by eye.

The holdout must be **large enough**. The first attempt used two instruments,
both came back strongly positive, and it was tempting to stop there. Expanding
to ten showed 5 positive, 5 negative, sign test p = 0.62 — the two-instrument
version had landed on the two best of ten. A holdout that is too small is not a
check; it is another lottery ticket.

---

## 9. Threats to validity

Stated plainly, because a study that lists none has not looked.

1. **Single broker, single price stream.** Results may not survive a different
   data source. Not yet tested.
2. **Constant costs.** Real spreads widen around news and in thin sessions,
   precisely when a volatility-driven model is most active. The bias is towards
   optimism.
3. **Survivorship in instrument choice.** The majors were chosen because history
   was available, not at random.
4. **Multiple comparisons.** Every configuration tried is a trial. The
   deflated Sharpe corrects for the arms within one programme, not for every
   configuration ever run across sessions. Section 8 is the fuller answer.
5. **Regime label instability.** Cluster identities can permute between refits.
   Names are derived from centroid geometry rather than cluster index to
   mitigate this, but a regime named "quiet range" in one fold is not guaranteed
   to be the same population as in the next.
6. **No slippage model for gaps.** Weekend gaps are executed at the next open
   with no penalty beyond the standard cost.

---

## References

- Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.*
- López de Prado, M. (2018). *Advances in Financial Machine Learning.*
  Triple-barrier labelling, purging and embargoing.
- Politis, D. and Romano, J. (1994). *The Stationary Bootstrap.*
- Harvey, C., Liu, Y. and Zhu, H. (2016). *… and the Cross-Section of Expected
  Returns.* On the multiple-testing problem in finance.
