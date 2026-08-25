# Quantitative Market Regime Research

A research framework for testing whether **market regimes** — persistent states
of trend, volatility and structure — make systematic strategies more robust out
of sample.

It is built as a study, not as a trading bot. Every number it reports comes from
purged walk-forward validation with an embargo, priced through an execution
model with realistic costs, and accompanied by a confidence interval and a
correction for how many configurations were tried. The framework is designed to
be able to return the answer "no", and on the sample studied here it does.

```
   data  ->  causal features  ->  regime detection  ->  triple-barrier labels
         ->  walk-forward CV  ->  directional model  ->  backtest with costs
         ->  risk analysis    ->  significance tests
```

---

## The research question

> Do market regimes identified from price, volatility and technical structure
> improve the robustness of systematic trading strategies?

"Improve" is defined before the results are in: a regime-conditioned arm must
raise the out-of-sample Sharpe ratio against an identical control arm
(`regime.method = none`) **and** do so on more than half of the folds. Same
features, same labels, same folds, same costs, same seed — the regime layer is
the only difference.

---

## Findings

Studied on **EURUSD H1, 2018–2026**: 6 expanding walk-forward folds, 24-bar
label horizon, 48-bar embargo, minimum holding period equal to the label
horizon, and 2.5 bps charged on every change in position (5.0 bps a round
trip). Four learners crossed with two regime settings; ~36,000 out-of-sample
bars per arm.

### 1. Every learner finds the same tiny directional edge

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

Directional precision lands between **50.7% and 51.4%** for every learner, from
a regularised logit to a tuned gradient-boosted ensemble. The spread across
model families is smaller than the spread across folds within any one of them.
Whatever signal is in these features, a linear model extracts essentially all of
it, and the choice of learner is not what is limiting the result.

### 2. The edge is real. The spread is bigger.

Running the identical signal with costs switched off separates the two:

| | Gross, per trade | Cost, per trade | Net, per trade | Sharpe |
|---|---|---|---|---|
| Reference | +3.70 bps | −5.00 bps | **−1.73 bps** | −0.47 |
| Zero-cost arm | +3.70 bps | 0 | +3.70 bps | **+0.85** |

The signal is genuinely worth something: with friction removed it earns a Sharpe
of 0.85 and never draws down more than 9%. It is simply worth less than the toll.
Closing a 1.3 bps per-trade gap needs roughly **35% more gross edge** — in
hit-rate terms, moving from about 51% to about 55%.

That is the central result, and it is why this repository prints return per
trade beside precision. A 51% hit rate is a fine headline for a classification
paper and a losing strategy in a brokerage account.

### 3. Transaction friction dominates everything else

An ablation on the identical window, learner, horizon, folds and seed, changing
only the named assumption:

| Arm | Precision | Long / short labels | Trades | Sharpe | CAGR | Max drawdown |
|---|---|---|---|---|---|---|
| **Reference** (symmetric barriers, 24-bar hold) | 51.1% | 49.9 / 50.1 | 1,191 | −0.47 | −3.9% | −26.8% |
| No minimum holding period | 51.1% | 49.9 / 50.1 | 4,905 | **−5.55** | −31.3% | −88.9% |
| Asymmetric barriers (1.5 / 1.0 ATR) | **53.0%** | 39.6 / 60.0 | 888 | −1.13 | −6.8% | −36.8% |
| Both defects together | 53.0% | 39.6 / 60.0 | 2,592 | −4.68 | −17.5% | −67.9% |
| Reference with **zero transaction cost** | 51.1% | 49.9 / 50.1 | 1,191 | **+0.85** | +6.5% | −8.7% |

Nothing about the model changes across those rows.

The third row is the one worth staring at. Making the barriers asymmetric
*raises* directional precision from 51.1% to 53.0% — and more than doubles the
loss. The closer barrier is hit more often, the label set tilts 60/40 short,
and a model that leans short in a market that did not fall scores better on
the classification metric while losing more money. If you needed one exhibit
for why accuracy is the wrong objective in this domain, it is that row.

### 4. Regime conditioning did not help here

In all four matched pairs, the k-means arm scored *below* its own control:

| Learner | Control | With regimes | Change |
|---|---|---|---|
| Random forest | −0.47 | −0.76 | −0.29 |
| LightGBM | −0.84 | −1.10 | −0.26 |
| XGBoost | −1.47 | −1.37 | +0.10 |
| Logistic regression | −1.64 | −1.79 | −0.15 |

Mean change **−0.15 Sharpe**; the regime layer helped in 1 of 4 pairs. On this
instrument, timeframe and cost structure, the answer to the research question is
**no** — regime conditioning adds parameters and turnover without adding
robustness.

That is a finding, not a failure. The framework was built to be capable of
returning it.

### What the finding is not

It is not "machine learning does not work in markets". It is a bounded claim
about one instrument, one timeframe, one feature set and one cost assumption.
The obvious places to push next are named in [docs/findings.md](docs/findings.md).

---

## The research console

```bash
qmr console          # or: streamlit run app/main.py
```

Seven views over the same library:

| Tab | What it is for |
|---|---|
| **Overview** | The question, the pipeline, and the design decisions that determine the answer |
| **Data** | Price and features, return distribution, and the target the models are asked to predict |
| **Regimes** | Fit a detector; measure persistence, distinctness and stability of the states it finds |
| **Run a study** | Configure and launch a walk-forward experiment, with a live log |
| **Results** | Full out-of-sample report: economics, fold stability, regime breakdown, feature importance, significance |
| **Model comparison** | Every stored study side by side, with regime arms paired against their own controls |
| **Signals** | Out-of-sample positions on the chart, bar by bar, with the model's conviction over time |

Everything the console can produce is reachable from the CLI, and every study it
runs is written to `experiments/<run_id>/` with the exact configuration that
produced it.

---

## Install

```bash
git clone https://github.com/Iftekhar-mobin/quant-market-regime-research.git
cd quant-market-regime-research
python -m pip install -e .
```

Python 3.10 or later. Optional extras: `pip install -e ".[sequence]"` for the
LSTM/GRU learners, `".[mt5]"` for MetaTrader 5 history export.

Trimmed sample datasets for EURUSD, GBPUSD and GOLD ship with the repository, so
a fresh clone runs immediately:

```bash
qmr datasets     # what history is available
qmr run          # one study with the default configuration
qmr console      # the research console
```

For the full history, drop broker CSVs named
`SYMBOL_TIMEFRAME_YYYYMMDD_YYYYMMDD.csv` into `data/raw`, or pull them directly:

```bash
qmr export-mt5 --symbols EURUSD,GBPUSD,GOLD --timeframes H1,H4,D1
```

---

## Command line

```bash
qmr datasets                                       # discovered market history
qmr run --set model.name=xgboost --set regime.method=kmeans
qmr compare --models logistic,xgboost,lightgbm --regimes none,kmeans
qmr results                                        # every stored run
qmr show <run_id>                                  # one run in full
```

Any configuration key can be overridden from the command line:

```bash
qmr run \
  --set data.symbol=GOLD \
  --set data.timeframe=H4 \
  --set labeling.horizon=12 \
  --set backtest.min_holding_bars=12 \
  --set validation.n_folds=8
```

---

## What the framework does that most do not

**Causal features, checked.** Swing points are reported on the bar that
*confirms* them, not the bar they occurred on. The conventional
`argrelextrema` pivot tells the model where the top was before the top formed —
the fastest route to a backtest that cannot be reproduced live.

**Everything refitted inside the fold.** The regime detector, the scaler, the
imputer and the classifier see the training window only. Clustering once on the
full sample and then evaluating "out of sample" within those clusters leaks the
entire future into the regime assignment, and produces the most convincing-looking
chart in the field.

**An embargo between train and test.** A label on the last training bar is a
function of the next *h* bars, which are the first bars of the test window. The
tail of each training window is purged.

**Path-aware labels.** The triple barrier follows each hypothetical position
forward until it touches a profit barrier, a loss barrier or the time limit, in
ATR units. A fixed-horizon return sign rewards the model for calling a move a
real position would have been stopped out of.

**Honest execution.** Fills at the next bar's open, never the close that produced
the signal. Costs on every change in position, so a long-to-short flip pays
twice. A minimum holding period, because a horizon-ahead forecast is one opinion.

**Benchmarks that are not handicapped.** Buy-and-hold plus four rule-based
strategies, priced through the identical execution model and costs.

**Significance, three ways.** A stationary-bootstrap confidence interval that
preserves autocorrelation; a probabilistic Sharpe ratio corrected for skew and
kurtosis; a deflated Sharpe ratio that discounts for the number of trials.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/methodology.md](docs/methodology.md) | Every assumption, and the reasoning behind it |
| [docs/architecture.md](docs/architecture.md) | Module layout, data flow, extension points |
| [docs/findings.md](docs/findings.md) | Full results and where the study goes next |

---

## Configuration

One `configs/default.yaml` drives everything, and a resolved copy is saved with
every run.

```yaml
data:      { symbol: EURUSD, timeframe: H1, warmup_bars: 300 }
features:  { blocks: [returns, trend, momentum, volatility, structure, volume, session] }
labeling:  { method: triple_barrier, horizon: 24, take_profit_atr: 1.0, stop_loss_atr: 1.0 }
regime:    { method: kmeans, n_regimes: 4, as_model_feature: true }
model:     { name: xgboost, decision_threshold: 0.50, class_balance: true }
validation:{ scheme: expanding, n_folds: 6, test_size: 0.12, embargo_bars: 48 }
backtest:  { cost_bps: 2.0, slippage_bps: 0.5, execution_lag: 1, min_holding_bars: 24 }
```

Learners available: `logistic`, `random_forest`, `hist_gradient_boosting`,
`xgboost`, `lightgbm`, `mlp`, and `lstm` / `gru` with the `sequence` extra.
Regime detectors: `none`, `rule`, `kmeans`, `gmm`.

---

## Author

**Iftekharul Mobin** — quantitative research, machine learning and financial
time series.
[GitHub](https://github.com/Iftekhar-mobin) ·
[Google Scholar](https://scholar.google.com/citations?user=xmFRahwAAAAJ)

## License

MIT — see [LICENSE](LICENSE).

> Research code. Nothing here is investment advice, and past out-of-sample
> performance is not a forecast of anything.
