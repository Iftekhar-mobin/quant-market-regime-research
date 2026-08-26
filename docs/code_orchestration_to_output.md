# From one command to one answer

### A complete walkthrough of how this project runs

---

## How to read this book

You type one command. Ninety seconds later a table of numbers appears. This book
explains every step in between — which file wakes up, which function it calls,
what that function is handed, what it gives back, what the data looks like at
that moment, and where it goes next.

The command being traced is:

```bash
qmr run --set data.start=2020-01-01 --set validation.n_folds=3 --set model.name=random_forest
```

**Every shape, column name and number in this book was captured from that exact
run.** Nothing is illustrative. If you run the same command you will see the
same things, and you are encouraged to — the last chapter shows you how to stop
the program at any stage and look.

Each stage is described in the same layout, so you can skim or study:

> **WHO CALLS IT** — the file and function above it in the chain
> **IT RECEIVES** — what is handed in
> **IT DOES** — the work, in plain English
> **IT RETURNS** — the exact object, with real shapes
> **WHERE IT GOES** — the next stage

Assumed: a little trading knowledge, a little Python. Not assumed: any
experience with projects made of many files. That last part is what this book is
really about.

---

## Chapter 1. The shape of the whole thing

Before the detail, the skeleton. Nine stages, in order:

```
   1. qmr                     the terminal word
   2. cli.py                  reads your command
   3. config.py               loads the settings
   4. runner.py               the conductor - everything below is called by it
      5. loader.py            price data          ->  38,684 bars x 5 columns
      6. pipeline.py          features            ->  38,165 bars x 84 columns
      7. targets.py           labels              ->  38,128 answers
      8. walk_forward.py      the fold schedule   ->  3 train/test splits
      9. per fold:
            detectors.py      market regimes      ->  4 states
            zoo.py            fit + predict       ->  probabilities
            engine.py         backtest            ->  equity, trades
     10. metrics.py           performance
         classification.py    prediction quality
         significance.py      is it real?
     11. store.py             write 14 files to disk
```

Two ideas to hold on to, because everything else follows from them.

**The conductor pattern.** `runner.py` does almost no work itself. It calls the
specialists in order and passes each one's output to the next. This is why the
project has many small files instead of one big one: each specialist can be
read, tested and replaced on its own, and the conductor stays short enough to
hold in your head.

**Everything is a table.** Between stages, the data is always a pandas
`DataFrame` — rows labelled by timestamp, columns labelled by name. No custom
formats, no objects that only one function understands. That is why you can stop
the program anywhere and simply look at what it has.

---

## Chapter 2. `qmr` — how a word becomes a program

You type `qmr`. Your operating system has no idea what that is. So who does?

The answer is in `pyproject.toml`:

```toml
[project.scripts]
qmr = "qmr.cli:main"
```

When you ran `pip install -e .`, Python read that line and created a small
launcher program called `qmr` on your system PATH. All it does is import
`qmr/cli.py` and call the function `main` inside it.

> **Read as:** "the command `qmr` means: in the module `qmr.cli`, run `main`."

That is the entire mechanism. There is no magic — the `[project.scripts]` table
is the connection between a word in your terminal and a function in this
repository.

`streamlit run app/main.py` is the same idea by a different route: Streamlit is
the launcher, and `app/main.py` is the function it runs.

---

## Chapter 3. `cli.py` — reading your instructions

**FILE** `src/qmr/cli.py` · **FUNCTION** `main()`

> **WHO CALLS IT** the `qmr` launcher
> **IT RECEIVES** your command line, as a list of words
> **IT RETURNS** an exit code — 0 means success

### 3.1 Working out what you asked for

`argparse` is Python's standard tool for reading command lines. `build_parser()`
declares what is allowed:

```python
run = subparsers.add_parser("run", help="run one study")
run.set_defaults(func=cmd_run)
```

That last line is the important one. It attaches the *function* `cmd_run` to the
word `run`. Later, `main` simply does:

```python
return int(args.func(args))
```

`args.func` is whatever function the sub-command attached. Typing `qmr run`
calls `cmd_run`; typing `qmr datasets` calls `cmd_datasets`. Adding a new
command means writing a function and attaching it — nothing else changes.

### 3.2 Turning the terminal on

```python
ensure_directories()
configure_logging(level=..., log_file=LOG_DIR / "qmr.log")
```

`ensure_directories()` (in `paths.py`) creates `data/`, `experiments/`, `logs/`
if they are missing, so nothing later fails for a boring reason.

`configure_logging()` (in `logging_utils.py`) decides where messages go. Every
`log.info(...)` anywhere in the project now prints to your terminal *and* to
`logs/qmr.log`. This is why the pipeline never uses `print()` — logging can be
redirected, silenced or captured, and `print` cannot. The research console uses
exactly that ability to stream a running study into your browser.

### 3.3 Handing over

`cmd_run` does three things:

```python
config = _load_config(args)        # Chapter 4
result = run_experiment(config)    # Chapter 5 onwards - all the work
save_experiment(result)            # Chapter 14
```

**WHERE IT GOES:** `_load_config` → `Config.load()`.

---

## Chapter 4. `config.py` — one place for every setting

**FILE** `src/qmr/config.py` · **FUNCTION** `Config.load()`

> **IT RECEIVES** a path to a YAML file (default `configs/default.yaml`)
> **IT RETURNS** a `Config` object

### 4.1 What is on disk

`configs/default.yaml` is plain text:

```yaml
data:
  symbol: EURUSD
  timeframe: H1
  warmup_bars: 300

labeling:
  method: triple_barrier
  horizon: 24
  take_profit_atr: 1.0
  stop_loss_atr: 1.0
```

### 4.2 What it becomes

`Config.load()` reads that into a **tree of dataclasses** — one small class per
section:

```python
config.data.symbol            # 'EURUSD'
config.labeling.horizon       # 24
config.backtest.cost_bps      # 2.0
```

**Why not just use the dictionary?** Because a dictionary accepts anything. If
you wrote `symbal: EURUSD`, a dictionary-based program would silently ignore it
and study the default instrument, and you would never know. The typed version
refuses at load time:

```python
raise KeyError(f"Unknown key(s) {sorted(unknown)} in section '{key}'.")
```

**A misspelling becomes an error instead of a wrong answer.** On a project where
one setting can change the conclusion, that is worth the extra file.

### 4.3 Your `--set` flags

```python
config = config.with_overrides({"data.start": "2020-01-01", "validation.n_folds": 3})
```

`with_overrides` walks the dotted path, checks each level exists, and returns a
**copy**. The original is untouched, so an arm of a comparison can never
contaminate the next.

> **THE OBJECT NOW:** `Config(data=DataConfig(symbol='EURUSD', timeframe='H1',
> start='2020-01-01', warmup_bars=300), labeling=LabelConfig(method='triple_barrier',
> horizon=24, ...), model=ModelConfig(name='random_forest', decision_threshold=0.5,
> ...), validation=ValidationConfig(n_folds=3, embargo_bars=48, ...), ...)`

**WHERE IT GOES:** the whole object is handed to `run_experiment(config)`. From
here on, every stage reads its settings from it. Nothing in the pipeline has a
hard-coded number.

---

## Chapter 5. `runner.py` — the conductor takes over

**FILE** `src/qmr/experiments/runner.py` · **FUNCTION** `run_experiment(config)`

This one function is the spine of the project. Read it and you have read the
study. The remaining chapters are its stages, in the order it calls them.

It begins with a small piece of housekeeping worth understanding:

```python
def report(fraction, message):
    log.info(message)
    if progress is not None:
        progress(fraction, message)
```

`progress` is a *function passed in as an argument*. From the CLI it is `None`
and nothing extra happens. From the research console it is a function that moves
a progress bar. This is how the same code serves a terminal and a browser
without knowing which is watching — the runner announces what it is doing, and
whoever is listening decides what to do with that.

---

## Chapter 6. Stage one — getting the price data

**FILE** `src/qmr/data/loader.py` · **FUNCTION** `load_ohlcv(...)`

> **CALLED BY** `run_experiment`, line ~1 of the body
> **IT RECEIVES** `symbol='EURUSD'`, `timeframe='H1'`, `start='2020-01-01'`

### 6.1 Finding the file

`load_ohlcv` first calls `find_dataset()` in `data/catalog.py`. The catalogue
scans two folders — `data/raw` then `data/samples` — and reads each filename:

```
EURUSD_H1_20140525_20251021.csv
  ^      ^      ^        ^
symbol  timeframe  from    to
```

A regular expression pulls those four pieces out. This is why the naming
convention matters: the catalogue can answer *what data do I have, and covering
what period* without opening a single file. It returns a `DatasetInfo` record
holding the path.

### 6.2 Cleaning it

`read_ohlcv_csv(path)` then does the unglamorous work:

1. **Rename columns.** `Open`/`open`/`o` all become `open`. Exports that carry
   both `time` and `Time` collapse to one column.
2. **Parse timestamps.** Numbers are epoch seconds; text is a date string.
3. **Drop duplicates.** The repeated bar at a daylight-saving change is a
   re-send, not a second hour of trading.
4. **Drop impossible bars.** A bar whose high is below its low is a broker
   glitch.

### 6.3 What comes out

> **IT RETURNS** a `DataFrame`, shape **(38684, 5)**

```
                        open     high      low    close  volume
timestamp
2020-01-02 09:00:00  1.12083  1.12095  1.12036  1.12091  1104.0
2020-01-02 10:00:00  1.12091  1.12130  1.12045  1.12111  1354.0
2020-01-02 11:00:00  1.12109  1.12132  1.12004  1.12022  1078.0
```

- **Index:** named `timestamp`, type `datetime64[ns]`, sorted, no duplicates.
- **Columns:** `open, high, low, close, volume`, all `float64`.
- **Meaning:** one row is one hour of EURUSD. 38,684 hours ≈ 4.4 years of
  trading.

**This is the only shape the rest of the project ever has to handle.** Every
mess in the source data was dealt with here, once.

**WHERE IT GOES:** straight into feature construction.

---

## Chapter 7. Stage two — turning 5 columns into 84

**FILE** `src/qmr/features/pipeline.py` · **FUNCTION** `build_features(...)`

> **IT RECEIVES** the price frame **(38684, 5)** and `config.features`
> **IT RETURNS** a `DataFrame`, shape **(38165, 84)**

### 7.1 What a "feature" is

A feature is a number describing the market at one bar, which the model may use
to predict. `close` is not useful on its own — 1.12 means nothing to a model
without context. `rsi_14 = 70.9` means something. The job of this stage is to
turn raw prices into the second kind of number.

### 7.2 How the work is divided

`build_features` calls seven block functions and glues the results together:

```python
parts = [frame.copy()]
if "returns"    in config.blocks: parts.append(_returns_block(frame, ...))
if "trend"      in config.blocks: parts.append(_trend_block(frame, ...))
if "momentum"   in config.blocks: parts.append(_momentum_block(frame))
if "volatility" in config.blocks: parts.append(_volatility_block(frame, ...))
if "structure"  in config.blocks: parts.append(_structure_block(frame))
if "volume"     in config.blocks: parts.append(_volume_block(frame))
if "session"    in config.blocks: parts.append(_session_block(frame))
parts.append(_regime_descriptors(frame))

features = pd.concat(parts, axis=1)
```

`pd.concat(..., axis=1)` glues tables side by side, matching rows by timestamp.
Each block returns a DataFrame with the same index, so they line up exactly.

**Why blocks?** Because switching one off is a legitimate experiment. Removing
`volume` from `config.features.blocks` answers "does volume information help
here at all?" with no code change.

### 7.3 What each block calls

The blocks are recipes; the cooking happens in
`src/qmr/features/indicators.py`. For instance `_momentum_block`:

```python
out = {
    "rsi_14":     ind.rsi(close, 14),
    "stoch_k":    stoch["stoch_k"],
    "williams_r": ind.williams_r(high, low, close, 14),
    "cci_20":     ind.cci(high, low, close, 20),
    ...
}
```

`ind` is the indicators module. Each call returns one column, and the dictionary
becomes a DataFrame.

### 7.4 The warm-up, and why rows disappear

38,684 bars went in and **38,165** came out. 519 rows vanished. Where?

A 200-period moving average has no value until 200 bars have passed. Pandas
marks those `NaN` — "unknown". The pipeline throws them away:

```python
features = features.iloc[warmup_bars:]     # warmup_bars = 300
features = features.ffill().dropna()
```

An invented warm-up value is not a market observation, so it is discarded rather
than filled in. `ffill()` (forward fill) repeats the last known value across
small interior gaps — legitimate, because it copies the *past* forward. There is
deliberately no `bfill()`, which would copy the future backwards.

### 7.5 What comes out

> **84 columns = 5 original OHLCV + 79 features**

```
                       close     ret_1     rsi_14  atr_14_rel  trend_strength  vol_percentile
timestamp
2020-02-03 00:00:00  1.10832 -0.000893  70.916620    0.000893        2.737146           0.868
2020-02-03 01:00:00  1.10883  0.000460  72.970852    0.000874        2.944226           0.844
2020-02-03 02:00:00  1.10808 -0.000677  65.629448    0.000870        3.025204           0.912
```

Reading that first row as a trader: the last bar fell 8.9 basis points; RSI is
70.9 so it is overbought; the average range is 0.089% of price; the fast average
is 2.74 ATRs above the slow one, which is a strong uptrend; and volatility is at
the 87th percentile of its own recent history.

The first twelve feature names, in order:

```
ret_1, ret_3, ret_3_norm, ret_5, ret_5_norm, ret_10, ret_10_norm,
ret_20, ret_20_norm, ret_50, ret_50_norm, ret_skew_50
```

and the last six:

```
weekday_cos, is_london_ny_overlap,
trend_strength, vol_percentile, momentum_score, mean_reversion_score
```

Those final four have a special job — Chapter 10.

### 7.6 The rule that governs this whole file

Every feature at bar *t* uses only bars *t* and earlier. The clearest case is
`swing_points`, which reports a pivot on the bar that *confirms* it rather than
the bar it happened on, because a top is not knowable until the bars after it
have printed. Break that rule and every number in this book becomes fiction —
`scripts/demo_lookahead_bias.py` measures exactly how much.

**WHERE IT GOES:** the feature frame is passed to labelling — and, importantly,
it is passed *whole*, because labelling needs the OHLC columns too.

---

## Chapter 8. Stage three — writing down the right answers

**FILE** `src/qmr/labeling/targets.py` · **FUNCTION** `build_labels(...)`

> **IT RECEIVES** the feature frame **(38165, 84)** and `config.labeling`
> **IT RETURNS** a `LabelResult`

### 8.1 What a label is

To learn, a model needs to be told the right answer for past bars. The label is
that answer: for each bar, should you have been long, short, or flat?

**This is the one stage allowed to look into the future.** It has to be — the
right answer for today depends on what happened next. The whole apparatus of
Chapter 9 exists to make sure that permission does not leak into the features.

### 8.2 How the answer is decided

The default is the **triple barrier**, and it works exactly like a real trade:

```python
upper = entry + config.take_profit_atr * volatility   # take profit, 1 ATR up
lower = entry - config.stop_loss_atr  * volatility    # stop loss,   1 ATR down
```

then walk forward up to 24 bars and see which is touched first:

```python
for step in range(1, horizon + 1):
    j = i + step
    if high[j] >= upper:  outcome = 1;  break      # profit first  -> Long
    if low[j]  <= lower:  outcome = -1; break      # stop first    -> Short
```

`break` leaves the loop the moment a barrier is hit. If neither is touched
inside 24 bars, the label is 0 — the move was not decisive.

**Why not simply "did price rise over 24 bars?"** Because that calls a +30 pip
move a win even if it went 60 pips against you first — a trade you would have
been stopped out of. You know this as a trader. The triple barrier encodes it.

This is the one place in the pipeline that loops bar by bar rather than
operating on whole columns, because a *path* has to be walked in order.

### 8.3 What comes out

> **IT RETURNS** `LabelResult`, a record with five fields

```
labels          38,128 values of -1, 0 or +1
forward_return  the return the trade actually made
holding_bars    how long until a barrier was touched
barrier_hit     which barrier: 'upper', 'lower', 'both', 'time'
method          'triple_barrier'
```

Real distribution from this run:

```
labels:       {-1: 19145,  0: 23,  +1: 18960}
barrier_hit:  {'upper': 18960, 'lower': 18483, 'both': 662, 'time': 23}
```

```
                     label  fwd_ret  bars    hit
timestamp
2020-02-03 13:00:00     -1 -0.00086   3.0  lower
2020-02-03 14:00:00     -1 -0.00083   2.0  lower
2020-02-03 15:00:00     -1 -0.00081   1.0  lower
2020-02-03 16:00:00     -1 -0.00087   1.0   both
```

Three things a researcher checks immediately, and you should too:

1. **Balance.** 19,145 short against 18,960 long — 50.2% / 49.8%. Good. If the
   barriers were asymmetric this would tilt, and the model would inherit a
   directional bias that came from the labelling rule rather than the market.
2. **Only 23 time-outs.** Nearly every trade resolves at a barrier within 24
   bars, so the horizon is well matched to a 1 ATR move.
3. **`both` = 662.** Both barriers touched inside a single bar. The intrabar
   path is unknowable, so the code assumes the loss — conservative by design.

`hit='both'` on the 16:00 row is that case.

**WHERE IT GOES:** back to the runner, which now holds features *and* answers.

---

## Chapter 9. Stage four — aligning, and then cutting time into folds

### 9.1 Lining the two tables up

Features cover 38,165 bars. Labels cover 38,128 — the last 24 bars have no
answer yet, because their future has not happened. The runner reconciles them:

```python
labelled_index = features.index.intersection(label_result.labels.index)
features = features.loc[: labelled_index[-1]]
labels = label_result.labels.reindex(features.index)
```

> **RESULT:** features **(38141, 84)**, of which **38,128 carry an answer**.

Read those two numbers carefully, because they are not the same and the
difference matters.

`features.loc[: labelled_index[-1]]` keeps every bar up to the last one that
could be labelled. `reindex` then lines the labels up against that index, giving
`NaN` for the 13 bars that have a feature row but no answer.

Why keep unlabelled bars at all? Because under `meta` labelling (Chapter 15)
only the bars where a rule proposed a trade carry a target — perhaps a third of
them — while the backtest still needs an unbroken series of bars to compute
returns from. So the study keeps every bar, **trains on the labelled subset**,
and **scores every bar** in the test window:

```python
train_labels = labels.iloc[fold.train_slice].dropna().astype(int)
test_labels  = labels.iloc[fold.test_slice].fillna(0).astype(int)
```

### 9.2 Cutting time into folds

**FILE** `src/qmr/validation/walk_forward.py` · **FUNCTION** `walk_forward_splits(...)`

> **IT RECEIVES** the number of bars (38,128) and `config.validation`
> **IT RETURNS** a list of 3 `Fold` objects

The rule: **train on the past, test on the future that follows it.** Never
shuffle. Shuffling a price series means training on Tuesday to predict Monday.

```python
Fold(index=0, train_start=0, train_end=9535, test_start=9535,
     test_end=14112, embargo_bars=48)
```

Those are *row numbers*, not dates. Translated to real time:

```
 Fold  Train start   Train end            Train bars  Test start           Test end             Test bars  Embargo
    1  2020-02-03    2021-08-11 16:00           9487  2021-08-13 17:00     2022-05-09 19:00          4577       48
    2  2020-02-03    2023-02-21 18:00          19022  2023-02-23 19:00     2023-11-17 11:00          4577       48
    3  2020-02-03    2024-09-03 02:00          28558  2024-09-05 03:00     2025-06-03 17:00          4577       48
```

Read that table carefully; it is the study.

- Every training window **starts at the same date** and grows. This is the
  `expanding` scheme: each fold trains on everything known so far, which is how
  a real system would be maintained.
- Every test window comes **after** its training window.
- Fold 1 trains on 9,487 bars — but `train_end` is row 9,535. The missing 48
  are the **embargo**. (Of those 9,487 bars, 9,474 carry a label; the rest are
  scored but not trained on.)

### 9.3 The embargo, explained slowly

This is the subtlest idea in the project.

The label on the last training bar was computed by looking **24 bars into the
future**. Those 24 bars are the first bars of the test window. So a model
trained right up to the boundary has been told something about the test period —
not through its features, but through its answers.

The fix: throw away the last 48 bars of every training window.

```python
@property
def train_slice(self) -> slice:
    return slice(self.train_start, self.train_end - self.embargo_bars)
```

48 is twice the 24-bar label horizon. **The embargo must be at least the label
horizon**, or the training labels can see the test period.

Nothing warns you if you get this wrong. The results simply come out better than
they should, which is why this small piece of code matters more than it looks.

**WHERE IT GOES:** the fold list drives a loop. Everything in Chapters 10–13
happens three times.

---

## Chapter 10. Inside one fold — part 1: what kind of market is this?

**FILE** `src/qmr/regimes/detectors.py`

### 10.1 Slicing the data

```python
train_features_all = features.iloc[fold.train_slice]   # (9487, 84)
test_features_all  = features.iloc[fold.test_slice]    # (4577, 84)
```

### 10.2 Fitting the detector

```python
detector = build_detector(config.regime, random_state=7)
detector.fit(train_features_all)                       # TRAINING WINDOW ONLY
train_regimes = detector.predict(train_features_all)
test_regimes  = detector.predict(test_features_all)
```

`build_detector` is a **factory**: it reads `config.regime.method` and returns
the matching class.

```python
DETECTORS = {"none": NullRegimeDetector, "rule": RuleBasedRegimeDetector,
             "kmeans": KMeansRegimeDetector, "gmm": GaussianMixtureRegimeDetector}
```

All four expose the same two methods, so the runner never asks which one it
has. Changing `regime.method` in the YAML changes the algorithm and nothing
else. **That is what a factory is for.**

### 10.3 What the detector actually looks at

Not all 79 features — just the four descriptors from Chapter 7.5:

```
trend_strength         (EMA20 - EMA100) / ATR    signed trend, in volatility units
vol_percentile         rank of realised vol      0 = calmest, 1 = most stressed
momentum_score         (RSI - 50) / 50           centred momentum
mean_reversion_score   0.5 - Hurst exponent      above 0 = mean-reverting
```

K-means groups the 9,484 training bars into 4 clusters in this 4-dimensional
space. Because the axes are interpretable, the clusters can be *named* by where
their centre sits:

```python
{0: 'Range, high-volatility',
 1: 'Range, quiet',
 2: 'Downtrend, quiet',
 3: 'Uptrend, high-volatility'}
```

Clustering four meaningful axes gives you states you can defend. Clustering all
79 features would give you a partition nobody can explain.

> **IT RETURNS** a `Series` of integers 0–3, one per bar

```
train regime counts: {0: 2164, 1: 2655, 2: 2481, 3: 2187}   (of 9487)
test  regime counts: {0: 1353, 1:  939, 2: 1537, 3:  748}   (of 4577)
```

The training window is fairly evenly split; the test window leans towards
regimes 0 and 2 and away from 1 and 3. That difference is information — the
market changed between training and testing, which is the whole reason the
detector is refitted per fold rather than once.

### 10.4 Why `fit` and `predict` are separate

If you fitted the detector on *all* the data and then evaluated "out of sample"
inside those clusters, you would have used the future to decide what the past
was. The result looks superb and means nothing.

This is the most seductive version of the mistake, because the resulting chart —
price shaded by regime, boundaries falling exactly at the turning points — is
the most convincing-looking output in the entire field.

*(The Regimes tab in the console does fit on the whole window, because it is for
describing the market rather than scoring a strategy, and it says so on screen.)*

---

## Chapter 11. Inside one fold — part 2: the model

**FILE** `src/qmr/models/zoo.py`

### 11.1 Assembling the input

```python
train_X = train_features_all[model_features]          # 79 columns
test_X  = test_features_all[model_features]

if config.regime.as_model_feature:
    train_X = pd.concat([train_X, _one_hot_regimes(train_regimes, 4)], axis=1)
    test_X  = pd.concat([test_X,  _one_hot_regimes(test_regimes, 4)], axis=1)
```

**One-hot encoding** turns one column of category numbers into several columns
of 0/1. Regime 2 becomes `regime_0=0, regime_1=0, regime_2=1, regime_3=0`. A
model must not be told regime 3 is "more" than regime 1 — they are names, not
quantities, and one-hot encoding is how you say so.

> **RESULT:** `train_X` shape **(9487, 83)** = 79 features + 4 regime columns

### 11.2 Fitting

```python
model = build_model(config.model, seed=7).fit(train_X, train_labels)
```

`X` is the input, `y` the answer — the conventional names, from `y = f(x)`.

`DirectionalModel` is a **wrapper**. Inside, it builds a scikit-learn `Pipeline`:

```
SimpleImputer(median)  ->  RandomForestClassifier(400 trees, depth 8)
```

Three details that matter more than the algorithm:

1. **The imputer is inside the pipeline.** It learns the median from the
   *training* window only. Computing it over all the data would leak the test
   period's statistics.
2. **Class balancing.** `compute_sample_weight("balanced", ...)` up-weights rare
   classes. Without it, a model that always predicts the majority class scores a
   respectable accuracy for doing nothing.
3. **Labels are re-coded.** The pipeline speaks in −1/0/+1; scikit-learn wants
   0/1/2. The wrapper translates in both directions so nothing else has to.

> **THE OBJECT:** `DirectionalModel` labelled 'Random forest', wrapping a
> `Pipeline`

### 11.3 Predicting

```python
probabilities = model.predict_proba(test_X)
```

> **IT RETURNS** a `DataFrame`, shape **(4577, 3)**

```
                     short   flat   long
timestamp
2021-08-13 17:00:00  0.602  0.016  0.381
2021-08-13 18:00:00  0.606  0.014  0.380
2021-08-13 19:00:00  0.604  0.009  0.387
```

Every row sums to 1. The first row says: 60.2% chance short is right, 38.1%
long, 1.6% flat.

Even at its most confident the model is at 60/38 — and across the whole test
window it spends most of its time much closer to 50/50. **That is the honest
picture of financial prediction.** Nothing here is 90% confident, and anything
claiming to be is broken or leaking.

The seven learners in the registry — logistic regression, random forest,
XGBoost, LightGBM, a neural network, LSTM, GRU — all return exactly this table.
That uniformity is why swapping models is a one-word config change.

---

## Chapter 12. Inside one fold — part 3: probabilities become positions

**FILE** `src/qmr/backtest/engine.py` · **FUNCTION** `signals_from_probabilities`

```python
signal[(long_prob >= threshold) & (long_prob >= short_prob)] =  1.0
signal[(short_prob >= threshold) & (short_prob > long_prob)] = -1.0
```

With `threshold = 0.50`: take a position only when the winning class clears 50%.

> **IT RETURNS** a `Series` of −1 / 0 / +1

```
{-1: 2411, 0: 722, +1: 1444}     of 4577 bars
```

So on 3,855 of 4,577 bars the model was confident enough to want a position.
**The threshold is the strategy's main risk dial** — raise it and you trade
less, more selectively.

---

## Chapter 13. Inside one fold — part 4: does it make money?

**FILE** `src/qmr/backtest/engine.py` · **FUNCTION** `run_backtest(...)`

> **IT RECEIVES** the price frame for this fold, the signal series, `config.backtest`
> **IT RETURNS** a `BacktestResult`

These are the financial heart of the project. Eight lines:

```python
side = aligned.shift(config.execution_lag).fillna(0.0)      # 1
positions = side * size

execution_price = frame["open"]                              # 2
bar_return = execution_price.pct_change().shift(-1).fillna(0.0)

gross_return = positions * bar_return                        # 3

cost_rate = (config.cost_bps + config.slippage_bps) / 1e4    # 4
traded_notional = positions.diff().abs().fillna(positions.abs())
costs = traded_notional * cost_rate

net_return = gross_return - costs                            # 5
```

**1 — `shift(1)`, the most important character in the project.** The signal was
computed from a bar's *close*. You cannot trade a close you have not seen yet,
so the position starts on the *next* bar. This single line separates a realistic
backtest from a fantasy.

**2 — open-to-open returns.** The move from this bar's open to the next bar's
open, which is what a position entered at the open actually earns.

**3 — one multiplication does the whole thing.** Long (+1) earns the move, short
(−1) earns its negative, flat (0) earns nothing.

**4 — cost is charged on *change*, not on holding.** `positions.diff().abs()` is
how much the position moved. Holding costs nothing. Flipping long to short is a
change of 2, so it pays twice — correctly, because it is two transactions.

**5 — what is left after the broker.** Everything downstream uses this.

Before all of it, two optional filters run: `apply_session_filter` (only open
positions during liquid hours) and `enforce_min_holding` (hold a new position
for at least N bars, because a 24-bar-ahead forecast is one opinion, not 24).

> **IT RETURNS** `BacktestResult` containing:

```
equity            10000.00, 9994.45, ...          the account balance, bar by bar
returns           net return per bar
positions         -1 / 0 / +1 (times size)
trades            (134, 7)  one row per round trip
benchmark_equity  buy and hold, same window
metrics           the performance dictionary
```

The trades table, extracted by collapsing the bar-by-bar position series into
discrete round trips:

```
           entry_time           exit_time   side  entry_price  exit_price  bars    return
0 2021-08-13 18:00:00 2021-08-17 22:00:00  short      1.17913     1.17102    52  0.006658
1 2021-08-17 22:00:00 2021-08-18 22:00:00   long      1.17102     1.17115    24 -0.000389
```

Row 0: short at 1.17913, covered at 1.17102 fifty-two bars later, +0.67%. Row 1
flips straight to long and loses 0.04% over the 24-bar minimum hold — note that
the exit price of one trade is the entry price of the next, because a flip is a
single transaction at a single price (and pays cost twice, as Chapter 13 step 4
explains).

**WHERE IT GOES:** the fold's predictions and metrics are stored, and the loop
moves to fold 2.

---

## Chapter 14. Stitching the folds back together

After three folds, the runner has three separate prediction tables. It joins
them:

```python
predictions = pd.concat(oos_frames).sort_index()
predictions = predictions[~predictions.index.duplicated(keep="first")]
```

The second line matters: test windows can overlap, and scoring a bar twice would
double-count it.

> **RESULT:** a `DataFrame`, shape **(13731, 8)**

```
                     fold  label  prediction  regime   close  p_short  p_flat  p_long
timestamp
2021-08-13 17:00:00     1      1          -1       1  1.1791   0.6024  0.0162  0.3814
2021-08-13 18:00:00     1     -1          -1       1  1.1795   0.6062  0.0137  0.3801
2021-08-13 19:00:00     1     -1          -1       1  1.1799   0.6040  0.0093  0.3867
```

**This one table is the entire out-of-sample record**, and it is worth
understanding column by column:

| Column | Meaning |
|---|---|
| `fold` | which test window this bar came from |
| `label` | what actually happened (the right answer) |
| `prediction` | what the model said |
| `regime` | which market state was in force |
| `close` | the price |
| `p_short`, `p_flat`, `p_long` | the model's confidence in each |

Row 1 is instructive: the model was 60.2% confident of short and took the
position — and the label says the answer was **+1**. It was confidently wrong.
Rows 2 and 3 are the same call, and those were right.

That sequence is the study in miniature. The model is not guessing randomly; it
is right slightly more often than not. Slightly is not enough.

13,731 rows ≈ 3 folds × 4,577 bars. Every one is a bar the model that predicted
it had never seen.

Then the whole out-of-sample series is backtested once, end to end, and that
result is the headline.

---

## Chapter 15. Stage five — scoring it

The runner now calls four specialists.

### 15.1 Did it predict well? — `evaluation/classification.py`

```python
classification = classification_summary(predictions["label"], predictions["prediction"])
```

Returns accuracy, balanced accuracy, Matthews correlation, per-class precision
and recall, and the one that matters:

```
directional_precision   of the bars where a position was taken, what share were right
```

**Accuracy is reported and then largely ignored.** On a three-class target where
flat is common, a model reaches a fine accuracy by never taking a position.
Precision on the bars actually traded is the number that has to beat costs.

### 15.2 Did it make money? — `backtest/metrics.py`

`performance_metrics(...)` returns ~23 numbers: Sharpe, Sortino, Calmar, maximum
drawdown, profit factor, hit rate, exposure, turnover, VaR, expected shortfall.

For this run:

```
sharpe -0.486   cagr -3.95%   max_drawdown -17.5%   trades 447   precision 50.3%
```

and fold by fold:

```
 fold  train_bars  test_bars  sharpe  trades
    1        9487       4577  -1.193     134
    2       19022       4577  -1.609     164
    3       28558       4577  +1.004     150
```

Two losing folds and one good one. The pooled number hides that, which is why
the framework reports the folds separately and runs a stability test on them.

### 15.3 Where did it work? — `regimes/analysis.py`

```
                     Regime  Bars  Share  Sharpe  Return/bar (bps)  Signal rate  Precision
              Range, quiet  4112  0.299  -0.030            -0.003        0.882      0.488
  Uptrend, high-volatility  3662  0.267   1.664             0.219        0.771      0.488
    Range, high-volatility  3025  0.220  -0.950            -0.137        0.712      0.509
          Downtrend, quiet  2932  0.214  -3.610            -0.408        0.789      0.541
```

This is the table the project's research question turns on, and it repays a
close look.

Sharpe swings enormously across regimes, from **+1.66** in high-volatility
uptrends to **−3.61** in quiet downtrends. That looks like a strong regime
effect — until you read the precision column, which barely moves (48.8% to
54.1%) and is *highest* in the worst-performing regime.

So the model is not more accurate in the profitable regime. The Sharpe
difference is coming from what the market did, not from the model knowing
anything extra about it. The regimes separate *market conditions*, which the
features already encode; they do not separate the bars where the model is right
from the bars where it is wrong. That is why conditioning on them does not
reliably help — and, as [findings.md](findings.md) §4 shows, why the sign of
that effect is not even stable.

### 15.4 Is it real? — `evaluation/significance.py`

```python
significance = significance_report(backtest.returns, fold_metrics, n_trials=3, ...)
```

Returns:

```
sharpe, lower, upper           bootstrap confidence interval
probabilistic_sharpe           P(true Sharpe > 0), corrected for fat tails
deflated_sharpe                discounted for the number of configurations tried
fold_stability                 mean, spread, share of profitable folds
```

The interval comes from a **stationary bootstrap**, which resamples in blocks so
the autocorrelation of the return stream survives. Resampling individual bars
would give an interval far too narrow and a false sense of certainty.

### 15.5 Compared with what? — `models/baselines.py`

Five rule-based strategies — buy and hold, MA crossover, RSI mean reversion,
Donchian breakout, ADX-filtered trend — are run over the identical window,
through the identical execution model and costs. If an 80-feature ensemble
cannot beat a moving-average crossover after costs, that is the finding.

---

## Chapter 16. Stage six — writing it all down

**FILE** `src/qmr/experiments/store.py` · **FUNCTION** `save_experiment(result)`

Creates `experiments/<run_id>/` and writes **14 files**:

```
config.yaml              ~1.4 KB       the exact settings that produced this
summary.json             ~8.4 KB       headline metrics, significance, benchmarks
predictions.parquet      ~1.6 MB       the 13,731-row out-of-sample record
equity.parquet           ~2.1 MB       equity, drawdown, benchmark, per bar
price.parquet            ~1.2 MB       the price data used
trades.csv               ~120 KB       every round trip
fold_metrics.csv         ~1.2 KB       per-fold economics
fold_layout.csv          ~0.7 KB       the walk-forward schedule
regime_table.csv         ~0.4 KB       regime characterisation
regime_performance.csv   ~0.8 KB       performance by regime
regime_transitions.csv   ~0.5 KB       state transition probabilities
confusion.csv            ~0.1 KB       the confusion matrix
threshold_curve.csv      ~1.7 KB       precision against coverage
feature_importance.csv   ~2.9 KB       what the model leaned on
```

Two deliberate choices:

**Plain formats.** CSV, JSON and Parquet — readable by Excel, pandas, R, or a
text editor. A study that can only be read back by the code that wrote it is not
reproducible in any useful sense.

**The fitted model is not saved.** A study is reproduced by re-running its
`config.yaml`, not by unpickling an estimator whose library version has since
moved on.

`summary.json` top-level keys:

```
run_id, created_at, duration_seconds, label, symbol, timeframe, model,
regime_method, n_regimes, specialised_models, labeling, decision_threshold,
oos_bars, oos_start, oos_end, metrics, benchmark_metrics, classification,
significance, baselines
```

---

## Chapter 17. The console reads the same files back

`streamlit run app/main.py` starts a web page. It computes **nothing** the CLI
cannot.

```
app/main.py       page setup, the sidebar, seven tabs
app/state.py      cached data access (@st.cache_data - remember, do not recompute)
app/theme.py      the palette and every chart
app/tabs/*.py     one file per tab, each exposing render(selection, config)
```

The Results tab does exactly this:

```python
record = load_experiment(run_id)      # store.py reads the 14 files back
st.plotly_chart(theme.equity_chart(record["equity"]["equity"], ...))
```

It reads the artefacts and draws them. The Run tab calls the same
`run_experiment` the CLI calls, passing a `progress` function so the browser can
show a bar (Chapter 5).

**Every chart is built in `app/theme.py`**, never in a tab. That is what keeps
forty figures across seven tabs looking like one instrument.

---

## Chapter 18. The whole journey on one page

```
YOU TYPE
    qmr run --set data.start=2020-01-01 --set validation.n_folds=3
        |
pyproject.toml [project.scripts]  ->  qmr.cli:main
        |
cli.py main() -> cmd_run()
        |
config.py Config.load()           ->  Config object (typed settings tree)
        |
runner.py run_experiment(config)  ->  THE CONDUCTOR
        |
        +-- loader.py    load_ohlcv()       ->  (38684, 5)   price
        +-- pipeline.py  build_features()   ->  (38165, 84)  features
        +-- targets.py   build_labels()     ->  38128 answers
        +-- align                           ->  (38141, 84), 38128 labelled
        +-- walk_forward.py  splits()       ->  3 Folds
        |
        +-- FOR EACH FOLD (x3):
        |     detectors.py  fit/predict     ->  regimes 0-3
        |     zoo.py        fit             ->  DirectionalModel
        |     zoo.py        predict_proba   ->  (4577, 3) probabilities
        |     engine.py     signals_from_probabilities  ->  -1/0/+1
        |     engine.py     run_backtest    ->  fold economics
        |
        +-- concat folds                    ->  (13731, 8) predictions
        +-- engine.py    run_backtest()     ->  equity, 447 trades
        +-- metrics.py, classification.py, significance.py, baselines.py
        |
        v
ExperimentResult
        |
store.py save_experiment()  ->  experiments/<run_id>/  (14 files)
        |
        +--> terminal table
        +--> research console reads the same files
```

---

## Chapter 19. Every file, one line each

**The conductor**

| File | Job |
|---|---|
| `experiments/runner.py` | Calls every stage in order. Read this to understand the study. |
| `experiments/store.py` | Writes results to disk; reads them back. |
| `cli.py` | Turns terminal words into function calls. |
| `config.py` | Typed settings; rejects misspellings at load time. |
| `paths.py` | Where everything lives on disk. |
| `logging_utils.py` | Where messages go, including into the browser. |

**The specialists, in pipeline order**

| File | Input | Output |
|---|---|---|
| `data/catalog.py` | folders | which datasets exist |
| `data/loader.py` | a CSV | clean OHLCV, (38684, 5) |
| `data/mt5_export.py` | MetaTrader 5 | fresh CSVs (optional) |
| `features/indicators.py` | price columns | one indicator column each |
| `features/pipeline.py` | OHLCV | (38165, 84) features |
| `labeling/targets.py` | features | 38,128 answers |
| `validation/walk_forward.py` | a row count | 3 train/test folds |
| `regimes/detectors.py` | 4 descriptors | a regime per bar |
| `regimes/analysis.py` | regimes | persistence, transitions, stats |
| `models/zoo.py` | features + labels | probabilities |
| `models/baselines.py` | OHLCV | rule-based benchmark signals |
| `backtest/engine.py` | signals + prices | equity, trades |
| `backtest/metrics.py` | returns | ~23 performance numbers |
| `evaluation/classification.py` | predictions | precision, recall, confusion |
| `evaluation/significance.py` | returns | confidence intervals, deflated Sharpe |

**The interface**

| File | Job |
|---|---|
| `app/main.py` | Page setup, sidebar, seven tabs. |
| `app/state.py` | Cached data access shared by the tabs. |
| `app/theme.py` | The palette and every chart in the console. |
| `app/tabs/*.py` | One module per tab. |

---

## Chapter 20. Watch it yourself

Reading is not believing. Stop the program at any stage and look.

```bash
python
```

```python
from qmr.config import Config
from qmr.data.loader import load_ohlcv
from qmr.features import build_features
from qmr.features.pipeline import feature_columns
from qmr.labeling import build_labels
from qmr.validation import walk_forward_splits, describe_folds

cfg = Config.load().with_overrides({"data.start": "2020-01-01",
                                    "validation.n_folds": 3})

# Chapter 6
price = load_ohlcv("EURUSD", "H1", start="2020-01-01")
print(price.shape, price.columns.tolist())
print(price.head())

# Chapter 7
features = build_features(price, cfg.features, warmup_bars=300)
print(features.shape, len(feature_columns(features)), "model inputs")
print(features[["close", "rsi_14", "trend_strength"]].tail())

# Chapter 8
labels = build_labels(features, cfg.labeling)
print(labels.summary())
print(labels.labels.value_counts())
print(labels.barrier_hit.value_counts())

# Chapter 9
folds = walk_forward_splits(len(labels.labels), cfg.validation)
print(describe_folds(folds, features.index))
```

Then, when you want the whole thing but want to *watch* it:

```bash
qmr run -v --set data.start=2020-01-01 --set validation.n_folds=3
```

`-v` turns on the full log, and every stage announces itself as it runs. Line
them up against the chapters here.

---

## A closing thought

You may have noticed the answer this study produces is a bad one — a negative
Sharpe ratio, a model that predicts barely better than a coin flip. That is not
a defect in the machinery. **It is the machinery working.**

Almost everything in this codebase exists to stop a result from looking better
than it is: features that cannot see the future, detectors refitted inside every
fold, an embargo between training and testing, execution at the next bar's open,
costs on every change of position, confidence intervals that respect
autocorrelation, and a Sharpe ratio discounted for the number of configurations
tried.

Take any one of those away and the numbers improve. That is precisely why they
are there.

The hard part of quantitative research is not building a model. It is knowing
whether the number in front of you is real. Now you know where, in this project,
that question gets asked — and you can go and ask it of somebody else's code.

---

*Next: [novice_learner.md](novice_learner.md) teaches the Python itself.
[methodology.md](methodology.md) argues each assumption.
[architecture.md](architecture.md) is the reference map.
[findings.md](findings.md) reports what the study found.*
