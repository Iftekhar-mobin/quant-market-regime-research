# Reading this codebase without being a programmer

This guide is for someone who understands **trading** — moving averages, stop
losses, support and resistance — but is new to **Python**. It teaches the
language through this repository rather than in the abstract, because you
already know what the code is *trying* to do, and that is most of the battle.

You do not need to read the whole thing. Work through Part 1 and Part 2, then
use Part 6 as a map and come back to Part 7 whenever something looks like
gibberish.

---

## Part 0. Get something on screen first

Nothing kills motivation like reading code that has never run. Do this first.

```bash
cd quant-market-regime-research
python -m pip install -e .
qmr console
```

A browser opens. Click through the tabs. **Everything you see there is produced
by the files in `src/qmr/`** — the whole point of the tour below is to connect
what you just looked at to the code that made it.

Then run one study from the terminal, so you can see the same numbers arrive
without a browser:

```bash
qmr run --set model.name=random_forest --set validation.n_folds=3
```

That single command touches roughly 80% of this codebase. By the end of Part 3
you will know what each stage of its output means.

---

## Part 1. The five ideas you actually need

Python has a lot of features. You need five to read this repository.

### 1.1 A `Series` is one column of a spreadsheet

If you have used Excel or MetaTrader, you already have the mental model. A
`Series` is a single labelled column — closing prices, say — where the labels
are timestamps.

```python
close = frame["close"]        # one column: the close price of every bar
close.iloc[-1]                # the most recent value
close.mean()                  # the average over the whole column
```

### 1.2 A `DataFrame` is the whole spreadsheet

Many columns sharing one set of row labels. In this project every DataFrame of
market data is indexed by **timestamp** and has the columns `open`, `high`,
`low`, `close`, `volume`.

```python
frame["close"]                # -> a Series
frame[["open", "close"]]      # -> a smaller DataFrame
frame.iloc[-100:]             # the last 100 rows
```

`pd` is the conventional short name for the pandas library, which provides
both. You will see `import pandas as pd` at the top of nearly every file.

### 1.3 Operations apply to the whole column at once

**This is the single biggest shift if you come from MQL4, Pine Script, or VBA.**

In those languages you write a loop: *for each bar, do something*. In pandas you
almost never do. You write one expression and it applies to every row.

```python
# Not this (the way most trading platforms make you think):
for i in range(len(close)):
    change[i] = close[i] - close[i - 1]

# This:
change = close.diff()
```

`diff()` subtracts each value from the one before it, for all 73,419 bars of
EURUSD H1 history, in one line. This is called *vectorised* code, and it is both
shorter and far faster — measured on this dataset, the loop above takes 8.9 ms
and `diff()` takes 0.32 ms, a 28× difference. On the heavier operations the gap
is wider still.

The three vectorised operations that carry most of this codebase:

| Code | In plain English | Trading example |
|---|---|---|
| `close.diff()` | this bar minus the previous bar | bar-to-bar change |
| `close.shift(1)` | the value from **1 bar ago**, lined up on this bar | previous close |
| `close.rolling(20).mean()` | the average of the last 20 bars | a 20-period SMA |

`shift()` is the one to really understand — see Part 7.1. It is how the code
avoids cheating.

### 1.4 A function is a named recipe

```python
def atr(high, low, close, window=14):
    ...
    return result
```

- `def` starts the definition. `atr` is the name.
- `high, low, close, window` are the **parameters** — the ingredients.
- `window=14` means *if the caller does not say, use 14*.
- `return` hands the answer back.

You call it by name: `atr(frame["high"], frame["low"], frame["close"], 20)`.

The odd-looking colons and arrows are **type hints** — a note to the reader
about what goes in and what comes out:

```python
def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
```

Read that as: "three price columns and a whole number go in; one column comes
out." Python does not enforce it. It is documentation that your editor can
check. **You can mentally delete everything after a `:` and before a `,` when
you are first reading a line.**

### 1.5 A class is a thing that remembers

A function forgets everything when it returns. A **class** describes an object
that holds onto state.

```python
detector = KMeansRegimeDetector(features=[...], n_regimes=4)
detector.fit(training_data)      # learns where the regime boundaries are, remembers them
regimes = detector.predict(test_data)   # applies what it remembered
```

`fit` then `predict` is the standard pattern in machine learning, and it is the
reason regimes in this project are honest: `fit` is only ever shown the training
window (see Part 3.4).

Some classes here are just **labelled bags of values**, marked `@dataclass`:

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    cost_bps: float = 2.0
    slippage_bps: float = 0.5
```

That is nothing more than a settings record with defaults. `10_000.0` is just
`10000.0` — the underscore is a thousands separator for human eyes.

---

## Part 2. How to read any file in this project

Every file has the same shape. Here is the top of
`src/qmr/features/indicators.py`:

```python
"""Technical indicator primitives.                    <- 1. what the file is for

Every function here is *causal*: the value at bar ``t`` uses information from
bars ``<= t`` only. ...
"""

from __future__ import annotations                    <- 2. ignore this line

import numpy as np                                    <- 3. tools being borrowed
import pandas as pd


def true_range(high, low, close):                     <- 4. the actual work
    ...
```

1. **The triple-quoted block at the top is a docstring.** In this repository the
   docstrings explain *why*, not what. They are the most valuable thing in the
   file for a learner. Read them first, always.
2. `from __future__ import annotations` is a compatibility line that affects
   type hints only. It sits at the top of every real module here. Ignore it
   forever.
3. `import` lines borrow other people's code. `as np` gives it a short nickname.
4. Then functions, usually smallest first.

**A leading underscore means "internal".** `_returns_block` is a helper meant to
be used only inside its own file; `build_features` is meant to be used by
everyone. Nothing enforces this — it is a convention, and a useful signal about
where to start reading (start with the ones *without* underscores).

---

## Part 3. A guided tour: follow one bar through the machine

Take a single EURUSD hourly bar. Here is everywhere it goes.

### 3.1 It gets loaded and cleaned — `src/qmr/data/loader.py`

A broker CSV is messier than you would hope. Column names differ between
exports, timestamps arrive as either `2024-01-05 13:00` or `1704459600`, and the
same bar sometimes appears twice around daylight-saving changes.

`read_ohlcv_csv` fixes all of that once, so no other file has to think about it.
The important line:

```python
frame = frame.loc[valid]
```

where `valid` marks bars that are not obviously broken — a bar whose high is
below its low is a data glitch, not a market event.

**Concept: a boolean mask.** `frame["high"] >= frame["low"]` does not give you
`True` or `False`. It gives you a **column of** `True`/`False`, one per bar.
Feeding that column back into the frame keeps only the `True` rows. You will see
this pattern everywhere; it is pandas' version of "filter".

### 3.2 It becomes ~80 features — `src/qmr/features/indicators.py`

Read `rsi` first. You already know what RSI means, so every line teaches you
Python for free:

```python
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()                      # bar-to-bar change
    gain = delta.clip(lower=0.0)              # keep the ups, zero the downs
    loss = (-delta).clip(lower=0.0)           # flip sign, keep the downs
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)
```

Line by line:

- `clip(lower=0.0)` — floor every value at zero. Losses become 0, gains survive.
- `(-delta)` — negate the whole column, so down-moves become positive.
- `.ewm(...).mean()` — an exponentially weighted average. This is Wilder's
  smoothing, the same one your platform uses for RSI.
- `.replace(0.0, np.nan)` — guard against dividing by zero. `np.nan` means
  "not a number", pandas' way of saying *missing*. Division by `nan` gives `nan`
  rather than crashing.
- `.fillna(50.0)` — where the answer was undefined, use the neutral value 50.

Six lines, one classic indicator, and you have now met masks, arithmetic on
whole columns, missing-value handling, and method chaining.

**Method chaining** is the `.this().that().other()` style. Read it left to
right, like a pipeline: take the gains, smooth them, take the mean.

### 3.3 The one function worth understanding properly — `swing_points`

This is where the project differs from most trading code you will find online,
and it is worth the effort.

```python
def swing_points(high, low, left=5, right=5):
    window = left + right + 1
    rolling_high = high.rolling(window).max()
    centre_high = high.shift(right)
    is_swing_high = (centre_high >= rolling_high).astype(float)
```

The question is: *is this bar a local top?*

The obvious answer — "it is higher than the 5 bars either side" — has a fatal
problem. **You cannot know the 5 bars to the right until 5 bars later.** Code
that marks the top on the bar it occurred is telling your model where the top
was before the top had formed. Backtest results built on it are fiction, and it
is the most common bug in retail trading research.

So this function deliberately reports the pivot **late**: `shift(right)` moves
the candidate 5 bars forward, and the flag lands on the bar that *confirms* the
pivot, not the pivot itself. You lose 5 bars of timeliness and gain a result
that is actually reproducible.

The word for this is **causality**: only use information you could have had at
the time. Almost every design decision in this repository is downstream of it.

### 3.4 It gets a regime — `src/qmr/regimes/detectors.py`

The code asks "what kind of market is this?" and answers with one of four
states, described by four numbers (`trend_strength`, `vol_percentile`,
`momentum_score`, `mean_reversion_score`).

The mechanism is k-means clustering, but the mechanism matters less than the
discipline around it:

```python
detector.fit(train_features)          # learns the boundaries — training data only
train_regimes = detector.predict(train_features)
test_regimes  = detector.predict(test_features)   # applies them, learns nothing new
```

If you fit on *all* the data and then test "out of sample" inside those
clusters, you have used the future to decide what the past was. The result looks
brilliant and means nothing. Keeping `fit` and `predict` separate is what
prevents it.

### 3.5 It gets a label — `src/qmr/labeling/targets.py`

The model needs to know the right answer for each bar to learn from. This is the
one place in the codebase that is *allowed* to look forward.

The **triple barrier** works exactly like a trade you would actually place:

- an upper barrier 1 ATR above entry (take profit),
- a lower barrier 1 ATR below (stop loss),
- a time limit of 24 bars.

Run forward from the bar and see which one gets hit first. That is the label:
`+1`, `-1`, or `0`.

Why not just "did price go up over the next 24 bars?" Because that calls a
+30 pip move a win even if it went 60 pips against you first — a trade you would
have been stopped out of. You know this from trading. The label knows it too.

Almost everything else in this project is vectorised (Part 1.3), but a path has
to be walked in order, so this is the one place in the feature-and-label path
that loops bar by bar:

```python
for step in range(1, horizon + 1):
    j = i + step
    if high[j] >= upper:
        outcome = 1
        break                 # first touch wins; stop looking
```

`break` means leave the loop immediately.

### 3.6 It lands in a training or a testing window — `src/qmr/validation/walk_forward.py`

The rule: **always train on the past, always test on the future that follows.**

Normal machine-learning practice shuffles the data and splits it randomly. On a
price series that means training on Tuesday to predict Monday. Every result is
optimistic and you cannot tell by how much.

Instead this file produces `Fold` objects:

```python
@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_bars: int
```

`frozen=True` means once created it cannot be modified — a safety rail.

The **embargo** is the subtle bit. A label on the last training bar depends on
the next 24 bars, which are the first bars of the test window. So the tail of
the training window gets thrown away. Without it the model gets a peek at the
test period through the back door.

### 3.7 A model makes a prediction — `src/qmr/models/zoo.py`

Seven different learners (logistic regression, random forest, XGBoost, LightGBM,
a neural network, LSTM, GRU) all wear the same two-method costume:

```python
model.fit(X, y)              # X = the features, y = the labels
model.predict_proba(X)       # -> probabilities for short / flat / long
```

Because they all look identical from outside, the experiment runner can swap one
for another without changing a line. That is the entire reason the file exists.

`X` and `y` are the conventional names: `X` for inputs, `y` for the thing being
predicted. It comes from `y = f(x)` in maths.

### 3.8 The prediction becomes a position — `src/qmr/backtest/engine.py`

```python
signal[(long_probability >= threshold) & (long_probability >= short_probability)] = 1.0
```

Read it as: *where the model is confident enough about "long" and long beats
short, set the signal to +1.* The `&` means "and" for whole columns. (Ordinary
`and` does not work on columns — see Part 7.3.)

### 3.9 The position earns or loses money

These eight lines are the financial heart of the project. Everything else is
preparation.

```python
positions = aligned.shift(config.execution_lag).fillna(0.0) * config.position_size

execution_price = frame["open"]
bar_return = execution_price.pct_change().shift(-1).fillna(0.0)

gross_return = positions * bar_return

cost_rate = (config.cost_bps + config.slippage_bps) / 1e4
traded_notional = positions.diff().abs().fillna(positions.abs())
costs = traded_notional * cost_rate

net_return = gross_return - costs
```

In trading terms:

1. **`positions = aligned.shift(1)`** — the signal computed on this bar's close
   is not acted on until the *next* bar. You cannot trade a close you have not
   seen yet. This one `shift(1)` is the difference between a realistic backtest
   and a fantasy.
2. **`bar_return`** — the move from this bar's open to the next bar's open,
   which is what a position actually earns given the timing above.
3. **`gross_return = positions * bar_return`** — long (+1) earns the move, short
   (−1) earns its negative, flat (0) earns nothing. Multiplication does the
   whole thing.
4. **`traded_notional = positions.diff().abs()`** — how much you *changed* your
   position. Holding costs nothing; changing costs. Flipping long to short is a
   change of 2, so it pays twice — correctly, because it is two transactions.
5. **`net_return`** — what is left after the broker.

Read those five points until they are obvious. If you can explain them to
someone else, you understand the backtest.

---

## Part 4. What the numbers at the end mean

`src/qmr/backtest/metrics.py` turns a stream of per-bar returns into the report.
The ones worth knowing:

| Metric | Plain English | Why it is there |
|---|---|---|
| **Sharpe ratio** | return per unit of wobble, annualised | Above 1 is good. Below 0 lost money. |
| **Max drawdown** | worst peak-to-trough fall | The number that decides whether you could actually have held it |
| **Profit factor** | gross wins ÷ gross losses | Below 1 loses money |
| **Directional precision** | of the bars you traded, what share were right | The number that must beat costs |
| **Exposure** | share of bars holding any position | 100% means always in the market |
| **Annual turnover** | how much you trade per year | High turnover is where profits go to die |

The one to be suspicious of is **accuracy**. On a three-class target where
"flat" is common, a model gets a fine accuracy by never taking a position at
all. This project reports it and then largely ignores it — see
[findings.md](findings.md) §2 for the arithmetic on why a 51% hit rate still
loses money.

---

## Part 5. Where the pieces are joined — `src/qmr/experiments/runner.py`

This is the conductor. Read it once you have met the individual sections; it is
mostly a list of calls to everything above.

```
load_ohlcv          -> price data
build_features      -> ~80 columns
build_labels        -> the right answers
walk_forward_splits -> the fold schedule

for each fold:
    detector.fit(train)          regimes, training window only
    model.fit(train_X, y)        learner, training window only
    model.predict_proba(test_X)  predictions on unseen bars

stitch the predictions -> run_backtest -> metrics -> significance
```

The `for fold in folds:` loop is the shape of the whole study. Everything inside
it happens six times, once per fold, and nothing inside it is ever allowed to
see the test window while fitting.

---

## Part 6. A reading order

Do not read alphabetically. Read in this order, and stop when you have had
enough — the first four alone will teach you a great deal.

| # | File | Difficulty | Why read it |
|---|---|---|---|
| 1 | `src/qmr/features/indicators.py` | ●○○ | Indicators you already know, written in Python. The best entry point. |
| 2 | `src/qmr/models/baselines.py` | ●○○ | Complete trading strategies in 5 lines each. Very readable. |
| 3 | `src/qmr/backtest/engine.py` | ●●○ | The money. Worth real effort. |
| 4 | `src/qmr/labeling/targets.py` | ●●○ | The only forward-looking code, and the only real loop. |
| 5 | `src/qmr/validation/walk_forward.py` | ●●○ | Short, and the idea matters more than the code. |
| 6 | `src/qmr/backtest/metrics.py` | ●●○ | Formulas you have seen in every performance report. |
| 7 | `src/qmr/data/loader.py` | ●●○ | Unglamorous, and 90% of real data work looks like this. |
| 8 | `src/qmr/regimes/detectors.py` | ●●● | Your first real look at classes and inheritance. |
| 9 | `src/qmr/features/pipeline.py` | ●●● | Long, but it is just the indicators assembled. |
| 10 | `src/qmr/models/zoo.py` | ●●● | Wrapping seven libraries in one interface. |
| 11 | `src/qmr/experiments/runner.py` | ●●● | Everything at once. Read last. |
| 12 | `app/` | ●●● | Streamlit. A different skill; leave it for later. |

Start with `baselines.py` if `indicators.py` feels heavy. Here is an entire
trend-following strategy:

```python
def ma_crossover(frame: pd.DataFrame, fast: int = 50, slow: int = 200, **_: object) -> pd.Series:
    """Classic trend following: long above the slow average, short below."""
    close = frame["close"]
    fast_ma = close.ewm(span=fast, adjust=False).mean()
    slow_ma = close.ewm(span=slow, adjust=False).mean()
    return pd.Series(np.sign(fast_ma - slow_ma), index=frame.index, name="signal").fillna(0.0)
```

`np.sign` returns +1 for positive, −1 for negative, 0 for zero. So: long when
the fast average is above the slow one, short when below. That is the whole
strategy, and this project prices it through exactly the same costs as the
machine-learning models.

(The `**_: object` in the signature is a catch-all for keyword arguments this
particular strategy does not use. Every baseline is called the same way, so each
has to tolerate arguments meant for the others. Ignore it.)

---

## Part 7. Things that will confuse you

### 7.1 `shift(1)` versus `shift(-1)`

The most important two characters in the project.

```python
close.shift(1)     # yesterday's value, sitting on today's row  -> looking BACK, safe
close.shift(-1)    # tomorrow's value, sitting on today's row   -> looking FORWARD, dangerous
```

`shift(1)` is legitimate: today you know yesterday's close. A **negative** shift
uses information from the future. There are exactly two in the whole codebase,
and you can check for yourself:

```bash
git grep -n "shift(-" -- src/
```

One is in labelling (`shift(-config.horizon)`, allowed — it is the answer key)
and one is in the backtest (`shift(-1)`, to line up the return a position
earns). Anywhere else it would be a bug.

**If you only remember one thing from this guide, remember to check the sign of
every `shift()` you ever write.**

### 7.2 `NaN` everywhere at the start

A 200-bar moving average has no value for the first 199 bars. Pandas fills those
with `NaN` — "not a number", meaning *unknown*.

The project throws away the leading block rather than filling it in
(`warmup_bars: 300` in the config), because an invented warm-up value is not a
market observation. It will also forward-fill interior gaps (`ffill()` — repeat
the last known value) but never backward-fill, because that would copy the
future into the past.

### 7.3 `&` and `|` instead of `and` and `or`

```python
buy = (rsi < 30) & (close > ma)      # correct
buy = (rsi < 30) and (close > ma)    # crashes
```

`and` works on one true/false value. `&` works on whole columns, element by
element. **The brackets around each condition are required** — without them
Python groups the operators wrongly and you get a confusing error.

### 7.4 `df.iloc[5]` versus `df.loc["2024-01-05"]`

- `iloc` = **i**nteger location. "The 6th row." (Counting starts at 0.)
- `loc` = label location. "The row labelled 2024-01-05 13:00."

`iloc[-1]` is the last row; `iloc[-100:]` is the last hundred.

### 7.5 The `-> pd.Series` arrows and `int | None`

Type hints again. `int | None` means "a whole number, or nothing at all". Skip
them while learning; they never change what the code does.

### 7.6 `f"..."` strings

```python
print(f"Loaded {len(frame)} bars for {symbol}")
```

The `f` lets you drop variables straight into the text inside `{}`.
`f"{value:.2f}"` formats to 2 decimal places, `f"{value:,}"` adds thousands
separators, `f"{value:.1%}"` turns 0.514 into "51.4%".

### 7.7 `@dataclass`, `@property`, `@st.cache_data`

Lines starting with `@` are **decorators** — a label attached to what follows
that modifies its behaviour.

- `@dataclass` — "this class is a settings record; write the boilerplate for
  me."
- `@property` — "this method can be read like a value": `fold.train_size`, not
  `fold.train_size()`.
- `@st.cache_data` — "remember the result so the app does not recompute it."

You can read past all of them on a first pass.

---

## Part 8. Exercises that actually teach

Do these in order. Each is small and each one teaches something the reading
cannot.

**1. Print something.** Add a line to `rsi` in `indicators.py`:

```python
print(f"RSI computed over {len(close)} bars, latest value {close.iloc[-1]:.5f}")
```

Run `qmr run` and watch it appear. You are now editing the code.

**2. Change a number.** In `configs/default.yaml` set `cost_bps: 0.0` and run a
study. Compare the Sharpe ratio to the default. You have just reproduced the
central finding of the whole project — the signal is real, the spread is bigger.

**3. Break causality on purpose and measure what it is worth.** This is the most
valuable exercise here, and it comes as a script so you never have to edit
`src/`:

```bash
python scripts/demo_lookahead_bias.py --start 2020-01-01
```

It runs three full studies that differ in exactly one thing — what the features
are allowed to know — and prints them side by side:

```
Arm         Precision    Sharpe      CAGR    Max DD   Trades   Top feature
------------------------------------------------------------------------------
honest          50.7%     -1.13     -8.7%    -19.7%      455   realised_vol_14
swing           51.0%     -0.76     -6.0%    -15.9%      456   realised_vol_14
future          76.9%     +3.87     36.0%     -5.1%      525   tomorrow_close
```

*(Your exact numbers will differ slightly — the pipeline has been refactored
since this table was captured. The gaps between the rows are what matter, and
those are stable.)*

- **honest** — the pipeline as shipped.
- **swing** — swing pivots reported on the bar they occurred instead of the bar
  that confirms them. A 5-bar leak, through only 5 of the ~84 features.
- **future** — the price move over the next 5 bars handed to the model as an
  input. The answer, supplied as a question.

Two things to take from that table.

*The subtle leak is the dangerous one.* It is worth only +0.38 Sharpe. It does
not produce an absurd equity curve; it just makes everything a little better,
which is precisely why you would believe it and ship it. Most real leaks look
like this.

*The blatant leak is what a fantasy looks like.* Sharpe 3.87, a 5% drawdown, and
`tomorrow_close` sitting at the top of the feature-importance chart. If a retail
backtest ever shows you numbers like that, this is very often why.

Neither one raised an error or printed a warning. Nothing in the output says
"this result is fake" — you have to know to check. That is the whole lesson.

Open `scripts/demo_lookahead_bias.py` afterwards and read `leaky_swing_points`:
the only difference from the shipped version is that two lines no longer say
`.shift(right)`.

(You can run one arm at a time with `--arm swing`, or on other data with
`--symbol GOLD`. Your exact numbers will differ from the table above if you
change the window or the instrument; the *ordering* will not.)

**4. Write your own indicator.** Add to `indicators.py`:

```python
def price_position(close: pd.Series, window: int = 50) -> pd.Series:
    """Where price sits in its recent range: 0 at the low, 1 at the high."""
    lowest = close.rolling(window).min()
    highest = close.rolling(window).max()
    return (close - lowest) / (highest - lowest)
```

Then use it in `_momentum_block` in `features/pipeline.py` and see whether it
appears in the feature-importance chart in the console.

**4b. Try a different question.** The framework can also ask the model *"will this rule-based trade work?"* instead of *"which way next?"* — that is meta-labelling. Compare the two:

```bash
qmr run --set validation.n_folds=3
qmr run --set validation.n_folds=3 --set labeling.method=meta
```

Watch the trade count collapse. Fewer, more selective trades pay less
spread — which, as [findings.md](findings.md) shows, is the whole game
here.

**5. Write your own strategy.** Add a function to `models/baselines.py` and
register it, and it will be priced against every model automatically, through
the same costs. Try: long when RSI is above 50, short when below.

```python
def rsi_trend(frame: pd.DataFrame, window: int = 14, **_: object) -> pd.Series:
    """Long while RSI is above the midline, short below."""
    rsi_series = ind.rsi(frame["close"], window)
    return pd.Series(np.sign(rsi_series - 50.0), index=frame.index).fillna(0.0)
```

Then add an entry to the `BASELINES` dictionary at the bottom of the file. Each
entry is `key: (display name, the function, one-line description)`:

```python
    "rsi_trend": (
        "RSI trend (14)",
        rsi_trend,
        "Long above the RSI midline, short below.",
    ),
```

Run `qmr run` and it appears in the benchmark table automatically — nothing else
needs changing. That is what the registry pattern buys you.

**6. Change one thing and defend it.** Set `labeling.horizon` to 48 and
`backtest.min_holding_bars` to 48, run the study, and write down in one sentence
why the number of trades halved.

---

## Part 9. When you get stuck

- **Read the docstring first.** In this repository they explain reasoning, not
  mechanics. If a function looks arbitrary, the docstring probably says why.
- **Print things.** `print(frame.head())`, `print(series.describe())`,
  `print(len(x))`. Unglamorous and effective.
- **Poke at it interactively:**

  ```bash
  python
  ```
  ```python
  from qmr.data import load_ohlcv
  from qmr.features import build_features
  df = load_ohlcv("EURUSD", "H1")
  print(df.tail())
  f = build_features(df)
  print(f.columns.tolist())
  print(f[["close", "rsi_14", "trend_strength"]].tail(10))
  ```

  Ten minutes of this teaches more than an hour of reading.
- **The error message is usually right.** Read the *last* line first; that is
  what actually went wrong. The lines above it are the trail of how it got there.

---

## Part 10. What to learn next, in order

1. **pandas** — the single highest-value skill here. The official *10 minutes to
   pandas* guide, then just use it.
2. **Python basics** — functions, lists, dictionaries, loops. You have already
   met all four above.
3. **scikit-learn** — `fit` / `predict`, train/test splits, why overfitting
   happens.
4. **Then, and only then, the finance-specific material** — López de Prado's
   *Advances in Financial Machine Learning* for triple-barrier labelling and
   purged validation, both of which are implemented here.

A last piece of advice, and the reason this project is shaped the way it is:
**the hard part of quantitative research is not the model.** It is knowing
whether the number you are looking at is real. Nearly all of the effort in this
codebase goes on that question — causal features, refitting inside each fold,
embargoes, realistic costs, confidence intervals. The machine learning is the
easy 10%.

---

*Read next: [code_orchestration_to_output.md](code_orchestration_to_output.md)
traces one command end to end - every file that wakes up, every function it
calls, and the exact shape of the data at each step. This guide taught you the
language; that one shows you the machine.*

*See also: [methodology.md](methodology.md) for the assumptions and why they
were chosen, [architecture.md](architecture.md) for how the modules fit
together, and [findings.md](findings.md) for what the study actually found.*
