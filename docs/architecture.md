# Architecture

## Layout

```
quant-market-regime-research/
├── app/                       Streamlit research console
│   ├── main.py                entry point, sidebar, tab layout
│   ├── theme.py               palette, Plotly template, every chart builder
│   ├── state.py               cached data access shared by the tabs
│   └── tabs/                  one module per tab, each exposing render()
├── configs/default.yaml       the resolved default study configuration
├── data/
│   ├── raw/                   full local history (not version controlled)
│   ├── samples/               trimmed datasets shipped with the repository
│   └── processed/             intermediate artefacts
├── docs/                      methodology, architecture, findings
├── experiments/               one directory per run (not version controlled)
├── scripts/                   maintenance utilities
└── src/qmr/                   the library
    ├── config.py              typed configuration tree
    ├── paths.py               filesystem layout
    ├── logging_utils.py       logging, plus the callback sink the console uses
    ├── cli.py                 command-line entry point
    ├── data/                  catalogue, loader, MetaTrader 5 export
    ├── features/              indicator primitives and the feature pipeline
    ├── regimes/               detectors and regime characterisation
    ├── labeling/              triple-barrier and directional targets
    ├── models/                learner registry and rule-based benchmarks
    ├── validation/            purged walk-forward splitter
    ├── backtest/              execution engine and performance metrics
    ├── evaluation/            classification diagnostics and significance tests
    └── experiments/           orchestration and persistence
```

## Data flow

```
    configs/default.yaml
            |
        Config  (typed, validated at load; unknown keys are an error)
            |
    ┌───────┴────────────────────────────────────────────────┐
    │  run_experiment(config)                                 │
    │                                                          │
    │  load_ohlcv          -> canonical OHLCV frame            │
    │  build_features      -> ~80 causal features + 4 descriptors
    │  build_labels        -> triple-barrier targets           │
    │  walk_forward_splits -> embargoed folds                  │
    │                                                          │
    │  for each fold:                                          │
    │      build_detector().fit(train)      regimes, train only│
    │      build_model().fit(train_X, y)    learner, train only│
    │      predict_proba(test_X)            out-of-sample      │
    │      run_backtest(fold slice)         per-fold economics │
    │                                                          │
    │  concat OOS predictions -> run_backtest -> metrics       │
    │  classification diagnostics, regime breakdown            │
    │  significance_report, rule-based benchmarks              │
    └──────────────────────────┬───────────────────────────────┘
                               │
                     ExperimentResult
                               │
              save_experiment  ->  experiments/<run_id>/
                               │
              ┌────────────────┴────────────────┐
          qmr CLI                     Streamlit console
```

Both entry points call the same `run_experiment` and read the same artefacts.
Nothing the console can produce is unreachable from the command line, which is
what makes a result reproducible rather than a screenshot.

## Module contracts

| Module | Input | Output |
|---|---|---|
| `data.catalog` | directories | `DatasetInfo` records |
| `data.loader` | CSV path or symbol | DatetimeIndex-ed OHLCV frame |
| `features.pipeline` | OHLCV, `FeatureConfig` | features + regime descriptors |
| `labeling.targets` | features, `LabelConfig` | `LabelResult` |
| `regimes.detectors` | descriptor frame | integer regime series |
| `validation.walk_forward` | sample length | list of `Fold` |
| `models.zoo` | features, labels | `DirectionalModel` with `predict_proba` |
| `backtest.engine` | OHLCV, signals | `BacktestResult` |
| `evaluation.*` | predictions, returns | metric dictionaries |
| `experiments.runner` | `Config` | `ExperimentResult` |
| `experiments.store` | `ExperimentResult` | files on disk |

Every stage consumes and returns plain pandas objects, so any one of them can be
driven from a notebook without the rest.

## Extension points

**A new learner.** Add a `ModelSpec` to `MODEL_REGISTRY` and a branch in
`_tabular_estimator`. It then appears in the CLI, the console and the comparison
view with no further changes.

**A new regime detector.** Subclass `RegimeDetector`, implement `fit` and
`predict`, register it in `DETECTORS`.

**A new benchmark.** Add a function to `models.baselines` and an entry in
`BASELINES`. It is priced through the same execution model automatically.

**A new feature block.** Add a `_block` function in `features.pipeline`, append
its name to `FEATURE_BLOCKS`, and wire it into `build_features`. Removing a block
from the config is then a valid ablation.

**A new instrument.** Drop a CSV named
`SYMBOL_TIMEFRAME_YYYYMMDD_YYYYMMDD.csv` into `data/raw`. The catalogue picks it
up on the next scan.

## Experiment artefacts

Each run writes plain formats to `experiments/<run_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | the exact configuration that produced the run |
| `summary.json` | headline metrics, significance, benchmark comparison |
| `predictions.parquet` | per-bar out-of-sample record |
| `equity.parquet` | equity, drawdown, benchmark |
| `fold_metrics.csv` | per-fold economics |
| `fold_layout.csv` | walk-forward schedule with real timestamps |
| `trades.csv` | round trips |
| `regime_*.csv` | regime characterisation, performance, transitions |
| `feature_importance.csv` | importance averaged across folds |
| `threshold_curve.csv` | precision against coverage |
| `confusion.csv` | classification confusion matrix |

Fitted models are deliberately **not** persisted. A study is reproduced by
rerunning its configuration, not by unpickling an estimator whose library
version has since moved on.

## Design notes

**Configuration is typed, not a dictionary.** A misspelled key raises at load
time instead of silently changing the meaning of a study.

**Logging is a stream, not a return value.** `qmr.logging_utils.CallbackHandler`
lets the console tail a running study without the pipeline knowing a browser
exists.

**Charts live in one module.** `app/theme.py` owns the palette and every figure
builder, which is what keeps forty charts across seven tabs reading as one
instrument. No tab constructs a Plotly figure directly.
