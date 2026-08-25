# Reports

Result tables from the studies discussed in [../docs/findings.md](../docs/findings.md).
Each is reproducible from the CLI; the commands that produced them are below.

| File | Study |
|---|---|
| `model_comparison.csv` | Four learners crossed with two regime settings, EURUSD H1 2018–2026 |
| `execution_ablation.csv` | One signal, five execution and labelling assumptions |
| `timeframe_comparison.csv` | The same pipeline at H1, H4 and D1 |

## Reproducing them

```bash
# model_comparison.csv
qmr compare --set data.start=2018-01-01 \
            --models logistic,random_forest,xgboost,lightgbm \
            --regimes none,kmeans \
            --output reports/model_comparison.csv

# execution_ablation.csv — one arm per row, e.g. the zero-cost arm
qmr run --set data.start=2018-01-01 --set model.name=random_forest \
        --set regime.method=none \
        --set backtest.cost_bps=0 --set backtest.slippage_bps=0

# timeframe_comparison.csv — one arm per timeframe, holding matched to the horizon
qmr run --set data.timeframe=D1 --set data.start=2005-01-01 \
        --set labeling.horizon=10 --set backtest.min_holding_bars=10 \
        --set validation.embargo_bars=20 --set model.name=random_forest \
        --set regime.method=none
```

Full per-run artefacts (predictions, equity curves, trades, fold layouts) are
written to `experiments/<run_id>/` and are not version controlled.
