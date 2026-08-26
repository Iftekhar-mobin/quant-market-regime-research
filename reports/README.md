# Reports

Result tables from the studies discussed in [../docs/findings.md](../docs/findings.md).
Every one is reproducible from the CLI or from `scripts/`.

| File | Study |
|---|---|
| `model_comparison.csv` | Four learners crossed with two regime settings, EURUSD H1 |
| `execution_ablation.csv` | One signal, five execution and labelling assumptions |
| `timeframe_comparison.csv` | The same pipeline at H1, H4 and D1 |
| `improvement_ablation.csv` | **The full search ledger** — every configuration tried, including the failures, plus the cross-instrument holdout |

`improvement_ablation.csv` is the important one. It exists so the number of
configurations tried is auditable rather than implied: 28 search arms and 10
holdout runs, each with its Sharpe, confidence interval, deflated Sharpe and the
exact settings that produced it. The headline result of the whole programme is
that the best of those 28 failed its holdout.

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

# improvement_ablation.csv — the search, then the holdout
python scripts/run_improvement_study.py --fresh
python scripts/run_improvement_study.py --holdout
```

Full per-run artefacts (predictions, equity curves, trades, fold layouts) go to
`experiments/<run_id>/` and are not version controlled.
