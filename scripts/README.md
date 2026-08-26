# Scripts

Standalone utilities. None of them modify anything under `src/`.

| Script | What it does |
|---|---|
| `prepare_sample_data.py` | Cut the trimmed sample datasets in `data/samples` from the full local history in `data/raw`. |
| `run_improvement_study.py` | Search for a configuration that clears zero, logging every arm attempted to `reports/improvement_ablation.csv` so the trial count stays auditable. |
| `demo_lookahead_bias.py` | Run the same study three times with different amounts of look-ahead bias and price the damage. Teaching aid for [../docs/novice_learner.md](../docs/novice_learner.md). |

```bash
python scripts/prepare_sample_data.py --bars 15000
python scripts/demo_lookahead_bias.py --start 2020-01-01
python scripts/run_improvement_study.py --fresh
```
