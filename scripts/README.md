# Scripts

Standalone utilities. None of them modify anything under `src/`.

| Script | What it does |
|---|---|
| `prepare_sample_data.py` | Cut the trimmed sample datasets in `data/samples` from the full local history in `data/raw`. |
| `demo_lookahead_bias.py` | Run the same study three times with different amounts of look-ahead bias and price the damage. Teaching aid for [../docs/novice_learner.md](../docs/novice_learner.md). |

```bash
python scripts/prepare_sample_data.py --bars 15000
python scripts/demo_lookahead_bias.py --start 2020-01-01
```
