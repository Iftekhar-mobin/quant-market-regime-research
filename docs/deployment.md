# Deploying the research console

The console is a Streamlit app. It needs a host that will run Python.

## Where it is hosted, and why

**Streamlit Community Cloud** — free, runs this repository unchanged.

The obvious alternative, a Hugging Face Space, no longer works without paying:

| Hugging Face SDK | Status (checked August 2026) |
|---|---|
| `streamlit` | **Retired.** The API rejects it: *expected one of "gradio", "docker", "static"* |
| `docker` | Requires a **PRO subscription** — `402 Payment Required` on the free tier |
| `gradio` | Requires a **PRO subscription** — same 402 |
| `static` | Free, but serves no Python; no running models |

That was verified by creating a probe Space for each SDK against the live API,
not by reading documentation. **Rewriting the interface in Gradio would not have
avoided the charge**, which is the useful thing to know before spending a day on
it: on Hugging Face today, anything that executes Python needs a paid plan.

The Hugging Face path is kept in `deploy/huggingface/` regardless, switched to a
Docker Space, so it works the moment a PRO plan exists.

---

## Streamlit Community Cloud

### One-time setup

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Authorise Streamlit to read your repositories. Public-repo access is enough.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `Iftekhar-mobin/quant-market-regime-research` |
   | Branch | `main` |
   | Main file path | `app/main.py` |

5. Click **Deploy**. The first build takes a few minutes while it installs the
   dependencies.

### What the deployed app has

- The three sample datasets in `data/samples` (EURUSD, GBPUSD, GOLD H1, 15,000
  bars each), so every tab works immediately.
- One pre-computed study, `experiments/demo_eurusd_h1`, so the Results, Model
  comparison and Signals tabs are populated on a cold start rather than empty.
- Full interactivity: visitors can configure and run their own walk-forward
  study from the **Run a study** tab. Expect one to two minutes per study on the
  free tier.

### Configuration that matters

| File | Why it is there |
|---|---|
| `requirements.txt` | What Cloud installs. Keep it in step with `pyproject.toml`. |
| `runtime.txt` | Pins the interpreter (`python-3.11`). Cloud otherwise picks a newer version than some boosting wheels prefer. |
| `.streamlit/config.toml` | The light theme, so the hosted app matches local. |

### Resource limits

The free tier gives about 1 GB of memory. The shipped samples are sized for it.
Pointing a study at years of minute data would exhaust it — that is what the
local install is for.

### Redeploying

Push to `main`. Cloud rebuilds automatically. If a change alters dependencies,
use **Manage app → Reboot** to force a clean install.

---

## Hugging Face Space (needs PRO)

Already built and tested; only the subscription is missing.

```bash
huggingface-cli login          # write token, in your own terminal
python deploy/huggingface/deploy_space.py --repo <username>/quant-market-regime-research
```

The script stages a curated payload — library, console, configs, sample data and
the seed study, about 6 MB — and uploads it as a Docker Space. `--dry-run` builds
it without uploading so you can inspect what would go.

`deploy/huggingface/Dockerfile` installs `libgomp1` (LightGBM links against it and
the slim base image omits it), runs as uid 1000 so the pipeline can write its
output directories, and serves on port 7860 as Spaces expect.

---

## Running it locally

Nothing above is required to use the console:

```bash
pip install -e .
qmr console
```
