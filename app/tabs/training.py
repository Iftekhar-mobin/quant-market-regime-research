"""Run a study: configure, launch and watch a walk-forward experiment."""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st
import theme


def _model_options() -> tuple[list[str], dict[str, str]]:
    from qmr.models.zoo import MODEL_REGISTRY, is_available

    labels: dict[str, str] = {}
    keys: list[str] = []
    for key, spec in MODEL_REGISTRY.items():
        available = is_available(key)
        labels[key] = spec.label if available else f"{spec.label}  (needs {spec.requires})"
        if available:
            keys.append(key)
        else:
            keys.append(key)  # still listed, so the missing dependency is visible
    return keys, labels


def render(selection: dict, config) -> None:
    from qmr.experiments import run_experiment, save_experiment
    from qmr.logging_utils import CallbackHandler
    from qmr.models.zoo import MODEL_REGISTRY, is_available

    st.markdown(
        theme.lede(
            "Configure one study and run it. The model, the feature scaler and the "
            "regime detector are refitted inside every fold on the training window "
            "alone, with an embargo separating it from the test window, so every "
            "number the study reports is out of sample by construction."
        ),
        unsafe_allow_html=True,
    )

    model_keys, model_labels = _model_options()

    left, middle, right = st.columns(3, gap="large")

    with left:
        st.markdown("##### Model")
        model_name = st.selectbox(
            "Learner",
            model_keys,
            index=model_keys.index(config.model.name) if config.model.name in model_keys else 0,
            format_func=lambda k: model_labels[k],
            key="tr_model",
        )
        st.caption(MODEL_REGISTRY[model_name].description)

        decision_threshold = st.slider(
            "Decision threshold",
            0.34,
            0.80,
            float(config.model.decision_threshold),
            0.01,
            key="tr_threshold",
            help="Minimum class probability before a position is taken. The main "
            "risk dial: higher means fewer, higher-conviction positions.",
        )
        class_balance = st.checkbox(
            "Balance the training classes",
            value=config.model.class_balance,
            key="tr_balance",
            help="Without this the model learns to predict the majority class and "
            "reports a respectable accuracy for taking no positions at all.",
        )

    with middle:
        st.markdown("##### Regimes")
        regime_method = st.selectbox(
            "Detector",
            ["none", "rule", "kmeans", "gmm"],
            index=["none", "rule", "kmeans", "gmm"].index(config.regime.method),
            key="tr_regime",
            help="`none` is the control arm — run it first, then compare.",
        )
        n_regimes = st.slider(
            "Number of regimes",
            2,
            8,
            int(config.regime.n_regimes),
            disabled=regime_method in {"none", "rule"},
            key="tr_nregimes",
        )
        as_feature = st.checkbox(
            "Give the regime to the model as a feature",
            value=config.regime.as_model_feature,
            disabled=regime_method == "none",
            key="tr_asfeature",
        )
        specialised = st.checkbox(
            "Fit one model per regime",
            value=config.regime.specialised_models,
            disabled=regime_method == "none",
            key="tr_specialised",
            help="The strong form of the hypothesis: each state learns its own "
            "relationship. Costs training data — regimes with too few bars fall "
            "back to the pooled model.",
        )

    with right:
        st.markdown("##### Labels and validation")
        label_method = st.selectbox(
            "Labelling",
            ["triple_barrier", "directional"],
            index=0 if config.labeling.method == "triple_barrier" else 1,
            key="tr_label",
        )
        horizon = st.number_input(
            "Horizon (bars)", 4, 200, int(config.labeling.horizon), 4, key="tr_horizon"
        )
        n_folds = st.slider("Walk-forward folds", 2, 12, int(config.validation.n_folds), key="tr_folds")
        scheme = st.selectbox(
            "Scheme",
            ["expanding", "rolling"],
            index=0 if config.validation.scheme == "expanding" else 1,
            key="tr_scheme",
            help="Expanding trains on everything so far, as a production model "
            "would. Rolling uses a fixed window, which probes whether the edge "
            "survives without ancient history.",
        )

    with st.expander("Execution assumptions", expanded=False):
        cost_columns = st.columns(3)
        with cost_columns[0]:
            cost_bps = st.number_input(
                "Round-trip cost (bps)", 0.0, 20.0, float(config.backtest.cost_bps), 0.5,
                key="tr_cost",
            )
        with cost_columns[1]:
            slippage_bps = st.number_input(
                "Slippage (bps)", 0.0, 20.0, float(config.backtest.slippage_bps), 0.5,
                key="tr_slip",
            )
        with cost_columns[2]:
            min_holding = st.number_input(
                "Minimum holding (bars)", 0, 200, int(config.backtest.min_holding_bars), 1,
                key="tr_minhold",
                help="Set this to the label horizon. A horizon-ahead forecast is "
                "one opinion, not one per bar.",
            )
        st.caption(
            "Signals are executed at the open of the following bar, never the "
            "close that produced them. Costs are charged on every change in "
            "position, so a flip from long to short pays twice."
        )

    overrides = {
        "data.symbol": selection["symbol"],
        "data.timeframe": selection["timeframe"],
        "data.start": selection["start"],
        "data.end": selection["end"],
        "model.name": model_name,
        "model.decision_threshold": float(decision_threshold),
        "model.class_balance": bool(class_balance),
        "regime.method": regime_method,
        "regime.n_regimes": int(n_regimes),
        "regime.as_model_feature": bool(as_feature),
        "regime.specialised_models": bool(specialised),
        "labeling.method": label_method,
        "labeling.horizon": int(horizon),
        "validation.n_folds": int(n_folds),
        "validation.scheme": scheme,
        "validation.embargo_bars": int(horizon) * 2,
        "backtest.cost_bps": float(cost_bps),
        "backtest.slippage_bps": float(slippage_bps),
        "backtest.min_holding_bars": int(min_holding),
    }

    study_config = config.with_overrides(overrides)

    st.divider()
    launch, preview = st.columns([1, 3])
    with launch:
        start_run = st.button("Run study", type="primary", use_container_width=True, key="tr_run")
    with preview:
        st.caption(
            f"{selection['symbol']} {selection['timeframe']} · {model_labels[model_name]} · "
            f"{regime_method} regimes · {n_folds} folds · "
            f"embargo {int(horizon) * 2} bars"
        )

    if not is_available(model_name):
        spec = MODEL_REGISTRY[model_name]
        st.error(
            f"{spec.label} needs the optional dependency `{spec.requires}`. "
            f"Install it with `pip install {spec.requires}`, or choose another learner."
        )
        return

    with st.expander("Resolved configuration", expanded=False):
        st.code(study_config.to_yaml(), language="yaml")
        st.caption(
            "This exact file is written next to the run, so the study can be "
            "reproduced headlessly with `qmr run -c <that file>`."
        )

    if not start_run:
        return

    # -- run ---------------------------------------------------------------
    progress_bar = st.progress(0.0, text="Starting...")
    log_area = st.empty()
    lines: list[str] = []

    def on_progress(fraction: float, message: str) -> None:
        progress_bar.progress(fraction, text=message)

    def on_log(line: str) -> None:
        lines.append(line)
        log_area.code("\n".join(lines[-18:]), language="text")

    handler = CallbackHandler(on_log)
    root_logger = logging.getLogger("qmr")
    root_logger.addHandler(handler)

    started = datetime.now()
    try:
        with st.spinner("Running the walk-forward study..."):
            result = run_experiment(study_config, progress=on_progress)
        directory = save_experiment(result)
    except Exception as exc:
        progress_bar.empty()
        st.error(f"The study failed: {exc}")
        st.exception(exc)
        return
    finally:
        root_logger.removeHandler(handler)

    progress_bar.empty()
    st.success(
        f"Finished in {(datetime.now() - started).total_seconds():.1f}s — "
        f"saved as `{result.run_id}`"
    )
    st.session_state["last_run_id"] = result.run_id

    headline = result.headline()
    st.markdown(
        theme.stat_tiles(
            [
                (
                    "Sharpe ratio",
                    f"{headline['sharpe']:+.2f}",
                    "annualised, out of sample, after costs",
                ),
                ("CAGR", theme.format_metric("cagr", headline["cagr"]), "compound annual growth"),
                (
                    "Maximum drawdown",
                    theme.format_metric("max_drawdown", headline["max_drawdown"]),
                    "deepest peak-to-trough loss",
                ),
                (
                    "Directional precision",
                    theme.format_metric(
                        "directional_precision", headline["directional_precision"]
                    ),
                    "correct calls, on bars traded",
                ),
                ("Trades", f"{result.metrics['trades']:,.0f}", "round trips"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        theme.equity_chart(
            result.backtest.equity,
            result.backtest.benchmark_equity,
            title="Out-of-sample equity against buy-and-hold",
        ),
        use_container_width=True,
    )

    st.info(
        "The full report — per-fold results, regime breakdown, feature "
        "importance and significance tests — is on the **Results** tab.",
    )
    st.caption(f"Artefacts written to `{directory}`")
