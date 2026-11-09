"""
quantlab graphical interface (Streamlit): everything the CLI does, plus charts.

    python -m quantlab ui            (or: python -m streamlit run quantlab/ui/app.py)

Pages: Backtest, Optimization (walk-forward), Chart (VWAP + support / resistance).
The pages call the same library functions as the CLI; the labels are in French.
"""

from __future__ import annotations

import hashlib
import io
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # `streamlit run` puts only this file's folder on the path
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from quantlab.cli import format_value, parse_value  # noqa: E402
from quantlab.data import loading  # noqa: E402
from quantlab.execution.risk import RiskLimits  # noqa: E402
from quantlab.strategies.params import (  # noqa: E402
    GROUPS,
    Param,
    ParamError,
    defaults,
    load_preset,
    param_range,
    save_preset,
)
from quantlab.strategies.registry import STRATEGIES, build_strategy  # noqa: E402
from quantlab.ui import charts  # noqa: E402

logger = logging.getLogger("quantlab.ui")

SOURCE_LABELS = {
    "yfinance": "yfinance (2 min, ~60 jours)",
    "file": "Fichier (parquet / csv)",
}
STRATEGY_LABELS = {
    "trend": "Trend following",
    "mean_reversion": "Mean reversion",
    "arima": "ARIMA",
    "ml_hybrid": "ML Hybrid",
    "cascade": "Cascade",
    "pair": "Pair trading (2 symboles)",
}
OBJECTIVES = ("total_return_pct", "sharpe_ratio", "sortino_ratio", "profit_factor", "total_pnl_net", "win_rate")
#: versioned parameter presets, `<strategy>-<name>.json`
PRESETS_DIR = _ROOT / "presets"


# --- shared widgets ----------------------------------------------------------------------------------------


#: display names of the parameter groups
SECTION_LABELS = {
    "signal": "Signal",
    "indicateurs": "Indicateurs",
    "modèle": "Modèle",
    "cible": "Cible",
    "taille": "Taille",
    "sorties": "Sorties",
}
NUMERIC = ("int", "float")


def sheet(name: str) -> tuple[Param, ...]:
    return STRATEGIES[name].params


def group_label(group: str) -> str:
    section, _, model = group.partition(":")
    return SECTION_LABELS.get(section, section) + (f" : {model}" if model else "")


def grouped(params: Sequence[Param]) -> dict[str, list[Param]]:
    """Parameters by group, in the GROUPS order (declaration order inside a group)."""
    order = {section: i for i, section in enumerate(GROUPS)}
    groups: dict[str, list[Param]] = {}
    for param in sorted(params, key=lambda p: order.get(p.section, len(order))):
        groups.setdefault(param.group, []).append(param)
    return groups


def to_widget(param: Param, value: Any) -> Any:
    """A parameter value as its widget holds it (numbers: 0 for "none"; tuples as text)."""
    if param.kind == "int":
        return int(value or 0)
    if param.kind == "float":
        return float(value or 0.0)
    if param.kind == "tuple":
        return ", ".join(map(str, value or ()))
    return value


def from_widget(param: Param, raw: Any) -> Any:
    """The widget value as a parameter value (0 means "none" for an optional number); raises ParamError."""
    if param.optional and param.kind in NUMERIC and raw == 0:
        return None
    return param.convert(raw)


def param_widget(param: Param, key: str) -> Any:
    """The widget of one parameter; its value is kept in `st.session_state[key]` (presets write it there)."""
    if key not in st.session_state:
        st.session_state[key] = to_widget(param, param.default)
    label = param.name
    help_text = param.help
    if param.kind == "bool":
        return st.checkbox(label, key=key, help=help_text)
    if param.kind == "int":
        low = None if param.min is None else int(param.min)
        high = None if param.max is None else int(param.max)
        return st.number_input(label, min_value=low, max_value=high, step=1, key=key, help=help_text)
    if param.kind == "float":
        flow = None if param.min is None or (param.optional and param.min > 0) else float(param.min)
        fhigh = None if param.max is None else float(param.max)
        return st.number_input(label, min_value=flow, max_value=fhigh, format="%g", key=key, help=help_text)
    if param.kind == "choice":
        options = [*param.choices, *([None] if param.optional else [])]
        return st.selectbox(label, options, key=key, help=help_text, format_func=lambda v: "aucune" if v is None else v)
    return st.text_input(label, key=key, help=help_text + " Valeurs séparées par des virgules.")


def widget_key(key: str, name: str, param: Param) -> str:
    return f"{key}_{name}_{param.name}"


def set_widgets(key: str, name: str, values: Mapping[str, Any]) -> None:
    """Write parameter values into the widgets (before they are drawn: presets, reset)."""
    for param in sheet(name):
        st.session_state[widget_key(key, name, param)] = to_widget(param, values.get(param.name, param.default))


def preset_paths(name: str) -> list[Path]:
    return sorted(PRESETS_DIR.glob(f"{name}-*.json"))


def preset_controls(name: str, key: str) -> None:
    """Load a preset of `presets/` or the defaults into the widgets (drawn above them)."""
    paths = {p.stem: p for p in preset_paths(name)}
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    chosen = c1.selectbox("Preset", list(paths), index=None, placeholder="aucun", key=f"{key}_{name}_preset")
    if c2.button("Charger", key=f"{key}_{name}_load", disabled=chosen is None) and chosen is not None:
        try:
            _, values = load_preset(paths[chosen])
            set_widgets(key, name, STRATEGIES[name].validate(values))
            st.toast(f"Preset {chosen} chargé")
        except ParamError as exc:
            st.error(f"Preset {chosen} invalide : {exc}")
    if c3.button("Défauts", key=f"{key}_{name}_reset", help="Remettre toutes les valeurs par défaut"):
        set_widgets(key, name, {})


def param_inputs(name: str, key: str, presets: bool = True) -> dict[str, Any]:
    """
    Form of every parameter of a strategy, generated from its sheet (one tab per group); returns the values
    that differ from the defaults. Errors (bounds, rules linking several values) are shown under the form.
    """
    if presets:
        preset_controls(name, key)
    chosen: dict[str, Any] = {}
    errors: list[str] = []
    groups = grouped(sheet(name))
    for tab, params in zip(st.tabs([group_label(g) for g in groups]), groups.values(), strict=True):
        with tab:
            columns = st.columns(3)
            for i, param in enumerate(params):
                with columns[i % 3]:
                    raw = param_widget(param, widget_key(key, name, param))
                try:
                    value = from_widget(param, raw)
                except ParamError as exc:
                    errors.append(str(exc))
                    continue
                if value != param.default:
                    chosen[param.name] = value
    if STRATEGIES[name].factory.is_group:
        st.caption("Stop et objectif : sur le P&L combiné des deux jambes, de même notionnel, clôturées ensemble.")
    if chosen:
        st.caption("Différent des défauts : " + ", ".join(f"{k}={format_value(v)}" for k, v in chosen.items()))
    try:
        values = STRATEGIES[name].validate(chosen)
        full = {**defaults(sheet(name)), **values}
        for warning in STRATEGIES[name].factory.warnings(full):
            st.warning(warning)
    except ParamError as exc:
        errors.append(str(exc))
    for error in errors:
        st.error(error)
    if presets:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        label = c1.text_input("Enregistrer comme preset", key=f"{key}_{name}_save_as", placeholder="ex. stops-larges")
        slug = "-".join(label.strip().lower().split())
        if c2.button("Enregistrer", key=f"{key}_{name}_save", disabled=not slug or bool(errors)):
            path = PRESETS_DIR / f"{name}-{slug}.json"
            save_preset(path, name, chosen)
            st.toast(f"Preset enregistré : {path.name}")
    return chosen


def parse_extra(text: str) -> dict[str, Any]:
    """Free `key=value` lines (same syntax as `--param`)."""
    extra: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Paramètre invalide (attendu clé=valeur) : {line!r}")
        extra[key.strip()] = parse_value(value.strip())
    return extra


def grid_inputs(name: str) -> dict[str, list[Any]]:
    """Ranges (numbers) or value lists (booleans, choices) of the parameters to optimize."""
    tunable = [p for p in sheet(name) if p.kind in (*NUMERIC, "bool", "choice")]
    by_name = {p.name: p for p in tunable}
    selected = st.multiselect(
        "Paramètres à optimiser", list(by_name), key=f"wf_grid_{name}", help="Les autres gardent la valeur ci-dessus."
    )
    grid: dict[str, list[Any]] = {}
    for param_name in selected:
        param = by_name[param_name]
        key = f"wf_range_{name}_{param_name}"
        if param.kind in NUMERIC:
            c0, c1, c2, c3 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
            c0.markdown(f"`{param_name}` · {param.help}")
            default = float(param.default or (param.min or 0) or 1)
            start = c1.number_input("min", value=default, format="%g", key=f"{key}_min")
            stop = c2.number_input("max", value=default * 2, format="%g", key=f"{key}_max")
            step = c3.number_input("pas", value=1.0 if param.kind == "int" else default / 2 or 0.1, format="%g",
                                   key=f"{key}_step")  # fmt: skip
            try:
                grid[param_name] = param_range(param, start, stop, step)
            except ParamError as exc:
                st.error(str(exc))
        else:
            options = [True, False] if param.kind == "bool" else [*param.choices, *([None] if param.optional else [])]
            grid[param_name] = st.multiselect(param_name, options, default=options, key=key, help=param.help)
    if grid:
        st.caption(" · ".join(f"{k} : {', '.join(map(format_value, v))}" for k, v in grid.items()))
    return grid


def data_inputs(key: str, multi_symbol: bool = True, default_days: int = 55) -> dict[str, Any]:
    columns = st.columns([2, 2, 1])
    with columns[0]:
        symbols_text = st.text_input(
            "Symbole(s)",
            value="AAPL",
            key=f"{key}_symbols",
            help="Séparés par des virgules ; deux symboles pour le pair trading (ex. AAPL, MSFT).",
        )
    with columns[1]:
        source = st.selectbox(
            "Source", list(loading.SOURCES), format_func=SOURCE_LABELS.__getitem__, key=f"{key}_source"
        )
    with columns[2]:
        days = int(st.number_input("Jours d'historique", min_value=1, value=default_days, key=f"{key}_days"))
    file_path: str | None = None
    if source == "file":
        uploaded = st.file_uploader("Fichier de barres", type=["parquet", "csv"], key=f"{key}_upload")
        typed = st.text_input("…ou chemin du fichier", key=f"{key}_path", placeholder="tests/fixtures/aapl_1m.parquet")
        if uploaded is not None:
            file_path = save_upload(uploaded.getvalue(), uploaded.name)
        elif typed.strip():
            file_path = typed.strip()
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    if not multi_symbol:
        symbols = symbols[:1]
    return {"symbols": symbols, "source": source, "days": days, "file": file_path}


def save_upload(content: bytes, name: str) -> str:
    """Uploaded files are written once (by content hash) to a temporary folder."""
    digest = hashlib.sha256(content).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / "quantlab_uploads" / f"{digest}{Path(name).suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return str(path)


def execution_inputs(key: str) -> dict[str, Any]:
    """Capital, simulated costs and risk limits."""
    with st.expander("Exécution et risque", expanded=False):
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input("Capital ($)", min_value=1.0, value=100_000.0, step=10_000.0, key=f"{key}_capital")
        fee_bps = c2.number_input("Frais (bp par côté)", min_value=0.0, value=1.0, key=f"{key}_fee")
        slippage_bps = c3.number_input("Slippage simulé (bp)", min_value=0.0, value=1.0, key=f"{key}_slip")
        st.caption("Stop, objectif et durée de détention sont des paramètres de chaque stratégie (section Sorties).")
        c1, c2, c3 = st.columns(3)
        max_order = c1.number_input("Max actions par ordre (0 = illimité)", min_value=0, value=0, key=f"{key}_maxo")
        max_position = c2.number_input(
            "Max actions par symbole (0 = illimité)", min_value=0, value=0, key=f"{key}_maxp"
        )
        max_loss = c3.number_input("Coupe-circuit perte jour (%, 0 = aucun)", min_value=0.0, value=0.0, key=f"{key}_dl")
    return {
        "capital": float(capital),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "risk": RiskLimits(
            max_order_quantity=int(max_order) or None,
            max_position_quantity=int(max_position) or None,
            max_daily_loss_pct=max_loss / 100 or None,
        ),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def cached_bars(symbols: tuple[str, ...], source: str, days: int, file: str | None) -> pd.DataFrame:
    return loading.load_bars(list(symbols), source, days, file)


def get_bars(data: dict[str, Any]) -> pd.DataFrame:
    if not data["symbols"]:
        raise ValueError("Indiquez au moins un symbole.")
    with st.spinner(f"Chargement des données {data['source']}…"):
        return cached_bars(tuple(data["symbols"]), data["source"], data["days"], data["file"])


def metric_tiles(metrics: dict[str, float]) -> None:
    tiles = [
        ("Rendement", f"{metrics['total_return_pct']:+.3f} %"),
        ("P&L net", f"{metrics['total_pnl_net']:+,.2f} $"),
        ("Trades", f"{metrics['total_trades']:.0f}"),
        ("Réussite", f"{metrics['win_rate']:.1f} %"),
        ("Profit factor", f"{metrics['profit_factor']:.2f}"),
        ("Sharpe", f"{metrics['sharpe_ratio']:.2f}"),
        ("Drawdown max", f"{metrics['max_drawdown_pct']:.3f} %"),
    ]
    show_tiles(tiles)


def show_tiles(tiles: list[tuple[str, str]], per_row: int = 4) -> None:
    """Metric tiles, at most `per_row` per row so the values are never truncated."""
    for start in range(0, len(tiles), per_row):
        row = tiles[start : start + per_row]
        for column, (label, value) in zip(st.columns(per_row), row, strict=False):
            column.metric(label, value)


def show_error(exc: Exception) -> None:
    logger.exception("UI action failed")
    st.error(f"{type(exc).__name__} : {exc}")


# --- pages -------------------------------------------------------------------------------------------------


def page_backtest() -> None:
    st.header("Backtest")
    st.caption(
        "Entraîne les stratégies sur le début des données puis les fait trader barre par barre sur la fin, "
        "avec le même moteur que le live (exécution à l'open suivant, frais, slippage, stop / objectif par position)."
    )
    names = st.multiselect(
        "Stratégie(s)",
        list(STRATEGIES),
        default=["trend"],
        format_func=STRATEGY_LABELS.__getitem__,
        help="Plusieurs stratégies = un portefeuille.",
    )
    data = data_inputs("bt")
    c1, c2, c3 = st.columns(3)
    split_mode = c1.radio("Bloc de test", ["Derniers jours", "Dernière fraction"], key="bt_split", horizontal=True)
    if split_mode == "Derniers jours":
        test_days, test_fraction = c2.number_input("Jours de test", min_value=0.1, value=5.0), None
    else:
        test_days, test_fraction = 5.0, c2.slider("Fraction de test", 0.05, 0.9, 0.2)
    each = c3.checkbox("Aussi chaque stratégie seule", value=True, disabled=len(names) < 2)
    bar_by_bar = c3.checkbox("Chemin live barre à barre (plus lent)")
    params: dict[str, dict[str, Any]] = {}
    for name in names:
        with st.expander(f"Paramètres : {STRATEGY_LABELS[name]}"):
            params[name] = param_inputs(name, "bt")
    execution = execution_inputs("bt")

    if st.button("Lancer le backtest", type="primary", disabled=not names):
        try:
            st.session_state["backtest"] = run_backtests(names, params, data, test_days, test_fraction, each,
                                                         bar_by_bar, execution)  # fmt: skip
        except Exception as exc:
            show_error(exc)
    results = st.session_state.get("backtest")
    if results:
        show_backtest(results)


def run_backtests(
    names: list[str],
    params: dict[str, dict[str, Any]],
    data: dict[str, Any],
    test_days: float,
    test_fraction: float | None,
    each: bool,
    bar_by_bar: bool,
    execution: dict[str, Any],
) -> dict[str, Any]:
    from quantlab.backtest.run import backtest

    bars = get_bars(data)
    train, test = loading.split(bars, test_days, test_fraction)
    if train.empty or test.empty:
        raise ValueError(f"Découpage vide : {len(train)} barres d'entraînement, {len(test)} de test.")
    runs = [[n] for n in names] if each and len(names) > 1 else []
    runs.append(list(names))
    results = {}
    progress = st.progress(0.0, text="Backtest…")
    for i, run_names in enumerate(runs):
        label = " + ".join(STRATEGY_LABELS[n].split(" (")[0] for n in run_names)
        progress.progress(i / len(runs), text=f"Backtest : {label}")
        strategies = [build_strategy(n, data["symbols"], params[n]) for n in run_names]
        results[label] = backtest(
            strategies,
            train,
            test,
            capital=execution["capital"],
            fee_bps=execution["fee_bps"],
            slippage_bps=execution["slippage_bps"],
            risk_limits=execution["risk"],
            precompute_features=not bar_by_bar,
        )
    progress.empty()
    return {"results": results, "train": (train.index[0], train.index[-1], len(train)),
            "test": (test.index[0], test.index[-1], len(test))}  # fmt: skip


def show_backtest(state: dict[str, Any]) -> None:
    results = state["results"]
    t0, t1, n_train = state["train"]
    s0, s1, n_test = state["test"]
    st.caption(f"Entraînement : {t0} → {t1} ({n_train} barres) · Test : {s0} → {s1} ({n_test} barres)")
    labels = list(results)
    main = labels[-1]
    st.subheader(main)
    metric_tiles(results[main].metrics)
    st.altair_chart(charts.equity_chart({label: results[label].equity for label in labels}), width="stretch")
    st.altair_chart(charts.drawdown_chart(results[main].equity), width="stretch")
    if len(labels) > 1:
        st.dataframe(pd.DataFrame({k: v.metrics for k, v in results.items()}).T, width="stretch")

    trades = results[main].trades_frame()
    st.subheader(f"Trades ({len(trades)})")
    if trades.empty:
        st.info("Aucun trade.")
        return
    st.altair_chart(charts.trade_pnl_chart(trades), width="stretch")
    st.dataframe(trades, width="stretch", height=260, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Trades (CSV)", trades.to_csv(index=False), "trades.csv", "text/csv")
    c2.download_button("Courbe de capital (CSV)", results[main].equity.to_csv(), "equity.csv", "text/csv")


def page_optimize() -> None:
    st.header("Optimisation walk-forward")
    st.caption(
        "Chaque fold : [entraînement][embargo][validation][embargo][test]. La meilleure combinaison sur la "
        "validation est ré-entraînée puis évaluée une seule fois sur le test."
    )
    name = st.selectbox("Stratégie", list(STRATEGIES), format_func=STRATEGY_LABELS.__getitem__, key="wf_strategy")
    data = data_inputs("wf")
    with st.expander("Paramètres fixes", expanded=False):
        fixed = param_inputs(name, "wf")
    st.markdown("**Grille** : plage min / max / pas pour les nombres, liste de valeurs sinon.")
    grid = grid_inputs(name)
    c1, c2, c3, c4 = st.columns(4)
    train_size = int(c1.number_input("Barres d'entraînement", min_value=10, value=3000, step=100))
    validation_size = int(c2.number_input("Barres de validation", min_value=10, value=1000, step=100))
    test_size = int(c3.number_input("Barres de test", min_value=10, value=1000, step=100))
    embargo = int(c4.number_input("Embargo (barres)", min_value=0, value=30))
    c1, c2, c3, c4 = st.columns(4)
    step = int(c1.number_input("Pas entre folds (0 = taille du test)", min_value=0, value=0))
    objective = c2.selectbox("Objectif", OBJECTIVES)
    min_trades = int(c3.number_input("Trades minimum", min_value=0, value=5))
    anchored = c4.checkbox("Entraînement ancré (fenêtre croissante)")

    if st.button("Lancer l'optimisation", type="primary", disabled=not grid):
        try:
            from quantlab.backtest.walk_forward import WalkForwardOptimizer

            bars = get_bars(data)
            optimizer = WalkForwardOptimizer(
                strategy_factory=lambda **p: build_strategy(name, data["symbols"], {**fixed, **p}),
                param_grid=grid,
                train_size=train_size,
                validation_size=validation_size,
                test_size=test_size,
                step=step or None,
                embargo=embargo,
                anchored=anchored,
                objective=objective,
                min_trades=min_trades,
            )
            n = len(optimizer.candidates)
            with st.spinner(f"{n} combinaisons par fold…"):
                st.session_state["optimize"] = optimizer.run(bars)
        except Exception as exc:
            show_error(exc)
    if not grid:
        st.info("Renseignez au moins un paramètre de la grille.")
    result = st.session_state.get("optimize")
    if result is not None:
        summary = result.summary()
        st.metric(f"Score moyen hors échantillon ({result.objective})", f"{result.mean_test_score:,.4f}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Score de test par fold**")
            st.altair_chart(charts.value_bars(summary["test_score"], result.objective, "Fold"), width="stretch")
        with c2:
            st.markdown("**Stabilité des paramètres choisis** (nombre de folds)")
            stability = result.param_stability()
            st.altair_chart(charts.value_bars(stability, "Folds", "Paramètres", fmt="d"), width="stretch")
        st.dataframe(summary, width="stretch")


def page_chart() -> None:
    from quantlab.analysis.plot import levels_figure, session_bars

    st.header("Graphique de séance")
    st.caption(
        "Bougies, volume, VWAP et ses bandes, et les meilleures lignes de support / résistance. Ces lignes "
        "utilisent des barres futures : elles servent à l'analyse visuelle, jamais de signal."
    )
    data = data_inputs("chart", multi_symbol=False, default_days=3)
    c1, c2 = st.columns(2)
    day = c1.date_input("Séance (vide = la dernière)", value=None)
    n_lines = int(c2.number_input("Lignes de support / résistance", min_value=0, max_value=10, value=3))
    if st.button("Tracer", type="primary"):
        try:
            bars, chosen = session_bars(get_bars(data), day.isoformat() if day else None)
            with st.spinner("Calcul des niveaux…"):
                figure = levels_figure(bars, f"{data['symbols'][0]} {chosen} (New York time)", n_lines)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            st.session_state["chart"] = (figure, buffer.getvalue(), f"{data['symbols'][0]}_{chosen}.png")
        except Exception as exc:
            show_error(exc)
    if "chart" in st.session_state:
        figure, png, filename = st.session_state["chart"]
        st.pyplot(figure)
        st.download_button("Télécharger (PNG)", png, filename, "image/png")


def main() -> None:
    st.set_page_config(page_title="quantlab", page_icon=":chart_with_upwards_trend:", layout="wide")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    navigation = st.navigation(
        [
            st.Page(page_backtest, title="Backtest", icon=":material/query_stats:", default=True),
            st.Page(page_optimize, title="Optimisation", icon=":material/tune:", url_path="optimisation"),
            st.Page(page_chart, title="Graphique", icon=":material/candlestick_chart:", url_path="graphique"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
