"""MomentumAlpha research pipeline with executable timing and walk-forward evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import uuid
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from backtest import (
    bootstrap_mean_difference_ci,
    bootstrap_sharpe_ci,
    compute_benchmark_returns,
    cost_adjusted_portfolio_returns,
    compute_information_coefficient,
    compute_metrics,
    get_model_weights,
    monte_carlo_test,
    walk_forward_backtest,
)
from data_loader import load_market_data
from dolt_data_loader import load_dolt_market_data
from experiment_ledger import ExperimentLedger
from evidence import assess_run_evidence
from factor_attribution import analyze_portfolio_against_factors
from factor_data import load_french_daily_factors
from features import (
    compute_all_features,
    compute_lagged_target_features,
    prepare_weekly_dataset,
    compute_weekly_cross_sectional_features,
)
from modeling import build_classifier_models
from market_data_quality import sanitize_benchmark_ohlc
from research import (
    build_market_data_quality_report,
    complete_evaluation_weeks,
    fingerprint_market_data,
    validate_weekly_panel,
)
from run_validation import sha256_file, validate_run_directory
from visualizations import (
    plot_annual_returns_comparison,
    plot_cumulative_alpha,
    plot_cumulative_returns,
    plot_drawdown,
    plot_eda_prices,
    plot_feature_importance,
    plot_ic_over_time,
    plot_monthly_returns_heatmap,
    plot_return_correlation,
    plot_returns_distribution,
    plot_rolling_sharpe,
    plot_stock_selection_frequency,
    plot_stock_selection_heatmap,
)


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "JNJ", "BRK-B"]
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_START_DATE = "2012-01-01"
DEFAULT_INITIAL_TRAIN_END = "2022-12-31"
RISK_FREE_RATE = 0.04
RANDOM_STATE = 42


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def _classification_metrics(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    if predictions.empty:
        return {"rows": 0}
    y_true = predictions["target"].astype(int)
    probabilities = predictions["ensemble_prob"].astype(float).clip(1e-7, 1 - 1e-7)
    y_pred = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | int | None] = {
        "rows": int(len(predictions)),
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "roc_auc": None,
        "average_precision": None,
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
    if "ensemble_prob_raw" in predictions:
        raw_probability = predictions["ensemble_prob_raw"].astype(float).clip(1e-7, 1 - 1e-7)
        metrics["raw_ensemble_brier_score"] = float(brier_score_loss(y_true, raw_probability))
        metrics["raw_ensemble_log_loss"] = float(log_loss(y_true, raw_probability, labels=[0, 1]))
        metrics["raw_ensemble_expected_calibration_error_10_bins"] = _expected_calibration_error(
            y_true, raw_probability
        )
    metrics["expected_calibration_error_10_bins"] = _expected_calibration_error(y_true, probabilities)
    return metrics


def _expected_calibration_error(y_true: pd.Series, probabilities: pd.Series) -> float:
    bins = pd.cut(probabilities, bins=np.linspace(0, 1, 11), include_lowest=True)
    grouped = pd.DataFrame({"target": y_true.to_numpy(), "probability": probabilities.to_numpy(), "bin": bins}).groupby(
        "bin", observed=False
    )
    counts = grouped.size()
    weights = counts / max(1, int(counts.sum()))
    differences = (grouped["target"].mean() - grouped["probability"].mean()).abs().fillna(0.0)
    return float((weights * differences).sum())


def _calibration_curve_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column in (
        ("raw_ensemble", "ensemble_prob_raw"),
        ("calibrated_ensemble", "ensemble_prob"),
    ):
        if column not in predictions:
            continue
        values = predictions[["target", column]].dropna().copy()
        if values.empty:
            continue
        values["probability_bin"] = pd.cut(
            values[column].astype(float).clip(0, 1),
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
        )
        grouped = values.groupby("probability_bin", observed=False)
        for probability_bin, rows_in_bin in grouped:
            if rows_in_bin.empty:
                continue
            rows.append({
                "calibration_type": label,
                "probability_bin": str(probability_bin),
                "observations": int(len(rows_in_bin)),
                "mean_predicted_probability": float(rows_in_bin[column].mean()),
                "observed_positive_rate": float(rows_in_bin["target"].mean()),
            })
    return pd.DataFrame(rows)


def _phase_performance(
    strategy_gross: pd.Series,
    strategy_net: pd.Series,
    benchmarks: dict[str, pd.Series],
    phase_by_week: pd.Series,
    predictions: pd.DataFrame,
    information_coefficient: pd.DataFrame,
    *,
    bootstrap_samples: int,
    risk_free_rate: float,
    random_state: int,
) -> dict[str, dict]:
    """Summarize development and final-holdout results independently."""
    output: dict[str, dict] = {}
    for phase in ("development", "final_holdout"):
        phase_index = phase_by_week.index[phase_by_week.eq(phase)]
        net = strategy_net.reindex(phase_index).dropna()
        gross = strategy_gross.reindex(net.index).dropna()
        if net.empty:
            output[phase] = {"weeks": 0, "status": "No completed observations in this phase."}
            continue
        benchmark_summary = {
            name: compute_metrics(series.reindex(net.index).dropna(), risk_free_rate)
            for name, series in benchmarks.items()
            if not name.endswith("_gross")
        }
        phase_predictions = predictions.loc[predictions["evaluation_phase"].eq(phase)]
        phase_ic = information_coefficient.loc[
            information_coefficient.index.isin(net.index)
        ] if not information_coefficient.empty else information_coefficient
        entry = {
            "weeks": int(len(net)),
            "period_start": net.index.min(),
            "period_end": net.index.max(),
            "strategy_before_costs": compute_metrics(gross, risk_free_rate),
            "strategy_after_costs": compute_metrics(net, risk_free_rate),
            "benchmarks_after_costs": benchmark_summary,
            "classification": _classification_metrics(phase_predictions),
            "information_coefficient_mean": (
                float(phase_ic["ic"].mean()) if not phase_ic.empty else None
            ),
            "sharpe_95pct_block_bootstrap_ci": (
                list(bootstrap_sharpe_ci(
                    net,
                    n_bootstrap=bootstrap_samples,
                    risk_free_rate=risk_free_rate,
                    random_state=random_state,
                )) if len(net) >= 2 else None
            ),
            "evidence_note": (
                "A short period has high uncertainty; at least 52 weekly observations are preferred."
                if len(net) < 52 else None
            ),
        }
        excess_intervals = {}
        for name, series in benchmarks.items():
            if name.endswith("_gross"):
                continue
            paired = pd.concat({"strategy": net, "benchmark": series.reindex(net.index)}, axis=1).dropna()
            if len(paired) >= 2:
                excess_intervals[name] = list(bootstrap_mean_difference_ci(
                    paired["strategy"],
                    paired["benchmark"],
                    n_bootstrap=bootstrap_samples,
                    random_state=random_state,
                ))
        entry["annualized_excess_return_95pct_ci"] = excess_intervals
        output[phase] = entry
    return output


def _calendar_year_performance(
    strategy: pd.Series,
    benchmarks: dict[str, pd.Series],
    risk_free_rate: float,
) -> pd.DataFrame:
    rows = []
    for year in sorted(set(strategy.index.year)):
        year_index = strategy.index[strategy.index.year == year]
        series_map = {"MomentumAlpha_net": strategy.reindex(year_index)}
        series_map.update({name: values.reindex(year_index) for name, values in benchmarks.items() if not name.endswith("_gross")})
        for name, values in series_map.items():
            metrics = compute_metrics(values.dropna(), risk_free_rate)
            rows.append({"calendar_year": int(year), "portfolio": name, **metrics})
    return pd.DataFrame(rows)


def _cost_stress_performance(
    *,
    portfolio: pd.DataFrame,
    weekly_data: pd.DataFrame,
    tickers: list[str],
    spy_data: pd.DataFrame,
    phase_by_week: pd.Series,
    base_cost_bps: float,
    top_n: int,
    risk_free_rate: float,
    spy_dividends: pd.Series | None = None,
) -> pd.DataFrame:
    raw_scenarios = [
        ("frictionless", 0.0),
        ("base_cost", float(base_cost_bps)),
        ("2x_base_cost", float(base_cost_bps) * 2),
        ("5x_base_cost", float(base_cost_bps) * 5),
    ]
    scenarios: list[tuple[str, float]] = []
    seen_costs: set[float] = set()
    for name, cost_bps in raw_scenarios:
        if cost_bps not in seen_costs:
            scenarios.append((name, cost_bps))
            seen_costs.add(cost_bps)

    rows: list[dict] = []
    evaluation_weeks = pd.DatetimeIndex(phase_by_week.index)
    for scenario, cost_bps in scenarios:
        strategy = cost_adjusted_portfolio_returns(portfolio, cost_bps)
        baselines = compute_benchmark_returns(
            weekly_data,
            tickers,
            spy_data,
            top_n=top_n,
            transaction_cost_bps=cost_bps,
            evaluation_weeks=evaluation_weeks,
            spy_dividends=spy_dividends,
        )
        for phase in ("development", "final_holdout"):
            phase_weeks = phase_by_week.index[phase_by_week.eq(phase)]
            portfolio_series = {"MomentumAlpha": strategy}
            portfolio_series.update({
                name: series for name, series in baselines.items() if not name.endswith("_gross")
            })
            for portfolio_name, series in portfolio_series.items():
                matched = series.reindex(phase_weeks).dropna()
                rows.append({
                    "cost_scenario": scenario,
                    "transaction_cost_bps": cost_bps,
                    "evaluation_phase": phase,
                    "portfolio": portfolio_name,
                    "weekly_observations": int(len(matched)),
                    **compute_metrics(matched, risk_free_rate),
                })
    return pd.DataFrame(rows)


def _installed_versions() -> dict[str, str | None]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "ta": "ta",
        "yfinance": "yfinance",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "duckdb": "duckdb",
    }
    versions = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _backend_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in (
        "api.py", "main.py", "MomentumAlpha.py", "backtest.py", "data_loader.py", "dolt_data_loader.py", "features.py",
        "experiment_ledger.py", "evidence.py", "factor_attribution.py", "factor_data.py",
        "market_data_quality.py", "modeling.py", "research.py", "run_validation.py",
        "visualizations.py", "requirements.txt",
    ):
        path = PROJECT_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _aggregate_importance(model_importances: dict, model_name: str = "LightGBM") -> dict[str, float]:
    observations: dict[str, list[float]] = {}
    for by_model in model_importances.values():
        for feature, value in by_model.get(model_name, {}).items():
            observations.setdefault(feature, []).append(float(value))
    return {name: float(np.mean(values)) for name, values in observations.items() if values}


def _feature_importance_stability(model_importances: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fit_date, by_model in model_importances.items():
        for model_name, importance_map in by_model.items():
            for feature, value in importance_map.items():
                rows.append({
                    "fit_date": fit_date,
                    "model": model_name,
                    "feature": feature,
                    "importance": float(value),
                })
    by_fit = pd.DataFrame(rows)
    if by_fit.empty:
        return by_fit, by_fit.copy()
    by_fit["importance_rank"] = by_fit.groupby(["fit_date", "model"])["importance"].rank(
        method="average", ascending=False
    )
    summary = by_fit.groupby(["model", "feature"], as_index=False).agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        mean_rank=("importance_rank", "mean"),
        folds_observed=("fit_date", "nunique"),
        nonzero_fold_fraction=("importance", lambda values: float((values > 0).mean())),
    )
    return by_fit, summary.sort_values(["model", "mean_rank", "feature"])


def _plot_outputs(
    *,
    stock_data: dict[str, pd.DataFrame],
    tickers: list[str],
    results: dict,
    benchmarks: dict[str, pd.Series],
    ic: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    benchmark_series = {
        name: benchmarks[name]
        for name in ("equal_weight", "spy", "raw_momentum")
        if name in benchmarks
    }
    plot_eda_prices(stock_data, tickers, save_path=str(plot_dir / "eda_prices.png"))
    plot_return_correlation(stock_data, tickers, save_path=str(plot_dir / "eda_correlation.png"))
    plot_cumulative_returns(
        results["returns_before_costs"],
        results["returns_after_costs"],
        benchmark_series,
        save_path=str(plot_dir / "cumulative_returns.png"),
    )
    plot_drawdown(results["returns_after_costs"], title="Strategy (After Costs)", save_path=str(plot_dir / "drawdown.png"))
    plot_returns_distribution(results["returns_after_costs"], save_path=str(plot_dir / "returns_dist.png"))
    plot_rolling_sharpe(results["returns_after_costs"], save_path=str(plot_dir / "rolling_sharpe.png"))

    importance = _aggregate_importance(results["model_importances"])
    if importance:
        plot_feature_importance(importance, top_n=25, save_path=str(plot_dir / "feature_importance.png"))
    plot_stock_selection_heatmap(results["selected_stocks"], tickers, save_path=str(plot_dir / "selection_heatmap.png"))
    plot_monthly_returns_heatmap(results["returns_after_costs"], save_path=str(plot_dir / "monthly_heatmap.png"))
    if not ic.empty:
        plot_ic_over_time(ic, save_path=str(plot_dir / "ic_over_time.png"))
    if "equal_weight" in benchmark_series:
        plot_cumulative_alpha(
            results["returns_after_costs"],
            benchmark_series["equal_weight"],
            save_path=str(plot_dir / "cumulative_alpha.png"),
        )
    plot_annual_returns_comparison(
        results["returns_after_costs"], benchmark_series, save_path=str(plot_dir / "annual_returns.png")
    )
    plot_stock_selection_frequency(results["selected_stocks"], tickers, save_path=str(plot_dir / "selection_frequency.png"))


def _save_shap_plot(
    results: dict,
    weekly_data: pd.DataFrame,
    plot_dir: Path,
) -> str | None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    model = results["last_models"].get("LightGBM")
    feature_names = results["last_feature_cols"]
    if model is None or not feature_names:
        return "SHAP skipped: no fitted LightGBM model is available."
    try:
        import shap

        last_week = results["predictions"]["week"].max()
        rows = weekly_data.loc[weekly_data.index == last_week]
        features = rows[feature_names].apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan).fillna(results["last_feature_medians"])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, features, max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close("all")
        return None
    except Exception as exc:
        return f"SHAP skipped: {exc}"


def run_pipeline(args: argparse.Namespace) -> dict:
    dynamic_universe = args.data_source == "dolt"
    tickers = list(args.tickers) if not dynamic_universe else []
    if not dynamic_universe:
        if not tickers or len(set(tickers)) != len(tickers):
            raise ValueError("Supply a non-empty universe of unique ticker symbols.")
        if set(tickers) & {"SPY", "^VIX"}:
            raise ValueError("SPY and ^VIX are reserved for benchmark and market-regime data.")
        if not 1 <= args.top_n <= len(tickers):
            raise ValueError("top_n must be between one and the number of tickers.")
    if args.retrain_freq < 1 or args.rebalance_freq < 1:
        raise ValueError("Retraining and rebalance frequencies must be positive.")
    if (
        not np.isfinite(args.transaction_cost_bps)
        or args.transaction_cost_bps < 0
        or not np.isfinite(args.confidence_threshold)
        or not 0 <= args.confidence_threshold <= 1
    ):
        raise ValueError("Transaction costs must be non-negative and confidence threshold must be in [0, 1].")
    if args.bootstrap_samples < 1 or args.random_simulations < 1:
        raise ValueError("Bootstrap and random-simulation counts must be positive.")
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    if args.final_holdout_weeks < 1:
        raise ValueError("final_holdout_weeks must be a positive integer.")
    if args.evaluation_mode not in {"development", "final_holdout", "both"}:
        raise ValueError("evaluation_mode must be development, final_holdout, or both.")
    if not np.isfinite(args.risk_free_rate) or args.risk_free_rate <= -1:
        raise ValueError("risk_free_rate must exceed -100%.")

    print("Loading validated market data...")
    data_metadata: dict = {"provider": "Yahoo Finance via yfinance"}
    dividends: dict[str, pd.Series] = {}
    spy_dividends: pd.Series | None = None
    universe_by_week: dict[pd.Timestamp, set[str]] | None = None
    if dynamic_universe:
        bundle = load_dolt_market_data(
            project_dir=PROJECT_DIR,
            universe_size=args.universe_size,
            min_price=args.min_price,
            min_dollar_volume=args.min_dollar_volume,
            min_history_bars=args.min_history_bars,
            gap_days=args.ticker_gap_days,
            refresh=args.refresh_dolt_data,
            dolt_dir=args.dolt_dir,
            dolt_executable=args.dolt_executable,
        )
        stock_data = bundle.stock_data
        spy_data = bundle.spy_data
        spy_data, benchmark_quality = sanitize_benchmark_ohlc(spy_data)
        vix_data = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        dividends = bundle.dividends
        spy_dividends = bundle.spy_dividends
        universe_by_week = bundle.universe_by_week
        data_metadata = bundle.metadata
        data_metadata["benchmark_data_quality"] = benchmark_quality
        tickers = sorted(stock_data)
        if not 1 <= args.top_n <= args.universe_size:
            raise ValueError("top_n must be no larger than --universe-size.")
        print(
            f"Loaded {len(tickers):,} security eras and {len(universe_by_week):,} weekly liquidity rosters "
            f"from source revision {data_metadata.get('source_revision')}."
        )
        print("No VIX history is provided by this stock database; SPY trend and market return features remain available.")
    else:
        stock_data, spy_data, vix_data = load_market_data(
            tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_file=args.cache_file,
            refresh=args.refresh_cache,
        )
    earliest = min(frame.index.min() for frame in stock_data.values() if not frame.empty)
    latest = max(frame.index.max() for frame in stock_data.values() if not frame.empty)
    print(f"Stock coverage: {earliest.date()} through {latest.date()} ({len(tickers):,} security eras)")
    print(
        "Equity OHLC prices are split-adjusted; dividend cash is included in open-to-open labels; "
        "price gaps are not forward-filled."
        if dynamic_universe else
        "Equity OHLC prices are split/dividend adjusted; price gaps are not forward-filled."
    )

    print("Computing daily technical and cross-sectional features...")
    all_features = compute_all_features(
        stock_data, spy_data, vix_data, tickers,
        include_cross_sectional=not dynamic_universe,
    )
    weekly_data = prepare_weekly_dataset(all_features, stock_data, tickers, dividends=dividends)
    weekly_data = compute_lagged_target_features(weekly_data, tickers)
    if dynamic_universe:
        row_membership = [
            str(ticker) in universe_by_week.get(pd.Timestamp(week), set())
            for week, ticker in zip(weekly_data.index, weekly_data["ticker"], strict=True)
        ]
        weekly_data = weekly_data.loc[row_membership].copy()
        weekly_data = compute_weekly_cross_sectional_features(weekly_data)
        start_limit = pd.Timestamp(args.start_date)
        weekly_data = weekly_data.loc[weekly_data.index >= start_limit]
        if args.end_date:
            weekly_data = weekly_data.loc[weekly_data.index < pd.Timestamp(args.end_date)]
        universe_by_week = {
            week: names for week, names in universe_by_week.items()
            if week >= start_limit and (args.end_date is None or week < pd.Timestamp(args.end_date))
        }
    panel_diagnostics = validate_weekly_panel(
        weekly_data, tickers, universe_by_week=universe_by_week
    )
    market_data_quality = build_market_data_quality_report(stock_data, spy_data, vix_data)

    excluded = {
        "ticker", "forward_return", "target", "close", "entry_price", "exit_price",
        "signal_date", "entry_date", "label_end_date", "cash_dividends",
    }
    feature_cols = [
        name for name in weekly_data.select_dtypes(include=[np.number, "bool"]).columns
        if name not in excluded
    ]
    split_basis_sensitive_features: list[str] = []
    if dynamic_universe:
        # Retrospective split adjustment changes absolute dollar/volume scales
        # before the split. Keep scale-free features so later corporate actions
        # cannot leak into historical model inputs.
        split_basis_sensitive_features = [
            name for name in ("macd", "macd_signal", "macd_hist", "force_index")
            if name in feature_cols
        ]
        feature_cols = [name for name in feature_cols if name not in split_basis_sensitive_features]
    if not feature_cols:
        raise RuntimeError("Feature generation returned no numeric model features.")
    candidate_oos_weeks = complete_evaluation_weeks(
        weekly_data,
        tickers,
        after_date=args.initial_train_end,
        universe_by_week=universe_by_week,
        min_universe_size=args.universe_size if dynamic_universe else None,
    )
    if len(candidate_oos_weeks) < 2:
        raise RuntimeError("At least two complete out-of-sample weeks are needed to define an evaluation split.")
    if args.final_holdout_start_date:
        holdout_start = pd.Timestamp(args.final_holdout_start_date)
    else:
        reserved_holdout_count = min(args.final_holdout_weeks, max(1, len(candidate_oos_weeks) // 3))
        holdout_start = candidate_oos_weeks[-reserved_holdout_count]
    reserved_holdout_weeks = candidate_oos_weeks[candidate_oos_weeks >= holdout_start]
    development_candidate_weeks = candidate_oos_weeks[candidate_oos_weeks < holdout_start]
    if not len(reserved_holdout_weeks) or not len(development_candidate_weeks):
        raise ValueError(
            "The final holdout split must leave at least one complete development week and one complete holdout week."
        )

    output_root = Path(args.output_dir)
    experiment_ledger = ExperimentLedger(output_root)
    experiment_ledger.import_existing_runs()
    requested_holdout_end = weekly_data.loc[
        weekly_data.index.isin(reserved_holdout_weeks), "label_end_date"
    ].max()
    prior_holdout_overlaps: list[dict[str, str]] = []
    if args.evaluation_mode in {"both", "final_holdout"}:
        prior_holdout_overlaps = experiment_ledger.assert_holdout_is_fresh(
            holdout_start,
            requested_holdout_end,
            allow_consumed=args.allow_consumed_holdout,
        )
        if prior_holdout_overlaps:
            print("WARNING: Re-evaluating a previously consumed holdout; this result is not independent.")

    evaluation_start = holdout_start if args.evaluation_mode == "final_holdout" else None
    evaluation_end = holdout_start if args.evaluation_mode == "development" else None
    print(
        f"Weekly panel: {len(weekly_data):,} rows, {weekly_data.index.nunique():,} signal weeks, "
        f"{len(feature_cols)} candidate features."
    )

    print(
        f"Running {args.evaluation_mode} evaluation; development ends {holdout_start.date()}, "
        f"reserved final holdout has {len(reserved_holdout_weeks)} completed weeks."
    )
    models = build_classifier_models(random_state=args.random_state, n_jobs=args.n_jobs)
    results = walk_forward_backtest(
        weekly_data,
        feature_cols,
        models,
        tickers,
        initial_train_end=args.initial_train_end,
        retrain_freq=args.retrain_freq,
        top_n=args.top_n,
        transaction_cost_bps=args.transaction_cost_bps,
        turnover_penalty=args.turnover_penalty,
        regime_filter=args.regime_filter,
        use_regression_blend=args.regression_blend,
        confidence_threshold=args.confidence_threshold,
        rebalance_freq=args.rebalance_freq,
        evaluation_start_date=evaluation_start,
        evaluation_end_date=evaluation_end,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        universe_by_week=universe_by_week,
        min_universe_size=args.universe_size if dynamic_universe else 2,
    )

    test_index = pd.DatetimeIndex(results["returns_after_costs"].index).sort_values()
    if len(test_index) < 2:
        raise RuntimeError("At least two completed out-of-sample weeks are required for research evaluation.")
    if args.evaluation_mode == "development":
        phase_labels = np.repeat("development", len(test_index))
    elif args.evaluation_mode == "final_holdout":
        phase_labels = np.repeat("final_holdout", len(test_index))
    else:
        phase_labels = np.where(test_index >= holdout_start, "final_holdout", "development")
    phase_by_week = pd.Series(phase_labels, index=test_index, name="evaluation_phase")
    results["predictions"]["evaluation_phase"] = results["predictions"]["week"].map(phase_by_week)
    results["portfolio"]["evaluation_phase"] = phase_by_week.reindex(results["portfolio"].index)
    feature_importance_by_fit, feature_importance_stability = _feature_importance_stability(
        results["model_importances"]
    )

    benchmarks = compute_benchmark_returns(
        weekly_data,
        tickers,
        spy_data,
        top_n=args.top_n,
        transaction_cost_bps=args.transaction_cost_bps,
        evaluation_weeks=results["returns_after_costs"].index,
        spy_dividends=spy_dividends,
    )
    for name in benchmarks:
        benchmarks[name] = benchmarks[name].reindex(test_index).dropna()

    risk_free_rate = args.risk_free_rate
    strategy_before = compute_metrics(results["returns_before_costs"], risk_free_rate)
    strategy_after = compute_metrics(results["returns_after_costs"], risk_free_rate)
    benchmark_metrics = {
        name: compute_metrics(series, risk_free_rate)
        for name, series in benchmarks.items()
        if not name.endswith("_gross")
    }
    sharpe_ci = bootstrap_sharpe_ci(
        results["returns_after_costs"],
        n_bootstrap=args.bootstrap_samples,
        risk_free_rate=risk_free_rate,
        random_state=args.random_state,
    )
    excess_return_intervals = {}
    for name in ("equal_weight", "spy", "raw_momentum"):
        if name in benchmarks and len(benchmarks[name].dropna()) >= 2:
            excess_return_intervals[name] = bootstrap_mean_difference_ci(
                results["returns_after_costs"],
                benchmarks[name],
                n_bootstrap=args.bootstrap_samples,
                random_state=args.random_state,
            )

    random_test_data = weekly_data.loc[weekly_data.index.isin(test_index)]
    random_pvalue, random_sharpes = monte_carlo_test(
        random_test_data,
        strategy_after["Sharpe Ratio"],
        tickers,
        n_sims=args.random_simulations,
        top_n=args.top_n,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=risk_free_rate,
        random_state=args.random_state,
        universe_by_week=universe_by_week,
    )
    ic = compute_information_coefficient(results["predictions"])
    ic_mean = float(ic["ic"].mean()) if not ic.empty else None
    ic_std = float(ic["ic"].std(ddof=1)) if len(ic) > 1 else None
    ic_information_ratio = float(ic_mean / ic_std) if ic_mean is not None and ic_std else None

    classification = _classification_metrics(results["predictions"])
    calibration_curve = _calibration_curve_table(results["predictions"])
    selected = results["predictions"].loc[results["predictions"]["selected"].eq(1)]
    selection_precision = float(selected["actual_return"].gt(0).mean()) if len(selected) else None
    phase_performance = _phase_performance(
        results["returns_before_costs"],
        results["returns_after_costs"],
        benchmarks,
        phase_by_week,
        results["predictions"],
        ic,
        bootstrap_samples=args.bootstrap_samples,
        risk_free_rate=risk_free_rate,
        random_state=args.random_state,
    )
    calendar_year_performance = _calendar_year_performance(
        results["returns_after_costs"], benchmarks, risk_free_rate
    )
    cost_sensitivity = _cost_stress_performance(
        portfolio=results["portfolio"],
        weekly_data=weekly_data,
        tickers=tickers,
        spy_data=spy_data,
        phase_by_week=phase_by_week,
        base_cost_bps=args.transaction_cost_bps,
        top_n=args.top_n,
        risk_free_rate=risk_free_rate,
        spy_dividends=spy_dividends,
    )
    configured_cost_returns = cost_adjusted_portfolio_returns(
        results["portfolio"], args.transaction_cost_bps
    )
    if not np.allclose(
        configured_cost_returns.to_numpy(),
        results["returns_after_costs"].reindex(configured_cost_returns.index).to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Stored portfolio returns do not reconcile to the configured transaction-cost model.")

    factor_windows = pd.DataFrame()
    factor_estimates = pd.DataFrame()
    factor_research: dict = {"status": "unavailable"}
    factor_data_fingerprint: str | None = None
    try:
        daily_factors, factor_data_metadata = load_french_daily_factors(
            cache_dir=args.factor_cache_dir,
            refresh=args.refresh_factor_data,
        )
        factor_windows, factor_estimates, factor_diagnostics = analyze_portfolio_against_factors(
            results["portfolio"], daily_factors
        )
        factor_research = {"status": "complete", **factor_diagnostics, "data": factor_data_metadata}
        factor_data_fingerprint = factor_data_metadata["fingerprint_sha256"]
    except Exception as exc:
        # Factor attribution is an independent diagnostic. A source outage must
        # not destroy an otherwise complete market-data/model research run.
        factor_research = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "data_library_url": "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html",
        }

    market_price_fingerprint = fingerprint_market_data(stock_data, spy_data, vix_data)
    data_fingerprint = market_price_fingerprint
    if dynamic_universe:
        data_fingerprint = hashlib.sha256(
            (
                market_price_fingerprint
                + str(data_metadata.get("cache_fingerprint_sha256", ""))
                + str(data_metadata.get("source_revision", ""))
            ).encode("utf-8")
        ).hexdigest()
    source_fingerprint = _backend_source_fingerprint()
    reproducibility_config = {
        key: value for key, value in vars(args).items()
        if key not in {
            "output_dir", "cache_file", "refresh_cache", "factor_cache_dir",
            "refresh_factor_data", "run_id",
        }
    }
    if dynamic_universe:
        reproducibility_config["tickers"] = "weekly liquidity-ranked security eras"
    run_payload = json.dumps(
        {
            "data_fingerprint": data_fingerprint,
            "source_data_revision": data_metadata.get("source_revision"),
            "factor_data_fingerprint": factor_data_fingerprint,
            "source_fingerprint": source_fingerprint,
            "python": platform.python_version(),
            "packages": _installed_versions(),
            "configuration": reproducibility_config,
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    experiment_fingerprint = hashlib.sha256(run_payload.encode("utf-8")).hexdigest()
    run_id = args.run_id or uuid.uuid4().hex[:16]
    if not run_id.replace("-", "").isalnum() or len(run_id) > 64:
        raise ValueError("run_id must contain only letters, numbers, or hyphens and be at most 64 characters.")
    output_dir = output_root / f"run-{run_id}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run output already exists: {output_dir}")
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    market_quality_summary = {
        "symbols": int(len(market_data_quality)),
        "symbols_with_missing_spy_sessions": int(market_data_quality["missing_spy_sessions"].gt(0).sum()),
        "symbols_with_invalid_prices": int(market_data_quality["invalid_price_rows"].gt(0).sum()),
        "symbols_with_ohlc_inconsistencies": int(market_data_quality["ohlc_inconsistent_rows"].gt(0).sum()),
    }
    performance = {
        "strategy_before_costs": strategy_before,
        "strategy_after_costs": strategy_after,
        "benchmarks_after_costs": benchmark_metrics,
        "classification": classification,
        "probability_calibration_curve": calibration_curve.to_dict(orient="records"),
        "selection_positive_return_rate": selection_precision,
        "sharpe_95pct_block_bootstrap_ci": list(sharpe_ci),
        "annualized_excess_return_95pct_ci": excess_return_intervals,
        "random_pick_cost_matched_p_value": random_pvalue,
        "random_pick_sharpe_distribution": {
            "p05": float(np.quantile(random_sharpes, 0.05)),
            "median": float(np.quantile(random_sharpes, 0.50)),
            "p95": float(np.quantile(random_sharpes, 0.95)),
        },
        "information_coefficient_mean": ic_mean,
        "information_coefficient_ir": ic_information_ratio,
        "mean_turnover": float(results["turnover"].mean()),
        "total_transaction_cost_fraction": float(results["portfolio"]["transaction_cost"].sum()),
        "skipped_weeks": results["skipped_weeks"],
        "development": phase_performance["development"],
        "final_holdout": phase_performance["final_holdout"],
        "cost_sensitivity": cost_sensitivity.to_dict(orient="records"),
        "calendar_year_performance": calendar_year_performance.to_dict(orient="records"),
        "feature_importance_stability_top_20": feature_importance_stability.head(20).to_dict(orient="records"),
    }

    data_summary = {
        **data_metadata,
        "fingerprint_sha256": data_fingerprint,
        "market_price_fingerprint_sha256": market_price_fingerprint,
        "requested_start": args.start_date,
        "requested_end_exclusive": args.end_date,
        "observed_start": earliest,
        "observed_end": latest,
        "tickers": tickers,
        "signal_start": results["returns_after_costs"].index.min(),
        "signal_end": results["returns_after_costs"].index.max(),
        "target": "next-session-open to following-rebalance-open total return, including cash dividends",
        "quality_summary": market_quality_summary,
        "panel_validation": panel_diagnostics,
    }
    if not dynamic_universe:
        data_summary.update({
            "provider": "Yahoo Finance via yfinance",
            "equity_prices": "Yahoo Finance auto-adjusted OHLC; volume is reported as supplied by the provider, and history can be revised retrospectively.",
            "universe_policy": (
                "Fixed user-supplied ticker universe. Historical index membership and delisted names are not "
                "reconstructed, so survivorship bias remains a limitation."
            ),
            "known_data_and_execution_limits": [
                "The fixed ticker list does not reproduce point-in-time index membership or delisted securities.",
                "Yahoo Finance history is not a point-in-time institutional data feed and may be revised.",
                "Constant basis-point costs do not model per-stock spreads, market impact, liquidity, or capacity.",
                "The long-only historical simulation is not evidence of live execution or investment performance.",
            ],
        })
    else:
        data_summary.update({
            "equity_prices": data_metadata.get("prices"),
            "universe_policy": data_metadata.get("universe_policy"),
            "known_data_and_execution_limits": [
                *data_metadata.get("limitations", []),
                "Constant basis-point costs do not model per-stock spreads, market impact, or capacity.",
                "The long-only historical simulation is not evidence of live execution or investment performance.",
            ],
        })

    summary = {
        "project": "MomentumAlpha",
        "run_id": run_id,
        "experiment_fingerprint_sha256": experiment_fingerprint,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "factor_attribution": factor_research,
        "data": data_summary,
        "configuration": {
            "initial_train_end": args.initial_train_end,
            "final_holdout_start": holdout_start,
            "final_holdout_end": requested_holdout_end,
            "reserved_final_holdout_weeks": int(len(reserved_holdout_weeks)),
            "evaluated_final_holdout_weeks": int(phase_by_week.eq("final_holdout").sum()),
            "holdout_split_method": (
                "explicit date" if args.final_holdout_start_date else "most recent completed out-of-sample weeks"
            ),
            "evaluation_mode": args.evaluation_mode,
            "allow_consumed_holdout": args.allow_consumed_holdout,
            "retrain_every_weeks": args.retrain_freq,
            "top_n": args.top_n,
            "transaction_cost_bps_per_traded_notional": args.transaction_cost_bps,
            "risk_free_rate": risk_free_rate,
            "turnover_hysteresis": args.turnover_penalty,
            "regime_filter": args.regime_filter,
            "regression_rank_blend": args.regression_blend,
            "confidence_threshold": args.confidence_threshold,
            "rebalance_every_weeks": args.rebalance_freq,
            "random_state": args.random_state,
            "data_source": args.data_source,
            "universe_size": args.universe_size if dynamic_universe else len(tickers),
            "minimum_price": args.min_price if dynamic_universe else None,
            "minimum_60d_dollar_volume": args.min_dollar_volume if dynamic_universe else None,
            "minimum_history_bars": args.min_history_bars if dynamic_universe else None,
            "excluded_split_basis_sensitive_features": split_basis_sensitive_features,
            "models": list(models),
            "ensemble_weights": get_model_weights(list(models)),
            "probability_calibration": (
                "Per-model Platt scaling is fitted on a recent chronological calibration block when both labels "
                "are present; model fitting and preprocessing use earlier rows only, and failures are recorded."
            ),
            "candidate_features": len(feature_cols),
            "features_per_fit": [record["feature_count"] for record in results["fit_records"]],
        },
        "performance": performance,
        "fit_windows": results["fit_records"],
        "research_protocol": {
            "walk_forward": "Expanding training window with periodic retraining.",
            "label_purge": "Training labels are eligible only when label_end_date is no later than the signal date.",
            "feature_preprocessing": "Missing-value filtering, medians, and correlation pruning are fitted on model-training rows only.",
            "model_calibration": "Chronological Platt calibration block with fully matured labels; calibration failures are recorded.",
            "development_and_holdout": (
                "The default run evaluates development data only. The reserved latest period is evaluated only "
                "with --evaluation-mode final_holdout after model and strategy choices are frozen."
            ),
            "cost_stress": "Portfolio decisions are frozen while costs are recomputed at 0x, 1x, 2x, and 5x base costs.",
            "parameter_selection": "No automatic tuning is performed on the final holdout.",
        },
        "environment": {
            "python": platform.python_version(),
            "packages": _installed_versions(),
            "platform": platform.platform(),
            "git_revision": _git_revision(),
            "backend_source_fingerprint_sha256": source_fingerprint,
        },
        "evaluation_note": (
            (
                "The final holdout was evaluated in this run and is now consumed for strategy selection. "
                "Use a later, non-overlapping period for any fresh confirmatory claim. These results are "
                "historical research, not a live-trading performance claim."
            )
            if args.evaluation_mode in {"both", "final_holdout"}
            else (
                "The final holdout was reserved but not evaluated. Keep model and strategy choices frozen "
                "before opening it. These results are historical research, not a live-trading performance claim."
            )
        ),
    }
    summary["search_diagnostics"] = {
        "status": "not_estimable",
        "reason": "Experiment-history diagnostics have not yet completed.",
        "selection_sample": "development phase only; final-holdout rows are excluded",
    }
    summary["research_assessment"] = assess_run_evidence(summary)

    # Persist line-level outputs from the exact run; no static report files are touched.
    predictions = results["predictions"].copy()
    predictions["week_start_date"] = predictions["week"]
    predictions["predicted_prob"] = predictions["ensemble_prob"]
    predictions["predicted_class"] = (predictions["ensemble_prob"] >= 0.5).astype(int)
    csv_columns = [
        "week_start_date", "signal_date", "entry_date", "label_end_date", "ticker",
        "predicted_prob", "ensemble_prob_raw", "actual_return", "predicted_class", "rank", "selected",
        "weight", "rank_score", "decision_score", "rebalance", "exposure", "evaluation_phase",
    ]
    csv_columns.extend(
        column for column in predictions.columns
        if column.startswith("prob_") and column not in csv_columns
    )
    predictions[csv_columns].to_csv(output_dir / "predictions.csv", index=False)
    results["portfolio"].to_csv(output_dir / "portfolio_returns.csv", index_label="signal_week")
    ic.to_csv(output_dir / "information_coefficient.csv", index_label="signal_week")
    pd.DataFrame(results["fit_records"]).to_csv(output_dir / "walk_forward_fits.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    calendar_year_performance.to_csv(output_dir / "calendar_year_performance.csv", index=False)
    market_data_quality.to_csv(output_dir / "market_data_quality.csv", index_label="ticker")
    calibration_curve.to_csv(output_dir / "calibration_curve.csv", index=False)
    feature_importance_by_fit.to_csv(output_dir / "feature_importance_by_fit.csv", index=False)
    feature_importance_stability.to_csv(output_dir / "feature_importance_stability.csv", index=False)
    if not factor_windows.empty:
        factor_windows.to_csv(output_dir / "factor_aligned_returns.csv", index=False)
    if not factor_estimates.empty:
        factor_estimates.to_csv(output_dir / "factor_attribution.csv", index=False)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)

    run_manifest = {
        "run_id": run_id,
        "run_at_utc": summary["run_at_utc"],
        "data_fingerprint_sha256": data_fingerprint,
        "factor_data_fingerprint_sha256": factor_data_fingerprint,
        "experiment_fingerprint_sha256": experiment_fingerprint,
        "backend_source_fingerprint_sha256": source_fingerprint,
        "git_revision": _git_revision(),
        "configuration": _json_safe(reproducibility_config),
        "evaluation_split": {
            "development_weeks": int(phase_by_week.eq("development").sum()),
            "final_holdout_start": _json_safe(holdout_start),
            "final_holdout_end": _json_safe(requested_holdout_end),
            "reserved_final_holdout_weeks": int(len(reserved_holdout_weeks)),
            "evaluated_final_holdout_weeks": int(phase_by_week.eq("final_holdout").sum()),
            "evaluation_mode": args.evaluation_mode,
            "allow_consumed_holdout": args.allow_consumed_holdout,
        },
        "packages": _installed_versions(),
        "artifacts": [
            "predictions.csv",
            "portfolio_returns.csv",
            "information_coefficient.csv",
            "walk_forward_fits.csv",
            "cost_sensitivity.csv",
            "calendar_year_performance.csv",
            "market_data_quality.csv",
            "calibration_curve.csv",
            "feature_importance_by_fit.csv",
            "feature_importance_stability.csv",
            "run_summary.json",
        ],
    }
    if not factor_windows.empty:
        run_manifest["artifacts"].extend(["factor_aligned_returns.csv", "factor_attribution.csv"])
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(run_manifest), stream, indent=2, allow_nan=False)

    try:
        history = experiment_ledger.record_completed(run_manifest, summary, output_dir)
        history["reused_previous_holdout"] = bool(prior_holdout_overlaps)
        search_diagnostics = experiment_ledger.search_diagnostics(
            history["study_family_fingerprint_sha256"],
            risk_free_rate=risk_free_rate,
        )
        summary["search_diagnostics"] = search_diagnostics
        summary["experiment_history"] = history
        summary["research_assessment"] = assess_run_evidence(
            summary,
            pbo=search_diagnostics.get("pbo"),
            dsr=search_diagnostics.get("deflated_sharpe"),
        )
        run_manifest["search_diagnostics"] = search_diagnostics
        run_manifest["experiment_history"] = history
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(run_manifest), stream, indent=2, allow_nan=False)
    except Exception as exc:
        print(f"WARNING: Run completed, but experiment history could not be updated: {exc}")

    # The summary is rewritten after ledger diagnostics. Hash the final versions
    # of every declared result so later API downloads and manual audits can
    # detect missing or modified artifacts. The manifest cannot hash itself.
    run_manifest["artifact_integrity"] = {
        "algorithm": "SHA-256",
        "scope": "every manifest-listed artifact, including the final summary; excludes the manifest itself",
    }
    run_manifest["artifact_checksums_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in run_manifest["artifacts"]
        if (output_dir / name).is_file()
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(run_manifest), stream, indent=2, allow_nan=False)
    artifact_verification = validate_run_directory(output_dir, expected_run_id=run_id)
    if not artifact_verification["valid"]:
        raise RuntimeError(
            "Completed run failed its artifact integrity check: "
            + "; ".join(artifact_verification["errors"])
        )

    performance_frame = pd.DataFrame({
        "Strategy (gross)": strategy_before,
        "Strategy (net)": strategy_after,
        **{
            name.replace("_", " ").title(): metrics
            for name, metrics in benchmark_metrics.items()
        },
    })
    print("\nPerformance (all strategies use matched open-to-open periods):")
    print(performance_frame.round(4).to_string())
    print(f"\nAfter-cost Sharpe 95% moving-block interval: [{sharpe_ci[0]:.3f}, {sharpe_ci[1]:.3f}]")
    print(f"Cost-matched random-pick comparison: p={random_pvalue:.4f}")
    print(f"Mean weekly information coefficient: {ic_mean if ic_mean is not None else float('nan'):.4f}")
    if args.evaluation_mode == "development":
        print(
            f"Reserved final holdout begins {holdout_start.date()} "
            f"({len(reserved_holdout_weeks)} weeks); it was not evaluated."
        )
    else:
        print(
            f"Final holdout evaluation from {holdout_start.date()} "
            f"({int(phase_by_week.eq('final_holdout').sum())} completed weeks)."
        )
    print("Cost sensitivity and year-by-year results are saved with this run.")
    print(f"Verified {artifact_verification['checked_artifacts']} run artifacts and SHA-256 checksums.")
    print(f"Saved run outputs under: {output_dir.resolve()}")

    if not args.skip_plots:
        _plot_outputs(
            stock_data=stock_data,
            tickers=tickers,
            results=results,
            benchmarks=benchmarks,
            ic=ic,
            plot_dir=plot_dir,
        )
    if args.shap:
        note = _save_shap_plot(results, weekly_data, plot_dir)
        if note:
            print(note)

    return {"summary": summary, "results": results, "benchmarks": benchmarks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MomentumAlpha walk-forward stock-selection research pipeline."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--data-source",
        choices=("yahoo", "dolt"),
        default="yahoo",
        help="Use the fixed Yahoo universe or the cached broad historical DoltHub common-stock panel.",
    )
    parser.add_argument("--universe-size", type=int, default=50, help="Weekly top-liquidity universe size for --data-source dolt.")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum contemporaneous raw close for Dolt universe eligibility.")
    parser.add_argument("--min-dollar-volume", type=float, default=5_000_000.0, help="Minimum 60-session average dollar volume for Dolt eligibility.")
    parser.add_argument("--min-history-bars", type=int, default=252, help="Minimum security-era bars before Dolt universe eligibility.")
    parser.add_argument("--ticker-gap-days", type=int, default=180, help="Split reused ticker symbols into eras after this many calendar days without bars.")
    parser.add_argument("--dolt-dir", default=str(PROJECT_DIR / ".cache" / "datasets" / "stocks"))
    parser.add_argument("--dolt-executable", default=str(PROJECT_DIR / ".cache" / "tools" / "dolt" / "dolt-windows-amd64" / "bin" / "dolt.exe"))
    parser.add_argument("--refresh-dolt-data", action="store_true", help="Re-export and rebuild the Dolt snapshot cache.")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional exclusive Yahoo Finance end date; defaults to latest available data.",
    )
    parser.add_argument("--initial-train-end", default=DEFAULT_INITIAL_TRAIN_END)
    parser.add_argument(
        "--final-holdout-weeks",
        type=int,
        default=52,
        help="Reserve the most recent completed out-of-sample weeks for a separate final evaluation.",
    )
    parser.add_argument(
        "--final-holdout-start-date",
        default=None,
        help="Optional explicit holdout start date; overrides the latest-weeks split.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("development", "final_holdout", "both"),
        default="development",
        help="Use development by default to keep the final holdout uninspected until choices are frozen.",
    )
    parser.add_argument("--retrain-freq", type=int, default=13)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--transaction-cost-bps", type=float, default=10)
    parser.add_argument("--risk-free-rate", type=float, default=RISK_FREE_RATE)
    parser.add_argument("--rebalance-freq", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--turnover-penalty", action="store_true")
    parser.add_argument("--regime-filter", action="store_true")
    parser.add_argument("--regression-blend", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-simulations", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--cache-file", default=str(PROJECT_DIR / ".cache" / "market_data.zip"))
    parser.add_argument(
        "--factor-cache-dir",
        default=str(PROJECT_DIR / ".cache" / "fama_french"),
        help="Directory for the original French Data Library factor archives.",
    )
    parser.add_argument(
        "--refresh-factor-data",
        action="store_true",
        help="Fetch the latest factor archives rather than reusing the cached vintage.",
    )
    parser.add_argument("--run-id", default=None, help="Optional unique run ID, useful to track API-launched runs.")
    parser.add_argument(
        "--allow-consumed-holdout",
        action="store_true",
        help=(
            "Permit a final holdout that overlaps a previously evaluated interval. "
            "The resulting run is recorded as a non-independent recheck."
        ),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / ".runs"),
        help="Parent folder for run-specific outputs stored under a unique run-ID folder.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--shap", action="store_true", help="Optionally create SHAP values for the latest fitted LightGBM model.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Fixed universe for --data-source yahoo; ignored for Dolt's weekly liquidity universe.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(args)
    except Exception as exc:
        raise SystemExit(f"MomentumAlpha run failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
