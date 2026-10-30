"""Factor-aware returns and HAC alpha attribution for completed research runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


FACTOR_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
PERIODS_PER_YEAR = 52


def align_factor_intervals(
    portfolio: pd.DataFrame,
    daily_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compound daily factors over each portfolio's open-to-open label window.

    French factors are close-to-close. We use observations after the entry date
    through the exit date as a close-to-close proxy for the portfolio's open-
    to-open period, and record that approximation in the result metadata.
    """
    required_portfolio = {"entry_date", "label_end_date", "net_return"}
    required_factors = {*FACTOR_COLUMNS, "RF"}
    if not required_portfolio.issubset(portfolio.columns):
        raise ValueError(f"Portfolio data is missing: {sorted(required_portfolio - set(portfolio.columns))}")
    if not required_factors.issubset(daily_factors.columns):
        raise ValueError(f"Factor history is missing: {sorted(required_factors - set(daily_factors.columns))}")
    if portfolio.empty or daily_factors.empty:
        raise ValueError("Portfolio returns and factor history must both be non-empty.")

    factors = daily_factors.loc[:, [*FACTOR_COLUMNS, "RF"]].copy()
    factors.index = pd.DatetimeIndex(pd.to_datetime(factors.index, errors="coerce")).tz_localize(None).normalize()
    factors = factors.loc[~factors.index.isna()].sort_index()
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, Any]] = []
    unmatched = 0
    for _, source in portfolio.iterrows():
        entry = pd.Timestamp(source["entry_date"]).tz_localize(None).normalize()
        exit_date = pd.Timestamp(source["label_end_date"]).tz_localize(None).normalize()
        interval = factors.loc[(factors.index > entry) & (factors.index <= exit_date)]
        if interval.empty:
            unmatched += 1
            continue
        compounded = (1.0 + interval).prod(axis=0) - 1.0
        if not np.isfinite(compounded.to_numpy(dtype=float)).all():
            unmatched += 1
            continue
        record: dict[str, Any] = {
            "signal_week": pd.Timestamp(source.name) if source.name is not None else pd.Timestamp(source.get("signal_week")),
            "entry_date": entry,
            "label_end_date": exit_date,
            "evaluation_phase": source.get("evaluation_phase", "development"),
            "net_return": float(source["net_return"]),
            "factor_rf": float(compounded["RF"]),
            "excess_return": float(source["net_return"] - compounded["RF"]),
            "factor_sessions": int(len(interval)),
        }
        record.update({name: float(compounded[name]) for name in FACTOR_COLUMNS})
        rows.append(record)

    aligned = pd.DataFrame(rows)
    if aligned.empty:
        raise ValueError("No portfolio holding intervals overlap the factor history.")
    aligned = aligned.sort_values("signal_week").reset_index(drop=True)
    coverage = {
        "portfolio_intervals": int(len(portfolio)),
        "matched_intervals": int(len(aligned)),
        "unmatched_intervals": int(unmatched),
        "coverage_fraction": float(len(aligned) / len(portfolio)),
        "first_matched_exit": aligned["label_end_date"].min().date().isoformat(),
        "last_matched_exit": aligned["label_end_date"].max().date().isoformat(),
        "factor_window_method": "compound daily close-to-close factors for entry_date < date <= label_end_date",
        "execution_alignment_note": (
            "The portfolio target is open-to-open; daily factor returns are close-to-close, so attribution is an approximation."
        ),
    }
    return aligned, coverage


def _newey_west_ols(
    design: np.ndarray,
    target: np.ndarray,
    *,
    max_lags: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    observations, parameters = design.shape
    degrees_freedom = observations - parameters
    if degrees_freedom < 2:
        raise ValueError("Too few aligned weekly observations for a factor regression.")
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residuals = target - design @ beta
    score = design * residuals[:, None]
    meat = score.T @ score
    lag_count = min(max(0, int(max_lags)), observations - 1)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        autocovariance = score[lag:].T @ score[:-lag]
        meat += weight * (autocovariance + autocovariance.T)
    bread = np.linalg.pinv(design.T @ design)
    covariance = (observations / degrees_freedom) * bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    residual_sum = float(np.sum(residuals**2))
    total_sum = float(np.sum((target - target.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
    return beta, standard_errors, r_squared, degrees_freedom


def factor_regression(
    aligned: pd.DataFrame,
    *,
    max_lags: int = 4,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate weekly FF5+momentum alpha with Newey-West standard errors.

    Reported p-values are nominal and do not adjust for strategy/model search.
    They are diagnostics, not a standalone pass/fail claim of alpha.
    """
    if max_lags < 0 or min_observations < 1:
        raise ValueError("max_lags must be non-negative and min_observations positive.")
    required = {"excess_return", *FACTOR_COLUMNS}
    if aligned.empty or not required.issubset(aligned.columns):
        raise ValueError("Aligned returns are missing factor-regression fields.")

    records: list[dict[str, Any]] = []
    phases: dict[str, dict[str, Any]] = {}
    groups = [("all", aligned)]
    if "evaluation_phase" in aligned.columns:
        groups.extend(
            (str(phase), phase_frame)
            for phase, phase_frame in aligned.groupby("evaluation_phase", sort=True)
        )

    for phase, frame in groups:
        clean = frame[["excess_return", *FACTOR_COLUMNS]].apply(pd.to_numeric, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(clean)
        phase_result: dict[str, Any] = {
            "status": "insufficient_observations",
            "observations": int(n),
            "minimum_observations": int(min_observations),
            "nominal_p_values_adjust_for_search": False,
        }
        if n < max(min_observations, len(FACTOR_COLUMNS) + 3):
            phases[phase] = phase_result
            continue

        design = np.column_stack([np.ones(n), clean.loc[:, FACTOR_COLUMNS].to_numpy(dtype=float)])
        rank = int(np.linalg.matrix_rank(design))
        condition_number = float(np.linalg.cond(design))
        phase_result["design_rank"] = rank
        phase_result["design_parameters"] = int(design.shape[1])
        phase_result["design_condition_number"] = condition_number
        if rank < design.shape[1]:
            phase_result["status"] = "rank_deficient_design"
            phases[phase] = phase_result
            continue
        if not np.isfinite(condition_number) or condition_number > 1e10:
            phase_result["status"] = "ill_conditioned_design"
            phases[phase] = phase_result
            continue
        target = clean["excess_return"].to_numpy(dtype=float)
        beta, standard_errors, r_squared, degrees_freedom = _newey_west_ols(
            design, target, max_lags=max_lags
        )
        critical = float(student_t.ppf(0.975, degrees_freedom))
        terms = ["alpha", *FACTOR_COLUMNS]
        term_records = []
        for index, term in enumerate(terms):
            estimate = float(beta[index])
            standard_error = float(standard_errors[index])
            t_stat = estimate / standard_error if standard_error > 0 else None
            p_value = (
                float(2.0 * student_t.sf(abs(t_stat), degrees_freedom))
                if t_stat is not None else None
            )
            scale = PERIODS_PER_YEAR if term == "alpha" else 1.0
            record = {
                "phase": phase,
                "term": term,
                "estimate": estimate * scale,
                "hac_standard_error": standard_error * scale,
                "t_statistic": t_stat,
                "nominal_p_value": p_value,
                "ci95_low": (estimate - critical * standard_error) * scale,
                "ci95_high": (estimate + critical * standard_error) * scale,
                "annualized": term == "alpha",
            }
            records.append(record)
            term_records.append(record)
        phase_result.update({
            "status": "estimated",
            "observations": int(n),
            "hac_lags": int(min(max_lags, n - 1)),
            "r_squared": float(r_squared),
            "alpha_annualized_arithmetic": float(beta[0] * PERIODS_PER_YEAR),
            "alpha_t_statistic_hac": term_records[0]["t_statistic"],
            "alpha_nominal_p_value_hac": term_records[0]["nominal_p_value"],
            "alpha_annualized_ci95_hac": [term_records[0]["ci95_low"], term_records[0]["ci95_high"]],
            "nominal_p_values_adjust_for_search": False,
        })
        phases[phase] = phase_result

    return pd.DataFrame(records), {
        "model": "Fama-French five factors plus momentum",
        "standard_errors": "Newey-West HAC",
        "frequency": "weekly compounded factor returns",
        "alpha_annualization": "arithmetic weekly alpha multiplied by 52",
        "multiple_testing_correction": "none; nominal p-values only",
        "phases": phases,
    }


def analyze_portfolio_against_factors(
    portfolio: pd.DataFrame,
    daily_factors: pd.DataFrame,
    *,
    max_lags: int = 4,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    aligned, coverage = align_factor_intervals(portfolio, daily_factors)
    estimates, regression = factor_regression(
        aligned, max_lags=max_lags, min_observations=min_observations
    )
    return aligned, estimates, {"coverage": coverage, **regression}
