"""Search-aware diagnostics for deciding whether a backtest merits more trust."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    *,
    n_slices: int = 8,
    minimum_periods_per_slice: int = 8,
    risk_free_period_return: float = 0.0,
) -> dict[str, Any]:
    """Estimate PBO with combinatorially symmetric cross-validation.

    Rows are consecutive periods and columns are distinct strategy trials.
    The function uses contiguous slices, selects the best in-sample Sharpe for
    each split, and checks where that winner ranks in the complementary data.
    """
    if n_slices < 4 or n_slices % 2:
        raise ValueError("n_slices must be an even integer of at least four.")
    if minimum_periods_per_slice < 1:
        raise ValueError("minimum_periods_per_slice must be positive.")
    matrix = returns.copy().apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    matrix = matrix.loc[:, matrix.nunique(dropna=True) > 1]
    trials = matrix.shape[1]
    periods = matrix.shape[0]
    result: dict[str, Any] = {
        "method": "CSCV probability of backtest overfitting",
        "status": "not_estimable",
        "trials": int(trials),
        "complete_periods": int(periods),
        "n_slices": int(n_slices),
    }
    if trials < 2:
        result["reason"] = "At least two distinct strategy configurations are required."
        return result
    if periods < n_slices * minimum_periods_per_slice:
        result["reason"] = "Too few aligned periods for the requested contiguous CSCV slices."
        return result

    chunks = [chunk for chunk in np.array_split(np.arange(periods), n_slices) if len(chunk)]
    if len(chunks) != n_slices or min(map(len, chunks)) < minimum_periods_per_slice:
        result["reason"] = "Each CSCV slice must contain the configured minimum number of periods."
        return result

    values = matrix.to_numpy(dtype=float)

    def scores(indices: np.ndarray) -> np.ndarray:
        sample = values[indices]
        std = sample.std(axis=0, ddof=1)
        means = (sample - risk_free_period_return).mean(axis=0)
        return np.divide(means, std, out=np.full(trials, -np.inf), where=std > 0)

    selected_ranks: list[float] = []
    logits: list[float] = []
    block_ids = range(n_slices)
    for in_sample_blocks in combinations(block_ids, n_slices // 2):
        is_mask = np.zeros(n_slices, dtype=bool)
        is_mask[list(in_sample_blocks)] = True
        in_indices = np.concatenate([chunks[index] for index in range(n_slices) if is_mask[index]])
        out_indices = np.concatenate([chunks[index] for index in range(n_slices) if not is_mask[index]])
        in_scores = scores(in_indices)
        winner = int(np.argmax(in_scores))
        out_scores = scores(out_indices)
        out_rank = float(rankdata(out_scores, method="average")[winner])
        percentile = float(np.clip(out_rank / (trials + 1.0), 1e-12, 1.0 - 1e-12))
        selected_ranks.append(percentile)
        logits.append(math.log(percentile / (1.0 - percentile)))

    result.update({
        "status": "estimated",
        "cscv_combinations": int(len(logits)),
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "median_selected_oos_rank_percentile": float(np.median(selected_ranks)),
        "interpretation": "Estimated fraction of in-sample winners that rank below the OOS median; exploratory, not a pass/fail guarantee.",
    })
    return result


def deflated_sharpe_probability(
    selected_returns: pd.Series,
    trial_sharpes: list[float] | np.ndarray,
    *,
    risk_free_period_return: float = 0.0,
) -> dict[str, Any]:
    """Compute a Deflated-Sharpe-style probability against the trial winner hurdle.

    Sharpe inputs are per-period (not annualized). The result remains nominal
    because trial dependence must be estimated or conservatively specified.
    """
    values = pd.to_numeric(pd.Series(selected_returns), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    if len(values) < 3 or len(sharpes) < 2:
        return {
            "status": "not_estimable",
            "reason": "At least three return periods and two tracked strategy trials are required.",
            "trials": int(len(sharpes)),
            "observations": int(len(values)),
        }
    excess = values - float(risk_free_period_return)
    sigma = float(excess.std(ddof=1))
    if sigma <= 0:
        return {"status": "not_estimable", "reason": "Selected strategy has zero return variance."}
    observed = float(excess.mean() / sigma)
    trial_dispersion = float(np.std(sharpes, ddof=1))
    if trial_dispersion <= 0:
        return {"status": "not_estimable", "reason": "Tracked trial Sharpes have zero dispersion."}

    # Expected maximum of N approximately normal trial Sharpes under the null.
    gamma = 0.5772156649015329
    n_trials = len(sharpes)
    expected_max = trial_dispersion * (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    centered_moments = excess / sigma
    skew = float(np.mean(centered_moments**3))
    kurtosis = float(np.mean(centered_moments**4))
    variance_term = 1.0 - skew * observed + ((kurtosis - 1.0) / 4.0) * observed**2
    if variance_term <= 0:
        return {"status": "not_estimable", "reason": "Non-normal Sharpe variance correction was non-positive."}
    z_score = (observed - expected_max) * math.sqrt(len(excess) - 1.0) / math.sqrt(variance_term)
    return {
        "status": "estimated",
        "observations": int(len(excess)),
        "trials": int(n_trials),
        "observed_sharpe_per_period": observed,
        "expected_max_null_sharpe_per_period": float(expected_max),
        "deflated_sharpe_probability": float(norm.cdf(z_score)),
        "z_score": float(z_score),
        "trial_sharpe_dispersion": trial_dispersion,
        "trial_dependence_adjustment": "none; trial count is an upper-bound approximation unless effective independent trials are supplied",
    }


def assess_run_evidence(summary: dict, *, pbo: dict | None = None, dsr: dict | None = None) -> dict[str, Any]:
    """Produce a conservative, explainable result status from one completed run."""
    performance = summary.get("performance", {})
    configuration = summary.get("configuration", {})
    data = summary.get("data", {})
    history = summary.get("experiment_history", {})
    final = performance.get("final_holdout", {})
    final_weeks = int(final.get("strategy_after_costs", {}).get("Num Weeks", 0) or 0)
    mode = configuration.get("evaluation_mode", "development")
    reasons: list[str] = []
    if mode not in {"both", "final_holdout"} or final_weeks == 0:
        reasons.append("The final holdout was not evaluated, so a confirmatory conclusion is not available.")
    elif final_weeks < 104:
        reasons.append(f"The final holdout has {final_weeks} weeks; it is too short for a strong standalone claim.")
    if mode in {"both", "final_holdout"} and final_weeks:
        intervals = final.get("annualized_excess_return_95pct_ci", {})
        for benchmark in ("equal_weight", "spy"):
            interval = intervals.get(benchmark)
            if not interval or interval[0] <= 0:
                reasons.append(f"After-cost excess return versus {benchmark} is not clearly positive at the 95% interval.")
        factor = summary.get("factor_attribution", {}).get("phases", {}).get("final_holdout", {})
        factor_interval = factor.get("alpha_annualized_ci95_hac")
        if not factor_interval or factor_interval[0] <= 0:
            reasons.append("Factor-adjusted holdout alpha does not have a positive 95% HAC lower bound.")
        if factor.get("nominal_p_values_adjust_for_search") is False:
            reasons.append("Factor alpha p-values are nominal; model and strategy search has not been corrected.")
        reasons.append("The final holdout has been inspected and is now consumed; further tuning needs a new period.")
        if history.get("reused_previous_holdout"):
            reasons.append("This run reused a previously inspected final-holdout interval and is not an independent recheck.")

    universe_policy = str(data.get("universe_policy", ""))
    if "Fixed user-supplied" in universe_policy or "survivorship" in universe_policy.lower():
        reasons.append("The fixed historical ticker list retains survivorship bias; point-in-time membership is unavailable.")
    if not pbo or pbo.get("status") != "estimated":
        reasons.append("PBO is not estimable until multiple distinct strategy configurations share aligned return history.")
    if not dsr or dsr.get("status") != "estimated":
        reasons.append("Deflated Sharpe is not estimable until the experiment ledger contains multiple strategy trials.")

    status = "inconclusive" if reasons else "candidate_edge_requires_independent_confirmation"
    return {
        "status": status,
        "claim": (
            "The run does not establish a robust, independent trading edge."
            if reasons else
            "The result passed the configured evidence checks, but is still a historical research result."
        ),
        "reasons": reasons,
        "final_holdout_weeks": final_weeks,
        "final_holdout_evaluated": bool(mode in {"both", "final_holdout"} and final_weeks),
        "reused_previous_holdout": bool(history.get("reused_previous_holdout", False)),
        "pbo": pbo or {"status": "not_estimable"},
        "deflated_sharpe": dsr or {"status": "not_estimable"},
        "evidence_gate_version": 1,
    }
