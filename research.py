"""Research protocol checks and reproducibility helpers for MomentumAlpha."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from data_loader import ohlc_inconsistency_mask


def validate_weekly_panel(
    weekly_data: pd.DataFrame,
    tickers: list[str],
    *,
    universe_by_week: dict[pd.Timestamp, set[str]] | None = None,
) -> dict[str, int | float | str]:
    """Fail fast on malformed labels, duplicate rows, or impossible timestamps."""
    required = {
        "ticker", "target", "forward_return", "signal_date", "entry_date", "label_end_date"
    }
    missing = required - set(weekly_data.columns)
    if missing:
        raise ValueError(f"Weekly panel is missing required fields: {', '.join(sorted(missing))}")
    if weekly_data.empty or not tickers:
        raise ValueError("Weekly panel and ticker universe must be non-empty.")

    panel = weekly_data.copy()
    panel.index = pd.to_datetime(panel.index, errors="coerce")
    if panel.index.isna().any():
        raise ValueError("Weekly panel contains invalid signal-week index values.")
    duplicate_rows = panel.reset_index(names="signal_week").duplicated(
        ["signal_week", "ticker"], keep=False
    )
    if duplicate_rows.any():
        sample = panel.reset_index(names="signal_week").loc[duplicate_rows, ["signal_week", "ticker"]]
        raise ValueError(f"Weekly panel has duplicate (week, ticker) rows: {sample.head(5).to_dict('records')}")

    for column in ("signal_date", "entry_date", "label_end_date"):
        panel[column] = pd.to_datetime(panel[column], errors="coerce")
    panel["forward_return"] = pd.to_numeric(panel["forward_return"], errors="coerce")
    panel["target"] = pd.to_numeric(panel["target"], errors="coerce")
    observed_targets = set(panel["target"].dropna().unique())
    if not observed_targets.issubset({0, 1, 0.0, 1.0}):
        raise ValueError("Classification targets must contain only 0 and 1 values.")
    labeled = panel["forward_return"].notna() & panel["target"].notna()
    if not np.isfinite(panel.loc[labeled, "forward_return"]).all():
        raise ValueError("Completed weekly labels contain non-finite forward returns.")
    invalid_timeline = labeled & (
        panel["signal_date"].isna()
        | panel["entry_date"].isna()
        | panel["label_end_date"].isna()
        | (panel["signal_date"] >= panel["entry_date"])
        | (panel["entry_date"] >= panel["label_end_date"])
    )
    if invalid_timeline.any():
        raise ValueError("A completed weekly label has missing or impossible signal/entry/exit dates.")
    if (panel.loc[labeled, "forward_return"] <= -1).any():
        raise ValueError("A stock forward return is at or below -100%; inspect source prices and labels.")
    expected_target = panel.loc[labeled, "forward_return"].gt(0).astype(int)
    actual_target = panel.loc[labeled, "target"].astype(int)
    if not expected_target.equals(actual_target):
        raise ValueError("Some classification targets disagree with the sign of their forward return.")

    counts = panel.groupby(level=0)["ticker"].nunique()
    expected = set(tickers)
    incomplete = 0
    misaligned_timing = 0
    for week, group in panel.groupby(level=0):
        if universe_by_week is None:
            incomplete += int(set(group["ticker"].astype(str)) != expected)
        else:
            requested = universe_by_week.get(pd.Timestamp(week), set())
            incomplete += int(set(group["ticker"].astype(str)) != set(requested))
        completed_group = group.loc[group["forward_return"].notna() & group["target"].notna()]
        dates = completed_group[["signal_date", "entry_date", "label_end_date"]]
        if not dates.empty and any(dates[column].nunique(dropna=False) != 1 for column in dates.columns):
            misaligned_timing += 1
    return {
        "rows": int(len(panel)),
        "signal_weeks": int(panel.index.nunique()),
        "completed_labels": int(labeled.sum()),
        "candidate_features": int(max(0, len(panel.columns) - len(required))),
        "weeks_with_incomplete_universe": int(incomplete),
        "weeks_with_misaligned_execution_dates": int(misaligned_timing),
        "minimum_tickers_per_week": int(counts.min()),
        "maximum_tickers_per_week": int(counts.max()),
    }


def complete_evaluation_weeks(
    weekly_data: pd.DataFrame,
    tickers: list[str],
    *,
    after_date: str | pd.Timestamp,
    universe_by_week: dict[pd.Timestamp, set[str]] | None = None,
    min_universe_size: int | None = None,
) -> pd.DatetimeIndex:
    """Return fully labeled weeks with one synchronized execution window per universe."""
    cutoff = pd.Timestamp(after_date)
    expected = set(tickers)
    complete_weeks = []
    for week, group in weekly_data.groupby(level=0, sort=True):
        if pd.Timestamp(week) <= cutoff:
            continue
        if universe_by_week is None:
            week_expected = expected
        else:
            week_expected = set(universe_by_week.get(pd.Timestamp(week), set()))
            if len(week_expected) < (min_universe_size or 1):
                continue
        if set(group["ticker"].astype(str)) != week_expected:
            continue
        if group["ticker"].nunique() != len(week_expected):
            continue
        if group["forward_return"].isna().any() or group["target"].isna().any():
            continue
        if any(
            group[column].isna().any() or group[column].nunique() != 1
            for column in ("signal_date", "entry_date", "label_end_date")
        ):
            continue
        complete_weeks.append(pd.Timestamp(week))
    return pd.DatetimeIndex(complete_weeks).sort_values()


def build_market_data_quality_report(
    stock_data: dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize coverage and OHLC consistency against SPY trading sessions."""
    if spy_data.empty:
        raise ValueError("SPY data is empty; it cannot serve as the session calendar.")
    sessions = pd.DatetimeIndex(pd.to_datetime(spy_data.index)).tz_localize(None).normalize().unique().sort_values()
    rows: list[dict] = []
    frames = {**stock_data, "SPY": spy_data, "^VIX": vix_data}
    for ticker, source in frames.items():
        frame = source.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame = frame.sort_index()
        fields = [name for name in ("Open", "High", "Low", "Close") if name in frame]
        prices = frame[fields].apply(pd.to_numeric, errors="coerce")
        valid_prices = prices.gt(0).all(axis=1) & prices.notna().all(axis=1)
        if {"Open", "High", "Low", "Close"}.issubset(frame.columns):
            ohlc_consistent = ~ohlc_inconsistency_mask(prices)
        else:
            ohlc_consistent = pd.Series(True, index=frame.index)
        expected = sessions[(sessions >= frame.index.min()) & (sessions <= frame.index.max())] if len(frame) else sessions[:0]
        covered = expected.intersection(frame.index)
        missing_sessions = len(expected.difference(covered))
        rows.append({
            "ticker": ticker,
            "rows": int(len(frame)),
            "start_date": frame.index.min() if len(frame) else pd.NaT,
            "end_date": frame.index.max() if len(frame) else pd.NaT,
            "expected_spy_sessions": int(len(expected)),
            "covered_spy_sessions": int(len(covered)),
            "missing_spy_sessions": int(missing_sessions),
            "session_coverage": float(len(covered) / len(expected)) if len(expected) else None,
            "invalid_price_rows": int((~valid_prices).sum()) if len(frame) else 0,
            "ohlc_inconsistent_rows": int((~ohlc_consistent).sum()) if len(frame) else 0,
            "duplicate_dates": int(frame.index.duplicated().sum()),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_index()


def fingerprint_market_data(
    stock_data: dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
) -> str:
    """Return a stable SHA-256 fingerprint of the cleaned price inputs."""
    digest = hashlib.sha256()
    frames = [(ticker, stock_data[ticker]) for ticker in sorted(stock_data)]
    frames.extend((("SPY", spy_data), ("^VIX", vix_data)))
    for ticker, source in frames:
        frame = source.sort_index()
        digest.update(ticker.encode("utf-8"))
        digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
        hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
        digest.update(hashed.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()
