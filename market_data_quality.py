"""Auditable, minimal repairs for benchmark bars used in research features.

The public Dolt snapshot contains a small number of malformed SPY range bars.
MomentumAlpha uses SPY close for market features and open for its benchmark, so
the source open/close are retained while an invalid high/low envelope is
expanded only to the minimum interval consistent with the reported prices.
The untouched source export remains in the cache and each correction is
recorded in run metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def sanitize_benchmark_ohlc(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize SPY sessions and minimally repair malformed high/low bounds.

    Open and close drive execution and market features, so rows missing either
    are excluded. High and low are not used by this project's SPY features or
    SPY return benchmark; when their envelope is malformed, they are replaced
    with the smallest range enclosing the positive values reported for that
    bar. This is a bound repair, not recovery of the true intraday extremes.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Benchmark data must be a non-empty DataFrame.")

    data = frame.copy()
    if "date" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index("date")
    data.columns = [str(column).strip().title() for column in data.columns]
    missing = set(PRICE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Benchmark data is missing OHLC columns: {', '.join(sorted(missing))}.")

    dates = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))
    invalid_date = dates.isna()
    invalid_date_rows = int(invalid_date.sum())
    data = data.loc[~invalid_date].copy()
    dates = dates[~invalid_date]
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    data.index = dates.normalize()

    duplicate_dates_removed = int(data.index.duplicated(keep="last").sum())
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    for column in PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    valid_execution_prices = (
        data[["Open", "Close"]].notna().all(axis=1)
        & data[["Open", "Close"]].gt(0).all(axis=1)
    )
    invalid_execution_rows = int((~valid_execution_prices).sum())
    data = data.loc[valid_execution_prices].copy()
    if data.empty:
        raise ValueError("Benchmark data has no rows with valid positive open and close prices.")

    reported = data.loc[:, PRICE_COLUMNS].where(data.loc[:, PRICE_COLUMNS].gt(0))
    required_high = reported.max(axis=1)
    required_low = reported.min(axis=1)
    high = data["High"]
    low = data["Low"]
    malformed = (
        high.isna()
        | low.isna()
        | high.lt(required_high)
        | low.gt(required_low)
        | high.lt(low)
    )
    repair_examples: list[dict[str, Any]] = []
    for date, row in data.loc[malformed].head(20).iterrows():
        positive_values = row.loc[list(PRICE_COLUMNS)].where(row.loc[list(PRICE_COLUMNS)].gt(0)).dropna()
        repair_examples.append({
            "date": date.date().isoformat(),
            "source_high": None if pd.isna(row["High"]) else float(row["High"]),
            "source_low": None if pd.isna(row["Low"]) else float(row["Low"]),
            "minimum_valid_high": float(positive_values.max()),
            "minimum_valid_low": float(positive_values.min()),
        })
    if malformed.any():
        data.loc[malformed, "High"] = required_high.loc[malformed]
        data.loc[malformed, "Low"] = required_low.loc[malformed]

    invalid_after = (
        data["High"].isna()
        | data["Low"].isna()
        | data["High"].lt(data[["Open", "Low", "Close"]].max(axis=1))
        | data["Low"].gt(data[["Open", "High", "Close"]].min(axis=1))
    )
    if invalid_after.any():
        raise ValueError("Benchmark OHLC envelope remains invalid after conservative repair.")

    metadata = {
        "source_rows": int(len(frame)),
        "invalid_date_rows_removed": invalid_date_rows,
        "duplicate_dates_removed": duplicate_dates_removed,
        "invalid_open_close_rows_removed": invalid_execution_rows,
        "ohlc_envelope_repairs": int(malformed.sum()),
        "repair_method": (
            "For malformed range bars, set High/Low to the minimum/maximum positive reported OHLC values. "
            "This repairs only the envelope; actual intraday extremes are unknown."
        ),
        "repair_examples": repair_examples,
        "source_export_preserved": True,
    }
    return data, metadata
