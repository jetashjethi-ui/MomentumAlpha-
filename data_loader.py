"""Validated market-data loading and cache management for MomentumAlpha."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


STOCK_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MARKET_COLUMNS = ("Open", "High", "Low", "Close")


def ohlc_inconsistency_mask(prices: pd.DataFrame) -> pd.Series:
    """Flag material OHLC envelope violations, ignoring floating-point dust.

    Yahoo's adjusted OHLC fields are multiplied by an adjustment factor. When
    High or Low equals another field, independent floating-point operations
    can leave the values a few ulps apart. A scale-aware 1e-12 tolerance keeps
    those harmless artifacts from rejecting an otherwise valid history.
    """
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(prices.columns):
        raise ValueError("OHLC consistency checks require Open, High, Low, and Close.")

    numeric = prices.loc[:, ["Open", "High", "Low", "Close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = numeric.gt(0).all(axis=1) & numeric.notna().all(axis=1)
    scale = numeric.abs().max(axis=1).clip(lower=1.0)
    tolerance = scale * 1e-12
    required_high = numeric[["Open", "Low", "Close"]].max(axis=1)
    required_low = numeric[["Open", "High", "Close"]].min(axis=1)
    return valid & (
        (numeric["High"] + tolerance < required_high)
        | (numeric["Low"] - tolerance > required_low)
    )


def _clean_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    index = pd.to_datetime(frame.index, errors="coerce")
    valid = ~index.isna()
    frame = frame.loc[valid]
    index = index[valid]
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def normalize_yfinance_frame(
    frame: pd.DataFrame,
    ticker: str,
    *,
    require_volume: bool | None = None,
) -> pd.DataFrame:
    """Flatten yfinance's single- or multi-ticker columns into OHLCV fields."""
    if frame is None or frame.empty:
        raise ValueError(f"Yahoo Finance returned no rows for {ticker}.")

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        selected = None
        for level in range(data.columns.nlevels):
            if ticker in data.columns.get_level_values(level):
                selected = data.xs(ticker, axis=1, level=level, drop_level=True)
                break
        if selected is not None:
            data = selected
        else:
            flattened = []
            for column in data.columns:
                pieces = [str(piece) for piece in column if str(piece) != ticker]
                flattened.append(next((p for p in pieces if p in STOCK_COLUMNS + ("Adj Close",)), pieces[0] if pieces else str(column)))
            data.columns = flattened
    data.columns = [str(column) for column in data.columns]
    if len(set(data.columns)) != len(data.columns):
        data = data.loc[:, ~pd.Index(data.columns).duplicated(keep="first")]

    data = _clean_index(data)
    if require_volume is None:
        require_volume = ticker not in {"SPY", "^VIX"}
    required = STOCK_COLUMNS if require_volume else MARKET_COLUMNS
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{ticker} data is missing required columns: {', '.join(missing)}")

    data = data.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    price_columns = data[["Open", "High", "Low", "Close"]]
    valid_prices = price_columns.gt(0).all(axis=1) & price_columns.notna().all(axis=1)
    required_high = price_columns[["Open", "Low", "Close"]].max(axis=1)
    required_low = price_columns[["Open", "High", "Close"]].min(axis=1)
    scale = price_columns.abs().max(axis=1).clip(lower=1.0)
    tolerance = scale * 1e-12
    inconsistent_ohlc = ohlc_inconsistency_mask(price_columns)
    if inconsistent_ohlc.any():
        bad_dates = data.index[inconsistent_ohlc].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"{ticker} has inconsistent OHLC prices on {bad_dates}.")

    # Snap only sub-tolerance rounding artifacts to the exact OHLC envelope.
    # Materially inconsistent bars still fail above and are never rewritten.
    high_roundoff = valid_prices & data["High"].lt(required_high) & (required_high - data["High"]).le(tolerance)
    low_roundoff = valid_prices & data["Low"].gt(required_low) & (data["Low"] - required_low).le(tolerance)
    data.loc[high_roundoff, "High"] = required_high.loc[high_roundoff]
    data.loc[low_roundoff, "Low"] = required_low.loc[low_roundoff]
    data = data.loc[valid_prices]
    if require_volume:
        data = data.loc[data["Volume"].fillna(0).gt(0)]
    if data.empty:
        raise ValueError(f"{ticker} has no valid OHLC rows after cleaning.")
    return data


def _as_date(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    date = pd.Timestamp(value)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    return date.normalize()


def _valid_cache(
    cached: Any,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame] | None:
    if not isinstance(cached, dict):
        return None
    stock_data = cached.get("stock_data")
    spy_data = cached.get("spy_data")
    vix_data = cached.get("vix_data")
    if not isinstance(stock_data, dict) or not isinstance(spy_data, pd.DataFrame) or not isinstance(vix_data, pd.DataFrame):
        return None
    if any(ticker not in stock_data for ticker in tickers):
        return None

    try:
        cleaned_stocks = {ticker: normalize_yfinance_frame(stock_data[ticker], ticker) for ticker in tickers}
        spy = normalize_yfinance_frame(spy_data, "SPY", require_volume=False)
        vix = normalize_yfinance_frame(vix_data, "^VIX", require_volume=False)
    except (KeyError, TypeError, ValueError):
        return None

    ends = [frame.index.max() for frame in [*cleaned_stocks.values(), spy, vix]]
    # Some symbols (for example META) began trading after the requested start.
    # Validate requested-start coverage against the market calendars, while
    # allowing a stock's legitimate IPO date to be later than that date.
    if max(spy.index.min(), vix.index.min()) > start + pd.Timedelta(days=7):
        return None
    latest = min(ends)
    if end is not None:
        if latest < end - pd.Timedelta(days=7):
            return None
    else:
        freshest_allowed = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        if latest < freshest_allowed:
            return None
    def restrict(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame.index >= start
        if end is not None:
            mask &= frame.index < end
        return frame.loc[mask]

    cleaned_stocks = {ticker: restrict(frame) for ticker, frame in cleaned_stocks.items()}
    spy = restrict(spy)
    vix = restrict(vix)
    if any(frame.empty for frame in [*cleaned_stocks.values(), spy, vix]):
        return None
    return cleaned_stocks, spy, vix


def _read_cache_archive(path: Path) -> dict[str, Any]:
    """Read the CSV/JSON cache format without deserializing executable objects."""
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported market-data cache schema.")

        def read_frame(member: str) -> pd.DataFrame:
            return pd.read_csv(BytesIO(archive.read(member)), index_col=0, parse_dates=[0])

        return {
            "stock_data": {
                ticker: read_frame(member)
                for ticker, member in manifest["stock_files"].items()
            },
            "spy_data": read_frame(manifest["spy_file"]),
            "vix_data": read_frame(manifest["vix_file"]),
        }


def _download_one(ticker: str, start: str, end: str | None, *, auto_adjust: bool) -> pd.DataFrame:
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                timeout=30,
            )
            return normalize_yfinance_frame(
                frame,
                ticker,
                require_volume=ticker not in {"SPY", "^VIX"},
            )
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"{ticker} failed after 3 download attempts: {last_error}") from last_error


def _atomic_cache_archive(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip.tmp", delete=False) as temporary:
            temporary_name = temporary.name
        stock_files = {ticker: f"stocks/{index}.csv" for index, ticker in enumerate(value["stock_data"])}
        manifest = {
            "schema_version": 1,
            "stock_files": stock_files,
            "spy_file": "SPY.csv",
            "vix_file": "VIX.csv",
            **value["metadata"],
        }
        with ZipFile(temporary_name, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
            for ticker, member in stock_files.items():
                archive.writestr(member, value["stock_data"][ticker].to_csv(index_label="Date"))
            archive.writestr("SPY.csv", value["spy_data"].to_csv(index_label="Date"))
            archive.writestr("VIX.csv", value["vix_data"].to_csv(index_label="Date"))
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_market_data(
    tickers: list[str],
    start_date: str = "2017-01-01",
    end_date: str | None = None,
    *,
    cache_file: str | Path = ".cache/market_data.zip",
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load adjusted equity prices and reported volume plus SPY and VIX with checked cache coverage.

    ``end_date`` follows yfinance's exclusive end-date convention. When omitted,
    the latest available market data is requested and stale caches are refreshed.
    """
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be a non-empty list of unique symbols.")
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start is None or (end is not None and end <= start):
        raise ValueError("end_date must be later than start_date.")

    cache_path = Path(cache_file)
    if cache_path.exists() and not refresh:
        try:
            cached = _read_cache_archive(cache_path)
            usable = _valid_cache(cached, tickers, start, end)
            if usable is not None:
                print(f"Using validated market-data cache: {cache_path}")
                return usable
            print("Market-data cache is incomplete or stale; refreshing it.")
        except Exception as exc:  # Corrupt caches should not silently stop a run.
            print(f"Market-data cache could not be read ({exc}); downloading fresh data.")

    stock_data: dict[str, pd.DataFrame] = {}
    request_start = start.strftime("%Y-%m-%d")
    request_end = end.strftime("%Y-%m-%d") if end is not None else None
    failures: list[str] = []
    symbols = [*tickers, "SPY", "^VIX"]
    print(f"Downloading {len(symbols)} symbols from Yahoo Finance...")
    for ticker in symbols:
        try:
            frame = _download_one(ticker, request_start, request_end, auto_adjust=ticker != "^VIX")
            if ticker in tickers:
                stock_data[ticker] = frame
            elif ticker == "SPY":
                spy_data = frame
            else:
                vix_data = frame
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(f"Market data download failed; no partial run was started:\n  - {details}")

    if set(stock_data) != set(tickers):
        raise RuntimeError("The downloaded stock universe is incomplete.")

    def restrict(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame.index >= start
        if end is not None:
            mask &= frame.index < end
        return frame.loc[mask]

    stock_data = {ticker: restrict(frame) for ticker, frame in stock_data.items()}
    spy_data = restrict(spy_data)
    vix_data = restrict(vix_data)
    if any(frame.empty for frame in [*stock_data.values(), spy_data, vix_data]):
        raise RuntimeError("Downloaded market data does not cover the requested date range.")

    metadata = {
        "provider": "Yahoo Finance via yfinance",
        "start_date": request_start,
        "end_date_exclusive": request_end,
        "equity_prices_auto_adjusted": True,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_cache_archive(
        {"stock_data": stock_data, "spy_data": spy_data, "vix_data": vix_data, "metadata": metadata},
        cache_path,
    )
    print(f"Saved validated market data to {cache_path}")
    return stock_data, spy_data, vix_data
