"""Prepare a broad, point-in-time-ish U.S. common-stock panel from DoltHub.

The community source is useful for research breadth and includes inactive
symbols, but it is not a CRSP/Norgate substitute: its identifiers and issuer
metadata are not guaranteed to be permanent or historically point-in-time.
Raw and derived data stay under the ignored .cache directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


SOURCE_URL = "https://www.dolthub.com/repositories/post-no-preference/stocks"
SOURCE_ATTRIBUTION = "post-no-preference/stocks on DoltHub; accessed as a public database snapshot."
KNOWN_LIMITATIONS = [
    "Community-maintained daily data, not a licensed institutional point-in-time feed.",
    "The source starts in 2011 and omits earlier delisted history; it is not fully survivor-bias-free.",
    "Ticker-based source keys are not permanent security identifiers; long ticker gaps are split into eras, but some reuses may remain unresolved.",
    "Historical issuer metadata and index membership are not reconstructed; the universe uses contemporaneous liquidity and source listing flags.",
    "Data may contain vendor errors or revisions; all derived results are historical research, not live-trading evidence.",
]


@dataclass
class DoltMarketData:
    stock_data: dict[str, pd.DataFrame]
    spy_data: pd.DataFrame
    dividends: dict[str, pd.Series]
    spy_dividends: pd.Series
    universe_by_week: dict[pd.Timestamp, set[str]]
    metadata: dict[str, Any]


def _safe_sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _run_dolt(executable: Path, database_dir: Path, arguments: list[str]) -> subprocess.CompletedProcess:
    if not executable.is_file():
        raise FileNotFoundError(f"Dolt executable was not found: {executable}")
    if not database_dir.is_dir():
        raise FileNotFoundError(
            f"Dolt stock snapshot was not found: {database_dir}. Clone post-no-preference/stocks into .cache/datasets/stocks."
        )
    result = subprocess.run(
        [str(executable), "-C", str(database_dir), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Dolt command failed.")
    return result


def _snapshot_revision(executable: Path, database_dir: Path) -> str:
    result = _run_dolt(executable, database_dir, ["log", "-n", "1", "--oneline"])
    first_line = result.stdout.strip().splitlines()[0]
    return first_line.split()[0]


def _export_parquet(
    executable: Path,
    database_dir: Path,
    query: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output:
        result = subprocess.run(
            [
                str(executable), "-C", str(database_dir), "sql", "-q", query,
                "-r", "parquet",
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        temporary.unlink(missing_ok=True)
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "Dolt could not export the public stock data.")
    with temporary.open("rb") as stream:
        if stream.read(4) != b"PAR1":
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Dolt returned an invalid Parquet export.")
    os.replace(temporary, destination)


def _materialize_source(
    *,
    executable: Path,
    database_dir: Path,
    cache_dir: Path,
    refresh: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = cache_dir / "source_manifest.json"
    raw_paths = {
        "stocks": cache_dir / "eligible_common_stock_ohlcv.parquet",
        "splits": cache_dir / "split_events.parquet",
        "dividends": cache_dir / "dividend_events.parquet",
        "spy": cache_dir / "spy_ohlcv.parquet",
    }
    existing: dict[str, Any] = {}
    if source_manifest_path.is_file():
        try:
            existing = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}

    can_read_revision = executable.is_file() and database_dir.is_dir()
    revision = _snapshot_revision(executable, database_dir) if can_read_revision else existing.get("source_revision")
    complete = all(path.is_file() and path.stat().st_size > 4 for path in raw_paths.values())
    snapshot_changed = revision is not None and existing.get("source_revision") != revision
    if complete and not refresh and not snapshot_changed:
        return existing
    if not can_read_revision:
        raise FileNotFoundError(
            "The cached source export is missing or stale and Dolt is unavailable. "
            "Keep the downloaded .cache/datasets/stocks snapshot and .cache/tools/dolt folder, or install Dolt and clone the source."
        )

    stock_query = """
        SELECT o.date, o.act_symbol, o.open, o.high, o.low, o.close, o.volume
        FROM ohlcv AS o
        INNER JOIN symbol AS s ON s.act_symbol = o.act_symbol
        WHERE s.is_etf = 0
          AND s.is_test_issue = 0
          AND s.listing_exchange IN ('NASDAQ', 'NYSE', 'NYSE MKT')
          AND (s.security_name LIKE '%Common Stock%' OR s.security_name LIKE '%Ordinary Share%')
    """.strip()
    split_query = "SELECT act_symbol, ex_date, to_factor, for_factor FROM split"
    dividend_query = "SELECT act_symbol, ex_date, amount FROM dividend"
    spy_query = "SELECT date, act_symbol, open, high, low, close, volume FROM ohlcv WHERE act_symbol = 'SPY'"

    print("Exporting 2011+ common-stock bars and corporate actions from the local public snapshot...")
    for key, query in (
        ("stocks", stock_query),
        ("splits", split_query),
        ("dividends", dividend_query),
        ("spy", spy_query),
    ):
        if refresh or snapshot_changed or not raw_paths[key].is_file():
            _export_parquet(executable, database_dir, query, raw_paths[key])

    manifest = {
        "provider": "DoltHub public community database",
        "source_url": SOURCE_URL,
        "source_attribution": SOURCE_ATTRIBUTION,
        "source_revision": revision,
        "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tables": ["ohlcv", "symbol", "split", "dividend"],
        "raw_exports": {key: str(path.relative_to(cache_dir)) for key, path in raw_paths.items()},
        "limitations": KNOWN_LIMITATIONS,
        "redistribution_note": "Raw data is not copied into the tracked project files; confirm the source terms before redistributing a bundle containing market data.",
    }
    temporary_manifest = source_manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, source_manifest_path)
    return manifest


def _duckdb_connection(cache_dir: Path):
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Dolt data preparation requires DuckDB. Install the project requirements first.") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(cache_dir / "momentumalpha_research.duckdb"))
    connection.execute("SET memory_limit = '4GB'")
    connection.execute("SET threads = 4")
    connection.execute(f"SET temp_directory = '{_safe_sql_path(cache_dir / 'duckdb_tmp')}'")
    return connection


def _prepare_research_tables(
    connection,
    *,
    cache_dir: Path,
    source_manifest: dict[str, Any],
    universe_size: int,
    min_price: float,
    min_dollar_volume: float,
    min_history_bars: int,
    gap_days: int,
    refresh: bool,
) -> None:
    config = {
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": source_manifest.get("source_revision"),
        "universe_size": int(universe_size),
        "min_price": float(min_price),
        "min_dollar_volume": float(min_dollar_volume),
        "min_history_bars": int(min_history_bars),
        "gap_days": int(gap_days),
    }
    config_path = cache_dir / "prepared_config.json"
    roster_path = cache_dir / "weekly_universe.parquet"
    bars_path = cache_dir / "selected_adjusted_bars.parquet"
    event_path = cache_dir / "selected_dividends.parquet"
    metadata_path = cache_dir / "prepared_data_manifest.json"
    if not refresh and all(path.is_file() and path.stat().st_size > 4 for path in (roster_path, bars_path, event_path)):
        try:
            if json.loads(config_path.read_text(encoding="utf-8")) == config:
                return
        except (OSError, json.JSONDecodeError):
            pass

    source_dir = Path(cache_dir)
    raw_stocks = _safe_sql_path(source_dir / "eligible_common_stock_ohlcv.parquet")
    raw_splits = _safe_sql_path(source_dir / "split_events.parquet")
    raw_dividends = _safe_sql_path(source_dir / "dividend_events.parquet")
    raw_spy = _safe_sql_path(source_dir / "spy_ohlcv.parquet")
    connection.execute("DROP TABLE IF EXISTS ma_epoched")
    connection.execute("DROP TABLE IF EXISTS ma_split_events")
    connection.execute("DROP TABLE IF EXISTS ma_split_future")
    connection.execute("DROP TABLE IF EXISTS ma_adjusted_bars")
    connection.execute("DROP TABLE IF EXISTS ma_unresolved_jumps")
    connection.execute("DROP TABLE IF EXISTS ma_research_bars")
    connection.execute("DROP TABLE IF EXISTS ma_roster")
    connection.execute("DROP TABLE IF EXISTS ma_selected_bars")
    connection.execute("DROP TABLE IF EXISTS ma_selected_dividends")

    print("Building ticker eras, split-adjusted prices, and the dynamic liquidity universe...")
    connection.execute(f"""
        CREATE TEMP TABLE ma_epoched AS
        WITH casted AS (
            SELECT CAST(date AS DATE) AS date, CAST(act_symbol AS VARCHAR) AS act_symbol,
                   CAST(open AS DOUBLE) AS open, CAST(high AS DOUBLE) AS high,
                   CAST(low AS DOUBLE) AS low, CAST(close AS DOUBLE) AS close,
                   CAST(volume AS DOUBLE) AS volume
            FROM read_parquet('{raw_stocks}')
        ), lagged AS (
            SELECT *, LAG(date) OVER (PARTITION BY act_symbol ORDER BY date) AS previous_date
            FROM casted
        ), marked AS (
            SELECT *, CASE WHEN previous_date IS NULL OR date - previous_date > {int(gap_days)}
                           THEN 1 ELSE 0 END AS new_epoch
            FROM lagged
        ), numbered AS (
            SELECT *, SUM(new_epoch) OVER (
                PARTITION BY act_symbol ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS epoch_number
            FROM marked
        ), ranges AS (
            SELECT act_symbol, epoch_number, MIN(date) AS epoch_start, MAX(date) AS epoch_end
            FROM numbered GROUP BY act_symbol, epoch_number
        )
        SELECT n.date, n.act_symbol, n.open, n.high, n.low, n.close, n.volume,
               n.epoch_number, r.epoch_start, r.epoch_end,
               n.act_symbol || '@' || STRFTIME(r.epoch_start, '%Y%m%d') AS security_id
        FROM numbered n JOIN ranges r USING (act_symbol, epoch_number)
    """)

    connection.execute(f"""
        CREATE TEMP TABLE ma_split_events AS
        WITH split_values AS (
            SELECT CAST(act_symbol AS VARCHAR) AS act_symbol, CAST(ex_date AS DATE) AS ex_date,
                   CAST(to_factor AS DOUBLE) / NULLIF(CAST(for_factor AS DOUBLE), 0) AS split_ratio
            FROM read_parquet('{raw_splits}')
            WHERE to_factor > 0 AND for_factor > 0
        ), mapped AS (
            SELECT s.act_symbol, s.ex_date, s.split_ratio, e.security_id
            FROM split_values s JOIN (
                SELECT DISTINCT security_id, act_symbol, epoch_start, epoch_end FROM ma_epoched
            ) e ON e.act_symbol = s.act_symbol AND s.ex_date >= e.epoch_start AND s.ex_date <= e.epoch_end
            WHERE ABS(s.split_ratio - 1.0) > 1e-12
        ), checked AS (
            SELECT m.*, p.date AS previous_bar_date, p.close AS previous_close,
                   n.date AS action_bar_date, n.close AS action_close
            FROM mapped m
            ASOF LEFT JOIN ma_epoched p
              ON m.security_id = p.security_id AND m.ex_date > p.date
            ASOF LEFT JOIN ma_epoched n
              ON m.security_id = n.security_id AND m.ex_date <= n.date
        )
        SELECT security_id, ex_date, LN(split_ratio) AS log_ratio
        FROM checked
        WHERE previous_close > 0 AND action_close > 0
          AND action_bar_date <= ex_date + INTERVAL 5 DAY
          AND ABS(LN((action_close / previous_close) / (1.0 / split_ratio))) <= 0.35
    """)
    connection.execute("""
        CREATE TEMP TABLE ma_split_future AS
        SELECT security_id, ex_date,
               EXP(SUM(log_ratio) OVER (
                   PARTITION BY security_id ORDER BY ex_date DESC
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )) AS future_split_factor
        FROM ma_split_events
    """)

    connection.execute("""
        CREATE TEMP TABLE ma_adjusted_bars AS
        SELECT b.date, b.act_symbol, b.security_id, b.close AS raw_close,
               b.open / COALESCE(s.future_split_factor, 1.0) AS "Open",
               b.high / COALESCE(s.future_split_factor, 1.0) AS "High",
               b.low / COALESCE(s.future_split_factor, 1.0) AS "Low",
               b.close / COALESCE(s.future_split_factor, 1.0) AS "Close",
               b.volume * COALESCE(s.future_split_factor, 1.0) AS "Volume"
        FROM ma_epoched b
        ASOF LEFT JOIN ma_split_future s
          ON b.security_id = s.security_id AND b.date < s.ex_date
        WHERE b.open > 0 AND b.high > 0 AND b.low > 0 AND b.close > 0 AND b.volume > 0
          AND b.high >= GREATEST(b.open, b.low, b.close)
          AND b.low <= LEAST(b.open, b.high, b.close)
    """)

    # Reject security eras with extreme daily discontinuities that remain after
    # split events have been corroborated against the source prices. This
    # conservative quarantine prevents obvious corporate-action/vendor errors
    # from becoming labels or rolling features.
    connection.execute("""
        CREATE TEMP TABLE ma_unresolved_jumps AS
        WITH returns AS (
            SELECT security_id, date, "Close" AS close,
                   LAG("Close") OVER (PARTITION BY security_id ORDER BY date) AS previous_close
            FROM ma_adjusted_bars
        )
        SELECT security_id, date, close / previous_close AS price_ratio
        FROM returns
        WHERE previous_close > 0 AND (close / previous_close > 5.0 OR close / previous_close < 0.2)
    """)
    connection.execute("""
        CREATE TEMP TABLE ma_research_bars AS
        SELECT b.* FROM ma_adjusted_bars b
        ANTI JOIN (SELECT DISTINCT security_id FROM ma_unresolved_jumps) q
          ON b.security_id = q.security_id
    """)

    connection.execute(f"""
        CREATE TEMP TABLE ma_roster AS
        WITH measured AS (
            SELECT *,
                   AVG("Close" * "Volume") OVER (
                       PARTITION BY security_id ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
                   ) AS adv_60d,
                   COUNT(*) OVER (
                       PARTITION BY security_id ORDER BY date ROWS BETWEEN {int(min_history_bars)-1} PRECEDING AND CURRENT ROW
                   ) AS history_bars
            FROM ma_research_bars
        ),
        weekly AS (
            SELECT *, CAST(DATE_TRUNC('week', date) AS DATE) + INTERVAL 4 DAY AS signal_week,
                   ROW_NUMBER() OVER (PARTITION BY security_id, DATE_TRUNC('week', date) ORDER BY date DESC) AS last_bar_of_week
            FROM measured
        ),
        eligible AS (
            SELECT * FROM weekly
            WHERE last_bar_of_week = 1 AND history_bars >= {int(min_history_bars)}
              AND raw_close >= {float(min_price)} AND adv_60d >= {float(min_dollar_volume)}
        ),
        ranked AS (
            SELECT signal_week, date AS signal_date, security_id AS ticker, adv_60d,
                   ROW_NUMBER() OVER (PARTITION BY signal_week ORDER BY adv_60d DESC, security_id) AS liquidity_rank
            FROM eligible
        )
        SELECT signal_week, signal_date, ticker, adv_60d, liquidity_rank
        FROM ranked WHERE liquidity_rank <= {int(universe_size)}
    """)
    roster_count = int(connection.execute("SELECT COUNT(*) FROM ma_roster").fetchone()[0])
    if roster_count == 0:
        raise RuntimeError("No weekly candidates met the requested liquidity/history rules.")

    connection.execute("""
        CREATE TEMP TABLE ma_selected_bars AS
        SELECT b.* FROM ma_research_bars b
        SEMI JOIN (SELECT DISTINCT ticker FROM ma_roster) r ON b.security_id = r.ticker
    """)
    connection.execute(f"""
        CREATE TEMP TABLE ma_selected_dividends AS
        WITH dividend_values AS (
            SELECT CAST(act_symbol AS VARCHAR) AS act_symbol, CAST(ex_date AS DATE) AS ex_date,
                   CAST(amount AS DOUBLE) AS amount
            FROM read_parquet('{raw_dividends}')
            WHERE amount > 0
        ),
        mapped AS (
            SELECT d.act_symbol, d.ex_date, d.amount, e.security_id
            FROM dividend_values d JOIN (
                SELECT DISTINCT security_id, act_symbol, epoch_start, epoch_end FROM ma_epoched
            ) e ON e.act_symbol = d.act_symbol AND d.ex_date >= e.epoch_start AND d.ex_date <= e.epoch_end
            SEMI JOIN (SELECT DISTINCT ticker FROM ma_roster) r ON e.security_id = r.ticker
        )
        SELECT d.security_id AS ticker, d.ex_date,
               d.amount / COALESCE(s.future_split_factor, 1.0) AS amount
        FROM mapped d
        ASOF LEFT JOIN ma_split_future s
          ON d.security_id = s.security_id AND d.ex_date < s.ex_date
    """)

    connection.execute(f"""
        COPY (SELECT * FROM ma_roster ORDER BY signal_week, liquidity_rank)
        TO '{_safe_sql_path(roster_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.execute(f"""
        COPY (SELECT * FROM ma_selected_bars ORDER BY security_id, date)
        TO '{_safe_sql_path(bars_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.execute(f"""
        COPY (SELECT * FROM ma_selected_dividends ORDER BY ticker, ex_date)
        TO '{_safe_sql_path(event_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    raw_count = int(connection.execute("SELECT COUNT(*) FROM ma_epoched").fetchone()[0])
    security_count = int(connection.execute("SELECT COUNT(DISTINCT ticker) FROM ma_roster").fetchone()[0])
    week_count = int(connection.execute("SELECT COUNT(DISTINCT signal_week) FROM ma_roster").fetchone()[0])
    invalid_rows = int(connection.execute("SELECT COUNT(*) FROM ma_epoched") .fetchone()[0] - connection.execute("SELECT COUNT(*) FROM ma_adjusted_bars").fetchone()[0])
    quarantined_eras = int(connection.execute("SELECT COUNT(DISTINCT security_id) FROM ma_unresolved_jumps").fetchone()[0])
    unresolved_jump_rows = int(connection.execute("SELECT COUNT(*) FROM ma_unresolved_jumps").fetchone()[0])
    metadata = {
        "provider": "DoltHub public community database",
        "source_url": SOURCE_URL,
        "source_attribution": SOURCE_ATTRIBUTION,
        "source_revision": source_manifest.get("source_revision"),
        "universe_policy": (
            f"At each Friday signal, rank common stocks by trailing 60-trading-bar average dollar volume; "
            f"retain up to {universe_size} with contemporaneous raw close >= {min_price:g}, "
            f"ADV >= ${min_dollar_volume:,.0f}, and >= {min_history_bars} bars."
        ),
        "security_identity": f"Ticker eras are split after gaps longer than {gap_days} calendar days; the source has no permanent issue identifier.",
        "rows_in_source_filter": raw_count,
        "selected_security_eras": security_count,
        "weekly_roster_rows": roster_count,
        "weekly_signal_dates": week_count,
        "rows_removed_for_invalid_ohlcv": invalid_rows,
        "quarantined_security_eras_for_unresolved_extreme_jumps": quarantined_eras,
        "unresolved_extreme_jump_rows": unresolved_jump_rows,
        "extreme_jump_quarantine_rule": "Exclude an entire ticker era if any split-adjusted close-to-close move remains above 5x or below 0.2x after source split validation.",
        "source_split_events": int(connection.execute(f"SELECT COUNT(*) FROM read_parquet('{raw_splits}') WHERE to_factor > 0 AND for_factor > 0 AND ABS(CAST(to_factor AS DOUBLE) / CAST(for_factor AS DOUBLE) - 1.0) > 1e-12").fetchone()[0]),
        "split_events_confirmed_by_raw_price_jump": int(connection.execute("SELECT COUNT(*) FROM ma_split_events").fetchone()[0]),
        "split_events_rejected_as_unconfirmed_or_misaligned": int(connection.execute(f"SELECT COUNT(*) FROM read_parquet('{raw_splits}') WHERE to_factor > 0 AND for_factor > 0 AND ABS(CAST(to_factor AS DOUBLE) / CAST(for_factor AS DOUBLE) - 1.0) > 1e-12").fetchone()[0] - connection.execute("SELECT COUNT(*) FROM ma_split_events").fetchone()[0]),
        "split_validation_rule": "A source split is applied only when the first raw close on/after its ex-date is within five calendar days and its price ratio corroborates the announced split factor within a log-ratio tolerance of 0.35.",
        "prices": "Historical OHLC and volume are adjusted for subsequent stock splits. Cash dividends are retained separately and added to open-to-open return labels.",
        "benchmark": "SPY daily OHLC from the same source; its total return includes source cash dividends.",
        "limitations": KNOWN_LIMITATIONS,
        "redistribution_note": source_manifest.get("redistribution_note"),
        "cache_fingerprint_sha256": _fingerprint_files((roster_path, bars_path, event_path)),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_config = config_path.with_suffix(".json.tmp")
    temporary_config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    os.replace(temporary_config, config_path)


def _fingerprint_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_frames(connection, query: str, *, ticker_column: str | None = None) -> dict[str, pd.DataFrame] | pd.DataFrame:
    cursor = connection.execute(query)
    if ticker_column is None:
        return cursor.fetchdf()
    grouped: dict[str, list[pd.DataFrame]] = {}
    while True:
        chunk = cursor.fetch_df_chunk(1)
        if chunk is None or chunk.empty:
            break
        for ticker, rows in chunk.groupby(ticker_column, sort=False):
            grouped.setdefault(str(ticker), []).append(rows.drop(columns=[ticker_column]))
    return {
        ticker: pd.concat(parts, ignore_index=True).set_index("date").sort_index()
        for ticker, parts in grouped.items()
    }


def _read_dividends(connection, path: Path) -> dict[str, pd.Series]:
    rows = connection.execute(f"SELECT ticker, ex_date, amount FROM read_parquet('{_safe_sql_path(path)}') ORDER BY ticker, ex_date").fetchdf()
    if rows.empty:
        return {}
    return {
        str(ticker): pd.Series(group["amount"].to_numpy(dtype=float), index=pd.DatetimeIndex(group["ex_date"]), name="amount")
        for ticker, group in rows.groupby("ticker", sort=False)
    }


def load_dolt_market_data(
    *,
    project_dir: str | Path,
    universe_size: int = 50,
    min_price: float = 5.0,
    min_dollar_volume: float = 5_000_000.0,
    min_history_bars: int = 252,
    gap_days: int = 180,
    refresh: bool = False,
    dolt_dir: str | Path | None = None,
    dolt_executable: str | Path | None = None,
) -> DoltMarketData:
    """Return split-adjusted bars, dividends, SPY, and a weekly liquidity roster."""
    if universe_size < 2 or min_history_bars < 60 or min_price <= 0 or min_dollar_volume < 0 or gap_days < 30:
        raise ValueError("Invalid Dolt universe or data-quality thresholds.")
    root = Path(project_dir).resolve()
    cache_dir = root / ".cache" / "dolt_materialized"
    database_dir = Path(dolt_dir).resolve() if dolt_dir else root / ".cache" / "datasets" / "stocks"
    executable = Path(dolt_executable).resolve() if dolt_executable else root / ".cache" / "tools" / "dolt" / "dolt-windows-amd64" / "bin" / "dolt.exe"

    source_manifest = _materialize_source(
        executable=executable,
        database_dir=database_dir,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    connection = _duckdb_connection(cache_dir)
    try:
        _prepare_research_tables(
            connection,
            cache_dir=cache_dir,
            source_manifest=source_manifest,
            universe_size=universe_size,
            min_price=min_price,
            min_dollar_volume=min_dollar_volume,
            min_history_bars=min_history_bars,
            gap_days=gap_days,
            refresh=refresh,
        )
        bars_path = cache_dir / "selected_adjusted_bars.parquet"
        roster_path = cache_dir / "weekly_universe.parquet"
        dividend_path = cache_dir / "selected_dividends.parquet"
        stock_data = _read_frames(
            connection,
            f"SELECT date, security_id AS ticker, \"Open\", \"High\", \"Low\", \"Close\", \"Volume\" FROM read_parquet('{_safe_sql_path(bars_path)}') ORDER BY security_id, date",
            ticker_column="ticker",
        )
        # DataFrame and feature code expect clean, timezone-naive market sessions.
        for ticker, frame in stock_data.items():
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
            frame.columns.name = None
        spy_data = connection.execute(
            f"""
            WITH raw AS (
                SELECT CAST(date AS DATE) AS date, CAST(open AS DOUBLE) AS open,
                       CAST(high AS DOUBLE) AS high, CAST(low AS DOUBLE) AS low,
                       CAST(close AS DOUBLE) AS close
                FROM read_parquet('{_safe_sql_path(cache_dir / 'spy_ohlcv.parquet')}')
            ), split_events AS (
                SELECT CAST(ex_date AS DATE) AS ex_date,
                       EXP(SUM(LN(CAST(to_factor AS DOUBLE) / NULLIF(CAST(for_factor AS DOUBLE),0))) OVER (
                           ORDER BY CAST(ex_date AS DATE) DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                       )) AS future_split_factor
                FROM read_parquet('{_safe_sql_path(cache_dir / 'split_events.parquet')}')
                WHERE act_symbol='SPY' AND to_factor > 0 AND for_factor > 0
            )
            SELECT r.date,
                   r.open / COALESCE(s.future_split_factor,1.0) AS "Open",
                   r.high / COALESCE(s.future_split_factor,1.0) AS "High",
                   r.low / COALESCE(s.future_split_factor,1.0) AS "Low",
                   r.close / COALESCE(s.future_split_factor,1.0) AS "Close"
            FROM raw r ASOF LEFT JOIN split_events s ON r.date < s.ex_date
            ORDER BY r.date
            """
        ).fetchdf()
        spy_data = spy_data.set_index("date").sort_index()
        spy_data.index = pd.DatetimeIndex(pd.to_datetime(spy_data.index)).tz_localize(None).normalize()
        dividends = _read_dividends(connection, dividend_path)
        spy_events = connection.execute(
            f"SELECT ex_date, amount FROM read_parquet('{_safe_sql_path(cache_dir / 'dividend_events.parquet')}') WHERE act_symbol='SPY' AND amount > 0 ORDER BY ex_date"
        ).fetchdf()
        spy_dividends = pd.Series(
            spy_events["amount"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(spy_events["ex_date"]),
            name="amount",
        ) if not spy_events.empty else pd.Series(dtype=float, name="amount")
        roster = connection.execute(
            f"SELECT signal_week, ticker FROM read_parquet('{_safe_sql_path(roster_path)}') ORDER BY signal_week, liquidity_rank"
        ).fetchdf()
        roster["signal_week"] = pd.to_datetime(roster["signal_week"]).dt.tz_localize(None).dt.normalize()
        universe_by_week = {
            pd.Timestamp(week): set(group["ticker"].astype(str))
            for week, group in roster.groupby("signal_week", sort=True)
        }
        metadata_path = cache_dir / "prepared_data_manifest.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({
            "cache_directory": str(cache_dir),
            "stock_data_start": min((frame.index.min() for frame in stock_data.values() if not frame.empty), default=None),
            "stock_data_end": max((frame.index.max() for frame in stock_data.values() if not frame.empty), default=None),
            "loaded_symbols": len(stock_data),
            "market_data_fingerprint_sha256": _fingerprint_frames(stock_data, spy_data),
        })
        return DoltMarketData(
            stock_data=stock_data,
            spy_data=spy_data,
            dividends=dividends,
            spy_dividends=spy_dividends,
            universe_by_week=universe_by_week,
            metadata=metadata,
        )
    finally:
        connection.close()


def _fingerprint_frames(stock_data: dict[str, pd.DataFrame], spy_data: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for ticker, frame in sorted(stock_data.items()):
        digest.update(ticker.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype=np.uint64).tobytes())
    digest.update(pd.util.hash_pandas_object(spy_data, index=True, categorize=True).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()
