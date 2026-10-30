"""Persistent experiment history and final-holdout reuse protection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd


class HoldoutReuseError(ValueError):
    """Raised when a requested final holdout overlaps already-inspected data."""


class ExperimentLedger:
    """SQLite ledger for completed runs and explicitly consumed holdout windows."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)
        self.path = self.output_root / "experiment_ledger.sqlite3"
        self.output_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    experiment_fingerprint TEXT NOT NULL,
                    family_fingerprint TEXT NOT NULL,
                    data_fingerprint TEXT,
                    source_fingerprint TEXT,
                    evaluation_mode TEXT NOT NULL,
                    holdout_start TEXT,
                    holdout_end TEXT,
                    holdout_weeks INTEGER NOT NULL DEFAULT 0,
                    holdout_consumed INTEGER NOT NULL DEFAULT 0,
                    run_dir TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    strategy_summary_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS experiments_holdout_window
                    ON experiments(holdout_consumed, holdout_start, holdout_end);
                CREATE INDEX IF NOT EXISTS experiments_family
                    ON experiments(family_fingerprint, experiment_fingerprint);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @staticmethod
    def _family_fingerprint(manifest: dict, summary: dict) -> str:
        config = summary.get("configuration", {})
        data = summary.get("data", {})
        identity = {
            "data_fingerprint": manifest.get("data_fingerprint_sha256") or data.get("fingerprint_sha256"),
            "tickers": data.get("tickers", manifest.get("configuration", {}).get("tickers", [])),
            "requested_start": data.get("requested_start", manifest.get("configuration", {}).get("start_date")),
            "initial_train_end": config.get("initial_train_end", manifest.get("configuration", {}).get("initial_train_end")),
            "holdout_start": config.get("final_holdout_start", manifest.get("evaluation_split", {}).get("final_holdout_start")),
            "holdout_end": summary.get("performance", {}).get("final_holdout", {}).get("period_end"),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _record(self, manifest: dict, summary: dict, run_dir: Path) -> None:
        run_id = str(manifest.get("run_id") or summary.get("run_id") or run_dir.name.removeprefix("run-"))
        performance = summary.get("performance", {})
        final = performance.get("final_holdout", {})
        config = summary.get("configuration", {})
        split = manifest.get("evaluation_split", {})
        mode = str(config.get("evaluation_mode", split.get("evaluation_mode", "development")))
        holdout_weeks = int(final.get("strategy_after_costs", {}).get("Num Weeks", 0) or 0)
        holdout_start_value = config.get("final_holdout_start", split.get("final_holdout_start"))
        holdout_end_value = final.get("period_end")
        holdout_start = str(holdout_start_value)[:10] if holdout_start_value else None
        holdout_end = str(holdout_end_value)[:10] if holdout_end_value else None
        consumed = int(mode in {"both", "final_holdout"} and holdout_weeks > 0)
        data_fingerprint = manifest.get("data_fingerprint_sha256") or summary.get("data", {}).get("fingerprint_sha256")
        experiment_fingerprint = (
            manifest.get("experiment_fingerprint_sha256")
            or summary.get("experiment_fingerprint_sha256")
            or hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        )
        family = self._family_fingerprint(manifest, summary)
        strategy_summary = {
            "strategy_after_costs": performance.get("strategy_after_costs"),
            "benchmarks_after_costs": performance.get("benchmarks_after_costs"),
            "random_pick_cost_matched_p_value": performance.get("random_pick_cost_matched_p_value"),
            "information_coefficient_mean": performance.get("information_coefficient_mean"),
            "factor_alpha": summary.get("factor_attribution", {}).get("phases", {}).get("all"),
        }
        created = str(summary.get("run_at_utc") or manifest.get("run_at_utc") or datetime.now(timezone.utc).isoformat())
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO experiments (
                    run_id, experiment_fingerprint, family_fingerprint,
                    data_fingerprint, source_fingerprint, evaluation_mode,
                    holdout_start, holdout_end, holdout_weeks, holdout_consumed,
                    run_dir, created_at_utc, strategy_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, experiment_fingerprint, family, data_fingerprint,
                    manifest.get("backend_source_fingerprint_sha256"), mode,
                    holdout_start if consumed else None, holdout_end if consumed else None,
                    holdout_weeks, consumed, str(run_dir.resolve()), created,
                    json.dumps(strategy_summary, sort_keys=True, allow_nan=False),
                ),
            )

    def import_existing_runs(self) -> int:
        """Index completed run folders, including runs created before the ledger existed."""
        imported = 0
        for run_dir in sorted(self.output_root.glob("run-*")):
            manifest_path = run_dir / "run_manifest.json"
            summary_path = run_dir / "run_summary.json"
            if not manifest_path.is_file() or not summary_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self._record(manifest, summary, run_dir)
                imported += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return imported

    def assert_holdout_is_fresh(
        self,
        start: str,
        end: str,
        *,
        allow_consumed: bool = False,
    ) -> list[dict[str, str]]:
        """Reject any new holdout interval overlapping a previously evaluated one."""
        start_day, end_day = str(start)[:10], str(end)[:10]
        if end_day < start_day:
            raise ValueError("Holdout end date must not precede its start date.")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT run_id, holdout_start, holdout_end, run_dir
                   FROM experiments
                   WHERE holdout_consumed = 1 AND holdout_start IS NOT NULL AND holdout_end IS NOT NULL
                     AND holdout_start <= ? AND holdout_end >= ?
                   ORDER BY holdout_start""",
                (end_day, start_day),
            ).fetchall()
        overlaps = [dict(row) for row in rows]
        if overlaps and not allow_consumed:
            descriptions = ", ".join(
                f"{row['run_id']} ({row['holdout_start']} to {row['holdout_end']})" for row in overlaps
            )
            raise HoldoutReuseError(
                "Requested final holdout overlaps previously evaluated data: "
                f"{descriptions}. Use development mode, move the holdout forward, or explicitly pass "
                "--allow-consumed-holdout to record a non-independent recheck."
            )
        return overlaps

    def record_completed(self, manifest: dict, summary: dict, run_dir: str | Path) -> dict[str, Any]:
        path = Path(run_dir)
        self._record(manifest, summary, path)
        experiment_fingerprint = manifest.get("experiment_fingerprint_sha256") or summary.get("experiment_fingerprint_sha256")
        family = self._family_fingerprint(manifest, summary)
        with self._connect() as connection:
            trial_count = connection.execute(
                "SELECT COUNT(DISTINCT experiment_fingerprint) FROM experiments WHERE family_fingerprint = ?",
                (family,),
            ).fetchone()[0]
            holdout_count = connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE family_fingerprint = ? AND holdout_consumed = 1",
                (family,),
            ).fetchone()[0]
        return {
            "ledger_path": str(self.path.resolve()),
            "study_family_fingerprint_sha256": family,
            "distinct_experiment_configurations": int(trial_count),
            "holdout_accesses_in_family": int(holdout_count),
            "current_experiment_fingerprint_sha256": experiment_fingerprint,
        }

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500.")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT run_id, experiment_fingerprint, family_fingerprint,
                          data_fingerprint, evaluation_mode, holdout_start, holdout_end,
                          holdout_weeks, holdout_consumed, run_dir, created_at_utc,
                          strategy_summary_json
                   FROM experiments ORDER BY created_at_utc DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        output = []
        for row in rows:
            value = dict(row)
            value["holdout_consumed"] = bool(value["holdout_consumed"])
            value["strategy_summary"] = json.loads(value.pop("strategy_summary_json"))
            output.append(value)
        return output

    def search_diagnostics(
        self,
        family_fingerprint: str,
        *,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 52,
    ) -> dict[str, Any]:
        """Compute PBO/DSR only when distinct recorded trials share aligned returns."""
        from evidence import deflated_sharpe_probability, probability_of_backtest_overfitting

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT run_id, experiment_fingerprint, run_dir, created_at_utc
                   FROM experiments WHERE family_fingerprint = ?
                   ORDER BY created_at_utc DESC""",
                (family_fingerprint,),
            ).fetchall()
        distinct: dict[str, sqlite3.Row] = {}
        for row in rows:
            distinct.setdefault(str(row["experiment_fingerprint"]), row)

        series_by_trial: dict[str, pd.Series] = {}
        excluded_without_phase_labels = 0
        for fingerprint, row in distinct.items():
            path = Path(row["run_dir"]) / "portfolio_returns.csv"
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(path, parse_dates=["signal_week"])
                if not {"signal_week", "net_return", "evaluation_phase"}.issubset(frame.columns):
                    # Legacy artifacts cannot be safely separated from a
                    # previously inspected final holdout.
                    excluded_without_phase_labels += 1
                    continue
                frame = frame.loc[frame["evaluation_phase"].eq("development")]
                if frame.empty:
                    continue
                returns = pd.to_numeric(frame["net_return"], errors="coerce")
                series = pd.Series(returns.to_numpy(dtype=float), index=frame["signal_week"])
                series = series.replace([float("inf"), float("-inf")], float("nan")).dropna()
                series_by_trial[fingerprint] = series[~series.index.duplicated(keep="last")]
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                continue

        result: dict[str, Any] = {
            "distinct_recorded_configurations": int(len(series_by_trial)),
            "family_fingerprint_sha256": family_fingerprint,
            "selection_sample": "development phase only; final-holdout rows are excluded",
            "runs_excluded_without_phase_labels": int(excluded_without_phase_labels),
        }
        if len(series_by_trial) < 2:
            result["pbo"] = {
                "status": "not_estimable",
                "reason": "At least two distinct completed strategy configurations are required.",
                "trials": int(len(series_by_trial)),
            }
            result["deflated_sharpe"] = {"status": "not_estimable", "trials": int(len(series_by_trial))}
            return result

        matrix = pd.concat(series_by_trial, axis=1, join="inner").dropna(axis=0, how="any")
        matrix.columns = list(series_by_trial)
        risk_free_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        result["pbo"] = probability_of_backtest_overfitting(
            matrix,
            risk_free_period_return=risk_free_period,
        )
        sharpe_values: list[float] = []
        for column in matrix:
            excess = matrix[column] - risk_free_period
            std = float(excess.std(ddof=1))
            sharpe_values.append(float(excess.mean() / std) if std else 0.0)
        selected_index = int(max(range(len(sharpe_values)), key=sharpe_values.__getitem__))
        selected_returns = matrix.iloc[:, selected_index]
        result["deflated_sharpe"] = deflated_sharpe_probability(
            selected_returns,
            sharpe_values,
            risk_free_period_return=risk_free_period,
        )
        result["aligned_periods"] = int(len(matrix))
        result["aligned_start"] = str(matrix.index.min())[:10] if len(matrix) else None
        result["aligned_end"] = str(matrix.index.max())[:10] if len(matrix) else None
        return result
