"""Local research-run API for MomentumAlpha.

Run with one worker bound to localhost:
    python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from experiment_ledger import ExperimentLedger


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / ".runs"
JOBS_DB = RUNS_DIR / "api_jobs.sqlite3"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "JNJ", "BRK-B"]
API_VERSION = "1.0.0"
MAX_LOG_CHARACTERS = 24_000
DOLT_CACHE_FILES = (
    "prepared_data_manifest.json",
    "selected_adjusted_bars.parquet",
    "weekly_universe.parquet",
    "selected_dividends.parquet",
)


class ResearchRunRequest(BaseModel):
    """Explicit, bounded model-run configuration; callers cannot set file paths."""

    model_config = ConfigDict(extra="forbid")

    start_date: date = date(2012, 1, 1)
    end_date: date | None = None
    data_source: Literal["yahoo", "dolt"] = "yahoo"
    initial_train_end: date = date(2022, 12, 31)
    final_holdout_weeks: int = Field(default=52, ge=1, le=260)
    final_holdout_start_date: date | None = None
    evaluation_mode: Literal["development", "final_holdout", "both"] = "development"
    retrain_freq: int = Field(default=13, ge=1, le=260)
    top_n: int = Field(default=2, ge=1, le=50)
    transaction_cost_bps: float = Field(default=10.0, ge=0, le=1000, allow_inf_nan=False)
    risk_free_rate: float = Field(default=0.04, gt=-1, le=1, allow_inf_nan=False)
    rebalance_freq: int = Field(default=1, ge=1, le=52)
    confidence_threshold: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    n_jobs: int = -1
    random_simulations: int = Field(default=2_000, ge=100, le=100_000)
    bootstrap_samples: int = Field(default=2_000, ge=100, le=100_000)
    random_state: int = 42
    tickers: list[str] = Field(default_factory=lambda: list(DEFAULT_TICKERS), min_length=1, max_length=50)
    universe_size: int = Field(default=50, ge=2, le=500)
    min_price: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    min_dollar_volume: float = Field(default=5_000_000.0, ge=0, allow_inf_nan=False)
    min_history_bars: int = Field(default=252, ge=60, le=5_000)
    ticker_gap_days: int = Field(default=180, ge=30, le=3_650)
    refresh_market_data: bool = False
    refresh_dolt_data: bool = False
    refresh_factor_data: bool = False
    allow_consumed_holdout: bool = False
    include_plots: bool = False

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, value: list[str]) -> list[str]:
        tickers = [ticker.strip().upper() for ticker in value]
        if any(not re.fullmatch(r"[A-Z0-9.^=-]{1,16}", ticker) for ticker in tickers):
            raise ValueError("Ticker symbols must contain 1–16 non-empty characters.")
        if len(set(tickers)) != len(tickers):
            raise ValueError("Ticker symbols must be unique.")
        if set(tickers) & {"SPY", "^VIX"}:
            raise ValueError("SPY and ^VIX are reserved for market data and benchmarks.")
        return tickers

    @model_validator(mode="after")
    def validate_dates_and_universe(self) -> "ResearchRunRequest":
        if self.end_date is not None and self.end_date <= self.start_date:
            raise ValueError("end_date must be later than start_date.")
        if self.initial_train_end <= self.start_date:
            raise ValueError("initial_train_end must be later than start_date.")
        if self.end_date is not None and self.end_date <= self.initial_train_end:
            raise ValueError("end_date must leave data after initial_train_end for evaluation.")
        if self.final_holdout_start_date is not None and self.final_holdout_start_date <= self.initial_train_end:
            raise ValueError("final_holdout_start_date must be later than initial_train_end.")
        if self.end_date is not None and self.final_holdout_start_date is not None:
            if self.final_holdout_start_date >= self.end_date:
                raise ValueError("final_holdout_start_date must precede end_date.")
        portfolio_capacity = self.universe_size if self.data_source == "dolt" else len(self.tickers)
        if self.top_n > portfolio_capacity:
            raise ValueError("top_n cannot exceed the selected data-source universe size.")
        if self.n_jobs == 0 or self.n_jobs < -1 or self.n_jobs > 256:
            raise ValueError("n_jobs must be -1 or between 1 and 256.")
        return self

    def cli_arguments(self, run_id: str) -> list[str]:
        values = [
            "--run-id", run_id,
            "--start-date", self.start_date.isoformat(),
            "--data-source", self.data_source,
            "--initial-train-end", self.initial_train_end.isoformat(),
            "--final-holdout-weeks", str(self.final_holdout_weeks),
            "--evaluation-mode", self.evaluation_mode,
            "--retrain-freq", str(self.retrain_freq),
            "--top-n", str(self.top_n),
            "--transaction-cost-bps", str(self.transaction_cost_bps),
            "--risk-free-rate", str(self.risk_free_rate),
            "--rebalance-freq", str(self.rebalance_freq),
            "--confidence-threshold", str(self.confidence_threshold),
            "--n-jobs", str(self.n_jobs),
            "--random-simulations", str(self.random_simulations),
            "--bootstrap-samples", str(self.bootstrap_samples),
            "--random-state", str(self.random_state),
        ]
        if self.data_source == "yahoo":
            values.extend(["--tickers", *self.tickers])
        else:
            values.extend([
                "--universe-size", str(self.universe_size),
                "--min-price", str(self.min_price),
                "--min-dollar-volume", str(self.min_dollar_volume),
                "--min-history-bars", str(self.min_history_bars),
                "--ticker-gap-days", str(self.ticker_gap_days),
            ])
        if self.end_date is not None:
            values.extend(["--end-date", self.end_date.isoformat()])
        if self.final_holdout_start_date is not None:
            values.extend(["--final-holdout-start-date", self.final_holdout_start_date.isoformat()])
        if self.refresh_market_data:
            values.append("--refresh-cache")
        if self.refresh_dolt_data:
            values.append("--refresh-dolt-data")
        if self.refresh_factor_data:
            values.append("--refresh-factor-data")
        if self.allow_consumed_holdout:
            values.append("--allow-consumed-holdout")
        if not self.include_plots:
            values.append("--skip-plots")
        return values


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    started_at_utc TEXT,
                    finished_at_utc TEXT,
                    error TEXT,
                    log_tail TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def mark_interrupted(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """UPDATE jobs SET status='interrupted', updated_at_utc=?, finished_at_utc=?,
                   error=COALESCE(error, 'API process restarted before this job completed.')
                   WHERE status IN ('queued','running')""",
                (now, now),
            )

    def create(self, run_id: str, request: ResearchRunRequest) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        payload = request.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (run_id,status,request_json,created_at_utc,updated_at_utc) VALUES (?,?,?,?,?)",
                (run_id, "queued", json.dumps(payload, sort_keys=True), now, now),
            )
        return self.get(run_id)  # type: ignore[return-value]

    def update(self, run_id: str, *, status_value: str, error: str | None = None, log_tail: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        started = now if status_value == "running" else None
        finished = now if status_value in {"completed", "failed", "interrupted"} else None
        with self._connect() as connection:
            connection.execute(
                """UPDATE jobs SET status=?, updated_at_utc=?,
                   started_at_utc=COALESCE(started_at_utc, ?),
                   finished_at_utc=COALESCE(?, finished_at_utc),
                   error=?, log_tail=? WHERE run_id=?""",
                (status_value, now, started, finished, error, log_tail, run_id),
            )

    def get(self, run_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["request"] = json.loads(value.pop("request_json"))
        return value

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM jobs ORDER BY created_at_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return [job for row in rows if (job := self.get(row["run_id"])) is not None]

    def active_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0])


class ResearchJobManager:
    def __init__(self, store: JobStore):
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="momentumalpha-run")
        self.lock = threading.Lock()

    def submit(self, request: ResearchRunRequest) -> dict:
        with self.lock:
            if self.store.active_count():
                raise HTTPException(status_code=409, detail="A research run is already queued or running.")
            run_id = uuid.uuid4().hex[:16]
            job = self.store.create(run_id, request)
            self.executor.submit(self._execute, run_id, request)
            return job

    def _execute(self, run_id: str, request: ResearchRunRequest) -> None:
        self.store.update(run_id, status_value="running")
        command = [sys.executable, str(PROJECT_DIR / "main.py"), *request.cli_arguments(run_id)]
        log_dir = RUNS_DIR / "job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"

        def read_log_tail() -> str:
            try:
                with log_path.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    size = stream.tell()
                    stream.seek(max(0, size - MAX_LOG_CHARACTERS), os.SEEK_SET)
                    return stream.read().decode("utf-8", errors="replace")
            except OSError:
                return ""

        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    timeout=6 * 60 * 60,
                    check=False,
                    env=os.environ.copy(),
                )
            log_tail = read_log_tail()
            if completed.returncode == 0:
                summary = RUNS_DIR / f"run-{run_id}" / "run_summary.json"
                if not summary.is_file():
                    self.store.update(
                        run_id,
                        status_value="failed",
                        error="Pipeline exited successfully but did not write its run summary.",
                        log_tail=log_tail,
                    )
                else:
                    from run_validation import validate_run_directory

                    verification = validate_run_directory(summary.parent, expected_run_id=run_id)
                    if not verification["valid"]:
                        self.store.update(
                            run_id,
                            status_value="failed",
                            error="Research pipeline finished, but run verification failed: "
                            + "; ".join(verification["errors"]),
                            log_tail=log_tail,
                        )
                        return
                    self.store.update(run_id, status_value="completed", log_tail=log_tail)
            else:
                self.store.update(
                    run_id,
                    status_value="failed",
                    error=f"Research pipeline exited with code {completed.returncode}.",
                    log_tail=log_tail,
                )
        except subprocess.TimeoutExpired:
            self.store.update(
                run_id,
                status_value="failed",
                error="Research run exceeded the six-hour service limit.",
                log_tail=read_log_tail(),
            )
        except Exception as exc:
            self.store.update(
                run_id,
                status_value="failed",
                error=f"{type(exc).__name__}: {exc}",
            )


store = JobStore(JOBS_DB)
store.mark_interrupted()
manager = ResearchJobManager(store)
ledger = ExperimentLedger(RUNS_DIR)
ledger.import_existing_runs()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    manager.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="MomentumAlpha Research API",
    description="Queue reproducible walk-forward research runs and inspect their evidence reports.",
    version=API_VERSION,
    lifespan=lifespan,
)


def _read_summary(run_id: str) -> dict | None:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id):
        return None
    path = RUNS_DIR / f"run-{run_id}" / "run_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "MomentumAlpha Research API", "version": API_VERSION}


@app.get("/ready")
def readiness() -> dict:
    writable = RUNS_DIR.exists() and os.access(RUNS_DIR, os.W_OK)
    ready = (PROJECT_DIR / "main.py").is_file() and writable
    if not ready:
        raise HTTPException(status_code=503, detail="Research engine or run storage is unavailable.")
    yahoo_cache = (PROJECT_DIR / ".cache" / "market_data.zip").is_file()
    dolt_cache_dir = PROJECT_DIR / ".cache" / "dolt_materialized"
    dolt_cache = all((dolt_cache_dir / name).is_file() for name in DOLT_CACHE_FILES)
    dolt_snapshot = (PROJECT_DIR / ".cache" / "datasets" / "stocks").is_dir()
    factor_cache = all((PROJECT_DIR / ".cache" / "fama_french" / name).is_file() for name in (
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "F-F_Momentum_Factor_daily_CSV.zip",
    ))
    return {
        "ready": True,
        "engine": "walk-forward research pipeline",
        "active_jobs": store.active_count(),
        "market_data_cache_present": yahoo_cache or dolt_cache,
        "data_sources": {
            "yahoo_cache_present": yahoo_cache,
            "dolt_materialized_cache_present": dolt_cache,
            "dolt_source_snapshot_present": dolt_snapshot,
        },
        "factor_cache_present": factor_cache,
    }


@app.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def submit_run(request: ResearchRunRequest) -> dict:
    """Queue a single isolated CLI run; only one model-training job can run at a time."""
    job = manager.submit(request)
    return {
        "run_id": job["run_id"],
        "status": job["status"],
        "status_url": f"/runs/{job['run_id']}",
        "result_url": f"/runs/{job['run_id']}/result",
    }


@app.get("/runs")
def list_runs(limit: int = 50) -> dict:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200.")
    jobs = store.list_jobs(limit)
    job_ids = {job["run_id"] for job in jobs}
    completed = []
    for run in ledger.list_runs(limit):
        if run["run_id"] in job_ids:
            continue
        value = {key: item for key, item in run.items() if key != "run_dir"}
        summary = _read_summary(run["run_id"])
        if summary is not None:
            data = summary.get("data", {})
            value["data_source"] = data.get("provider") or "Unknown source"
            value["signal_start"] = data.get("signal_start")
            value["signal_end"] = data.get("signal_end")
        completed.append(value)
    return {"jobs": jobs, "completed_runs": completed[:max(0, limit - len(jobs))]}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id):
        raise HTTPException(status_code=404, detail="Research run was not found.")
    job = store.get(run_id)
    if job is None:
        completed = next((run for run in ledger.list_runs(500) if run["run_id"] == run_id), None)
        if completed is None:
            raise HTTPException(status_code=404, detail="Research run was not found.")
        completed.pop("run_dir", None)
        return {"run_id": run_id, "status": "completed", "history": completed}
    return job


@app.get("/runs/{run_id}/result")
def get_run_result(run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id):
        raise HTTPException(status_code=404, detail="Research result was not found.")
    job = store.get(run_id)
    if job is not None and job["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Research run has not completed.")
    summary = _read_summary(run_id)
    if summary is None:
        if job is None:
            raise HTTPException(status_code=404, detail="Research result was not found.")
        raise HTTPException(status_code=409, detail=job.get("error") or "Research result is unavailable.")
    return summary


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id):
        raise HTTPException(status_code=404, detail="Research artifacts were not found.")
    run_dir = RUNS_DIR / f"run-{run_id}"
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Research artifacts were not found.")
    files = [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in sorted(run_dir.iterdir()) if path.is_file()
    ]
    return {"run_id": run_id, "artifacts": files}


@app.get("/runs/{run_id}/verification")
def verify_run(run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id):
        raise HTTPException(status_code=404, detail="Research run was not found.")
    from run_validation import validate_run_directory

    result = validate_run_directory(RUNS_DIR / f"run-{run_id}", expected_run_id=run_id)
    if not result["valid"] and not (RUNS_DIR / f"run-{run_id}").is_dir():
        raise HTTPException(status_code=404, detail="Research run was not found.")
    return result


@app.get("/runs/{run_id}/artifacts/{artifact_name}")
def get_artifact(run_id: str, artifact_name: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id) or Path(artifact_name).name != artifact_name:
        raise HTTPException(status_code=404, detail="Research artifact was not found.")
    run_dir = (RUNS_DIR / f"run-{run_id}").resolve()
    target = (run_dir / artifact_name).resolve()
    if target.parent != run_dir or not target.is_file():
        raise HTTPException(status_code=404, detail="Research artifact was not found.")
    return FileResponse(target, filename=target.name)


@app.get("/", include_in_schema=False)
def service_info() -> dict[str, str]:
    return {
        "service": "MomentumAlpha Research API",
        "version": API_VERSION,
        "health": "/health",
        "readiness": "/ready",
        "documentation": "/docs",
    }
