from __future__ import annotations

import asyncio
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
import httpx
from pydantic import ValidationError

from experiment_ledger import ExperimentLedger, HoldoutReuseError
from factor_attribution import FACTOR_COLUMNS, align_factor_intervals, factor_regression
from factor_data import ARCHIVES, _read_archive, load_french_daily_factors
from evidence import assess_run_evidence, deflated_sharpe_probability, probability_of_backtest_overfitting


def test_factor_archive_parser_converts_percent_to_decimal(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    raw = (
        "Vintage test\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "20260102,1.00,0.50,-0.25,0.10,0.20,0.01\n"
        "20260105,-99.99,0.00,0.00,0.00,0.00,0.01\n\nCopyright\n"
    )
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("sample.csv", raw)

    frame, member = _read_archive(archive_path, ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"))

    assert member == "sample.csv"
    assert frame.loc[pd.Timestamp("2026-01-02"), "Mkt-RF"] == pytest.approx(0.01)
    assert pd.isna(frame.loc[pd.Timestamp("2026-01-05"), "Mkt-RF"])


def test_cached_factor_loader_reports_content_stable_fingerprint(tmp_path: Path):
    for _key, (filename, _url, columns) in ARCHIVES.items():
        table = "," + ",".join(columns) + "\n20260102," + ",".join("0.10" for _ in columns) + "\n"
        with ZipFile(tmp_path / filename, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("sample.csv", "Test vintage\n\n" + table)

    first_factors, first_metadata = load_french_daily_factors(tmp_path)
    second_factors, second_metadata = load_french_daily_factors(tmp_path)

    assert first_factors.equals(second_factors)
    assert first_metadata["fingerprint_sha256"] == second_metadata["fingerprint_sha256"]
    assert first_metadata["archives"]["ff5_daily"]["acquisition"] == "cached"
    assert first_metadata["archives"]["ff5_daily"]["cache_file_mtime_utc"]


def test_factor_windows_use_only_dates_after_entry_through_exit():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    factors = pd.DataFrame(0.01, index=dates, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"])
    portfolio = pd.DataFrame(
        {
            "entry_date": ["2026-01-02"],
            "label_end_date": ["2026-01-06"],
            "net_return": [0.03],
            "evaluation_phase": ["final_holdout"],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )

    aligned, coverage = align_factor_intervals(portfolio, factors)

    assert aligned.loc[0, "factor_sessions"] == 2
    assert aligned.loc[0, "Mkt-RF"] == pytest.approx((1.01**2) - 1)
    assert coverage["matched_intervals"] == 1


def test_factor_regression_recovers_known_alpha_with_hac_errors():
    rng = np.random.default_rng(123)
    n = 260
    factors = rng.normal(0.001, 0.02, size=(n, 6))
    beta = np.array([0.002, 0.2, -0.1, 0.05, 0.03, 0.08, 0.25])
    design = np.column_stack([np.ones(n), factors])
    target = design @ beta + rng.normal(0, 0.001, size=n)
    frame = pd.DataFrame(factors, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    frame["excess_return"] = target

    estimates, summary = factor_regression(frame, max_lags=4, min_observations=30)

    alpha = estimates.query("phase == 'all' and term == 'alpha'").iloc[0]
    assert alpha["estimate"] == pytest.approx(beta[0] * 52, abs=0.002)
    assert alpha["hac_standard_error"] > 0
    assert summary["phases"]["all"]["status"] == "estimated"


def test_factor_regression_refuses_rank_deficient_design():
    rng = np.random.default_rng(124)
    market = rng.normal(0.001, 0.02, size=80)
    frame = pd.DataFrame({name: rng.normal(0.0, 0.01, size=80) for name in FACTOR_COLUMNS})
    frame["SMB"] = market
    frame["HML"] = market
    frame["excess_return"] = market * 0.1 + rng.normal(0.0, 0.001, size=80)

    _, summary = factor_regression(frame, min_observations=30)

    assert summary["phases"]["all"]["status"] == "rank_deficient_design"


def test_pbo_requires_multiple_strategies_then_estimates():
    one = pd.DataFrame({"only": np.linspace(-0.02, 0.03, 100)})
    assert probability_of_backtest_overfitting(one)["status"] == "not_estimable"

    rng = np.random.default_rng(44)
    many = pd.DataFrame(rng.normal(0.001, 0.02, size=(160, 5)), columns=list("abcde"))
    result = probability_of_backtest_overfitting(many, n_slices=8, minimum_periods_per_slice=8)
    assert result["status"] == "estimated"
    assert 0 <= result["pbo"] <= 1
    assert result["cscv_combinations"] == 70


def test_deflated_sharpe_and_evidence_gate_are_conservative():
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0.001, 0.02, size=156))
    assert deflated_sharpe_probability(returns, [0.001, 0.002])["status"] == "estimated"

    summary = {
        "configuration": {"evaluation_mode": "both"},
        "data": {"universe_policy": "Fixed user-supplied ticker universe; survivorship bias remains."},
        "performance": {
            "final_holdout": {
                "strategy_after_costs": {"Num Weeks": 52},
                "annualized_excess_return_95pct_ci": {
                    "equal_weight": [-0.01, 0.2],
                    "spy": [0.01, 0.2],
                },
            }
        },
        "factor_attribution": {
            "phases": {
                "final_holdout": {
                    "alpha_annualized_ci95_hac": [-0.01, 0.2],
                    "nominal_p_values_adjust_for_search": False,
                }
            }
        },
    }
    assessment = assess_run_evidence(summary)
    assert assessment["status"] == "inconclusive"
    assert any("consumed" in reason for reason in assessment["reasons"])
    summary["experiment_history"] = {"reused_previous_holdout": True}
    reused_assessment = assess_run_evidence(summary)
    assert reused_assessment["reused_previous_holdout"] is True
    assert any("reused a previously inspected" in reason for reason in reused_assessment["reasons"])


def test_experiment_ledger_blocks_overlapping_consumed_holdout(tmp_path: Path):
    run_dir = tmp_path / "run-old"
    run_dir.mkdir()
    manifest = {
        "run_id": "old-run",
        "data_fingerprint_sha256": "data-hash",
        "backend_source_fingerprint_sha256": "source-hash",
        "configuration": {"tickers": ["AAA"], "start_date": "2012-01-01"},
        "evaluation_split": {"evaluation_mode": "both", "final_holdout_start": "2025-01-03"},
    }
    summary = {
        "run_id": "old-run",
        "run_at_utc": "2026-01-01T00:00:00+00:00",
        "data": {"tickers": ["AAA"], "requested_start": "2012-01-01"},
        "configuration": {"evaluation_mode": "both", "initial_train_end": "2022-12-31", "final_holdout_start": "2025-01-03"},
        "performance": {"final_holdout": {"period_end": "2025-12-29", "strategy_after_costs": {"Num Weeks": 52}}},
    }
    (run_dir / "run_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(__import__("json").dumps(summary), encoding="utf-8")

    ledger = ExperimentLedger(tmp_path)
    assert ledger.import_existing_runs() == 1
    with pytest.raises(HoldoutReuseError):
        ledger.assert_holdout_is_fresh("2025-05-01", "2025-11-01")
    assert len(ledger.assert_holdout_is_fresh("2026-01-01", "2026-12-31")) == 0
    assert ledger.list_runs()[0]["holdout_consumed"] is True


def test_search_diagnostics_exclude_final_holdout_returns(tmp_path: Path):
    output_root = tmp_path / "runs"
    output_root.mkdir()
    ledger = ExperimentLedger(output_root)
    dates = pd.date_range("2020-01-06", periods=156, freq="W-MON")
    phases = np.array(["development"] * 104 + ["final_holdout"] * 52)
    base = np.sin(np.arange(156) / 4.0) * 0.01

    for index, development_shift in enumerate((0.0002, -0.0002)):
        run_id = f"variant-{index}"
        run_dir = output_root / f"run-{run_id}"
        run_dir.mkdir()
        manifest = {
            "run_id": run_id,
            "experiment_fingerprint_sha256": f"experiment-{index}",
            "data_fingerprint_sha256": "shared-data",
            "backend_source_fingerprint_sha256": "source",
            "configuration": {"tickers": ["AAA"], "start_date": "2012-01-01"},
            "evaluation_split": {"evaluation_mode": "both", "final_holdout_start": "2024-01-01"},
        }
        summary = {
            "run_id": run_id,
            "run_at_utc": f"2026-01-0{index + 1}T00:00:00+00:00",
            "data": {"tickers": ["AAA"], "requested_start": "2012-01-01"},
            "configuration": {
                "evaluation_mode": "both",
                "initial_train_end": "2019-12-31",
                "final_holdout_start": "2024-01-01",
            },
            "performance": {
                "final_holdout": {
                    "period_end": "2024-12-30",
                    "strategy_after_costs": {"Num Weeks": 52},
                }
            },
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        # Deliberately make the final holdout dominate each trial's full-period
        # Sharpe. Search diagnostics must only see the development window.
        returns = base + development_shift
        returns[104:] = 0.03 if index == 0 else -0.03
        pd.DataFrame({
            "signal_week": dates,
            "net_return": returns,
            "evaluation_phase": phases,
        }).to_csv(run_dir / "portfolio_returns.csv", index=False)
        ledger._record(manifest, summary, run_dir)

    family = ledger.list_runs()[0]["family_fingerprint"]
    diagnostics = ledger.search_diagnostics(family)

    assert diagnostics["aligned_periods"] == 104
    assert diagnostics["aligned_end"] == dates[103].date().isoformat()
    assert diagnostics["selection_sample"].startswith("development phase only")
    assert diagnostics["pbo"]["status"] == "estimated"


def test_api_health_and_request_validation_do_not_start_training():
    import api

    async def exercise_api():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            ready = await client.get("/ready")
            invalid = await client.post("/runs", json={"tickers": ["AAPL", "AAPL"]})
            invalid_chronology = await client.post("/runs", json={
                "start_date": "2012-01-01",
                "initial_train_end": "2022-12-31",
                "end_date": "2020-01-01",
            })
            path_traversal = await client.get("/runs/%2E%2E%2Fsecret/artifacts")
        return health, ready, invalid, invalid_chronology, path_traversal

    health, ready, invalid, invalid_chronology, path_traversal = asyncio.run(exercise_api())
    assert health.status_code == 200
    assert ready.json()["ready"] is True
    assert invalid.status_code == 422
    assert invalid_chronology.status_code == 422
    assert path_traversal.status_code == 404


def test_api_can_queue_dynamic_dolt_universe_without_ticker_arguments():
    from api import ResearchRunRequest

    request = ResearchRunRequest(data_source="dolt", top_n=5, universe_size=50)
    arguments = request.cli_arguments("sample-run")

    assert arguments[arguments.index("--data-source") + 1] == "dolt"
    assert arguments[arguments.index("--universe-size") + 1] == "50"
    assert "--tickers" not in arguments

    with pytest.raises(ValidationError):
        ResearchRunRequest(data_source="dolt", top_n=51, universe_size=50)
