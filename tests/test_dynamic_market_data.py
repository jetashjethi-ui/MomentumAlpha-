from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import compute_weekly_cross_sectional_features, prepare_weekly_dataset
from market_data_quality import sanitize_benchmark_ohlc
from research import complete_evaluation_weeks, validate_weekly_panel


def test_total_return_label_includes_only_dividends_after_entry():
    dates = pd.to_datetime([
        "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
        "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-19",
    ])
    bars = pd.DataFrame(
        {
            "Open": [95, 100, 101, 102, 103, 104, 110, 111, 112],
            "High": [96, 101, 102, 103, 104, 105, 111, 112, 113],
            "Low": [94, 99, 100, 101, 102, 103, 109, 110, 111],
            "Close": [95, 100, 101, 102, 103, 104, 110, 111, 112],
            "Volume": [1_000] * 9,
        },
        index=dates,
    )
    dividend_events = pd.Series(
        [7.0, 2.0],
        index=pd.to_datetime(["2024-01-08", "2024-01-10"]),
    )

    panel = prepare_weekly_dataset(
        {"AAA": bars[["Close"]]},
        {"AAA": bars},
        ["AAA"],
        dividends={"AAA": dividend_events},
    )

    first_signal = panel.loc[pd.Timestamp("2024-01-05")]
    assert first_signal["entry_date"] == pd.Timestamp("2024-01-08")
    assert first_signal["label_end_date"] == pd.Timestamp("2024-01-16")
    assert first_signal["cash_dividends"] == pytest.approx(2.0)
    assert first_signal["forward_return"] == pytest.approx((110.0 + 2.0) / 100.0 - 1.0)


def test_weekly_cross_sectional_ranks_are_computed_within_each_roster():
    week_a = pd.Timestamp("2024-01-05")
    week_b = pd.Timestamp("2024-01-12")
    panel = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B"],
            "return_5d": [0.1, 0.2, 0.4, 0.3],
            "return_20d": [0.0, 0.1, 0.2, 0.1],
            "rsi_14": [40.0, 60.0, 70.0, 50.0],
            "vol_20d": [0.2, 0.3, 0.4, 0.1],
            "rel_volume_20d": [1.0, 2.0, 1.5, 0.5],
        },
        index=pd.DatetimeIndex([week_a, week_a, week_b, week_b], name="week"),
    )

    output = compute_weekly_cross_sectional_features(panel)

    assert output.loc[week_a, "cs_ret5d_rank"].tolist() == pytest.approx([0.5, 1.0])
    assert output.loc[week_b, "cs_ret5d_rank"].tolist() == pytest.approx([1.0, 0.5])
    assert np.isfinite(output["cs_breadth"]).all()


def test_dynamic_week_validation_requires_exact_roster_and_synchronized_dates():
    week_a = pd.Timestamp("2024-01-05")
    week_b = pd.Timestamp("2024-01-12")
    panel = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B"],
            "target": [1, 0, 0, 1],
            "forward_return": [0.1, -0.1, -0.2, 0.2],
            "signal_date": [week_a, week_a, week_b, week_b],
            "entry_date": [week_a + pd.Timedelta(days=3)] * 2 + [week_b + pd.Timedelta(days=3)] * 2,
            "label_end_date": [week_a + pd.Timedelta(days=10)] * 2 + [week_b + pd.Timedelta(days=10)] * 2,
        },
        index=pd.DatetimeIndex([week_a, week_a, week_b, week_b], name="week"),
    )
    roster = {week_a: {"A", "B"}, week_b: {"A", "B"}}

    diagnostics = validate_weekly_panel(panel, ["A", "B"], universe_by_week=roster)
    complete = complete_evaluation_weeks(
        panel,
        ["A", "B"],
        after_date="2024-01-01",
        universe_by_week=roster,
        min_universe_size=2,
    )

    assert diagnostics["weeks_with_incomplete_universe"] == 0
    assert diagnostics["weeks_with_misaligned_execution_dates"] == 0
    assert complete.tolist() == [week_a, week_b]


def test_benchmark_range_repair_is_minimal_audited_and_keeps_execution_prices():
    source = pd.DataFrame(
        {
            "Open": [208.02, 210.0],
            "High": [206.7645, 211.0],
            "Low": [204.25, 209.0],
            "Close": [204.53, 210.5],
        },
        index=pd.to_datetime(["2015-07-08", "2015-07-09"]),
    )

    cleaned, audit = sanitize_benchmark_ohlc(source)

    assert cleaned.loc[pd.Timestamp("2015-07-08"), "Open"] == pytest.approx(208.02)
    assert cleaned.loc[pd.Timestamp("2015-07-08"), "Close"] == pytest.approx(204.53)
    assert cleaned.loc[pd.Timestamp("2015-07-08"), "High"] == pytest.approx(208.02)
    assert cleaned.loc[pd.Timestamp("2015-07-08"), "Low"] == pytest.approx(204.25)
    assert audit["ohlc_envelope_repairs"] == 1
    assert audit["repair_examples"][0]["date"] == "2015-07-08"
    assert audit["source_export_preserved"] is True
