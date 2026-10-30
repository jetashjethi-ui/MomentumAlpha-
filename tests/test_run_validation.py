from __future__ import annotations

import csv
import json
from pathlib import Path

from run_validation import sha256_file, validate_run_directory


def _write_valid_run(directory: Path) -> None:
    directory.mkdir()
    artifacts = {
        "predictions.csv": "ticker,predicted_prob,actual_return,evaluation_phase\nAAA,0.6,0.02,development\n",
        "portfolio_returns.csv": "signal_week,net_return,evaluation_phase\n2024-01-05,0.01,development\n",
        "walk_forward_fits.csv": "week,failures\n2024-01-05,[]\n",
        "run_summary.json": json.dumps({"run_id": "test-run", "configuration": {"evaluation_mode": "development"}}),
    }
    for name, content in artifacts.items():
        (directory / name).write_text(content, encoding="utf-8", newline="")
    manifest = {
        "run_id": "test-run",
        "artifacts": list(artifacts),
        "evaluation_split": {"evaluation_mode": "development", "evaluated_final_holdout_weeks": 0},
        "artifact_checksums_sha256": {name: sha256_file(directory / name) for name in artifacts},
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_run_validation_checks_declared_artifacts_and_checksums(tmp_path: Path):
    run_dir = tmp_path / "run-test"
    _write_valid_run(run_dir)

    result = validate_run_directory(run_dir, expected_run_id="test-run")

    assert result["valid"] is True
    assert result["checked_artifacts"] == 4
    assert result["warnings"] == []

    with (run_dir / "predictions.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["BBB", "0.5", "0.0", "development"])
    result = validate_run_directory(run_dir, expected_run_id="test-run")

    assert result["valid"] is False
    assert any("checksum mismatch" in error for error in result["errors"])


def test_run_validation_warns_for_legacy_manifest_without_hashes(tmp_path: Path):
    run_dir = tmp_path / "run-legacy"
    _write_valid_run(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_checksums_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_directory(run_dir)

    assert result["valid"] is True
    assert any("no artifact checksums" in warning for warning in result["warnings"])
