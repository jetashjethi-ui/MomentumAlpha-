"""Integrity checks for completed MomentumAlpha research runs."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
from typing import Any


CORE_CSV_ARTIFACTS = {"predictions.csv", "portfolio_returns.csv", "walk_forward_fits.csv"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is missing or invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object.")
        return None
    return value


def validate_run_directory(
    run_dir: str | Path,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Verify run identity, required results, CSV readability, and hashes.

    Legacy runs without a checksum map are still checked for completeness and
    parseability, but receive a warning because their artifacts cannot be
    cryptographically matched to their original manifest.
    """
    directory = Path(run_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not directory.is_dir():
        return {
            "valid": False,
            "run_dir": str(directory),
            "checked_artifacts": 0,
            "errors": ["Run directory does not exist."],
            "warnings": [],
        }

    manifest = _read_json(directory / "run_manifest.json", "run_manifest.json", errors)
    summary = _read_json(directory / "run_summary.json", "run_summary.json", errors)
    if manifest is None or summary is None:
        return {
            "valid": False,
            "run_dir": str(directory),
            "checked_artifacts": 0,
            "errors": errors,
            "warnings": warnings,
        }

    manifest_id = str(manifest.get("run_id", ""))
    summary_id = str(summary.get("run_id", ""))
    if not manifest_id or manifest_id != summary_id:
        errors.append("Run ID does not match between the manifest and summary.")
    if expected_run_id is not None and manifest_id != expected_run_id:
        errors.append("Run ID does not match the requested run.")

    artifact_names = manifest.get("artifacts")
    if not isinstance(artifact_names, list) or not artifact_names:
        errors.append("Manifest has no artifact list.")
        artifact_names = []
    elif any(not isinstance(name, str) for name in artifact_names):
        errors.append("Manifest artifact names must all be strings.")
        artifact_names = [name for name in artifact_names if isinstance(name, str)]
    if len(set(artifact_names)) != len(artifact_names):
        errors.append("Manifest contains duplicate artifact names.")
    for required in CORE_CSV_ARTIFACTS | {"run_summary.json"}:
        if required not in artifact_names:
            errors.append(f"Manifest is missing required artifact {required}.")

    checksums = manifest.get("artifact_checksums_sha256")
    if not isinstance(checksums, dict):
        checksums = {}
        warnings.append("This run has no artifact checksums; file completeness was checked without hash verification.")

    checked: list[dict[str, Any]] = []
    for name in artifact_names:
        if name in {".", ".."} or Path(name).name != name or PureWindowsPath(name).name != name:
            errors.append(f"Unsafe artifact path in manifest: {name!r}.")
            continue
        path = directory / name
        if not path.is_file():
            errors.append(f"Artifact is missing: {name}.")
            continue
        size = path.stat().st_size
        if size == 0:
            errors.append(f"Artifact is empty: {name}.")
            continue
        digest = sha256_file(path)
        expected_digest = checksums.get(name)
        if expected_digest is not None and digest.lower() != str(expected_digest).lower():
            errors.append(f"Artifact checksum mismatch: {name}.")
        elif checksums and expected_digest is None:
            errors.append(f"Artifact has no checksum entry: {name}.")

        artifact_check: dict[str, Any] = {
            "name": name,
            "size_bytes": size,
            "sha256": digest,
        }
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader, None)
                    if not header or any(not column.strip() for column in header):
                        raise ValueError("missing or empty header")
                    row_count = 0
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"row {row_count + 2} has a different column count than the header")
                        row_count += 1
                if name in CORE_CSV_ARTIFACTS and row_count == 0:
                    errors.append(f"Core artifact has no data rows: {name}.")
                artifact_check["rows"] = row_count
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                errors.append(f"CSV artifact is unreadable: {name}: {exc}")
        checked.append(artifact_check)

    if checksums:
        undeclared_hashes = sorted(set(checksums) - set(artifact_names))
        if undeclared_hashes:
            errors.append("Checksum map contains undeclared artifacts: " + ", ".join(undeclared_hashes) + ".")

    split = manifest.get("evaluation_split", {})
    configuration = summary.get("configuration", {})
    mode = configuration.get("evaluation_mode", split.get("evaluation_mode"))
    evaluated_holdout = int(split.get("evaluated_final_holdout_weeks", 0) or 0)
    if mode == "development" and evaluated_holdout != 0:
        errors.append("Development-only run reports evaluated final-holdout weeks.")
    if mode in {"final_holdout", "both"} and evaluated_holdout == 0:
        warnings.append("Run was configured to inspect the final holdout but reports zero completed holdout weeks.")

    return {
        "valid": not errors,
        "run_id": manifest_id or None,
        "run_dir": str(directory),
        "checked_artifacts": len(checked),
        "artifacts": checked,
        "evaluation_mode": mode,
        "evaluated_final_holdout_weeks": evaluated_holdout,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a saved MomentumAlpha research run.")
    parser.add_argument(
        "run",
        help="Run ID (for example 9908803147864be0) or path to a run-<id> folder.",
    )
    args = parser.parse_args(argv)
    value = Path(args.run)
    if value.is_dir():
        run_dir = value
        expected_run_id = None
    else:
        run_dir = Path(__file__).resolve().parent / ".runs" / f"run-{args.run}"
        expected_run_id = args.run
    result = validate_run_directory(run_dir, expected_run_id=expected_run_id)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
