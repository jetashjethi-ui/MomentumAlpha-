"""Reproducible access to the Kenneth R. French U.S. daily factor data."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


DATA_LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html"
ARCHIVES = {
    "ff5_daily": (
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    ),
    "momentum_daily": (
        "F-F_Momentum_Factor_daily_CSV.zip",
        "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip",
        ("Mom",),
    ),
}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_archive(archive_path: Path, required_columns: Iterable[str]) -> tuple[pd.DataFrame, str]:
    try:
        with ZipFile(archive_path, "r") as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError("Factor archive must contain exactly one CSV file.")
            raw = archive.read(csv_names[0]).decode("latin-1")
    except (BadZipFile, OSError) as exc:
        raise ValueError(f"Cannot read factor archive {archive_path.name}: {exc}") from exc

    lines = raw.splitlines()
    header_index = next(
        (
            index for index, line in enumerate(lines)
            if all(column in line for column in required_columns)
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"No daily factor header found in {archive_path.name}.")

    reader = csv.reader(StringIO("\n".join(lines[header_index:])))
    header = [item.strip() for item in next(reader)]
    date_column = header[0] if header and header[0] else "Date"
    columns = [date_column, *required_columns]
    positions = {name: header.index(name) for name in required_columns}
    records: list[tuple[pd.Timestamp, ...]] = []
    for row in reader:
        if not row or not row[0].strip().isdigit() or len(row[0].strip()) != 8:
            break
        date = pd.to_datetime(row[0].strip(), format="%Y%m%d", errors="coerce")
        if pd.isna(date):
            continue
        values: list[float] = []
        for name in required_columns:
            position = positions[name]
            try:
                value = float(row[position].strip())
            except (IndexError, ValueError):
                value = np.nan
            values.append(np.nan if value <= -99.0 else value / 100.0)
        records.append((date, *values))

    if not records:
        raise ValueError(f"No daily factor observations found in {archive_path.name}.")
    frame = pd.DataFrame(records, columns=columns).set_index(date_column)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame, csv_names[0]


def load_french_daily_factors(
    cache_dir: str | Path = ".cache/fama_french",
    *,
    refresh: bool = False,
    timeout: int = 45,
) -> tuple[pd.DataFrame, dict]:
    """Load FF5 and momentum daily factors, caching the original ZIP bytes.

    Factor values are converted from percentages to decimal returns. The
    returned metadata fingerprints the exact source archives used in a run.
    """
    root = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    archive_metadata: dict[str, dict] = {}

    for key, (filename, url, columns) in ARCHIVES.items():
        path = root / filename
        acquisition = "cached"
        refresh_fallback_error = None
        if refresh or not path.exists():
            request = Request(url, headers={"User-Agent": "MomentumAlpha research backend"})
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
            except Exception as exc:
                if not path.exists():
                    raise RuntimeError(f"Could not download French factor data ({key}): {exc}") from exc
                refresh_fallback_error = f"{type(exc).__name__}: {exc}"
            else:
                if not payload.startswith(b"PK\x03\x04"):
                    raise RuntimeError(f"Factor provider returned a non-ZIP response for {key}.")
                _atomic_bytes(path, payload)
                acquisition = "downloaded"

        frame, member = _read_archive(path, columns)
        payload_hash = sha256(path.read_bytes()).hexdigest()
        frames.append(frame)
        cached_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        archive_metadata[key] = {
            "url": url,
            "archive_member": member,
            "archive_sha256": payload_hash,
            "acquisition": acquisition,
            "cache_file_mtime_utc": cached_at,
            "refresh_fallback_error": refresh_fallback_error,
            "observations": int(len(frame)),
            "first_date": frame.index.min().date().isoformat(),
            "last_date": frame.index.max().date().isoformat(),
        }

    factors = pd.concat(frames, axis=1, join="inner").sort_index()
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna()
    if factors.empty or not factors.index.is_monotonic_increasing or factors.index.has_duplicates:
        raise RuntimeError("Merged French factor history is empty or has invalid dates.")

    metadata = {
        "provider": "Kenneth R. French Data Library",
        "data_library_url": DATA_LIBRARY_URL,
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_units": "decimal simple returns; provider percentage values divided by 100",
        "daily_factor_interval": "provider daily close-to-close factors",
        "observations": int(len(factors)),
        "first_date": factors.index.min().date().isoformat(),
        "last_date": factors.index.max().date().isoformat(),
        "archives": archive_metadata,
    }
    content_identity = {
        key: details["archive_sha256"]
        for key, details in sorted(archive_metadata.items())
    }
    metadata["fingerprint_sha256"] = sha256(
        json.dumps(content_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return factors, metadata
