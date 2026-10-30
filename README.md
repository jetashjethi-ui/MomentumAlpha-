# MomentumAlpha

MomentumAlpha is a research platform for weekly, cross-sectional stock selection. It combines a walk-forward research pipeline with a FastAPI service and a live Next.js dashboard.

## Start the app on Windows

1. In a PowerShell window, start the API from the Project3 folder:

   ```powershell
   Set-Location "$env:USERPROFILE\Desktop\Project3"
   .\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
   ```

2. In a second PowerShell window, start the dashboard:

   ```powershell
   Set-Location "$env:USERPROFILE\Desktop\Project3"
   .\frontend-terminal\run-dev.ps1
   ```

   The script installs the locked frontend dependencies if needed, then starts the dashboard.

3. Open <http://127.0.0.1:3000/>. API health is available at <http://127.0.0.1:8000/health> and interactive API documentation at <http://127.0.0.1:8000/docs>.

Starting either service does not train a model. A research run must be explicitly launched in the dashboard or through the Python CLI.

## Project layout

- `api.py` — FastAPI endpoints for health, run status, reports, and artifacts.
- `main.py` — research pipeline and command-line entry point.
- `data_loader.py`, `dolt_data_loader.py` — market data loading and validation.
- `features.py`, `factor_data.py` — feature engineering and benchmark data.
- `modeling.py`, `backtest.py` — model fitting and walk-forward portfolio simulation.
- `research.py`, `evidence.py`, `experiment_ledger.py` — diagnostics, evidence checks, and run tracking.
- `market_data_quality.py`, `factor_attribution.py`, `visualizations.py` — quality checks, attribution, and charts.
- `frontend-terminal/` — active MomentumAlpha Terminal dashboard.
- `tests/` — automated test suite (not run as part of this cleanup).
- `.cache/` — downloaded and prepared data cache.
- `.runs/` — saved run outputs, SQLite job/experiment records, and evidence.
- `archive/legacy-2026-10-30/` — older reports, exports, charts, scripts, and the retired static interface kept for reference.

## Build and compile checks

From the Project3 folder, compile the Python source without running the model:

```powershell
.\.venv\Scripts\python.exe -m compileall -q api.py main.py MomentumAlpha.py backtest.py data_loader.py dolt_data_loader.py evidence.py experiment_ledger.py factor_attribution.py factor_data.py features.py market_data_quality.py modeling.py research.py run_validation.py visualizations.py tests
```

To build the dashboard, open PowerShell in `frontend-terminal` and run:

```powershell
npx --yes pnpm@10 build
```

## Research status

The latest saved run is recorded as **inconclusive**. Its final holdout was reserved and not evaluated, and the report does not establish a robust independent trading edge. Treat the saved metrics as historical research evidence, not live-trading performance.
