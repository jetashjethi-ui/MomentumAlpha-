# MomentumAlpha Terminal (v0 frontend)

This is the v0-generated frontend integrated with the existing Project3 FastAPI backend. The visual components, colors, layout, motion, and typography are kept from the supplied v0 ZIP. Only the run-form defaults and allowed value ranges were aligned to the backend's validated request schema.

## Run locally on Windows

1. In a first PowerShell window, start the backend from the Project3 root:

   ```powershell
   Set-Location "$env:USERPROFILE\Desktop\Project3"
   .\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
   ```

2. In a second PowerShell window, start this frontend:

   ```powershell
   Set-Location "$env:USERPROFILE\Desktop\Project3"
   .\frontend-terminal\run-dev.ps1
   ```

3. Open `http://127.0.0.1:3000/`.

The frontend proxies `/health`, `/ready`, and `/runs` calls to `http://127.0.0.1:8000` through `next.config.mjs`. It uses real run summaries, verification reports, and CSV artifacts from the backend. It does not start model training automatically. Launching a development run is an explicit action in the Experiments section; final-holdout evaluation stays behind a separate confirmation.

The launch form defaults match the previously used Dolt development configuration: 2012 start, 2022 initial training cutoff, 50-security universe, 5 holdings, 13-week retraining, 10 bps costs, four jobs, and 2,000 bootstrap/random simulations. Advanced inputs follow the request bounds in `api.py`.
