# MomentumAlpha Terminal

This directory contains the Next.js dashboard. The main repository README has the project overview, backend setup, research status, API routes, and development commands.

The terminal reads real run summaries and artifacts from FastAPI. It does not launch model fitting on page load. To populate a fresh clone, start the backend and dashboard, then explicitly submit a run in the Experiments section.

## Local development

From the repository root, start FastAPI:

    .\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1

Then, in a second PowerShell window, run:

    .\frontend-terminal\run-dev.ps1

The script installs the locked dependencies when needed and proxies API routes to http://127.0.0.1:8000. The app opens at http://127.0.0.1:3000/.
