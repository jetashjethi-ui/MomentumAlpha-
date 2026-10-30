$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:MOMENTUM_API_URL = if ($env:MOMENTUM_API_URL) { $env:MOMENTUM_API_URL } else { 'http://127.0.0.1:8000' }

if (-not (Test-Path '.\node_modules\.bin\next.cmd')) {
  Write-Host 'Installing the frontend dependencies from the locked package list...'
  & npx --yes pnpm@10 install --frozen-lockfile
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host 'Starting MomentumAlpha Terminal at http://127.0.0.1:3000'
Write-Host "Proxying research API requests to $env:MOMENTUM_API_URL"
& '.\node_modules\.bin\next.cmd' dev --hostname 127.0.0.1 --port 3000
exit $LASTEXITCODE
