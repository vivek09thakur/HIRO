#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$python = @("$root\venv\Scripts\python.exe", "$root/venv/bin/python") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    throw "No venv found. Run: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
}

$port = if ($env:PORT) { $env:PORT } else { 8000 }
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $port is already in use."
}

Write-Host "HIRO -> http://127.0.0.1:$port"
& $python -m uvicorn app.api.main:app --reload --reload-dir app --reload-dir model --port $port
