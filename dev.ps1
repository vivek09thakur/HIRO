#Requires -Version 5.1
<#
Starts the FastAPI server (port 8000) and the Vite client (port 5173) together.
Ctrl+C stops both.
#>
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

$python = @("$root\venv\Scripts\python.exe", "$root/venv/bin/python") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    throw "No venv found. Run: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
}

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { $npm = Get-Command npm }

if (-not (Test-Path "$root\client\node_modules")) {
    Write-Host "Installing client dependencies..."
    & $npm.Source --prefix client install
}

$env:TF_CPP_MIN_LOG_LEVEL = '2'

$server = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'server.app.main:app', '--reload', '--port', '8000' `
    -WorkingDirectory $root -NoNewWindow -PassThru
$client = Start-Process -FilePath $npm.Source -ArgumentList 'run', 'dev' `
    -WorkingDirectory "$root\client" -NoNewWindow -PassThru

function Stop-Tree($proc) {
    if ($proc -and -not $proc.HasExited) {
        & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
    }
}

try {
    Write-Host ""
    Write-Host "HIRO server  -> http://127.0.0.1:8000"
    Write-Host "HIRO client  -> http://localhost:5173"
    Write-Host "Press Ctrl+C to stop both."
    Write-Host ""
    while (-not $server.HasExited -and -not $client.HasExited) {
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Stop-Tree $server
    Stop-Tree $client
}
