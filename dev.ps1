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
$healthUrl = 'http://127.0.0.1:8000/api/health'
$client = $null

foreach ($port in 8000, 5173) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Stop the other HIRO instance (or process) first."
    }
}

function Stop-Tree($proc) {
    if ($proc -and -not $proc.HasExited) {
        & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
    }
}

function Test-Server {
    try { return (Invoke-WebRequest -UseBasicParsing $healthUrl -TimeoutSec 2).StatusCode -eq 200 }
    catch { return $false }
}

$server = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'server.app.main:app', '--reload', '--reload-dir', 'server', '--port', '8000' `
    -WorkingDirectory $root -NoNewWindow -PassThru

try {
    Write-Host ""
    Write-Host "Starting HIRO server on http://127.0.0.1:8000 ..."
    while (-not (Test-Server)) {
        if ($server.HasExited) { throw "Server exited before becoming healthy (exit code $($server.ExitCode))." }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "Server is online. Starting client..."

    $client = Start-Process -FilePath $npm.Source -ArgumentList 'run', 'dev' `
        -WorkingDirectory "$root\client" -NoNewWindow -PassThru

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
    Stop-Tree $client
    Stop-Tree $server
}
