<#
.SYNOPSIS
Starts the local Changzhou industry-investment demo at http://127.0.0.1:8000.

.DESCRIPTION
Uses the named Conda environment directly, rather than the ``conda run``
wrapper. This avoids a Windows console-encoding issue with Unicode output.
The web server binds only to 127.0.0.1. Any credential file is read only when
explicitly configured in demo/.env by the backend; it is never sent to the
browser.
#>

[CmdletBinding()]
param(
    [string]$CondaEnvironment = "changzhou-rag-demo"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$conda = Get-Command conda -ErrorAction Stop
$environmentList = (& $conda.Source env list --json | ConvertFrom-Json).envs
$environmentRoot = $environmentList | Where-Object {
    (Split-Path -Leaf $_) -eq $CondaEnvironment
} | Select-Object -First 1

if (-not $environmentRoot) {
    throw "Conda environment '$CondaEnvironment' was not found. Run: conda create -n $CondaEnvironment python=3.11"
}

$projectPython = Join-Path $environmentRoot "python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "python.exe was not found in Conda environment '$CondaEnvironment'."
}

& $projectPython -c "import fastapi, httpx, jinja2, numpy, uvicorn"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing core web dependencies into $CondaEnvironment..."
    & $projectPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Core dependency installation failed."
    }
}

Set-Location -LiteralPath $projectRoot
Write-Host "Using Conda environment: $CondaEnvironment"
Write-Host "Local server: http://127.0.0.1:8000"
& $projectPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
