<#
.SYNOPSIS
Starts the local Changzhou industry-investment demo at http://127.0.0.1:8000.

.DESCRIPTION
Uses the named Conda environment directly, rather than the ``conda run``
wrapper. This avoids a Windows console-encoding issue with Unicode output.
The web server binds only to 127.0.0.1. No cloud credential is required for
the core workflow; optional local model/session settings in demo/.env are
never sent to the browser.
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

& $projectPython -c "import fastapi, httpx, jinja2, multipart, numpy, uvicorn; import docx, pptx"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing core web dependencies into $CondaEnvironment..."
    & $projectPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Core dependency installation failed."
    }
}

Set-Location -LiteralPath $projectRoot
$embeddingModel = Join-Path $projectRoot "models\bge-m3"
$rerankerModel = Join-Path $projectRoot "models\bge-reranker-v2-m3"
Write-Host "Using Conda environment: $CondaEnvironment"
Write-Host "Local server: http://127.0.0.1:8000"
if ((Test-Path -LiteralPath $embeddingModel) -and (Test-Path -LiteralPath $rerankerModel)) {
    Write-Host "Advanced RAG models detected: BGE-M3 + BGE Cross-Encoder"
} else {
    Write-Warning "BGE-M3 or BGE Cross-Encoder model is missing. The app can start with a labelled fallback, but run README model setup for full Advanced RAG."
}
& $projectPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
