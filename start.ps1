<#
Start the stack on the NVIDIA GPU when Docker can use one, otherwise on CPU.

  .\start.ps1            start (or apply .env changes)
  .\start.ps1 --build    rebuild images after code changes, then start

The choice is saved as COMPOSE_FILE in .env, so later plain `docker compose ...`
commands (exec, stop, logs, up) keep using the same GPU/CPU setup.
#>
Set-Location $PSScriptRoot

$gpuFiles = "docker-compose.yml;docker-compose.gpu.yml"
$cpuFiles = "docker-compose.yml"

function Save-ComposeFile([string]$value) {
    $path = Join-Path $PSScriptRoot ".env"
    $lines = @()
    if (Test-Path $path) { $lines = @(Get-Content $path | Where-Object { $_ -notmatch '^\s*COMPOSE_FILE\s*=' }) }
    $lines += "COMPOSE_FILE=$value"
    [IO.File]::WriteAllLines($path, [string[]]$lines)  # UTF-8 without BOM (docker compose can't read a BOM)
}

$useGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi -L *> $null
    $useGpu = ($LASTEXITCODE -eq 0)
}

if ($useGpu) {
    Write-Host "NVIDIA GPU found - starting with GPU support..." -ForegroundColor Green
    $env:COMPOSE_FILE = $gpuFiles
    $ErrorActionPreference = "Continue"
    docker compose up -d @args 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable log
    if ($LASTEXITCODE -ne 0) {
        # Fall back only when the failure is about the GPU; other failures are reported as they are
        if (($log | Out-String) -match '(?i)nvidia|gpu|device driver|could not select device') {
            Write-Host "Docker could not start with the GPU - falling back to CPU." -ForegroundColor Yellow
            $useGpu = $false
        } else {
            Write-Host "Startup failed (not a GPU problem) - see the output above." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
} else {
    Write-Host "No NVIDIA GPU found - starting on CPU (answers will be slow)." -ForegroundColor Yellow
}

if (-not $useGpu) {
    $env:COMPOSE_FILE = $cpuFiles
    docker compose up -d @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Save-ComposeFile $env:COMPOSE_FILE
$mode = if ($useGpu) { "GPU" } else { "CPU" }
Write-Host "`nRunning on $mode. App: http://localhost:3000" -ForegroundColor Green
Write-Host "Check where the model runs:  docker compose exec ollama ollama ps"
