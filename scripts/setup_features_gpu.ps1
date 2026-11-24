param([string]$Python = '.venv/Scripts/python.exe')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$OriginalTemp = $env:TEMP
$OriginalTmp = $env:TMP
$OriginalUvCache = $env:UV_CACHE_DIR
$BuildTemp = Join-Path $Root 'runtime/build-temp'
[System.IO.Directory]::CreateDirectory($BuildTemp) | Out-Null
$env:TEMP = $BuildTemp
$env:TMP = $BuildTemp
$env:UV_CACHE_DIR = Join-Path $Root 'runtime/uv-cache'
Push-Location -LiteralPath $Root
try {
    $GpuPython = Join-Path $Root 'runtime/gpu-env/Scripts/python.exe'
    $Requirements = if (Test-Path -LiteralPath 'requirements-features-gpu.lock') {
        'requirements-features-gpu.lock'
    } else {
        'requirements-features-gpu.txt'
    }
    if (-not (Test-Path -LiteralPath $GpuPython)) {
        & $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
        if ($LASTEXITCODE) { throw 'Python 3.12 is required' }
        & $Python -m venv runtime/gpu-env
        if ($LASTEXITCODE) { throw 'GPU environment creation failed' }
    }
    if (Test-Path -LiteralPath 'runtime/uv.exe') {
        & './runtime/uv.exe' pip install --python $GpuPython --link-mode copy -r $Requirements
        if ($LASTEXITCODE) { throw 'CUDA dependency installation failed' }
        & './runtime/uv.exe' pip install --python $GpuPython --link-mode copy --no-deps -e .
        if ($LASTEXITCODE) { throw 'Project installation failed' }
        & './runtime/uv.exe' pip check --python $GpuPython
    } else {
        & $GpuPython -m pip install -r $Requirements
        if ($LASTEXITCODE) { throw 'CUDA dependency installation failed' }
        & $GpuPython -m pip install --no-deps -e .
        if ($LASTEXITCODE) { throw 'Project installation failed' }
        & $GpuPython -m pip check
    }
    if ($LASTEXITCODE) { throw 'Dependency consistency check failed' }
    & $GpuPython -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones((8,8),device="cuda"); assert (x@x).sum().item()==512; print(torch.__version__, torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE) { throw 'CUDA device validation failed' }
} finally {
    $env:TEMP = $OriginalTemp
    $env:TMP = $OriginalTmp
    $env:UV_CACHE_DIR = $OriginalUvCache
    Pop-Location
}
