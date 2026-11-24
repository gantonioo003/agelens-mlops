param(
    [string]$Python = 'python',
    [switch]$SkipInstall
)
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
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        & $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
        if ($LASTEXITCODE) { throw 'Python 3.12 is required' }
        & $Python -m venv .venv
        if ($LASTEXITCODE) { throw 'Virtual environment creation failed' }
    }
    $EnvPython = Join-Path $Root '.venv/Scripts/python.exe'
    if (-not $SkipInstall) {
        if (Test-Path -LiteralPath 'runtime/uv.exe') {
            & './runtime/uv.exe' pip install --python $EnvPython --link-mode copy --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu' 'torchvision==0.23.0+cpu'
        } else {
            & $EnvPython -m pip install 'torch==2.8.0+cpu' 'torchvision==0.23.0+cpu' --index-url https://download.pytorch.org/whl/cpu
        }
        if ($LASTEXITCODE) { throw 'CPU PyTorch installation failed' }
        if (Test-Path -LiteralPath 'requirements-windows.lock') {
            if (Test-Path -LiteralPath 'runtime/uv.exe') {
                & './runtime/uv.exe' pip install --python $EnvPython --link-mode copy -r requirements-windows.lock
            } else {
                & $EnvPython -m pip install -r requirements-windows.lock
            }
        } else {
            & $EnvPython -m pip install -r requirements.txt
        }
        if ($LASTEXITCODE) { throw 'Scientific dependency installation failed' }
        if (Test-Path -LiteralPath 'runtime/uv.exe') {
            & './runtime/uv.exe' pip install --python $EnvPython --link-mode copy --no-deps -e .
        } else {
            & $EnvPython -m pip install --no-deps -e .
        }
        if ($LASTEXITCODE) { throw 'Project package installation failed' }
    }
    & $EnvPython -m scripts.init_local
    if ($LASTEXITCODE) { throw 'Local environment initialization failed' }
    & $EnvPython -m scripts.check_environment
    if ($LASTEXITCODE) { throw 'Environment check failed' }
} finally {
    $env:TEMP = $OriginalTemp
    $env:TMP = $OriginalTmp
    $env:UV_CACHE_DIR = $OriginalUvCache
    Pop-Location
}
