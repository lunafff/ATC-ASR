param(
    [ValidateSet('all', 'prepare', 'train', 'final', 'package')]
    [string]$Stage = 'all',
    [string]$Python = '',
    [string]$Config = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    if ($Python) {
        $pythonCommand = $Python
        $pythonPrefix = @()
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCommand = 'py'
        $pythonPrefix = @('-3.11')
    } else {
        $pythonCommand = 'python'
        $pythonPrefix = @()
    }
    $ErrorActionPreference = 'Continue'
    & $pythonCommand @pythonPrefix -c 'import sys; print(sys.version); sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 (recommended), or specify -Python with a Python 3.10-3.12 executable path.' }
    & $pythonCommand @pythonPrefix -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment.' }
    $ErrorActionPreference = 'Stop'
}
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
New-Item -ItemType Directory -Force 'runtime/logs' | Out-Null
$ErrorActionPreference = 'Continue'
if ($Stage -ne 'package') {
    # Explicit CUDA wheels: avoid accidentally installing a CPU-only Windows build.
    & $venvPython -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
$pipelineArgs = @('src/pipeline.py', '--stage', $Stage)
if ($Config) { $pipelineArgs += @('--config', $Config) }
& $venvPython -u @pipelineArgs 2>&1 | Tee-Object -FilePath 'runtime/logs/launcher.log' -Append
$pipelineExit = $LASTEXITCODE
if ($pipelineExit -ne 0) {
    Write-Host 'Run failed. Check runtime/failure.json and runtime/logs/launcher.log.'
    exit $pipelineExit
}
