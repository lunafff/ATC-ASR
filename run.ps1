param(
    [ValidateSet('all', 'prepare', 'train', 'final', 'package')]
    [string]$Stage = 'all',
    [string]$Python = '',
    [string]$Config = '',
    [ValidateSet('both', 'base', 'small')]
    [string]$Student = 'both'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Config -and $Student -eq 'both') {
    throw 'Custom -Config requires -Student base or -Student small.'
}
if (Test-Path -LiteralPath 'runtime/provenance/config.json') {
    throw 'Legacy runtime detected. Stop old training and archive runtime before starting the base + small suite.'
}

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
$ErrorActionPreference = 'Continue'
if ($Stage -ne 'package') {
    # Upgrade bundled pip before resolving modern normalized package metadata.
    & $venvPython -m pip install --upgrade pip --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    # General dependencies come from PyPI; require the exact CUDA wheel build.
    & $venvPython -m pip install 'torch==2.5.1+cu121' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $venvPython -m pip install -r requirements.txt --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies are inconsistent.' }
}
$students = if ($Student -eq 'both') { @('base', 'small') } else { @($Student) }
$previousWorkDir = $env:ATC_WORK_DIR
try {
    foreach ($studentName in $students) {
        $env:ATC_WORK_DIR = Join-Path $PSScriptRoot ('runtime/' + $studentName)
        $logDirectory = Join-Path $env:ATC_WORK_DIR 'logs'
        New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
        $studentConfig = if ($studentName -eq 'base') { 'configs/base.json' } else { 'configs/default.json' }
        if ($Config) { $studentConfig = $Config }
        $actualConfig = Get-Content -LiteralPath $studentConfig -Raw | ConvertFrom-Json
        if ($actualConfig.student -ne ('openai/whisper-' + $studentName)) {
            throw 'Student selection does not match config.'
        }
        Write-Host ('Starting ' + $studentName + ': CE / KD / KD+adv0.01 / KD+adv0.1; stage=' + $Stage)
        $pipelineArgs = @('src/pipeline.py', '--stage', $Stage, '--config', $studentConfig)
        & $venvPython -u @pipelineArgs 2>&1 | Tee-Object -FilePath (Join-Path $logDirectory 'launcher.log') -Append
        if ($LASTEXITCODE -ne 0) {
            throw ('Run failed: ' + $env:ATC_WORK_DIR + '. Check failure.json and logs/launcher.log; next student was not started.')
        }
    }
} finally {
    $env:ATC_WORK_DIR = $previousWorkDir
}
