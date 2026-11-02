$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$uiPythonCandidates = @(
    (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
    (Join-Path $PSScriptRoot '../.audit/venv/Scripts/python.exe')
)
$uiPython = $null
foreach ($candidate in $uiPythonCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        & $candidate -c 'import streamlit, torch, PIL, numpy' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $uiPython = $candidate
            break
        }
    }
}
if (-not $uiPython) {
    Write-Host 'Install the interface in a Python 3.12 environment first:'
    Write-Host '  python -m venv .venv'
    Write-Host '  .\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt'
    exit 1
}
Write-Host 'Open http://localhost:8501 in your browser. Press Ctrl+C here to stop.'
& $uiPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.port 8501
exit $LASTEXITCODE
