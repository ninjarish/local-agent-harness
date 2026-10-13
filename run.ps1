$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
python -m local_agent --workspace (Join-Path $projectRoot "workspace") @args

