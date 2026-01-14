# PowerShell script to run backend locally with correct PYTHONPATH
# This ensures absolute imports work correctly

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Get project root (two levels up from backend directory)
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"

Write-Host "Starting backend server..."
Write-Host "Script directory: $scriptDir"
Write-Host "Project root: $projectRoot"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host ""

# Change to backend directory
Set-Location $scriptDir

# Start uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
