# Runs a script on the GPU server (not on this PC), streaming output here.
# Usage: powershell -ExecutionPolicy Bypass -File run_on_server.ps1 [script.py] [args...]
# Default script: test_zone_counting_v1_ui.py (Flask UI on port 5004, forwarded to http://localhost:5004)

param(
    [string]$Script = "test_zone_counting_v1_ui.py",
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$ScriptArgs
)

$Remote = "aiteam@192.168.2.240"
$RemoteDir = "/home/aiteam/ZONE_COUNTING"
$Python = "/home/aiteam/anaconda3/bin/python"
$Port = 5004

# make sure the latest version of the script is on the server
scp -q -o BatchMode=yes -- (Join-Path $PSScriptRoot $Script) "${Remote}:$RemoteDir/$Script"
if ($LASTEXITCODE -ne 0) { Write-Host "Upload failed" -ForegroundColor Red; exit 1 }

Write-Host "Running $Script on GPU server (UI: http://localhost:$Port if it serves one). Ctrl+C stops it."
ssh -t -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -L "${Port}:localhost:${Port}" $Remote "cd $RemoteDir && $Python -u $Script $($ScriptArgs -join ' ')"
