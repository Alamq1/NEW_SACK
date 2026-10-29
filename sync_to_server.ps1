# Watches this folder and copies every created/changed file to the GPU server.
# Usage: powershell -ExecutionPolicy Bypass -File sync_to_server.ps1
# Deletions are NOT mirrored to the server (safer for datasets/weights).

$Local  = $PSScriptRoot
$Remote = "aiteam@192.168.2.240"
$RemoteDir = "/home/aiteam/ZONE_COUNTING"
$Exclude = '\\(\.git|__pycache__|\.vscode)(\\|$)|\.(tmp|swp|pyc)$|~$'

function Push-File($path) {
    if ($path -match $Exclude -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { return }
    $rel = $path.Substring($Local.Length).TrimStart('\') -replace '\\', '/'
    $dir = ($rel -replace '/?[^/]+$', '')
    $target = if ($dir) { "$RemoteDir/$dir" } else { $RemoteDir }
    ssh -o BatchMode=yes $Remote "mkdir -p '$target'"
    scp -q -o BatchMode=yes -- "$path" "${Remote}:$RemoteDir/$rel"
    if ($LASTEXITCODE -eq 0) { Write-Host "$(Get-Date -f HH:mm:ss) synced $rel" }
    else { Write-Host "$(Get-Date -f HH:mm:ss) FAILED $rel" -ForegroundColor Red }
}

$w = New-Object System.IO.FileSystemWatcher $Local
$w.IncludeSubdirectories = $true
$w.NotifyFilter = 'FileName, LastWrite, Size'
$w.EnableRaisingEvents = $true
Write-Host "Watching $Local -> ${Remote}:$RemoteDir  (Ctrl+C to stop)"

$pending = @{}
while ($true) {
    $r = $w.WaitForChanged('All', 1000)
    if (-not $r.TimedOut) { $pending[(Join-Path $Local $r.Name)] = Get-Date }
    # debounce: push files that have been quiet for 1.5s
    foreach ($p in @($pending.Keys)) {
        if (((Get-Date) - $pending[$p]).TotalSeconds -gt 1.5) { $pending.Remove($p); Push-File $p }
    }
}
