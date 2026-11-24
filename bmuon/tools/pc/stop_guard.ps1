# Stops the night guard (the runs on the cluster keep going; nothing restarts them until a guard
# runs again). Never run two guards on the same plan at once (e.g. old PC and new PC).
$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*night_guard*' -and $_.ProcessId -ne $PID }
if (-not $p) { Write-Output "no guard running"; exit 0 }
$p | ForEach-Object { Write-Output "stopping $($_.ProcessId) $($_.Name)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
