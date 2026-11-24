# Starts the night guard detached from any terminal (it survives closing VS Code / Claude Code).
# Run in PowerShell from anywhere. Edit $Repo if the repository is elsewhere.
#   powershell -ExecutionPolicy Bypass -File bmuon\tools\pc\start_guard.ps1
param([string]$Plan = "bmuon/experiments/wide_arch_runs.yml",
      [string]$Out = "bmuon/results/wide_arch_guard.out",
      [string]$Repo = "$env:USERPROFILE\OneDrive\Documents\EA\code")
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*night_guard.sh*' -and $_.Name -eq 'bash.exe' }
if ($running) { Write-Output "guard already running: $($running.ProcessId -join ', ')"; exit 1 }
$unix = '/' + $Repo.Substring(0, 1).ToLower() + $Repo.Substring(2).Replace('\', '/')
$cmd = "`"C:\Program Files\Git\bin\bash.exe`" -lc `"cd '$unix' && INTERVAL=600 WINDOW=1 exec bash bmuon/experiments/night_guard.sh $Plan >> $Out 2>&1 < /dev/null`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
Write-Output "started, pid $($r.ProcessId) (ReturnValue $($r.ReturnValue))"
