# Keeps the PC awake until a date (the guard stops working if the PC sleeps or hibernates).
#   powershell -ExecutionPolicy Bypass -File bmuon\tools\pc\keep_awake_until.ps1 -Until "2026-11-30 08:00"
# It cannot prevent hibernation on battery: keep the PC plugged in.
param([string]$Until)
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$end = [datetime]::Parse($Until)
while ((Get-Date) -lt $end) { [W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null; Start-Sleep 60 }
