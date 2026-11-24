# Old Windows PC, on the day of the move to the Mac: stops the night guard and packs into one
# archive on the Desktop what cannot go through GitHub (SSH key and config, Claude Code
# conversations and memory) plus bmuon/MAC_SETUP.md. Copy the archive to the Mac by USB stick
# (it contains the private SSH key: no e-mail, no cloud), then delete it.
#   powershell -ExecutionPolicy Bypass -File bmuon\tools\pc\prepare_mac_transfer.ps1
param([string]$Project = "$env:USERPROFILE\.claude\projects\c--Users-thoma-OneDrive-Documents-EA",
      [string]$Dest = "$env:USERPROFILE\transfert_mac.zip",   # outside OneDrive (the Desktop is synced)
      [switch]$KeepGuard)   # -KeepGuard: test run, the guard keeps running
$ErrorActionPreference = "Stop"
if (-not $KeepGuard) { & "$PSScriptRoot\stop_guard.ps1" }
$tmp = Join-Path $env:TEMP "transfert_mac"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force "$tmp\ssh", "$tmp\claude_project" | Out-Null
foreach ($f in "id_ed25519", "id_ed25519.pub", "config", "known_hosts") {
  Copy-Item "$env:USERPROFILE\.ssh\$f" "$tmp\ssh\"
}
Copy-Item -Recurse "$Project\*" "$tmp\claude_project\"
Copy-Item "$PSScriptRoot\..\..\MAC_SETUP.md" "$tmp\"
if (-not $KeepGuard) {
  "Night guard stopped on the old PC at $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File -Encoding utf8 "$tmp\GUARD_STOPPED.txt"
}
if (Test-Path $Dest) { Remove-Item -Force $Dest }
# zip entries with "/" separators (Compress-Archive of PowerShell 5.1 writes "\", which macOS unpacks badly)
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::Open($Dest, 'Create')
Get-ChildItem -Recurse -File $tmp | ForEach-Object {
  $rel = $_.FullName.Substring($tmp.Length + 1).Replace('\', '/')
  [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $rel, 'Optimal') | Out-Null
}
$zip.Dispose()
Remove-Item -Recurse -Force $tmp
$mb = [math]::Round((Get-Item $Dest).Length / 1MB, 1)
Write-Output "archive ready: $Dest ($mb MB)"
