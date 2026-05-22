# Dot-source this script to load variables from .env into the current PowerShell session.
# Usage:    . .\scripts\apply_env.ps1
# Note: lines starting with # are ignored. Values are not exported to child shells; use setx for permanence.

$envFile = Join-Path -Path (Resolve-Path ".").Path -ChildPath ".env"
if (-not (Test-Path $envFile)) {
    Write-Host ".env not found at project root: $envFile" -ForegroundColor Yellow
    return
}

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) { return }
    $parts = $line -split "=", 2
    if ($parts.Length -ne 2) { return }
    $key = $parts[0].Trim()
    $value = $parts[1].Trim('"')
    # set in current session
    Set-Item -Path "Env:$key" -Value $value
    Write-Host "Set Env:$key (masked) =" + ($value.Substring(0,[math]::Min(4,$value.Length))) + "..." -ForegroundColor Green
}

Write-Host "Environment variables loaded into current PowerShell session. Restart your shell or use setx to persist." -ForegroundColor Cyan
