Param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$addonDir = Join-Path $RepoRoot "blender_addons\project_r"
$zipPath  = Join-Path $RepoRoot "project_r.zip"

if (-Not (Test-Path $addonDir)) {
  Write-Error "Addon folder not found: $addonDir"
  exit 1
}

if (Test-Path $zipPath) {
  Remove-Item -Force $zipPath
}

# Create zip with folder root 'project_r' (so Blender installs it as an addon package)
Compress-Archive -Path $addonDir -DestinationPath $zipPath -Force
Write-Output "Wrote $zipPath"


