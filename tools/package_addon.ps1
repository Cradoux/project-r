Param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

# Package the Blender extension into project_r.zip from GIT-TRACKED files only.
#
# Canonical builder: `blender --command extension build` (reads blender_manifest.toml
# and its [build] excludes). This script is a no-Blender-on-PATH fallback that
# reproduces the same EXTENSION layout: blender_manifest.toml and the package files
# live at the ROOT of the zip (NOT inside a project_r/ subfolder) so Blender 4.2+
# "Install from Disk" recognises it as an extension. Listing files via `git ls-files`
# keeps the package in lock-step with what is committed on the current branch.

$ErrorActionPreference = "Stop"
$addonRel = "blender_addons/project_r"
$zipPath  = Join-Path $RepoRoot "project_r.zip"

Push-Location $RepoRoot
try {
  $tracked = & git ls-files -- $addonRel
} finally {
  Pop-Location
}
if (-not $tracked) {
  Write-Error "No git-tracked files found under $addonRel"
  exit 1
}
if (-not ($tracked | Where-Object { $_ -eq "$addonRel/blender_manifest.toml" })) {
  Write-Error "blender_manifest.toml is not git-tracked -- run 'git add $addonRel/blender_manifest.toml' first, or the zip would be an invalid extension."
  exit 1
}

# Stage the tracked files (preserving structure) at the staging ROOT so the zip is a
# flat extension package (manifest + code at the top level).
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("project_r_pkg_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $staging | Out-Null
try {
  foreach ($rel in $tracked) {
    $src = Join-Path $RepoRoot ($rel -replace '/', '\')
    $sub = $rel.Substring($addonRel.Length).TrimStart('/')
    $target = Join-Path $staging ($sub -replace '/', '\')
    $targetDir = Split-Path $target -Parent
    if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Force -Path $targetDir | Out-Null }
    Copy-Item -LiteralPath $src -Destination $target -Force
  }
  if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
  Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipPath -Force
}
finally {
  Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
}
Write-Output "Wrote $zipPath ($($tracked.Count) tracked files, extension layout)"
