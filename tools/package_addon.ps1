Param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

# Package the Blender addon into project_r.zip from GIT-TRACKED files only.
#
# Building from the working directory (the old approach) packaged whatever happened to
# be on disk -- untracked WIP, __pycache__, or files an external sync (OneDrive) drops
# in -- which silently polluted the zip. Listing tracked files via `git ls-files` keeps
# the package in lock-step with what is actually committed on the current branch.

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

# Stage the tracked files (preserving structure) under a temp 'project_r/' root so the
# zip installs as a proper Blender addon package.
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("project_r_pkg_" + [System.Guid]::NewGuid().ToString("N"))
$dest = Join-Path $staging "project_r"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
try {
  foreach ($rel in $tracked) {
    $src = Join-Path $RepoRoot ($rel -replace '/', '\')
    $sub = $rel.Substring($addonRel.Length).TrimStart('/')
    $target = Join-Path $dest ($sub -replace '/', '\')
    $targetDir = Split-Path $target -Parent
    if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Force -Path $targetDir | Out-Null }
    Copy-Item -LiteralPath $src -Destination $target -Force
  }
  if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
  Compress-Archive -Path $dest -DestinationPath $zipPath -Force
}
finally {
  Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
}
Write-Output "Wrote $zipPath ($($tracked.Count) tracked files)"
