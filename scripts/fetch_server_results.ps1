param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [Parameter(Mandatory = $true)]
    [string]$RemoteArchive,

    [string]$LocalRoot = "server_results"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp is not available. Install Windows OpenSSH Client or use Git Bash."
}

New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null

$archiveName = Split-Path -Leaf $RemoteArchive
$localArchive = Join-Path $LocalRoot $archiveName

scp "${Server}:${RemoteArchive}" $localArchive

if ($archiveName.EndsWith(".tar.gz")) {
    tar -xzf $localArchive -C $LocalRoot
}

Write-Host "Downloaded: $localArchive"
Write-Host "Extracted under: $LocalRoot"

