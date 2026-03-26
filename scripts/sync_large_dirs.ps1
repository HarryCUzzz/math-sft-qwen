param(
    [ValidateSet("pull", "push")]
    [string]$Direction = "pull",

    [Parameter(Mandatory = $true)]
    [string]$RemoteProjectDir,

    [string]$RemoteUser = "lyl",
    [string]$RemoteHost = "137.189.63.17",
    [int]$Port = 4536,
    [string]$KeyPath = "",
    [switch]$Delete,
    [string[]]$Folders = @("data", "outputs", "outputs_qwen35")
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$rsync = Get-Command rsync -ErrorAction SilentlyContinue
if (-not $rsync) {
    throw "rsync is not installed on this machine. Install a Windows rsync build first, then rerun this script."
}

$sshParts = @("ssh", "-p", "$Port")
if ($KeyPath) {
    if (-not (Test-Path $KeyPath)) {
        throw "SSH key not found: $KeyPath"
    }
    $sshParts += @("-i", $KeyPath)
}
$sshCommand = ($sshParts -join " ")
$remotePrefix = "{0}@{1}" -f $RemoteUser, $RemoteHost

Push-Location $repoRoot
try {
    foreach ($folder in $Folders) {
        $localPath = Join-Path $repoRoot $folder
        if (-not (Test-Path $localPath)) {
            New-Item -ItemType Directory -Path $localPath | Out-Null
        }

        $remoteDir = "$RemoteProjectDir/$folder"
        $rsyncArgs = @(
            "-az",
            "--partial",
            "--info=progress2",
            "-e", $sshCommand
        )
        if ($Delete) {
            $rsyncArgs += "--delete"
        }

        if ($Direction -eq "push") {
            $sshArgs = @("-p", "$Port")
            if ($KeyPath) {
                $sshArgs += @("-i", $KeyPath)
            }
            $sshArgs += @($remotePrefix, "mkdir -p '$remoteDir'")
            & ssh @sshArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to prepare remote directory: $remoteDir"
            }
            $source = "./$folder/"
            $destination = "${remotePrefix}:$remoteDir/"
        }
        else {
            $source = "${remotePrefix}:$remoteDir/"
            $destination = "./$folder/"
        }

        Write-Host "Syncing $folder ($Direction)..." -ForegroundColor Cyan
        & $rsync.Source @rsyncArgs $source $destination
        if ($LASTEXITCODE -ne 0) {
            throw "rsync failed for $folder"
        }
    }
}
finally {
    Pop-Location
}