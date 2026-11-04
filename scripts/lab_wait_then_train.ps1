# Start `train start` once another GPU program has been closed for a few minutes (GH-103).
#
# Run it from a scheduled task, so training outlives the SSH session that set it up and doesn't
# land on someone's active session. AGENTS.md, Lab PCs, has the task and how to stop it.
#
# It checks only until training starts: if the program is opened again later, training carries on.
# Every decision goes to logs\wait_for_gpu.log in the checkout.
param(
    # The program to wait for. Isaac Sim's is kit.exe.
    [string]$ProcessName = 'kit',
    # How many checks in a row must find it closed, so a quick close and reopen doesn't count.
    [int]$ClearChecks = 5,
    [int]$PollSeconds = 60,
    # What `train start` gets after --config. Relative to the checkout.
    [string]$StartArguments = 'fly_driver\configs\train_lab_vale.yaml'
)

$folder = Split-Path $PSScriptRoot -Parent
$log = Join-Path $folder 'logs\wait_for_gpu.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Write-Decision([string]$message) {
    "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $message | Add-Content -Path $log -Encoding utf8
}

Write-Decision "waiting for $ProcessName.exe to be closed for $ClearChecks checks, $PollSeconds s apart"
$clearCount = 0
$wasOpen = $null
while ($clearCount -lt $ClearChecks) {
    $isOpen = [bool](Get-Process -Name $ProcessName -ErrorAction SilentlyContinue)
    if ($isOpen -ne $wasOpen) {
        Write-Decision $(if ($isOpen) { "$ProcessName.exe is open" } else { "$ProcessName.exe is closed" })
        $wasOpen = $isOpen
    }
    if ($isOpen) { $clearCount = 0 } else { $clearCount += 1 }
    if ($clearCount -lt $ClearChecks) { Start-Sleep -Seconds $PollSeconds }
}

Write-Decision "clear: starting training"
Set-Location $folder
# The explicit .\ matters: cmd does not always search the current directory for a bare name.
& cmd /c ".\train.cmd start --config $StartArguments > logs\train_start.log 2>&1"
Write-Decision "train.cmd ended with exit code $LASTEXITCODE"
