# Edit only this configuration section before recording a model.
$Scenario = "AirSimNH"
$SceneExe = "D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$Algorithm = "ppo"
$Model = "D:\AirSim\rl_drone_navigation\experiments\airsimnh\ppo\curriculum_stage02_23m_10k_seed7_stable_v2_stage2_pilot\models\ppo_best.pt"
$PolicyMode = "stochastic"
$Episodes = 10
$StopAfterSuccess = $true
$MaxSteps = 110
$Seed = 7

$StartX = 85.413
$StartY = -15.334
$StartZ = -3.0
$TargetX = 107.635
$TargetY = -10.842
$TargetZ = -3.0

$CameraName = "0"
$VideoWidth = 960
$VideoHeight = 540
$VideoFps = 3

$PythonExe = "C:\Users\User\miniconda3\envs\airsim-rl\python.exe"
$AutoStartScene = $true
$CloseSceneAfterRun = $true
$AirSimHost = "127.0.0.1"
$AirSimPort = 41451
$ConnectionTimeoutSeconds = 180
$SceneWarmupSeconds = 5
# End of configuration section.

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Test-AirSimPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutMilliseconds = 1000
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait($TimeoutMilliseconds)) {
            return $false
        }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Stop-ConfiguredAirSimScene {
    param([string]$ExecutablePath)

    $sceneDirectory = (Resolve-Path -LiteralPath (Split-Path -Parent $ExecutablePath)).Path
    $sceneStem = [System.IO.Path]::GetFileNameWithoutExtension($ExecutablePath)
    $processIds = @()

    try {
        $processIds = @(
            Get-CimInstance Win32_Process -ErrorAction Stop |
                Where-Object {
                    $_.ExecutablePath -and
                    [System.IO.Path]::GetFullPath($_.ExecutablePath).StartsWith(
                        $sceneDirectory,
                        [System.StringComparison]::OrdinalIgnoreCase
                    )
                } |
                Select-Object -ExpandProperty ProcessId
        )
    }
    catch {
        Write-Warning "Could not inspect scene process paths: $($_.Exception.Message)"
    }

    if ($processIds.Count -eq 0) {
        $processIds = @(
            Get-Process -Name "$sceneStem*" -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Id
        )
    }

    foreach ($processId in ($processIds | Sort-Object -Unique)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    if ($processIds.Count -gt 0) {
        Write-Host "Closed AirSim scene: $sceneStem"
    }
}

if ($Algorithm -notin @("dqn", "ppo")) {
    throw "Algorithm must be 'dqn' or 'ppo'."
}
if ($PolicyMode -notin @("deterministic", "stochastic")) {
    throw "PolicyMode must be 'deterministic' or 'stochastic'."
}
if ($Algorithm -eq "dqn" -and $PolicyMode -eq "stochastic") {
    throw "Stochastic inference is supported only for PPO."
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model not found: $Model"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$modelDirectory = Split-Path -Parent (Split-Path -Parent $Model)
$outputDirectory = Join-Path $modelDirectory "recordings\${PolicyMode}_$timestamp"
$sceneStartedByScript = $false

if (-not (Test-AirSimPort -HostName $AirSimHost -Port $AirSimPort)) {
    if (-not $AutoStartScene) {
        throw "AirSim is not running. Start the configured scene or set AutoStartScene to true."
    }
    if (-not (Test-Path -LiteralPath $SceneExe -PathType Leaf)) {
        throw "Scene executable not found: $SceneExe"
    }

    Write-Host "Starting AirSim scene: $SceneExe"
    Start-Process `
        -FilePath $SceneExe `
        -WorkingDirectory (Split-Path -Parent $SceneExe) | Out-Null
    $sceneStartedByScript = $true
}
else {
    Write-Warning "AirSim is already running. Confirm that the open scene matches '$Scenario'."
}

try {
    Write-Host "Waiting for AirSim at ${AirSimHost}:${AirSimPort}..."
    $deadline = [DateTime]::UtcNow.AddSeconds($ConnectionTimeoutSeconds)
    while (-not (Test-AirSimPort -HostName $AirSimHost -Port $AirSimPort)) {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "AirSim did not become ready within $ConnectionTimeoutSeconds seconds."
        }
        Start-Sleep -Seconds 2
    }
    if ($SceneWarmupSeconds -gt 0) {
        Start-Sleep -Seconds $SceneWarmupSeconds
    }

    Write-Host ""
    Write-Host "Inference recording configuration"
    Write-Host "  Scenario:      $Scenario"
    Write-Host "  Algorithm:     $Algorithm"
    Write-Host "  Model:         $Model"
    Write-Host "  Policy mode:   $PolicyMode"
    Write-Host "  Attempts:      $Episodes"
    Write-Host "  Stop success:  $StopAfterSuccess"
    Write-Host "  Start:         ($StartX, $StartY, $StartZ)"
    Write-Host "  Target:        ($TargetX, $TargetY, $TargetZ)"
    Write-Host "  Output:        $outputDirectory"
    Write-Host ""

    $arguments = @(
        "src\infer_and_record.py",
        "--algorithm", $Algorithm,
        "--model", $Model,
        "--policy-mode", $PolicyMode,
        "--episodes", $Episodes,
        "--max-steps", $MaxSteps,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ,
        "--camera-name", $CameraName,
        "--video-width", $VideoWidth,
        "--video-height", $VideoHeight,
        "--fps", $VideoFps,
        "--seed", $Seed,
        "--output-dir", $outputDirectory
    )
    if ($StopAfterSuccess) {
        $arguments += "--stop-after-success"
    }

    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Inference recording failed with exit code $LASTEXITCODE."
    }
    Write-Host ""
    Write-Host "Recording complete: $outputDirectory"
}
finally {
    if ($CloseSceneAfterRun) {
        Stop-ConfiguredAirSimScene -ExecutablePath $SceneExe
    }
}
