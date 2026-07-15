# Edit only this configuration section for a new PPO experiment.
$Scenario = "AirSimNH"
$SceneExe = "D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$RunName = "stage02_10m_seed7"
$ResumeModel = "D:\AirSim\rl_drone_navigation\experiments\airsimnh\ppo\stage01_5m_seed7\models\ppo_final.pt"

$Episodes = 200
$MaxSteps = 150
$RolloutSteps = 512
$TargetX = 10.0
$TargetY = 0.0
$TargetZ = -3.0
$StartX = 0.0
$StartY = 0.0
$StartZ = -3.0

$LearningRate = 1e-4
$BatchSize = 64
$UpdateEpochs = 4
$CheckpointEvery = 10
$Seed = 7

$EvaluateAfterTraining = $true
$EvaluationEpisodes = 20
$RunSmokeTest = $true
$SmokeTestSteps = 3

$PythonExe = "C:\Users\User\miniconda3\envs\airsim-rl\python.exe"
$AutoStartScene = $true
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

function ConvertTo-ExperimentSlug {
    param([string]$Value)

    $slug = [System.Text.RegularExpressions.Regex]::Replace($Value.Trim().ToLowerInvariant(), "[^a-z0-9._-]+", "_")
    $slug = $slug.Trim([char[]]"._-")
    if ([string]::IsNullOrWhiteSpace($slug)) {
        throw "Scenario and RunName must contain at least one letter or number."
    }
    return $slug
}

if ([string]::IsNullOrWhiteSpace($Scenario)) {
    throw "Scenario cannot be empty."
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    throw "RunName cannot be empty. Use a descriptive name such as stage01_5m_seed7."
}
if ($Episodes -le 0 -or $MaxSteps -le 0 -or $RolloutSteps -le 0) {
    throw "Episodes, MaxSteps, and RolloutSteps must be positive."
}
if ($BatchSize -le 0 -or $BatchSize -gt $RolloutSteps) {
    throw "BatchSize must be positive and no larger than RolloutSteps."
}
if ($EvaluationEpisodes -le 0) {
    throw "EvaluationEpisodes must be positive."
}
if ([System.IO.Path]::IsPathRooted($PythonExe) -and -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

$resumeModelPath = $null
if (-not [string]::IsNullOrWhiteSpace($ResumeModel)) {
    if (-not (Test-Path -LiteralPath $ResumeModel -PathType Leaf)) {
        throw "Resume model not found: $ResumeModel"
    }
    $resumeModelPath = (Resolve-Path -LiteralPath $ResumeModel).Path
}

$scenarioSlug = ConvertTo-ExperimentSlug $Scenario
$runSlug = ConvertTo-ExperimentSlug $RunName
$runRoot = Join-Path $PSScriptRoot "experiments\$scenarioSlug\ppo\$runSlug"
if (Test-Path -LiteralPath $runRoot) {
    throw "Run already exists: $runRoot`nChoose a new RunName so existing results are not overwritten."
}

$sceneStartedByScript = $null
$portReady = Test-AirSimPort -HostName $AirSimHost -Port $AirSimPort

if (-not $portReady) {
    if (-not $AutoStartScene) {
        throw "AirSim is not running. Start the configured scene or set AutoStartScene to true."
    }
    if (-not (Test-Path -LiteralPath $SceneExe -PathType Leaf)) {
        throw "Scene executable not found: $SceneExe"
    }

    Write-Host "Starting AirSim scene: $SceneExe"
    $sceneStartedByScript = Start-Process `
        -FilePath $SceneExe `
        -WorkingDirectory (Split-Path -Parent $SceneExe) `
        -PassThru
}
else {
    Write-Warning "AirSim is already running. Confirm that the open scene matches '$Scenario'."
}

Write-Host "Waiting for AirSim at ${AirSimHost}:${AirSimPort}..."
$deadline = [DateTime]::UtcNow.AddSeconds($ConnectionTimeoutSeconds)
while (-not (Test-AirSimPort -HostName $AirSimHost -Port $AirSimPort)) {
    if ([DateTime]::UtcNow -ge $deadline) {
        throw "AirSim did not become ready within $ConnectionTimeoutSeconds seconds."
    }
    Start-Sleep -Seconds 2
}

if ($SceneWarmupSeconds -gt 0) {
    Write-Host "AirSim port is ready. Waiting $SceneWarmupSeconds more seconds for the scene..."
    Start-Sleep -Seconds $SceneWarmupSeconds
}

Write-Host ""
Write-Host "PPO training configuration"
Write-Host "  Scenario:       $scenarioSlug"
Write-Host "  Run:            $runSlug"
Write-Host "  Resume model:   $resumeModelPath"
Write-Host "  Target:         ($TargetX, $TargetY, $TargetZ)"
Write-Host "  Start:          ($StartX, $StartY, $StartZ)"
Write-Host "  Episodes:       $Episodes"
Write-Host "  Max steps:      $MaxSteps"
Write-Host "  Rollout steps:  $RolloutSteps"
Write-Host "  Learning rate:  $LearningRate"
Write-Host "  Output:         $runRoot"
Write-Host ""

if ($RunSmokeTest) {
    Write-Host "Running a clean-spawn smoke test..."
    $smokeArgs = @(
        "src\smoke_test_env.py",
        "--steps", $SmokeTestSteps,
        "--action", 5,
        "--require-clean",
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ
    )
    & $PythonExe @smokeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "AirSim spawn smoke test failed. Training was not started."
    }
    Write-Host "Spawn smoke test passed."
    Write-Host ""
}

$trainingArgs = @(
    "src\train_ppo.py",
    "--scenario", $scenarioSlug,
    "--run-name", $runSlug,
    "--episodes", $Episodes,
    "--max-steps", $MaxSteps,
    "--rollout-steps", $RolloutSteps,
    "--target-x", $TargetX,
    "--target-y", $TargetY,
    "--target-z", $TargetZ,
    "--start-x", $StartX,
    "--start-y", $StartY,
    "--start-z", $StartZ,
    "--learning-rate", $LearningRate,
    "--batch-size", $BatchSize,
    "--update-epochs", $UpdateEpochs,
    "--checkpoint-every", $CheckpointEvery,
    "--seed", $Seed
)

if ($null -ne $resumeModelPath) {
    $trainingArgs += @("--resume-model", $resumeModelPath)
}

& $PythonExe @trainingArgs
if ($LASTEXITCODE -ne 0) {
    throw "PPO training failed with exit code $LASTEXITCODE."
}

if ($EvaluateAfterTraining) {
    Write-Host ""
    Write-Host "Running $EvaluationEpisodes deterministic evaluation episodes..."
    $evaluationArgs = @(
        "src\evaluate.py",
        "--algorithm", "ppo",
        "--scenario", $scenarioSlug,
        "--run-name", $runSlug,
        "--episodes", $EvaluationEpisodes,
        "--max-steps", $MaxSteps,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ
    )
    & $PythonExe @evaluationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PPO evaluation failed with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Training complete. Results: $runRoot"
