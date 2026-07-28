# Edit only this configuration section for a new PPO experiment.
$Scenario = "AirSimNH"
$SceneExe = "D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$RunName = "curriculum_stage03_33m_30k_seed7_stable_v3_stage3_pilot"
$ResumeModel = "D:\AirSim\rl_drone_navigation\experiments\airsimnh\ppo\curriculum_stage02_23m_10k_seed7_stable_v3_stage2_pilot\models\ppo_best_deterministic.pt"
$ResumeOptimizer = $false

$TotalSteps = 30000
$Episodes = 100000
$MaxSteps = 150
$RolloutSteps = 500
$TargetX = 117.756
$TargetY = -19.034
$TargetZ = -3.0
$StartX = 85.413
$StartY = -15.334
$StartZ = -3.0

$LearningRate = 7.5e-5
$EntropyCoefStart = 0.01
$EntropyCoefEnd = 0.001
$BatchSize = 64
$UpdateEpochs = 4
$RewardScale = 0.1
$ValueLoss = "huber"
$BestWindow = 20
$BestMinEpisodes = 20
$CheckpointEvery = 100000
$CheckpointEverySteps = 2500
$CheckpointSweepEpisodes = 5
$Seed = 7

$RunCheckpointSweep = $true
$EvaluateAfterTraining = $true
$EvaluateBestModel = $true
$EvaluationEpisodes = 50
$RunSmokeTest = $true
$SmokeTestSteps = 3

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

function ConvertTo-ExperimentSlug {
    param([string]$Value)

    $slug = [System.Text.RegularExpressions.Regex]::Replace($Value.Trim().ToLowerInvariant(), "[^a-z0-9._-]+", "_")
    $slug = $slug.Trim([char[]]"._-")
    if ([string]::IsNullOrWhiteSpace($slug)) {
        throw "Scenario and RunName must contain at least one letter or number."
    }
    return $slug
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

    if ($processIds.Count -eq 0) {
        Write-Warning "No running process was found for the configured scene."
        return
    }

    foreach ($processId in ($processIds | Sort-Object -Unique)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Closed AirSim scene: $sceneStem"
}

if ([string]::IsNullOrWhiteSpace($Scenario)) {
    throw "Scenario cannot be empty."
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    throw "RunName cannot be empty. Use a descriptive name such as stage01_5m_seed7."
}
if ($TotalSteps -le 0 -or $Episodes -le 0 -or $MaxSteps -le 0 -or $RolloutSteps -le 0) {
    throw "TotalSteps, Episodes, MaxSteps, and RolloutSteps must be positive."
}
if ($BatchSize -le 0 -or $BatchSize -gt $RolloutSteps) {
    throw "BatchSize must be positive and no larger than RolloutSteps."
}
if ($RewardScale -le 0 -or $BestWindow -le 0 -or $BestMinEpisodes -le 0 -or $BestMinEpisodes -gt $BestWindow) {
    throw "Stable PPO reward scale and best-model window settings are invalid."
}
if ($EntropyCoefStart -lt 0 -or $EntropyCoefEnd -lt 0 -or $EntropyCoefEnd -gt $EntropyCoefStart) {
    throw "Entropy coefficients must satisfy 0 <= end <= start."
}
if ($CheckpointEverySteps -le 0 -or $CheckpointSweepEpisodes -le 0) {
    throw "CheckpointEverySteps and CheckpointSweepEpisodes must be positive."
}
if ($TotalSteps % $CheckpointEverySteps -ne 0) {
    throw "TotalSteps must be divisible by CheckpointEverySteps."
}
if ($ValueLoss -notin @("huber", "mse")) {
    throw "ValueLoss must be 'huber' or 'mse'."
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
    Write-Host "AirSim port is ready. Waiting $SceneWarmupSeconds more seconds for the scene..."
    Start-Sleep -Seconds $SceneWarmupSeconds
}

Write-Host ""
Write-Host "PPO training configuration"
Write-Host "  Scenario:       $scenarioSlug"
Write-Host "  Run:            $runSlug"
Write-Host "  Resume model:   $resumeModelPath"
Write-Host "  Resume optimizer: $ResumeOptimizer"
Write-Host "  Target:         ($TargetX, $TargetY, $TargetZ)"
Write-Host "  Start:          ($StartX, $StartY, $StartZ)"
Write-Host "  New step budget: $TotalSteps"
Write-Host "  Episode cap:    $Episodes"
Write-Host "  Max steps:      $MaxSteps"
Write-Host "  Rollout steps:  $RolloutSteps"
Write-Host "  Learning rate:  $LearningRate"
Write-Host "  Entropy:        $EntropyCoefStart -> $EntropyCoefEnd (linear)"
Write-Host "  Checkpoints:    every $CheckpointEverySteps steps"
Write-Host "  PPO stabilisation: reward scale=$RewardScale, value loss=$ValueLoss"
Write-Host "  Close scene:    $CloseSceneAfterRun"
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
    "--total-steps", $TotalSteps,
    "--max-steps", $MaxSteps,
    "--rollout-steps", $RolloutSteps,
    "--target-x", $TargetX,
    "--target-y", $TargetY,
    "--target-z", $TargetZ,
    "--start-x", $StartX,
    "--start-y", $StartY,
    "--start-z", $StartZ,
    "--learning-rate", $LearningRate,
    "--entropy-coef-start", $EntropyCoefStart,
    "--entropy-coef-end", $EntropyCoefEnd,
    "--reward-scale", $RewardScale,
    "--value-loss", $ValueLoss,
    "--best-window", $BestWindow,
    "--best-min-episodes", $BestMinEpisodes,
    "--batch-size", $BatchSize,
    "--update-epochs", $UpdateEpochs,
    "--checkpoint-every", $CheckpointEvery,
    "--checkpoint-every-steps", $CheckpointEverySteps,
    "--seed", $Seed
)

if ($null -ne $resumeModelPath) {
    $trainingArgs += @("--resume-model", $resumeModelPath)
    if ($ResumeOptimizer) {
        $trainingArgs += "--resume-optimizer"
    }
}

& $PythonExe @trainingArgs
if ($LASTEXITCODE -ne 0) {
    throw "PPO training failed with exit code $LASTEXITCODE."
}

if ($RunCheckpointSweep) {
    Write-Host ""
    Write-Host "Running deterministic checkpoint sweep..."
    $sweepArgs = @(
        "src\sweep_ppo_checkpoints.py",
        "--scenario", $scenarioSlug,
        "--run-name", $runSlug,
        "--episodes", $CheckpointSweepEpisodes,
        "--max-steps", $MaxSteps,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ,
        "--seed", $Seed
    )
    & $PythonExe @sweepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PPO checkpoint sweep failed with exit code $LASTEXITCODE."
    }
}

if ($EvaluateAfterTraining) {
    Write-Host ""
    Write-Host "Running $EvaluationEpisodes deterministic and stochastic evaluation episodes..."
    $evaluationModel = if ($EvaluateBestModel -and $RunCheckpointSweep) {
        Join-Path $runRoot "models\ppo_best_deterministic.pt"
    }
    elseif ($EvaluateBestModel) {
        Join-Path $runRoot "models\ppo_best.pt"
    }
    else {
        Join-Path $runRoot "models\ppo_final.pt"
    }
    $evaluationArgs = @(
        "src\evaluate.py",
        "--algorithm", "ppo",
        "--scenario", $scenarioSlug,
        "--run-name", $runSlug,
        "--model", $evaluationModel,
        "--policy-mode", "both",
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
}
finally {
    if ($CloseSceneAfterRun) {
        Stop-ConfiguredAirSimScene -ExecutablePath $SceneExe
    }
}
