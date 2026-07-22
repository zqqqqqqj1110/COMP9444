# Edit only this configuration section for a new PPO experiment.
$Scenario = "AirSimNH"
$SceneExe = "D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$RunName = "stage03_diagonal_10x5m_seed7"
$ResumeModel = "D:\AirSim\rl_drone_navigation\experiments\airsimnh\ppo\stage02_10m_seed7\models\ppo_final.pt"
$ResumeOptimizer = $false

$Episodes = 250
$MaxSteps = 200
$RolloutSteps = 512
$TargetX = 10.0
$TargetY = 5.0
$TargetZ = -3.0
$StartX = 0.0
$StartY = 0.0
$StartZ = -3.0

$LearningRate = 7.5e-5
$BatchSize = 64
$UpdateEpochs = 4
$RewardScale = 0.1
$ValueLoss = "huber"
$BestWindow = 20
$BestMinEpisodes = 20
$CheckpointEvery = 10
$Seed = 7

$EvaluateAfterTraining = $true
$EvaluateBestModel = $true
$EvaluationEpisodes = 20
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
if ($Episodes -le 0 -or $MaxSteps -le 0 -or $RolloutSteps -le 0) {
    throw "Episodes, MaxSteps, and RolloutSteps must be positive."
}
if ($BatchSize -le 0 -or $BatchSize -gt $RolloutSteps) {
    throw "BatchSize must be positive and no larger than RolloutSteps."
}
if ($RewardScale -le 0 -or $BestWindow -le 0 -or $BestMinEpisodes -le 0 -or $BestMinEpisodes -gt $BestWindow) {
    throw "Stable PPO reward scale and best-model window settings are invalid."
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
Write-Host "  Episodes:       $Episodes"
Write-Host "  Max steps:      $MaxSteps"
Write-Host "  Rollout steps:  $RolloutSteps"
Write-Host "  Learning rate:  $LearningRate"
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
    "--max-steps", $MaxSteps,
    "--rollout-steps", $RolloutSteps,
    "--target-x", $TargetX,
    "--target-y", $TargetY,
    "--target-z", $TargetZ,
    "--start-x", $StartX,
    "--start-y", $StartY,
    "--start-z", $StartZ,
    "--learning-rate", $LearningRate,
    "--reward-scale", $RewardScale,
    "--value-loss", $ValueLoss,
    "--best-window", $BestWindow,
    "--best-min-episodes", $BestMinEpisodes,
    "--batch-size", $BatchSize,
    "--update-epochs", $UpdateEpochs,
    "--checkpoint-every", $CheckpointEvery,
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

if ($EvaluateAfterTraining) {
    Write-Host ""
    Write-Host "Running $EvaluationEpisodes deterministic evaluation episodes..."
    $evaluationModel = if ($EvaluateBestModel) {
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
