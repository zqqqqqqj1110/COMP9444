# Edit this section on each machine before starting a comparison run.
$Scenario = "AirSimNH"
$SceneExe = "D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$Seed = 7
$RunTag = "stable_v2_stage2_pilot"
#$RunTag = "stable_v2"

# Enable any combination. A fresh run directory is required for every enabled job.
#$RunDqnScratch = $true
#$RunPpoScratch = $true
#$RunPpoCurriculum = $true
$RunDqnScratch = $false
$RunPpoScratch = $false
$RunPpoCurriculum = $true

$StartX = 85.413
$StartY = -15.334
$StartZ = -3.0

$Stage1TargetX = 95.190
$Stage1TargetY = -14.491
$Stage1TargetZ = -3.0
$Stage1TotalSteps = 5000
$Stage1MaxSteps = 70

$Stage2TargetX = 107.635
$Stage2TargetY = -10.842
$Stage2TargetZ = -3.0
$Stage2TotalSteps = 10000
$Stage2MaxSteps = 110

$FinalTargetX = 117.756
$FinalTargetY = -19.034
$FinalTargetZ = -3.0
$FinalTotalSteps = 45000
$FinalMaxSteps = 150
$Stage3TotalSteps = 30000

$PpoRolloutSteps = 500
$PpoBatchSize = 64
$PpoUpdateEpochs = 4
$PpoScratchLearningRate = 1e-4
$PpoStage1LearningRate = 1e-4
$PpoStage2LearningRate = 1e-4
$PpoStage3LearningRate = 7.5e-5
$PpoRewardScale = 0.1
$PpoValueLoss = "huber"
$PpoBestWindow = 20
$PpoBestMinEpisodes = 20
$CheckpointEverySteps = 5000

$EvaluateAt30000Steps = $true
$EvaluatePpoBestCheckpoints = $true
$IntermediateEvaluationEpisodes = 30
$FinalEvaluationEpisodes = 50
$RequireCurriculumGates = $true
#$CurriculumLastStage = 3
$CurriculumLastStage = 2
$CurriculumGateEpisodes = 20
$Stage1MinimumSuccessRate = 0.80
$Stage2MinimumSuccessRate = 0.70
$MaximumGateUnsafeRate = 0.20
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

    $slug = [System.Text.RegularExpressions.Regex]::Replace(
        $Value.Trim().ToLowerInvariant(),
        "[^a-z0-9._-]+",
        "_"
    )
    $slug = $slug.Trim([char[]]"._-")
    if ([string]::IsNullOrWhiteSpace($slug)) {
        throw "Scenario and run names must contain at least one letter or number."
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

    foreach ($processId in ($processIds | Sort-Object -Unique)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }

    if ($processIds.Count -gt 0) {
        Write-Host "Closed AirSim scene: $sceneStem"
    }
}

function Invoke-PythonCommand {
    param(
        [string]$Description,
        [object[]]$Arguments
    )

    Write-Host ""
    Write-Host $Description
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Invoke-TimedPythonTraining {
    param(
        [string]$Description,
        [string]$Algorithm,
        [string]$Method,
        [string]$Stage,
        [string]$RunName,
        [object[]]$Arguments
    )

    $startedAt = Get-Date
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $status = "completed"
    try {
        Invoke-PythonCommand -Description $Description -Arguments $Arguments
    }
    catch {
        $status = "failed"
        throw
    }
    finally {
        $stopwatch.Stop()
        $completedAt = Get-Date
        $timingRow = [pscustomobject]@{
            scenario = $script:ScenarioSlug
            seed = $Seed
            algorithm = $Algorithm
            method = $Method
            stage = $Stage
            run_name = $RunName
            status = $status
            started_at = $startedAt.ToString("o")
            completed_at = $completedAt.ToString("o")
            elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
            elapsed_hours = [Math]::Round($stopwatch.Elapsed.TotalHours, 6)
        }
        $append = Test-Path -LiteralPath $script:TimingLogPath -PathType Leaf
        $timingRow | Export-Csv `
            -LiteralPath $script:TimingLogPath `
            -NoTypeInformation `
            -Append:$append
        Write-Host "Recorded training time: $($stopwatch.Elapsed.TotalHours.ToString('F3')) hours"
    }
}

function Assert-NewRun {
    param(
        [string]$Algorithm,
        [string]$RunName
    )

    $runRoot = Join-Path $PSScriptRoot "experiments\$script:ScenarioSlug\$Algorithm\$RunName"
    if (Test-Path -LiteralPath $runRoot) {
        throw "Run already exists: $runRoot`nDisable it or choose a new seed/run name."
    }
}

function Invoke-Evaluation {
    param(
        [string]$Algorithm,
        [string]$RunName,
        [string]$ModelPath,
        [int]$Episodes,
        [int]$MaxSteps = $FinalMaxSteps,
        [double]$TargetX = $FinalTargetX,
        [double]$TargetY = $FinalTargetY,
        [double]$TargetZ = $FinalTargetZ
    )

    $arguments = @(
        "src\evaluate.py",
        "--algorithm", $Algorithm,
        "--scenario", $script:ScenarioSlug,
        "--run-name", $RunName,
        "--model", $ModelPath,
        "--episodes", $Episodes,
        "--max-steps", $MaxSteps,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ
    )
    Invoke-PythonCommand -Description "Evaluating $Algorithm run '$RunName'" -Arguments $arguments
}

function Assert-CurriculumGate {
    param(
        [string]$RunName,
        [string]$ModelPath,
        [int]$MaxSteps,
        [double]$TargetX,
        [double]$TargetY,
        [double]$TargetZ,
        [double]$MinimumSuccessRate
    )

    $gateRunName = "${RunName}_gate"
    Invoke-Evaluation `
        -Algorithm "ppo" `
        -RunName $gateRunName `
        -ModelPath $ModelPath `
        -Episodes $CurriculumGateEpisodes `
        -MaxSteps $MaxSteps `
        -TargetX $TargetX `
        -TargetY $TargetY `
        -TargetZ $TargetZ

    $logPath = Join-Path $PSScriptRoot (
        "experiments\$script:ScenarioSlug\ppo\$gateRunName\results\evaluation_log.csv"
    )
    $rows = @(Import-Csv -LiteralPath $logPath)
    if ($rows.Count -eq 0) {
        throw "Curriculum gate produced no evaluation rows: $logPath"
    }

    $successRate = ($rows | Measure-Object -Property success -Average).Average
    $unsafeCount = @(
        $rows | Where-Object {
            [int]$_.collision -ne 0 -or [int]$_.out_of_altitude -ne 0
        }
    ).Count
    $unsafeRate = $unsafeCount / $rows.Count
    Write-Host (
        "Curriculum gate: success={0:P1}, unsafe={1:P1}, required success>={2:P1}, unsafe<={3:P1}" -f `
        $successRate, $unsafeRate, $MinimumSuccessRate, $MaximumGateUnsafeRate
    )

    if ($RequireCurriculumGates -and (
        $successRate -lt $MinimumSuccessRate -or $unsafeRate -gt $MaximumGateUnsafeRate
    )) {
        throw "Curriculum gate failed for '$RunName'. Later stages were not started."
    }
}

function Invoke-PpoTraining {
    param(
        [string]$RunName,
        [string]$Method,
        [string]$Stage,
        [int]$TotalSteps,
        [int]$MaxSteps,
        [double]$TargetX,
        [double]$TargetY,
        [double]$TargetZ,
        [double]$LearningRate,
        [string]$ResumeModel = ""
    )

    $arguments = @(
        "src\train_ppo.py",
        "--scenario", $script:ScenarioSlug,
        "--run-name", $RunName,
        "--total-steps", $TotalSteps,
        "--max-steps", $MaxSteps,
        "--rollout-steps", $PpoRolloutSteps,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ,
        "--learning-rate", $LearningRate,
        "--reward-scale", $PpoRewardScale,
        "--value-loss", $PpoValueLoss,
        "--best-window", $PpoBestWindow,
        "--best-min-episodes", $PpoBestMinEpisodes,
        "--batch-size", $PpoBatchSize,
        "--update-epochs", $PpoUpdateEpochs,
        "--checkpoint-every", 100000,
        "--checkpoint-every-steps", $CheckpointEverySteps,
        "--seed", $Seed
    )
    if (-not [string]::IsNullOrWhiteSpace($ResumeModel)) {
        $arguments += @("--resume-model", $ResumeModel)
    }

    Invoke-TimedPythonTraining `
        -Description "Training PPO run '$RunName'" `
        -Algorithm "ppo" `
        -Method $Method `
        -Stage $Stage `
        -RunName $RunName `
        -Arguments $arguments
}

if ([string]::IsNullOrWhiteSpace($Scenario)) {
    throw "Scenario cannot be empty."
}
if (-not ($RunDqnScratch -or $RunPpoScratch -or $RunPpoCurriculum)) {
    throw "Enable at least one comparison method."
}
if ($FinalTotalSteps -ne 45000) {
    throw "The approved scratch comparison budget is 45000 steps."
}
if (($Stage1TotalSteps + $Stage2TotalSteps + $Stage3TotalSteps) -ne $FinalTotalSteps) {
    throw "Curriculum and scratch methods must have the same total interaction budget."
}
if ($CheckpointEverySteps -le 0 -or 30000 % $CheckpointEverySteps -ne 0) {
    throw "CheckpointEverySteps must be positive and divide 30000 exactly."
}
if ($PpoRolloutSteps -le 0 -or $PpoBatchSize -le 0) {
    throw "PPO rollout and batch sizes must be positive."
}
if ($PpoRewardScale -le 0 -or $PpoBestWindow -le 0 -or $PpoBestMinEpisodes -le 0) {
    throw "PPO reward scale and best-model window settings must be positive."
}
if ($PpoBestMinEpisodes -gt $PpoBestWindow) {
    throw "PpoBestMinEpisodes cannot exceed PpoBestWindow."
}
if ($CurriculumGateEpisodes -le 0 -or $Stage1MinimumSuccessRate -lt 0 -or $Stage1MinimumSuccessRate -gt 1 `
    -or $Stage2MinimumSuccessRate -lt 0 -or $Stage2MinimumSuccessRate -gt 1 `
    -or $MaximumGateUnsafeRate -lt 0 -or $MaximumGateUnsafeRate -gt 1) {
    throw "Curriculum gate settings are invalid."
}
if ($CurriculumLastStage -lt 1 -or $CurriculumLastStage -gt 3) {
    throw "CurriculumLastStage must be 1, 2, or 3."
}
if ([System.IO.Path]::IsPathRooted($PythonExe) -and -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $SceneExe -PathType Leaf) -and $AutoStartScene) {
    throw "Scene executable not found: $SceneExe"
}

$ScenarioSlug = ConvertTo-ExperimentSlug $Scenario
$RunTagSlug = ConvertTo-ExperimentSlug $RunTag
$DqnRun = "scratch_33m_45k_seed${Seed}_${RunTagSlug}"
$PpoScratchRun = "scratch_33m_45k_seed${Seed}_${RunTagSlug}"
$PpoStage1Run = "curriculum_stage01_10m_5k_seed${Seed}_${RunTagSlug}"
$PpoStage2Run = "curriculum_stage02_23m_10k_seed${Seed}_${RunTagSlug}"
$PpoStage3Run = "curriculum_stage03_33m_30k_seed${Seed}_${RunTagSlug}"
$scenarioRoot = Join-Path $PSScriptRoot "experiments\$ScenarioSlug"
New-Item -ItemType Directory -Path $scenarioRoot -Force | Out-Null
$TimingLogPath = Join-Path $scenarioRoot "comparison_seed${Seed}_${RunTagSlug}_training_times.csv"

if ($RunDqnScratch) {
    Assert-NewRun -Algorithm "dqn" -RunName $DqnRun
}
if ($RunPpoScratch) {
    Assert-NewRun -Algorithm "ppo" -RunName $PpoScratchRun
}
if ($RunPpoCurriculum) {
    Assert-NewRun -Algorithm "ppo" -RunName $PpoStage1Run
    if ($CurriculumLastStage -ge 2) {
        Assert-NewRun -Algorithm "ppo" -RunName $PpoStage2Run
    }
    if ($CurriculumLastStage -ge 3) {
        Assert-NewRun -Algorithm "ppo" -RunName $PpoStage3Run
    }
}

$portReady = Test-AirSimPort -HostName $AirSimHost -Port $AirSimPort
if (-not $portReady) {
    if (-not $AutoStartScene) {
        throw "AirSim is not running. Start the configured scene or enable AutoStartScene."
    }
    Write-Host "Starting AirSim scene: $SceneExe"
    Start-Process `
        -FilePath $SceneExe `
        -WorkingDirectory (Split-Path -Parent $SceneExe) | Out-Null
}
else {
    Write-Warning "AirSim is already running. Confirm that the open scene is '$Scenario'."
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
    Write-Host "Comparison experiment"
    Write-Host "  Scenario:          $ScenarioSlug"
    Write-Host "  Seed:              $Seed"
    Write-Host "  Run tag:           $RunTagSlug"
    Write-Host "  Start:             ($StartX, $StartY, $StartZ)"
    Write-Host "  Stage 1 target:    ($Stage1TargetX, $Stage1TargetY, $Stage1TargetZ)"
    Write-Host "  Stage 2 target:    ($Stage2TargetX, $Stage2TargetY, $Stage2TargetZ)"
    Write-Host "  Final target:      ($FinalTargetX, $FinalTargetY, $FinalTargetZ)"
    Write-Host "  Scratch budget:    $FinalTotalSteps"
    Write-Host "  Curriculum budget: $Stage1TotalSteps + $Stage2TotalSteps + $Stage3TotalSteps"
    Write-Host "  PPO stabilisation: reward scale=$PpoRewardScale, value loss=$PpoValueLoss, optimizer reset between stages"
    Write-Host "  Curriculum gates:  $RequireCurriculumGates"
    Write-Host "  Curriculum stop:   Stage $CurriculumLastStage"

    if ($RunSmokeTest) {
        $smokeArguments = @(
            "src\smoke_test_env.py",
            "--steps", $SmokeTestSteps,
            "--action", 5,
            "--require-clean",
            "--start-x", $StartX,
            "--start-y", $StartY,
            "--start-z", $StartZ,
            "--target-x", $FinalTargetX,
            "--target-y", $FinalTargetY,
            "--target-z", $FinalTargetZ
        )
        Invoke-PythonCommand -Description "Running the clean-spawn smoke test" -Arguments $smokeArguments
    }

    if ($RunDqnScratch) {
        $dqnArguments = @(
            "src\train_dqn.py",
            "--scenario", $ScenarioSlug,
            "--run-name", $DqnRun,
            "--total-steps", $FinalTotalSteps,
            "--max-steps", $FinalMaxSteps,
            "--start-x", $StartX,
            "--start-y", $StartY,
            "--start-z", $StartZ,
            "--target-x", $FinalTargetX,
            "--target-y", $FinalTargetY,
            "--target-z", $FinalTargetZ,
            "--checkpoint-every-steps", $CheckpointEverySteps,
            "--seed", $Seed
        )
        Invoke-TimedPythonTraining `
            -Description "Training DQN scratch run '$DqnRun'" `
            -Algorithm "dqn" `
            -Method "scratch" `
            -Stage "final_33m" `
            -RunName $DqnRun `
            -Arguments $dqnArguments

        if ($EvaluateAt30000Steps) {
            $checkpoint = "experiments\$ScenarioSlug\dqn\$DqnRun\models\dqn_step_0030000.pt"
            Invoke-Evaluation `
                -Algorithm "dqn" `
                -RunName "${DqnRun}_eval_at_30k" `
                -ModelPath $checkpoint `
                -Episodes $IntermediateEvaluationEpisodes
        }
        $finalModel = "experiments\$ScenarioSlug\dqn\$DqnRun\models\dqn_final.pt"
        Invoke-Evaluation -Algorithm "dqn" -RunName $DqnRun -ModelPath $finalModel -Episodes $FinalEvaluationEpisodes
    }

    if ($RunPpoScratch) {
        Invoke-PpoTraining `
            -RunName $PpoScratchRun `
            -Method "scratch" `
            -Stage "final_33m" `
            -TotalSteps $FinalTotalSteps `
            -MaxSteps $FinalMaxSteps `
            -TargetX $FinalTargetX `
            -TargetY $FinalTargetY `
            -TargetZ $FinalTargetZ `
            -LearningRate $PpoScratchLearningRate

        if ($EvaluateAt30000Steps) {
            $checkpoint = "experiments\$ScenarioSlug\ppo\$PpoScratchRun\models\ppo_step_0030000.pt"
            Invoke-Evaluation `
                -Algorithm "ppo" `
                -RunName "${PpoScratchRun}_eval_at_30k" `
                -ModelPath $checkpoint `
                -Episodes $IntermediateEvaluationEpisodes
        }
        $finalModel = "experiments\$ScenarioSlug\ppo\$PpoScratchRun\models\ppo_final.pt"
        Invoke-Evaluation -Algorithm "ppo" -RunName $PpoScratchRun -ModelPath $finalModel -Episodes $FinalEvaluationEpisodes
        if ($EvaluatePpoBestCheckpoints) {
            $bestModel = "experiments\$ScenarioSlug\ppo\$PpoScratchRun\models\ppo_best.pt"
            Invoke-Evaluation `
                -Algorithm "ppo" `
                -RunName "${PpoScratchRun}_best" `
                -ModelPath $bestModel `
                -Episodes $FinalEvaluationEpisodes
        }
    }

    if ($RunPpoCurriculum) {
        Invoke-PpoTraining `
            -RunName $PpoStage1Run `
            -Method "curriculum" `
            -Stage "stage01_10m" `
            -TotalSteps $Stage1TotalSteps `
            -MaxSteps $Stage1MaxSteps `
            -TargetX $Stage1TargetX `
            -TargetY $Stage1TargetY `
            -TargetZ $Stage1TargetZ `
            -LearningRate $PpoStage1LearningRate

        $stage1Model = (Resolve-Path -LiteralPath (
            "experiments\$ScenarioSlug\ppo\$PpoStage1Run\models\ppo_best.pt"
        )).Path
        Assert-CurriculumGate `
            -RunName $PpoStage1Run `
            -ModelPath $stage1Model `
            -MaxSteps $Stage1MaxSteps `
            -TargetX $Stage1TargetX `
            -TargetY $Stage1TargetY `
            -TargetZ $Stage1TargetZ `
            -MinimumSuccessRate $Stage1MinimumSuccessRate

        if ($CurriculumLastStage -ge 2) {
            Invoke-PpoTraining `
                -RunName $PpoStage2Run `
                -Method "curriculum" `
                -Stage "stage02_23m" `
                -TotalSteps $Stage2TotalSteps `
                -MaxSteps $Stage2MaxSteps `
                -TargetX $Stage2TargetX `
                -TargetY $Stage2TargetY `
                -TargetZ $Stage2TargetZ `
                -LearningRate $PpoStage2LearningRate `
                -ResumeModel $stage1Model

            $stage2Model = (Resolve-Path -LiteralPath (
                "experiments\$ScenarioSlug\ppo\$PpoStage2Run\models\ppo_best.pt"
            )).Path
            Assert-CurriculumGate `
                -RunName $PpoStage2Run `
                -ModelPath $stage2Model `
                -MaxSteps $Stage2MaxSteps `
                -TargetX $Stage2TargetX `
                -TargetY $Stage2TargetY `
                -TargetZ $Stage2TargetZ `
                -MinimumSuccessRate $Stage2MinimumSuccessRate

            if ($CurriculumLastStage -ge 3) {
                Invoke-PpoTraining `
                    -RunName $PpoStage3Run `
                    -Method "curriculum" `
                    -Stage "stage03_33m" `
                    -TotalSteps $Stage3TotalSteps `
                    -MaxSteps $FinalMaxSteps `
                    -TargetX $FinalTargetX `
                    -TargetY $FinalTargetY `
                    -TargetZ $FinalTargetZ `
                    -LearningRate $PpoStage3LearningRate `
                    -ResumeModel $stage2Model

                if ($EvaluateAt30000Steps) {
                    # The first 15k interactions are in Stages 1 and 2, so 15k into Stage 3 is 30k total.
                    $checkpoint = "experiments\$ScenarioSlug\ppo\$PpoStage3Run\models\ppo_step_0015000.pt"
                    Invoke-Evaluation `
                        -Algorithm "ppo" `
                        -RunName "curriculum_33m_45k_seed${Seed}_${RunTagSlug}_eval_at_30k" `
                        -ModelPath $checkpoint `
                        -Episodes $IntermediateEvaluationEpisodes
                }
                $finalModel = "experiments\$ScenarioSlug\ppo\$PpoStage3Run\models\ppo_final.pt"
                Invoke-Evaluation -Algorithm "ppo" -RunName $PpoStage3Run -ModelPath $finalModel -Episodes $FinalEvaluationEpisodes
                if ($EvaluatePpoBestCheckpoints) {
                    $bestModel = "experiments\$ScenarioSlug\ppo\$PpoStage3Run\models\ppo_best.pt"
                    Invoke-Evaluation `
                        -Algorithm "ppo" `
                        -RunName "${PpoStage3Run}_best" `
                        -ModelPath $bestModel `
                        -Episodes $FinalEvaluationEpisodes
                }
            }
            else {
                Write-Host "Curriculum stopped after Stage 2 by configuration."
            }
        }
        else {
            Write-Host "Curriculum stopped after Stage 1 by configuration."
        }
    }

    Write-Host ""
    Write-Host "Comparison experiment complete for scenario '$ScenarioSlug', seed $Seed."
    Write-Host "Training-time summary: $TimingLogPath"
}
finally {
    if ($CloseSceneAfterRun) {
        Stop-ConfiguredAirSimScene -ExecutablePath $SceneExe
    }
}
