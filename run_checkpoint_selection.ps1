# Select one validated model from each completed Seed 7 experiment.
$Scenario = "AirSimNH"
$SceneExe = "D:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$Seed = 7

$RunDqnScratch = $true
$RunPpoScratch = $true
$RunPpoCurriculum = $true

$DqnScratchRun = "scratch_33m_45k_seed7_stable_v3_scratch"
$PpoScratchRun = "scratch_33m_45k_seed7_stable_v3_scratch"
$PpoCurriculumRun = "curriculum_stage03_33m_30k_seed7_stable_v3_stage3_pilot"

$Stage1Episodes = 5
$TopK = 3
$Stage2Episodes = 30
$Stage2SeedOffset = 10000
$FinalTestEpisodes = 50
$FinalTestSeedOffset = 20000

$StartX = 85.413
$StartY = -15.334
$StartZ = -3.0
$TargetX = 117.756
$TargetY = -19.034
$TargetZ = -3.0
$MaxSteps = 150

$PythonExe = "C:\Users\User\miniconda3\envs\airsim-rl\python.exe"
$AutoStartScene = $true
$CloseSceneAfterRun = $true
$AirSimHost = "127.0.0.1"
$AirSimPort = 41451
$ConnectionTimeoutSeconds = 180
$SceneWarmupSeconds = 5
$RunSmokeTest = $true
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

function Invoke-PythonCommand {
    param(
        [string]$Description,
        [object[]]$Arguments
    )

    Write-Host ""
    Write-Host $Description
    & $PythonExe @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Assert-CompletedRun {
    param(
        [string]$Algorithm,
        [string]$RunName
    )

    $modelsDir = Join-Path $PSScriptRoot "experiments\airsimnh\$Algorithm\$RunName\models"
    $finalModel = Join-Path $modelsDir "${Algorithm}_final.pt"
    if (-not (Test-Path -LiteralPath $finalModel -PathType Leaf)) {
        throw "Completed model not found: $finalModel"
    }
    if (@(Get-ChildItem -LiteralPath $modelsDir -Filter "${Algorithm}_step_*.pt" -File).Count -eq 0) {
        throw "No step checkpoints found in: $modelsDir"
    }
}

function Assert-FinalTestOutputMissing {
    param(
        [string]$Algorithm,
        [string]$TrainingRun
    )

    $testSeed = $Seed + $FinalTestSeedOffset
    $testRun = "${TrainingRun}_validated_test_seed${testSeed}"
    $testRoot = Join-Path $PSScriptRoot "experiments\airsimnh\$Algorithm\$testRun"
    if (Test-Path -LiteralPath $testRoot) {
        throw "Final-test output already exists: $testRoot"
    }
}

function Invoke-TwoStageSelection {
    param(
        [string]$Algorithm,
        [string]$RunName
    )

    $arguments = @(
        "src\sweep_checkpoints.py",
        "--algorithm", $Algorithm,
        "--scenario", "airsimnh",
        "--run-name", $RunName,
        "--stage1-episodes", $Stage1Episodes,
        "--top-k", $TopK,
        "--stage2-episodes", $Stage2Episodes,
        "--stage2-seed-offset", $Stage2SeedOffset,
        "--max-steps", $MaxSteps,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ,
        "--seed", $Seed
    )
    Invoke-PythonCommand `
        -Description "Selecting the validated $Algorithm model for '$RunName'" `
        -Arguments $arguments
}

function Invoke-FinalTest {
    param(
        [string]$Algorithm,
        [string]$Method,
        [string]$TrainingRun
    )

    $testSeed = $Seed + $FinalTestSeedOffset
    $testRun = "${TrainingRun}_validated_test_seed${testSeed}"
    $testRoot = Join-Path $PSScriptRoot "experiments\airsimnh\$Algorithm\$testRun"
    if (Test-Path -LiteralPath $testRoot) {
        throw "Final-test output already exists: $testRoot"
    }

    $model = Join-Path $PSScriptRoot (
        "experiments\airsimnh\$Algorithm\$TrainingRun\models\${Algorithm}_best_deterministic.pt"
    )
    $policyMode = if ($Algorithm -eq "ppo") { "both" } else { "deterministic" }
    $arguments = @(
        "src\evaluate.py",
        "--algorithm", $Algorithm,
        "--scenario", "airsimnh",
        "--run-name", $testRun,
        "--model", $model,
        "--policy-mode", $policyMode,
        "--episodes", $FinalTestEpisodes,
        "--seed", $testSeed,
        "--max-steps", $MaxSteps,
        "--start-x", $StartX,
        "--start-y", $StartY,
        "--start-z", $StartZ,
        "--target-x", $TargetX,
        "--target-y", $TargetY,
        "--target-z", $TargetZ
    )
    Invoke-PythonCommand `
        -Description "Running the independent final test for $Method" `
        -Arguments $arguments

    return [pscustomobject]@{
        Algorithm = $Algorithm
        Method = $Method
        TrainingRun = $TrainingRun
        TestRun = $testRun
    }
}

function Export-ComparisonSummary {
    param([object[]]$CompletedTests)

    $rows = @()
    foreach ($test in $CompletedTests) {
        $summaryPath = Join-Path $PSScriptRoot (
            "experiments\airsimnh\$($test.Algorithm)\$($test.TestRun)\results\evaluation_summary.json"
        )
        $selectionPath = Join-Path $PSScriptRoot (
            "experiments\airsimnh\$($test.Algorithm)\$($test.TrainingRun)\results\checkpoint_sweep_two_stage_summary.json"
        )
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
        foreach ($mode in $summary.modes) {
            $rows += [pscustomobject]@{
                scenario = "airsimnh"
                seed = $Seed
                test_seed = $Seed + $FinalTestSeedOffset
                algorithm = $test.Algorithm
                method = $test.Method
                training_run = $test.TrainingRun
                policy_mode = $mode.policy_mode
                selected_checkpoint = [System.IO.Path]::GetFileName(
                    [string]$selection.selected_source_checkpoint
                )
                episodes = $mode.episodes
                success_rate = $mode.success_rate
                collision_rate = $mode.collision_rate
                altitude_violation_rate = $mode.altitude_violation_rate
                unsafe_rate = $mode.unsafe_rate
                timeout_rate = $mode.timeout_rate
                average_reward = $mode.average_reward
                average_steps = $mode.average_steps
                average_final_distance = $mode.average_final_distance
                average_path_length_m = $mode.average_path_length_m
                average_min_depth_m = $mode.average_min_depth_m
            }
        }
    }

    $outputPath = Join-Path $PSScriptRoot (
        "experiments\airsimnh\validated_comparison_seed${Seed}_test_seed$($Seed + $FinalTestSeedOffset).csv"
    )
    $rows | Export-Csv -LiteralPath $outputPath -NoTypeInformation
    return $outputPath
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if ($AutoStartScene -and -not (Test-Path -LiteralPath $SceneExe -PathType Leaf)) {
    throw "Scene executable not found: $SceneExe"
}
if ($Stage1Episodes -le 0 -or $TopK -le 0 -or $Stage2Episodes -le 0 `
    -or $FinalTestEpisodes -le 0) {
    throw "Selection and final-test episode counts must be positive."
}
if ($Stage2SeedOffset -eq 0 -or $FinalTestSeedOffset -eq 0 `
    -or $Stage2SeedOffset -eq $FinalTestSeedOffset) {
    throw "Selection and final-test seed offsets must be distinct and non-zero."
}

if ($RunDqnScratch) {
    Assert-CompletedRun -Algorithm "dqn" -RunName $DqnScratchRun
    Assert-FinalTestOutputMissing -Algorithm "dqn" -TrainingRun $DqnScratchRun
}
if ($RunPpoScratch) {
    Assert-CompletedRun -Algorithm "ppo" -RunName $PpoScratchRun
    Assert-FinalTestOutputMissing -Algorithm "ppo" -TrainingRun $PpoScratchRun
}
if ($RunPpoCurriculum) {
    Assert-CompletedRun -Algorithm "ppo" -RunName $PpoCurriculumRun
    Assert-FinalTestOutputMissing -Algorithm "ppo" -TrainingRun $PpoCurriculumRun
}

$portReady = Test-AirSimPort -HostName $AirSimHost -Port $AirSimPort
if (-not $portReady) {
    if (-not $AutoStartScene) {
        throw "AirSim is not running. Start it or set AutoStartScene to true."
    }
    Write-Host "Starting AirSim scene: $SceneExe"
    Start-Process `
        -FilePath $SceneExe `
        -WorkingDirectory (Split-Path -Parent $SceneExe) | Out-Null
}
else {
    Write-Warning "AirSim is already running. Confirm that the open scene is '$Scenario'."
}

$overallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
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
    Write-Host "Two-stage checkpoint selection"
    Write-Host "  Stage 1:       all checkpoints x $Stage1Episodes episodes"
    Write-Host "  Stage 2:       top $TopK x $Stage2Episodes episodes"
    Write-Host "  Final test:    selected model x $FinalTestEpisodes fresh-seed episodes"
    Write-Host "  Final route:   ($StartX, $StartY, $StartZ) -> ($TargetX, $TargetY, $TargetZ)"

    if ($RunSmokeTest) {
        $smokeArguments = @(
            "src\smoke_test_env.py",
            "--steps", 3,
            "--action", 5,
            "--require-clean",
            "--start-x", $StartX,
            "--start-y", $StartY,
            "--start-z", $StartZ,
            "--target-x", $TargetX,
            "--target-y", $TargetY,
            "--target-z", $TargetZ
        )
        Invoke-PythonCommand `
            -Description "Running the clean-spawn smoke test" `
            -Arguments $smokeArguments
    }

    $completedTests = @()
    if ($RunDqnScratch) {
        Invoke-TwoStageSelection -Algorithm "dqn" -RunName $DqnScratchRun
        $completedTests += Invoke-FinalTest `
            -Algorithm "dqn" `
            -Method "DQN Scratch" `
            -TrainingRun $DqnScratchRun
    }
    if ($RunPpoScratch) {
        Invoke-TwoStageSelection -Algorithm "ppo" -RunName $PpoScratchRun
        $completedTests += Invoke-FinalTest `
            -Algorithm "ppo" `
            -Method "PPO Scratch" `
            -TrainingRun $PpoScratchRun
    }
    if ($RunPpoCurriculum) {
        Invoke-TwoStageSelection -Algorithm "ppo" -RunName $PpoCurriculumRun
        $completedTests += Invoke-FinalTest `
            -Algorithm "ppo" `
            -Method "PPO Curriculum" `
            -TrainingRun $PpoCurriculumRun
    }

    $comparisonPath = Export-ComparisonSummary -CompletedTests $completedTests
    Write-Host ""
    Write-Host "Validated comparison complete."
    Write-Host "Comparison table: $comparisonPath"
}
finally {
    $overallStopwatch.Stop()
    Write-Host "Total selection/test time: $($overallStopwatch.Elapsed.TotalHours.ToString('F3')) hours"
    if ($CloseSceneAfterRun) {
        Stop-ConfiguredAirSimScene -ExecutablePath $SceneExe
    }
}
