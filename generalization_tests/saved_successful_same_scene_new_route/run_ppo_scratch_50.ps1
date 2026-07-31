param(
    [int]$Episodes = 50,
    [int]$Seed = 20007
)

$ErrorActionPreference = "Stop"
$TestRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $TestRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AirSimExe = Join-Path $ProjectRoot ".local\airsim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
$ModelPath = Join-Path $ProjectRoot "pretrained\airsimnh\ppo_scratch_seed7.pt"
$Route = Get-Content -LiteralPath (Join-Path $TestRoot "route.json") -Raw | ConvertFrom-Json
$ResultsRoot = Join-Path $TestRoot "results"
$MetricsRoot = Join-Path $TestRoot "metrics"

if ($Episodes -ne 50) { throw "The formal protocol requires exactly 50 episodes." }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path -LiteralPath $ModelPath)) { throw "Model not found: $ModelPath" }
if (Test-Path -LiteralPath $ResultsRoot) { Remove-Item -LiteralPath $ResultsRoot -Recurse -Force }
if (Test-Path -LiteralPath $MetricsRoot) { Remove-Item -LiteralPath $MetricsRoot -Recurse -Force }

Get-Process -Name "AirSimNH" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process -FilePath $AirSimExe -ArgumentList "-windowed", "-ResX=1280", "-ResY=720" -WindowStyle Normal
Start-Sleep -Seconds 12

& $Python -B (Join-Path $ProjectRoot "generalization_tests\show_route_markers.py") (Join-Path $TestRoot "route.json")
if ($LASTEXITCODE -ne 0) { throw "Could not display route markers." }

Write-Host "`n============================================================"
Write-Host "Model: PPO Scratch | deterministic | $Episodes episodes"
Write-Host "Route: saved successful same-scene new route"
Write-Host "============================================================"
& $Python (Join-Path $ProjectRoot "src\evaluate.py") `
    --algorithm ppo --scenario airsimnh --model $ModelPath `
    --policy-mode deterministic --episodes $Episodes --seed $Seed --max-steps $Route.max_steps `
    --start-x $Route.start.x --start-y $Route.start.y --start-z $Route.start.z `
    --target-x $Route.target.x --target-y $Route.target.y --target-z $Route.target.z `
    --results-dir (Join-Path $ResultsRoot "ppo_scratch") --show-progress
if ($LASTEXITCODE -ne 0) { throw "PPO Scratch evaluation failed." }

& $Python (Join-Path $TestRoot "summarize_results.py")
if ($LASTEXITCODE -ne 0) { throw "Could not summarize results." }
Write-Host "`nCompleted formal PPO Scratch same-scene new-route test: $TestRoot"
