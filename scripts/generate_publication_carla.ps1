param(
    [int]$Groups = 1500,
    [string]$OutputDir = "data/carla_tripaf_1024",
    [int]$MinVisibleTargets = 1,
    [int]$MaxSceneRetries = 6,
    [int]$MaxRefillRounds = 12
)

$ErrorActionPreference = "Stop"

Write-Host "Generating $Groups successful CARLA groups..."
python tools/generate_carla_dataset.py `
    --groups $Groups `
    --output-dir $OutputDir `
    --fog-min 20 `
    --fog-max 49.999 `
    --targets-min 4 `
    --targets-max 8 `
    --min-visible-targets $MinVisibleTargets `
    --max-scene-retries $MaxSceneRetries `
    --max-refill-rounds $MaxRefillRounds `
    --resume

if ($LASTEXITCODE -ne 0) {
    throw "CARLA generation failed with exit code $LASTEXITCODE"
}

Write-Host "Validating completed dataset..."
python tools/generate_carla_dataset.py `
    --groups $Groups `
    --output-dir $OutputDir `
    --min-visible-targets $MinVisibleTargets `
    --validate-only

if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed with exit code $LASTEXITCODE"
}

Write-Host "CARLA publication dataset passed validation."
