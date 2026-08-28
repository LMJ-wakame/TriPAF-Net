param(
    [string]$PythonExe = "python",
    [string]$YoloPython = "python",
    [string]$DataRoot = "data/carla_tripaf_1024",
    [string]$AdaptiveCheckpoint = "checkpoints/tripaf_v2/seed_42/best.pt",
    [string]$FixedCheckpoint = "checkpoints/tripaf_v2_fixed/seed_42/best.pt",
    [string]$YoloWeights = "yolov8/yolov8m.pt",
    [string]$OutputDir = "artifacts/evaluation",
    [switch]$SkipTraining,
    [switch]$SkipYolo
)

$ErrorActionPreference = "Stop"
$env:YOLO_CONFIG_DIR = (New-Item -ItemType Directory -Force "artifacts/ultralytics_config").FullName

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $Executable $($Arguments -join ' ')"
    }
}

if (-not $SkipTraining) {
    Invoke-Checked $PythonExe @(
        "-m", "training.train_tripaf_v2",
        "--config", "configs/tripaf_v2_fixed.yaml",
        "--hazy-dir", "$DataRoot/images/hazy",
        "--clean-dir", "$DataRoot/images/clear",
        "--metadata-csv", "$DataRoot/metadata.csv",
        "--output-dir", "checkpoints/tripaf_v2_fixed",
        "--no-adaptive-fusion", "--resume", "--amp"
    )
    Invoke-Checked $PythonExe @(
        "-m", "training.train_tripaf_v2",
        "--config", "configs/tripaf_v2.yaml",
        "--hazy-dir", "$DataRoot/images/hazy",
        "--clean-dir", "$DataRoot/images/clear",
        "--metadata-csv", "$DataRoot/metadata.csv",
        "--output-dir", "checkpoints/tripaf_v2",
        "--adaptive-fusion", "--resume", "--amp"
    )
}

Invoke-Checked $PythonExe @(
    "tools/evaluate_dehazing.py",
    "--pairs-dir", $DataRoot,
    "--split", "test",
    "--methods", "hazy,dcp_bcp,tripaf_v2_fixed,tripaf_v2",
    "--tripaf-v2-fixed-checkpoint", $FixedCheckpoint,
    "--tripaf-v2-checkpoint", $AdaptiveCheckpoint,
    "--output-dir", $OutputDir
)

if (-not $SkipYolo) {
    Invoke-Checked $YoloPython @(
        "tools/compare_yolov8_map.py",
        "--variant", "clear=$OutputDir/images/clear",
        "--variant", "hazy=$OutputDir/images/hazy",
        "--variant", "dcp_bcp=$OutputDir/images/dcp_bcp",
        "--variant", "tripaf_v2_fixed=$OutputDir/images/tripaf_v2_fixed",
        "--variant", "tripaf_v2=$OutputDir/images/tripaf_v2",
        "--label-dir", "$DataRoot/labels",
        "--weights", $YoloWeights,
        "--csv-out", "$OutputDir/native_gt_detection.csv"
    )
}

