param(
    [string]$Config = "configs/tripaf_v2_fixed.yaml",
    [string]$HazyDir = "data/carla_tripaf_1024/images/hazy",
    [string]$CleanDir = "data/carla_tripaf_1024/images/clear",
    [string]$MetadataCsv = "data/carla_tripaf_1024/metadata.csv",
    [string]$OutputDir = "checkpoints/tripaf_v2_fixed",
    [int]$Epochs = 20,
    [int]$BatchSize = 2,
    [int]$AccumulationSteps = 2,
    [int]$CropSize = 512,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

$trainingArgs = @(
    "-m", "training.train_tripaf_v2",
    "--config", $Config,
    "--hazy-dir", $HazyDir,
    "--clean-dir", $CleanDir,
    "--metadata-csv", $MetadataCsv,
    "--output-dir", $OutputDir,
    "--seed", "42",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--accumulation-steps", $AccumulationSteps,
    "--crop-size", $CropSize,
    "--workers", $Workers,
    "--no-adaptive-fusion",
    "--amp",
    "--resume"
)

python @trainingArgs
