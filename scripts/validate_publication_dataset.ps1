param([string]$Root = "data/carla_tripaf_1024")
$ErrorActionPreference = "Stop"
python scripts/validate_publication_dataset.py --root $Root
