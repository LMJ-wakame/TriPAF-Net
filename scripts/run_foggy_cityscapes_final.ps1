param(
    [string]$PythonExe = "C:\Users\Olivia\miniconda3\envs\dehaze_env\python.exe",
    [string]$DocumentPython = "C:\Users\Olivia\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$RawData = "data\foggy_cityscapes",
    [string]$Checkpoint = "checkpoints\tripaf_v2\seed_42\best.pt",
    [string]$Detector = "yolov8\yolov8m.pt",
    [string]$ReportTemplate = "artifacts\TriPAF-Net_v2_Report_before_foggy.docx"
)

& $PythonExe tools\prepare_foggy_cityscapes_yolo.py --raw-dir $RawData --output-dir data\foggy_cityscapes_yolo --beta 0.02
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe tools\check_foggy_labels.py --dataset data\foggy_cityscapes_yolo --output-dir results\foggy_label_visualization --samples 20 --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe tools\run_foggy_cityscapes_pipeline.py --source data\foggy_cityscapes_yolo\images\val --checkpoint $Checkpoint --output-dir results\foggy_detection --device cuda:0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe tools\evaluate_foggy_detection.py --weights $Detector --output-dir results\foggy_detection --imgsz 1024 --conf 0.001 --iou 0.7 --device 0 --batch 2 --half
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe tools\generate_foggy_figures.py --results results\foggy_detection --inputs results\foggy_detection\inputs --labels data\foggy_cityscapes_yolo\labels\val --weights $Detector --output figures\foggy_cityscapes --imgsz 1024 --device 0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $DocumentPython tools\update_final_report_foggy.py --input $ReportTemplate --output deliverables\TriPAF-Net_v2_Final_Report.docx --results results\foggy_detection --figures figures\foggy_cityscapes
exit $LASTEXITCODE
