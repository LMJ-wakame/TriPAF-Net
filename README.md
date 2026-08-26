# TriPAF-Net

**Adaptive Physical-Prior Fusion for Perception-Oriented Dehazing**

TriPAF-Net is a physics-guided dual-stream network for driving-scene dehazing. It encodes the dark-channel prior (DCP), bright-channel prior (BCP), and a deterministic sky mask as a parallel prior stream, then fuses them with RGB features through five input-conditioned blocks.

## Overwiew

### Highlights

- **Explicit physical-prior stream:** DCP, BCP, and the sky mask remain interpretable inputs rather than a one-shot preprocessing result.
- **Input-conditioned fusion:** channel, spatial, haze-context, and high-frequency evidence control prior and detail gates at five scales.
- **Matched fixed-gate control:** TriPAF-Net Fixed sets both gates to 0.5 while retaining the remaining network structure.
- **Hybrid reconstruction:** direct, learned-physical, and deterministic-prior candidates are combined before bounded inference stabilization.
- **Perception-oriented evaluation:** restoration metrics and frozen-detector metrics are reported separately.

  ### Method

```
    Hazy RGB ──> RGB encoder ───────────────────────────────┐
                                                           ├─> five-scale adaptive fusion ─> decoder ─> stabilized output
    DCP + BCP + sky mask ─> physical-prior encoder ────────┘
                             ↑
                     global haze descriptor
```

The adaptive model contains 13.23M parameters. The reported configuration uses base width 24, minimum transmission 0.08, residual scale 0.5, and five fusion levels. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the code-level description.

## Qualitative results

### CARLA restoration comparison

The fixed test IDs below cover low, medium, and high fog. Columns show the clear reference, hazy input, DCP/BCP, TriPAF-Net Fixed, and TriPAF-Net.

<img width="960" height="527" alt="carla" src="https://github.com/user-attachments/assets/73cb6670-e318-4e27-a29c-7228aa14243c" />

### Foggy Cityscapes detection examples

These are deterministically selected success cases. All variants use the same frozen YOLOv8m; matched TP is class-aware at IoU ≥ 0.50. Aggregate performance should be interpreted from the quantitative tables rather than from these examples alone.

<img width="2507" height="2151" alt="foggycity" src="https://github.com/user-attachments/assets/d4325076-74ac-48a0-86ff-21c74533a57b" />

## Quantitative results

### CARLA: 208 held-out pairs

| Input / Method   |     PSNR ↑ |   RGB MAE ↓ | CIEDE2000 ↓ | Precision ↑ |   Recall ↑ | mAP@0.50 ↑ | mAP@0.50:0.95 ↑ |
| ---------------- | ---------: | ----------: | ----------: | ----------: | ---------: | ---------: | --------------: |
| Foggy input      | **16.026** | **0.11879** | **10.8999** |      0.3341 |     0.1149 |     0.0948 |          0.0506 |
| DCP/BCP          |     14.397 |     0.16934 |     15.5854 |      0.2700 |     0.1175 | **0.1018** |          0.0548 |
| TriPAF-Net Fixed |     14.626 |     0.16575 |     14.9174 |      0.3736 |     0.1127 |     0.0996 |          0.0545 |
| TriPAF-Net       |     14.235 |     0.17335 |     15.6287 |  **0.3903** | **0.1326** |     0.1013 |      **0.0550** |


### Foggy Cityscapes: beta=0.02, 500 validation images

| Input / Method | Precision ↑ |   Recall ↑ | mAP@0.50 ↑ | mAP@0.50:0.95 ↑ |
| -------------- | ----------: | ---------: | ---------: | --------------: |
| Foggy original |      0.6231 |     0.3933 |     0.4275 |          0.2692 |
| DCP/BCP        |      0.6534 |     0.4090 |     0.4329 |          0.2806 |
| TriPAF-Net     |  **0.6571** | **0.4105** | **0.4381** |      **0.2823** |


### Efficiency

RTX 3060 Ti, batch size 1, FP16:

| Input | GFLOPs | Latency (ms) |       FPS | Peak memory (MB) |
| ----: | -----: | -----------: | --------: | ---------------: |
|  384² |   82.7 |        29.63 | **33.75** |              247 |
|  512² |  147.0 |        47.83 |     20.91 |              370 |
|  640² |  229.8 |        73.04 |     13.69 |              534 |
| 1024² |  588.2 |       177.43 |      5.64 |             1225 |

## Installation

```
conda env create -f environment.yml
conda activate tripafnet-v2
python -m pip install -r requirements.txt
python -m pip install -r requirements-yolo.txt
```

CARLA data generation additionally requires a compatible CARLA server and Python API.

## Single-image inference

```
python inference.py path/to/hazy.png outputs/dehazed.png --checkpoint checkpoints/tripaf_v2/seed_42/best.pt
```

The default output key is stable_image. Other reconstruction branches can be selected with the output-key argument.

## Training and evaluation

Windows PowerShell training entry points

```
./scripts/train/train_tripaf_v2_seed42.ps1
./scripts/train/train_tripaf_v2_fixed_seed42.ps1
```

Run restoration and frozen-detector evaluation using existing checkpoints

```
./run_pipeline.ps1 -SkipTraining
```

Data, checkpoint, YOLO-weight, and output paths can be overridden through script parameters. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository structure

```
  configs/                 adaptive and fixed-gate configurations
  datasets/                paired dehazing dataset loader
  evaluation/              shared evaluation protocol
  losses/                  restoration and task-aware losses
  metadata/                checkpoint records and split manifests
  models/                  TriPAF-Net implementation
  scripts/                 training and evaluation entry points
  tests/                   smoke and unit tests
  tools/                   dataset preparation and evaluation utilities
  training/                training loop
  utils/                   priors, inference, EMA, and image I/O
```

