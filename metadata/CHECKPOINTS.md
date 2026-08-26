# Model Checkpoints

The model checkpoints used for the reported experiments are not stored directly
in this Git repository.

| Model                 | Expected local path                           | SHA-256                                                      |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| TriPAF-Net (adaptive) | `checkpoints/tripaf_v2/seed_42/best.pt`       | `D2E803D77F0BBE8321D6AF2BBA28E42A25F1F088FF3D71BD2BA4806FC23F0C84` |
| TriPAF-Net Fixed      | `checkpoints/tripaf_v2_fixed/seed_42/best.pt` | `58FA4FB1589A0ED966947FEDBFD08B382D35D462ADE0D8AB5BC37AF`    |

## Checkpoint Information

Both checkpoints use the following configuration:

- Architecture: TriPAF-Net v2
- Checkpoint format version: 3
- Base channels: 24
- Residual scale: 0.5
- Minimum transmission: 0.08
- Training epochs: 20
- Random seed: 42
- Number of parameters: 13,383,307

The adaptive checkpoint uses `adaptive_fusion: true`, while the fixed-gate
checkpoint uses `adaptive_fusion: false`.

Checkpoints can be loaded with
`utils.inference_v2.load_v2_checkpoint`. The loader uses strict model-state
validation.

The dataset split manifests used for the reported experiments are available
under `metadata/splits/`.
