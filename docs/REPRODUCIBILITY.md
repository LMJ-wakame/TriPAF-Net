# Reproducibility record

## Paper protocol

- CARLA: 1,436 synchronized groups; 996 train, 232 validation, and 208 test.
- Image resolution: 1024 x 1024.
- Fog range: 20--50.
- Training: seed 42, 20 epochs, 512 x 512 crops, batch size 2, two gradient
  accumulation steps, AdamW, EMA 0.999, and mixed precision.
- Detection: frozen COCO-pretrained YOLOv8m, native CARLA boxes, 1024-pixel
  input, confidence 0.001, NMS IoU 0.7, batch size 2.
- Foggy Cityscapes: beta=0.02, 500 validation images, eight audited classes;
  `rider` is retained in labels but excluded from COCO aggregate detection
  metrics because there is no valid COCO counterpart.

