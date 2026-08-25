"""Guardrails for the locked validation/test protocol."""

from __future__ import annotations


def require_model_selection_split(split: str) -> None:
    if split.lower() == "test":
        raise ValueError(
            "Calibration, search, and checkpoint selection are forbidden on the test split."
        )
    if split.lower() not in {"train", "val", "validation"}:
        raise ValueError(f"Unknown split {split!r}")


def validate_yolo_row(
    class_id: int, x_center: float, y_center: float, width: float, height: float
) -> None:
    if class_id < 0 or not all(
        0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)
    ):
        raise ValueError("Invalid normalized YOLO coordinates")
    if width == 0.0 or height == 0.0:
        raise ValueError("YOLO boxes must have positive area")
