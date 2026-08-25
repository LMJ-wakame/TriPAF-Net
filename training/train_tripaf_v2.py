"""Configuration-driven, resumable training for official TriPAF-Net v2."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from datasets.paired_dehaze import PairedDehazeDataset
from losses.task_aware import (
    FrozenYOLOBackbone,
    FrozenYOLOTaskNetwork,
    detection_distillation_loss,
    detector_feature_loss,
)
from losses.tripaf import tripaf_v2_loss
from models.tripafnet_v2 import TriPAFNetV2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def move_batch(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


class FogSupervisedDataset(Dataset):
    """Attach CARLA fog density from the locked metadata manifest."""

    def __init__(
        self, dataset: PairedDehazeDataset, fog_by_id: dict[str, float]
    ) -> None:
        self.dataset = dataset
        self.fog_by_id = fog_by_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.dataset[index]
        identifier = str(sample["id"])
        if identifier not in self.fog_by_id:
            raise ValueError(f"No fog_density metadata for training pair {identifier}")
        sample["fog_density"] = torch.tensor(
            self.fog_by_id[identifier],
            dtype=torch.float32,
        )
        return sample


def read_locked_splits(
    metadata_csv: str | Path,
    available_ids: set[str],
) -> tuple[dict[str, list[str]], dict[str, float], str]:
    """Read one immutable split assignment shared by every training seed."""

    path = Path(metadata_csv)
    if not path.is_file():
        raise FileNotFoundError(
            f"Official training requires CARLA metadata with locked splits: {path}"
        )
    splits = {"train": [], "val": [], "test": []}
    fog_by_id: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = row.get("id", row.get("group_id", ""))
            split = row.get("split", "").lower()
            if not identifier or split not in splits:
                raise ValueError(
                    "Each metadata row needs id/group_id and split=train|val|test"
                )
            if identifier in fog_by_id:
                raise ValueError(f"Duplicate metadata group {identifier}")
            fog_by_id[identifier] = float(row["fog_density"])
            splits[split].append(identifier)
    assigned = set().union(*(set(values) for values in splits.values()))
    if assigned != available_ids:
        missing = sorted(available_ids - assigned)
        extra = sorted(assigned - available_ids)
        raise ValueError(
            "Locked metadata does not exactly match image pairs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if any(not splits[name] for name in splits):
        raise ValueError(
            "Locked train, validation, and test splits must all be nonempty"
        )
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Train/validation overlap in locked metadata")
    if set(splits["train"]) & set(splits["test"]):
        raise ValueError("Train/test overlap in locked metadata")
    if set(splits["val"]) & set(splits["test"]):
        raise ValueError("Validation/test overlap in locked metadata")
    return splits, fog_by_id, sha256(path)


@torch.no_grad()
def update_ema(ema_model: TriPAFNetV2, model: TriPAFNetV2, decay: float) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": train_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["loader"])


def save_checkpoint(
    path: Path,
    model: TriPAFNetV2,
    ema_model: TriPAFNetV2,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
    epoch: int,
    args: argparse.Namespace,
    score: float,
    best_score: float,
    split_hash: str,
) -> None:
    payload = {
        "format_version": 3,
        "architecture": "TriPAF-Net v2",
        "model_name": "tripaf_v2" if model.adaptive_fusion else "tripaf_v2_fixed",
        "fusion_mode": "adaptive" if model.adaptive_fusion else "fixed",
        "adaptive_fusion": model.adaptive_fusion,
        "detector_aware": args.detector_loss_weight > 0,
        "detector_loss_weight": args.detector_loss_weight,
        "base_channels": model.base_channels,
        "residual_scale": model.residual_scale,
        "t_min": model.t_min,
        "epoch": epoch,
        "validation_score": score,
        "best_validation_score": best_score,
        "seed": args.seed,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": args.config_sha256,
        "split_manifest_sha256": split_hash,
        # Inference reads the EMA weights; resume uses train_model.
        "model": ema_model.state_dict(),
        "train_model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": rng_state(train_generator),
        "args": vars(args),
    }
    torch.save(payload, path)


@torch.no_grad()
def validate(
    model: TriPAFNetV2,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    detector: FrozenYOLOBackbone | FrozenYOLOTaskNetwork | None,
    detector_mode: str,
    heavy_fog_weight: float,
    detector_weight: float,
) -> float:
    """Validation-only restoration, heavy-fog, and detector-consistency score."""

    model.eval()
    psnr_sum = 0.0
    mae_sum = 0.0
    heavy_psnr_sum = 0.0
    heavy_count = 0
    detector_sum = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device.type, enabled=amp):
            output = model(
                batch["hazy"],
                batch["dark"],
                batch["bright"],
                batch["sky"],
            )["image"]
        difference = output.float() - batch["clean"].float()
        mse = difference.square().mean((1, 2, 3)).clamp_min(1e-10)
        psnr_sum += (10.0 * torch.log10(1.0 / mse)).sum().item()
        per_image_psnr = 10.0 * torch.log10(1.0 / mse)
        heavy = batch["fog_density"] >= 30.0
        heavy_psnr_sum += per_image_psnr[heavy].sum().item()
        heavy_count += int(heavy.sum().item())
        mae_sum += difference.abs().mean((1, 2, 3)).sum().item()
        if detector is not None:
            detector_value = (
                detection_distillation_loss(detector, output, batch["clean"])
                if detector_mode == "distill"
                else detector_feature_loss(detector, output, batch["clean"])
            )
            detector_sum += float(detector_value) * output.shape[0]
        count += output.shape[0]
    global_psnr = psnr_sum / max(1, count)
    global_mae = mae_sum / max(1, count)
    heavy_psnr = heavy_psnr_sum / heavy_count if heavy_count else global_psnr
    detector_mean = detector_sum / max(1, count)
    return (
        global_psnr
        - 10.0 * global_mae
        + heavy_fog_weight * (heavy_psnr - global_psnr)
        - detector_weight * detector_mean
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp = bool(args.amp and device.type == "cuda")
    all_pairs = PairedDehazeDataset(
        args.hazy_dir,
        args.clean_dir,
        crop_size=args.crop_size,
        training=False,
    ).pairs
    available_ids = {pair[0] for pair in all_pairs}
    splits, fog_by_id, split_hash = read_locked_splits(args.metadata_csv, available_ids)
    train_base = PairedDehazeDataset(
        args.hazy_dir,
        args.clean_dir,
        splits["train"],
        args.crop_size,
        training=True,
    )
    val_base = PairedDehazeDataset(
        args.hazy_dir,
        args.clean_dir,
        splits["val"],
        args.crop_size,
        training=False,
    )
    train_set = FogSupervisedDataset(train_base, fog_by_id)
    val_set = FogSupervisedDataset(val_base, fog_by_id)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.workers,
        pin_memory=amp,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=amp,
    )
    run_dir = Path(args.output_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(
        json.dumps({"splits": splits, "metadata_sha256": split_hash}, indent=2) + "\n",
        encoding="utf-8",
    )

    model = TriPAFNetV2(
        args.base_channels,
        adaptive_fusion=args.adaptive_fusion,
        residual_scale=args.residual_scale,
        t_min=args.t_min,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, args.epochs * optimizer_steps_per_epoch),
        warmup_steps=args.warmup_epochs * optimizer_steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    detector = None
    if args.detector_loss_weight > 0:
        detector = (
            (
                FrozenYOLOTaskNetwork(args.detector_weights)
                if args.detector_mode == "distill"
                else FrozenYOLOBackbone(args.detector_weights)
            )
            .to(device)
            .eval()
        )

    start_epoch = 1
    best_score = float("-inf")
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if not args.resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError(
            f"Checkpoint already exists in {run_dir}; use --resume or a new --output-dir"
        )
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if bool(checkpoint.get("adaptive_fusion", True)) != args.adaptive_fusion:
            raise ValueError(
                "Refusing resume because the checkpoint fusion mode differs"
            )
        if checkpoint.get("config_sha256") != args.config_sha256:
            raise ValueError("Refusing resume because the training config changed")
        if checkpoint.get("split_manifest_sha256") != split_hash:
            raise ValueError(
                "Refusing resume because the locked split manifest changed"
            )
        model.load_state_dict(checkpoint["train_model"], strict=True)
        ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"], train_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_validation_score"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        detector_factor = min(1.0, epoch / max(1, args.detector_ramp_epochs))
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device.type, enabled=amp):
                outputs = model(
                    batch["hazy"],
                    batch["dark"],
                    batch["bright"],
                    batch["sky"],
                )
                outputs["input"] = batch["hazy"]
                total, _ = tripaf_v2_loss(
                    outputs,
                    batch["clean"],
                    batch["fog_density"],
                    args.loss_weights,
                )
                if detector is not None:
                    detector_loss = (
                        detection_distillation_loss(
                            detector, outputs["image"], batch["clean"]
                        )
                        if args.detector_mode == "distill"
                        else detector_feature_loss(
                            detector, outputs["image"], batch["clean"]
                        )
                    )
                    total = (
                        total
                        + args.detector_loss_weight * detector_factor * detector_loss
                    )
                total = total / args.accumulation_steps
            scaler.scale(total).backward()
            boundary = batch_index % args.accumulation_steps == 0
            boundary = boundary or batch_index == len(train_loader)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema_model, model, args.ema_decay)

        score = validate(
            ema_model,
            val_loader,
            device,
            amp,
            detector,
            args.detector_mode,
            args.selection_heavy_fog_weight,
            args.selection_detector_weight if detector is not None else 0.0,
        )
        is_best = score > best_score
        best_score = max(best_score, score)
        save_checkpoint(
            last_path,
            model,
            ema_model,
            optimizer,
            scheduler,
            scaler,
            train_generator,
            epoch,
            args,
            score,
            best_score,
            split_hash,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                train_generator,
                epoch,
                args,
                score,
                best_score,
                split_hash,
            )
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/tripaf_v2.yaml")
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = Path(config_args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    checkpoint_config = config.get("checkpoint", {})
    loss_config = config["loss"]
    evaluation_config = config["evaluation"]

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    publication_root = Path(data_config["publication_root"])
    parser.add_argument(
        "--hazy-dir",
        default=str(data_config.get("hazy_dir", publication_root / "images/hazy")),
    )
    parser.add_argument(
        "--clean-dir",
        default=str(data_config.get("clean_dir", publication_root / "images/clear")),
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(data_config.get("metadata_csv", publication_root / "metadata.csv")),
    )
    parser.add_argument(
        "--output-dir",
        default=str(checkpoint_config.get("output_dir", "checkpoints/tripaf_v2")),
    )
    parser.add_argument("--seed", type=int, default=int(config["project"]["seed"]))
    parser.add_argument("--epochs", type=int, default=int(train_config["epochs"]))
    parser.add_argument(
        "--batch-size", type=int, default=int(train_config["batch_size"])
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=int(train_config["accumulation_steps"]),
    )
    parser.add_argument("--crop-size", type=int, default=int(data_config["crop_size"]))
    parser.add_argument(
        "--base-channels", type=int, default=int(model_config["base_channels"])
    )
    parser.add_argument(
        "--residual-scale", type=float, default=float(model_config["residual_scale"])
    )
    parser.add_argument("--t-min", type=float, default=float(model_config["t_min"]))
    parser.add_argument(
        "--learning-rate", type=float, default=float(train_config["learning_rate"])
    )
    parser.add_argument(
        "--weight-decay", type=float, default=float(train_config["weight_decay"])
    )
    parser.add_argument(
        "--workers", type=int, default=int(train_config.get("workers", 2))
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=float(train_config["gradient_clip_norm"]),
    )
    parser.add_argument(
        "--ema-decay", type=float, default=float(train_config["ema_decay"])
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=int(train_config["warmup_epochs"])
    )
    parser.add_argument(
        "--detector-loss-weight", type=float, default=float(loss_config["detector"])
    )
    parser.add_argument("--detector-ramp-epochs", type=int, default=5)
    parser.add_argument(
        "--detector-mode", choices=("feature", "distill"), default="feature"
    )
    parser.add_argument("--detector-weights", default="yolov8/yolov8n.pt")
    parser.add_argument(
        "--selection-heavy-fog-weight",
        type=float,
        default=float(evaluation_config["selection_heavy_fog_weight"]),
    )
    parser.add_argument(
        "--selection-detector-weight",
        type=float,
        default=float(evaluation_config["selection_detector_weight"]),
    )
    parser.add_argument(
        "--adaptive-fusion",
        action=argparse.BooleanOptionalAction,
        default=bool(model_config["adaptive_fusion"]),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=bool(train_config["amp"]),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=bool(train_config["resume"]),
    )
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)
    if args.accumulation_steps < 1:
        parser.error("--accumulation-steps must be positive")
    args.loss_weights = {key: float(value) for key, value in loss_config.items()}
    args.config_sha256 = sha256(config_path)
    return args


if __name__ == "__main__":
    print(train(parse_args()))
