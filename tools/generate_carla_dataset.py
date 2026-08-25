"""Generate publication-scale paired clear/hazy CARLA data with native YOLO labels.

The default run creates 1,500 successful paired groups. A sample is counted only
when the clear/hazy pair is captured and at least ``--min-visible-targets``
valid projected ground-truth boxes are available. Failed scene attempts are
retried for the same sample id instead of consuming one of the requested groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import carla
except ImportError:  # pragma: no cover - requires a CARLA installation
    carla = None


# Native CARLA labels are mapped to the COCO ids used by the frozen detector.
# Placeholder names preserve a contiguous 0..7 YOLO dataset schema.
CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "unused_airplane",
    "bus",
    "unused_train",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--output-dir", default="data/carla_tripaf_1024")
    parser.add_argument("--groups", type=int, default=1500)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--fog-min", type=float, default=20.0)
    parser.add_argument("--fog-max", type=float, default=49.999)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--towns",
        default="Town01,Town03,Town04,Town05,Town10HD_Opt",
        help="Comma-separated CARLA maps; unavailable maps are skipped.",
    )
    parser.add_argument("--targets-min", type=int, default=4)
    parser.add_argument("--targets-max", type=int, default=8)
    parser.add_argument(
        "--min-visible-targets",
        type=int,
        default=1,
        help="Minimum projected GT boxes required before a sample is accepted.",
    )
    parser.add_argument(
        "--max-scene-retries",
        type=int,
        default=6,
        help="Maximum complete scene attempts for each sample id in one refill round.",
    )
    parser.add_argument(
        "--max-refill-rounds",
        type=int,
        default=12,
        help="Maximum refill passes over still-missing sample ids.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.groups <= 0:
        raise ValueError("groups must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (20.0 <= args.fog_min <= args.fog_max < 50.0):
        raise ValueError("publication fog range must satisfy 20 <= min <= max < 50")
    if (
        args.train_fraction <= 0
        or args.val_fraction < 0
        or args.train_fraction + args.val_fraction >= 1
    ):
        raise ValueError("invalid train/validation fractions")
    if args.targets_min <= 0 or args.targets_max < args.targets_min:
        raise ValueError("invalid target count range")
    if args.min_visible_targets <= 0:
        raise ValueError("min-visible-targets must be positive")
    if args.min_visible_targets > args.targets_max:
        raise ValueError("min-visible-targets cannot exceed targets-max")
    if args.max_scene_retries <= 0:
        raise ValueError("max-scene-retries must be positive")
    if args.max_refill_rounds <= 0:
        raise ValueError("max-refill-rounds must be positive")


def prepare_output(root: Path) -> None:
    for relative in ("images/clear", "images/hazy", "labels", "splits"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def split_for_index(index: int, args: argparse.Namespace) -> str:
    value = random.Random(args.seed * 1000003 + index).random()
    if value < args.train_fraction:
        return "train"
    if value < args.train_fraction + args.val_fraction:
        return "val"
    return "test"


def camera_calibration(width: int, height: int, fov: float) -> np.ndarray:
    calibration = np.identity(3, dtype=np.float64)
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    calibration[0, 0] = focal
    calibration[1, 1] = focal
    calibration[0, 2] = width / 2.0
    calibration[1, 2] = height / 2.0
    return calibration


def actor_class_id(actor: "carla.Actor") -> int | None:
    type_id = actor.type_id.lower()
    if type_id.startswith("walker.pedestrian"):
        return 0
    if not type_id.startswith("vehicle."):
        return None
    attributes = actor.attributes
    base_type = attributes.get("base_type", "").lower()
    text = f"{type_id} {base_type}"
    if "bicycle" in text or ("bike" in text and "motor" not in text):
        return 1
    if "motorcycle" in text or "motorbike" in text:
        return 3
    if "bus" in text or "coach" in text:
        return 5
    if "truck" in text or "firetruck" in text or "ambulance" in text:
        return 7
    return 2


def project_actor_box(
    actor: "carla.Actor",
    camera: "carla.Sensor",
    calibration: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    world_to_camera = np.asarray(
        camera.get_transform().get_inverse_matrix(), dtype=np.float64
    )
    vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    pixels = []
    for vertex in vertices:
        point = np.array([vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float64)
        camera_point = world_to_camera @ point
        camera_point = np.array([camera_point[1], -camera_point[2], camera_point[0]])
        if camera_point[2] <= 0.5:
            continue
        projected = calibration @ camera_point
        pixels.append((projected[0] / projected[2], projected[1] / projected[2]))
    if len(pixels) < 4:
        return None
    x_values, y_values = zip(*pixels)
    x1 = float(np.clip(min(x_values), 0, width - 1))
    y1 = float(np.clip(min(y_values), 0, height - 1))
    x2 = float(np.clip(max(x_values), 0, width - 1))
    y2 = float(np.clip(max(y_values), 0, height - 1))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    if (x2 - x1) * (y2 - y1) < 80:
        return None
    return x1, y1, x2, y2


def yolo_lines(
    actors: list["carla.Actor"],
    camera: "carla.Sensor",
    calibration: np.ndarray,
    width: int,
    height: int,
) -> list[str]:
    lines = []
    camera_location = camera.get_transform().location
    camera_forward = camera.get_transform().get_forward_vector()
    for actor in actors:
        if not actor.is_alive:
            continue
        direction = actor.get_transform().location - camera_location
        dot = (
            camera_forward.x * direction.x
            + camera_forward.y * direction.y
            + camera_forward.z * direction.z
        )
        if dot <= 0:
            continue
        class_id = actor_class_id(actor)
        box = project_actor_box(actor, camera, calibration, width, height)
        if class_id is None or box is None:
            continue
        x1, y1, x2, y2 = box
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            continue
        if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
            continue
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return lines


def drain(sensor_queue: queue.Queue) -> None:
    while True:
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            return


def capture(
    camera_queue: queue.Queue, world: "carla.World", timeout: float = 10.0
) -> "carla.Image":
    target_frame = world.tick()
    while True:
        image = camera_queue.get(timeout=timeout)
        if image.frame >= target_frame:
            return image


def set_weather(
    world: "carla.World",
    rng: random.Random,
    fog_density: float,
    base: dict | None = None,
) -> dict:
    if base is None:
        base = {
            "cloudiness": rng.uniform(0.0, 35.0),
            "sun_altitude_angle": rng.uniform(25.0, 75.0),
            "sun_azimuth_angle": rng.uniform(0.0, 360.0),
            "fog_distance": rng.uniform(0.0, 8.0),
            "fog_falloff": rng.uniform(0.05, 0.35),
            "scattering_intensity": rng.uniform(0.6, 1.0),
            "mie_scattering_scale": rng.uniform(0.0, 0.08),
        }
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=base["cloudiness"],
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=base["sun_azimuth_angle"],
            sun_altitude_angle=base["sun_altitude_angle"],
            fog_density=fog_density,
            fog_distance=base["fog_distance"],
            wetness=0.0,
            fog_falloff=base["fog_falloff"],
            scattering_intensity=base["scattering_intensity"],
            mie_scattering_scale=base["mie_scattering_scale"],
        )
    )
    return base


def setup_camera(
    world: "carla.World", ego: "carla.Actor", args: argparse.Namespace
) -> "carla.Sensor":
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(args.width))
    blueprint.set_attribute("image_size_y", str(args.height))
    blueprint.set_attribute("fov", str(args.fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    transform = carla.Transform(
        carla.Location(x=1.3, z=1.6), carla.Rotation(pitch=-3.0)
    )
    return world.spawn_actor(blueprint, transform, attach_to=ego)


def _road_aligned_vehicle_transform(
    world: "carla.World",
    ego_transform: "carla.Transform",
    distance: float,
    lateral: float,
) -> "carla.Transform":
    approximate = ego_transform.transform(carla.Location(x=distance, y=lateral, z=0.3))
    waypoint = world.get_map().get_waypoint(
        approximate,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return carla.Transform(
            approximate, carla.Rotation(yaw=ego_transform.rotation.yaw)
        )
    transform = waypoint.transform
    transform.location.z += 0.25
    return transform


def spawn_targets(
    world: "carla.World",
    ego_transform: "carla.Transform",
    rng: random.Random,
    args: argparse.Namespace,
) -> list["carla.Actor"]:
    library = world.get_blueprint_library()
    vehicle_blueprints = [bp for bp in library.filter("vehicle.*") if bp.id]
    walker_blueprints = list(library.filter("walker.pedestrian.*"))
    actors: list["carla.Actor"] = []
    desired = rng.randint(args.targets_min, args.targets_max)

    # More attempts than the old desired*3 loop substantially improves scene yield.
    max_spawn_attempts = max(desired * 10, 40)
    for _ in range(max_spawn_attempts):
        if len(actors) >= desired:
            break
        pedestrian = bool(walker_blueprints) and rng.random() < 0.25
        blueprint = rng.choice(walker_blueprints if pedestrian else vehicle_blueprints)
        distance = rng.uniform(8.0, 42.0)
        lateral = rng.uniform(-4.5, 4.5)

        if pedestrian:
            location = ego_transform.transform(
                carla.Location(x=distance, y=lateral, z=0.9)
            )
            transform = carla.Transform(
                location,
                carla.Rotation(yaw=ego_transform.rotation.yaw + rng.uniform(-30, 30)),
            )
        else:
            transform = _road_aligned_vehicle_transform(
                world, ego_transform, distance, lateral
            )
            transform.rotation.yaw += rng.uniform(-5.0, 5.0)

        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_simulate_physics(False)
        actors.append(actor)
    return actors


def destroy_attempt(world: "carla.World", camera, actors: list, ego) -> None:
    if camera is not None:
        try:
            camera.stop()
        except RuntimeError:
            pass
        if camera.is_alive:
            camera.destroy()
    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()
    if ego is not None and ego.is_alive:
        ego.destroy()
    world.tick()


def append_metadata(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file():
        return []
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_ids(metadata_path: Path) -> set[str]:
    return {row["id"] for row in read_metadata(metadata_path)}


def write_dataset_files(root: Path, metadata_path: Path) -> None:
    rows = read_metadata(metadata_path)
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        manifest_path = root / "splits" / f"{split}.csv"
        manifest_fields = [
            "id",
            "split",
            "town",
            "fog_density",
            "seed",
            "labels_visible",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=manifest_fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(sorted(selected, key=lambda row: row["id"]))
        for variant in ("clear", "hazy"):
            lines = [
                str((root / "images" / variant / f"{row['id']}.png").resolve())
                for row in selected
            ]
            (root / "splits" / f"{split}_{variant}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
    for variant in ("clear", "hazy"):
        yaml_text = (
            f"path: {root.resolve().as_posix()}\n"
            f"train: splits/train_{variant}.txt\n"
            f"val: splits/val_{variant}.txt\n"
            f"test: splits/test_{variant}.txt\n"
            f"names: {json.dumps(CLASS_NAMES)}\n"
        )
        (root / f"dataset_{variant}.yaml").write_text(yaml_text, encoding="utf-8")
    split_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "splits").glob("*.csv"))
    }
    if metadata_path.is_file():
        split_hashes["metadata.csv"] = hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
    (root / "splits" / "SHA256.json").write_text(
        json.dumps(split_hashes, indent=2) + "\n", encoding="utf-8"
    )
    (root / "labels" / "label_metadata.json").write_text(
        json.dumps(
            {
                "annotation_type": "carla_native_ground_truth",
                "classes": CLASS_NAMES,
                "warning": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def fog_band(value: float) -> str:
    if value < 30.0:
        return "20-29"
    if value < 40.0:
        return "30-39"
    return "40-49"


def validate_dataset(
    root: Path, expected_groups: int, min_visible_targets: int
) -> dict:
    metadata_path = root / "metadata.csv"
    rows = read_metadata(metadata_path)
    ids = [row.get("id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("metadata.csv contains duplicate sample ids")

    clear_files = {p.stem: p for p in (root / "images" / "clear").glob("*.png")}
    hazy_files = {p.stem: p for p in (root / "images" / "hazy").glob("*.png")}
    label_files = {p.stem: p for p in (root / "labels").glob("*.txt")}
    expected_ids = set(ids)

    problems = []
    if len(rows) != expected_groups:
        problems.append(f"metadata rows={len(rows)} expected={expected_groups}")
    for name, mapping in (
        ("clear", clear_files),
        ("hazy", hazy_files),
        ("labels", label_files),
    ):
        present_ids = set(mapping)
        missing = expected_ids - present_ids
        unexpected = present_ids - expected_ids
        if missing:
            problems.append(f"{name} missing {len(missing)} metadata-backed samples")
        if unexpected:
            problems.append(
                f"{name} contains {len(unexpected)} files not present in metadata"
            )

    class_counts: Counter[int] = Counter()
    fog_groups: Counter[str] = Counter()
    fog_objects: Counter[str] = Counter()
    total_objects = 0
    min_seen = None

    for row in rows:
        sample_id = row["id"]
        label_path = label_files.get(sample_id)
        if label_path is None:
            continue
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        min_seen = len(lines) if min_seen is None else min(min_seen, len(lines))
        if len(lines) < min_visible_targets:
            problems.append(
                f"{sample_id}: only {len(lines)} labels (< {min_visible_targets})"
            )
        band = fog_band(float(row["fog_density"]))
        fog_groups[band] += 1
        fog_objects[band] += len(lines)
        total_objects += len(lines)
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                problems.append(f"{sample_id}: malformed YOLO line {line!r}")
                continue
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                problems.append(f"{sample_id}: invalid class id {class_id}")
            if not all(0.0 <= value <= 1.0 for value in values):
                problems.append(f"{sample_id}: YOLO values out of range")
            if values[2] <= 0.0 or values[3] <= 0.0:
                problems.append(f"{sample_id}: non-positive box size")
            class_counts[class_id] += 1

    summary = {
        "status": "PASS" if not problems else "FAIL",
        "expected_groups": expected_groups,
        "metadata_rows": len(rows),
        "clear_files": len(clear_files),
        "hazy_files": len(hazy_files),
        "label_files": len(label_files),
        "minimum_labels_per_group": min_seen or 0,
        "total_objects": total_objects,
        "class_counts": {
            CLASS_NAMES[index]: class_counts.get(index, 0)
            for index in range(len(CLASS_NAMES))
            if not CLASS_NAMES[index].startswith("unused_")
        },
        "fog_groups": dict(fog_groups),
        "fog_objects": dict(fog_objects),
        "problems": problems[:100],
        "problem_count": len(problems),
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if problems:
        raise RuntimeError(
            f"Dataset validation failed with {len(problems)} problem(s); see dataset_summary.json"
        )
    return summary


def generate(args: argparse.Namespace) -> None:
    validate_args(args)
    root = Path(args.output_dir)
    prepare_output(root)

    if args.validate_only:
        validate_dataset(root, args.groups, args.min_visible_targets)
        return

    configuration = vars(args).copy()
    (root / "generation_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(
            json.dumps(
                {"status": "dry-run", **configuration, "classes": CLASS_NAMES}, indent=2
            )
        )
        return
    if carla is None:
        raise RuntimeError(
            "The CARLA Python package is required. Install the version matching the server."
        )

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(
        f"CARLA client={client.get_client_version()} server={client.get_server_version()}"
    )

    available = {Path(name).name for name in client.get_available_maps()}
    requested = [name.strip() for name in args.towns.split(",") if name.strip()]
    towns = [name for name in requested if name in available]
    if not towns:
        raise RuntimeError(
            f"None of the requested towns are available. Server maps: {sorted(available)}"
        )

    metadata_path = root / "metadata.csv"
    completed = completed_ids(metadata_path) if args.resume else set()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)
    calibration = camera_calibration(args.width, args.height, args.fov)

    try:
        for refill_round in range(args.max_refill_rounds):
            missing_indices = [
                index
                for index in range(1, args.groups + 1)
                if f"{index:06d}" not in completed
            ]
            if not missing_indices:
                break

            completed_before_round = len(completed)
            print(
                f"=== refill round {refill_round + 1}/{args.max_refill_rounds}: "
                f"{len(missing_indices)} missing ==="
            )

            for town_index, town in enumerate(towns):
                indices = [
                    index
                    for index in missing_indices
                    if (index - 1 + refill_round) % len(towns) == town_index
                ]
                if not indices:
                    continue

                world = client.load_world(town)
                original_settings = world.get_settings()
                original_weather = world.get_weather()
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                settings.deterministic_ragdolls = True
                world.apply_settings(settings)
                spawn_points = world.get_map().get_spawn_points()
                library = world.get_blueprint_library()

                try:
                    for index in indices:
                        sample_id = f"{index:06d}"
                        if sample_id in completed:
                            continue

                        accepted = False
                        last_error = None

                        for attempt in range(1, args.max_scene_retries + 1):
                            attempt_seed = (
                                args.seed
                                + index * 1_000_003
                                + refill_round * 1_000_033
                                + town_index * 193
                                + attempt * 9_176
                            )
                            rng = random.Random(attempt_seed)
                            actors: list["carla.Actor"] = []
                            camera = None
                            ego = None

                            try:
                                ego_blueprints = list(library.filter("vehicle.*"))
                                if not ego_blueprints:
                                    raise RuntimeError(
                                        "No vehicle blueprints available"
                                    )
                                rng.shuffle(ego_blueprints)
                                spawn_order = list(spawn_points)
                                rng.shuffle(spawn_order)

                                for spawn_number, transform in enumerate(spawn_order):
                                    blueprint = ego_blueprints[
                                        (attempt + spawn_number) % len(ego_blueprints)
                                    ]
                                    ego = world.try_spawn_actor(blueprint, transform)
                                    if ego is not None:
                                        break

                                if ego is None:
                                    raise RuntimeError("Could not spawn ego vehicle")

                                ego.set_simulate_physics(False)
                                camera = setup_camera(world, ego, args)
                                image_queue: queue.Queue = queue.Queue()
                                camera.listen(image_queue.put)

                                actors = spawn_targets(
                                    world, ego.get_transform(), rng, args
                                )
                                for _ in range(4):
                                    world.tick()
                                drain(image_queue)

                                labels = yolo_lines(
                                    actors,
                                    camera,
                                    calibration,
                                    args.width,
                                    args.height,
                                )
                                if len(labels) < args.min_visible_targets:
                                    raise RuntimeError(
                                        f"spawned={len(actors)}, visible={len(labels)}, "
                                        f"need={args.min_visible_targets}"
                                    )

                                base_weather = set_weather(world, rng, fog_density=0.0)
                                clear_image = capture(image_queue, world)
                                fog_density = rng.uniform(args.fog_min, args.fog_max)
                                set_weather(
                                    world,
                                    rng,
                                    fog_density=fog_density,
                                    base=base_weather,
                                )
                                hazy_image = capture(image_queue, world)

                                clear_path = (
                                    root / "images" / "clear" / f"{sample_id}.png"
                                )
                                hazy_path = (
                                    root / "images" / "hazy" / f"{sample_id}.png"
                                )
                                label_path = root / "labels" / f"{sample_id}.txt"

                                clear_image.save_to_disk(str(clear_path))
                                hazy_image.save_to_disk(str(hazy_path))
                                label_path.write_text(
                                    "\n".join(labels) + "\n",
                                    encoding="utf-8",
                                )

                                row = {
                                    "id": sample_id,
                                    "split": split_for_index(index, args),
                                    "town": town,
                                    "seed": attempt_seed,
                                    "fog_density": f"{fog_density:.6f}",
                                    "fog_distance": f"{base_weather['fog_distance']:.6f}",
                                    "fog_falloff": f"{base_weather['fog_falloff']:.6f}",
                                    "sun_altitude": f"{base_weather['sun_altitude_angle']:.6f}",
                                    "sun_azimuth": f"{base_weather['sun_azimuth_angle']:.6f}",
                                    "actors_spawned": len(actors),
                                    "labels_visible": len(labels),
                                    "width": args.width,
                                    "height": args.height,
                                }
                                append_metadata(metadata_path, row)
                                completed.add(sample_id)
                                accepted = True
                                print(
                                    f"[{len(completed):04d}/{args.groups:04d}] "
                                    f"{sample_id} {town} round={refill_round + 1} "
                                    f"attempt={attempt} fog={fog_density:.1f}% "
                                    f"labels={len(labels)}"
                                )
                                break

                            except (queue.Empty, RuntimeError) as error:
                                last_error = error
                                print(
                                    f"Retry {sample_id} {town} round {refill_round + 1} "
                                    f"attempt {attempt}/{args.max_scene_retries}: {error}",
                                    file=sys.stderr,
                                )
                            finally:
                                destroy_attempt(world, camera, actors, ego)

                        if not accepted:
                            print(
                                f"DEFER {sample_id}: {town} failed after "
                                f"{args.max_scene_retries} attempts ({last_error}); "
                                f"will retry in another town/refill round.",
                                file=sys.stderr,
                            )
                finally:
                    world.set_weather(original_weather)
                    world.apply_settings(original_settings)

            added = len(completed) - completed_before_round
            print(
                f"=== refill round {refill_round + 1} complete: "
                f"+{added}, total={len(completed)}/{args.groups} ==="
            )

    finally:
        traffic_manager.set_synchronous_mode(False)

    write_dataset_files(root, metadata_path)

    if len(completed) != args.groups:
        missing = [
            f"{index:06d}"
            for index in range(1, args.groups + 1)
            if f"{index:06d}" not in completed
        ]
        raise RuntimeError(
            f"Generation ended with {len(completed)}/{args.groups} valid groups "
            f"after {args.max_refill_rounds} refill rounds. "
            f"Still missing {len(missing)} ids. First missing ids: {missing[:20]}. "
            f"Run the same command again with --resume to continue."
        )

    validate_dataset(root, args.groups, args.min_visible_targets)
    print(f"Dataset complete: {root.resolve()} ({len(completed)} successful groups)")


if __name__ == "__main__":
    generate(parse_args())
