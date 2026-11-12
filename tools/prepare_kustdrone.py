#!/usr/bin/env python3
"""Prepare KUSTDrone for separate VISIBLE and FIR single-modality YOLO training.

Dataset layout expected after extraction:
  KUSTDrone/{train,test}/{Day,Night}/images/{VISIBLE,FIR}/*.jpg
  KUSTDrone/{train,test}/{Day,Night}/label/{VISIBLE,FIR}/*.xml

The public copy used by this project has no validation split. By default this
script takes 762 daytime and 221 nighttime samples from train to reproduce the
validation-set composition reported by the KUSTDrone paper.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
import xml.etree.ElementTree as ET

CLASS_NAMES = {
    0: "people",
    1: "car",
    2: "truck",
    3: "motorcycle",
}

NAME_ALIASES = {
    "people": "people",
    "peopel": "people",
    "peoepl": "people",
    "car": "car",
    "truck": "truck",
    "motorcycle": "motorcycle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "KUSTDrone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "KUSTDrone_YOLO",
    )
    parser.add_argument("--val-day", type=int, default=762)
    parser.add_argument("--val-night", type=int, default=221)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove generated visible/ and fir/ directories before rebuilding.",
    )
    return parser.parse_args()


def modality_source_name(modality: str) -> str:
    return "VISIBLE" if modality == "visible" else "FIR"


def canonical_id(stem: str) -> str:
    """Convert visible_00001 / fir_00001 to the shared sample id 00001."""
    return stem.split("_", 1)[1] if "_" in stem else stem


def index_files(directory: Path, suffix: str) -> dict[str, Path]:
    return {
        canonical_id(path.stem): path
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    }


def split_val_ids(ids: Iterable[str], count: int, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    if count > len(ids):
        raise ValueError(f"Requested {count} validation samples but only {len(ids)} are available")
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[count:]), sorted(ids[:count])


def xml_to_yolo(xml_path: Path, image_path: Path, output_path: Path) -> tuple[int, int]:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    width, height = image.size

    root = ET.parse(xml_path).getroot()
    lines: list[str] = []
    skipped = 0

    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        canonical = NAME_ALIASES.get(raw_name)
        if canonical is None:
            raise ValueError(f"Unknown class name {raw_name!r} in {xml_path}")
        class_id = next(key for key, value in CLASS_NAMES.items() if value == canonical)

        box = obj.find("bndbox")
        if box is None:
            skipped += 1
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            skipped += 1
            continue

        xmin = min(max(xmin, 0.0), float(width))
        xmax = min(max(xmax, 0.0), float(width))
        ymin = min(max(ymin, 0.0), float(height))
        ymax = min(max(ymax, 0.0), float(height))

        if xmax <= xmin or ymax <= ymin:
            skipped += 1
            continue

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), skipped


def link_or_fix_image(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()

    with Image.open(src) as image:
        orientation = image.getexif().get(274)
        if orientation and orientation != 1:
            corrected = ImageOps.exif_transpose(image)
            if corrected.mode != "RGB":
                corrected = corrected.convert("RGB")
            corrected.save(dst, quality=95)
            return True

    dst.symlink_to(src.resolve())
    return False


def write_yaml(path: Path, dataset_path: Path, test_path: str) -> None:
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        f"test: {test_path}",
        "",
        "channels: 3",
        "nc: 4",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in CLASS_NAMES.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"KUSTDrone source not found: {source}")

    if args.overwrite:
        for name in ("visible", "fir"):
            target = out / name
            if target.exists():
                shutil.rmtree(target)
    else:
        for name in ("visible", "fir"):
            if (out / name).exists():
                raise FileExistsError(
                    f"{out / name} already exists; pass --overwrite to rebuild"
                )

    split_ids: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}
    split_ids["train"]["Day"], split_ids["val"]["Day"] = split_val_ids(
        index_files(source / "train" / "Day" / "images" / "VISIBLE", ".jpg").keys(),
        args.val_day,
        args.seed,
    )
    split_ids["train"]["Night"], split_ids["val"]["Night"] = split_val_ids(
        index_files(source / "train" / "Night" / "images" / "VISIBLE", ".jpg").keys(),
        args.val_night,
        args.seed + 1,
    )
    split_ids["test"]["Day"] = sorted(
        index_files(source / "test" / "Day" / "images" / "VISIBLE", ".jpg")
    )
    split_ids["test"]["Night"] = sorted(
        index_files(source / "test" / "Night" / "images" / "VISIBLE", ".jpg")
    )

    yaml_dir = out / "yaml"
    stats: dict[str, dict[str, int]] = {}
    corrected = 0

    for modality in ("visible", "fir"):
        source_name = modality_source_name(modality)
        dataset_root = out / modality
        stats[modality] = {}

        for split in ("train", "val", "test"):
            # The public KUSTDrone archive has no val directory. Validation
            # samples are drawn from the original train split, while test
            # samples come from the original test split.
            source_split = "test" if split == "test" else "train"
            for phase, ids in split_ids[split].items():
                for sample_id in ids:
                    file_prefix = "visible" if modality == "visible" else "fir"
                    image_src = source / source_split / phase / "images" / source_name / f"{file_prefix}_{sample_id}.jpg"
                    label_src = source / source_split / phase / "label" / source_name / f"{file_prefix}_{sample_id}.xml"
                    if not image_src.is_file() or not label_src.is_file():
                        raise FileNotFoundError(
                            f"Missing pair for {modality} {split}/{phase}/{sample_id}: "
                            f"{image_src} / {label_src}"
                        )

                    image_dst = dataset_root / "images" / split / phase / image_src.name
                    label_dst = dataset_root / "labels" / split / phase / f"{image_src.stem}.txt"

                    corrected += int(link_or_fix_image(image_src, image_dst))
                    boxes, skipped = xml_to_yolo(label_src, image_src, label_dst)
                    stats[modality][f"{split}/{phase}/boxes"] = (
                        stats[modality].get(f"{split}/{phase}/boxes", 0) + boxes
                    )
                    stats[modality][f"{split}/{phase}/images"] = (
                        stats[modality].get(f"{split}/{phase}/images", 0) + 1
                    )
                    if skipped:
                        stats[modality][f"{split}/{phase}/skipped"] = (
                            stats[modality].get(f"{split}/{phase}/skipped", 0) + skipped
                        )

        write_yaml(yaml_dir / f"kustdrone_{modality}_all.yaml", dataset_root, "images/test")
        write_yaml(yaml_dir / f"kustdrone_{modality}_day.yaml", dataset_root, "images/test/Day")
        write_yaml(
            yaml_dir / f"kustdrone_{modality}_night.yaml",
            dataset_root,
            "images/test/Night",
        )

    summary = {
        "source": str(source),
        "output": str(out),
        "class_names": CLASS_NAMES,
        "split_sizes": {
            split: {phase: len(ids) for phase, ids in phase_ids.items()}
            for split, phase_ids in split_ids.items()
        },
        "stats": stats,
        "exif_corrected_images": corrected,
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Prepared KUSTDrone YOLO dataset at: {out}")
    print(json.dumps(summary["split_sizes"], indent=2))
    print("YAML files:")
    for path in sorted(yaml_dir.glob("*.yaml")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
