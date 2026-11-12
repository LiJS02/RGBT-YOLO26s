#!/usr/bin/env python3
"""Prepare RGBTDronePerson for separate visible/thermal YOLO training.

Source layout (inside the extracted zip):
  train/{visible,thermal,annotation}
  val/{visible,thermal,annotation}

The original archive contains only train and val. This script holds out 10% of
original train as validation and reserves the original val split as test.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps

CLASS_NAMES = {0: "person", 1: "crowd", 2: "rider", 3: "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson_YOLO",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.jpg") if path.is_file()}


def annotation_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.xml") if path.is_file()}


def xml_to_yolo(xml_path: Path, image_path: Path, output_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size

    root = ET.parse(xml_path).getroot()
    rows = []
    skipped = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in {value for value in CLASS_NAMES.values()}:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        class_id = next(index for index, value in CLASS_NAMES.items() if value == name)

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

        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        rows.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows), skipped


def link_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_yaml(path: Path, root: Path) -> None:
    lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "channels: 3",
        f"nc: {len(CLASS_NAMES)}",
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
        raise FileNotFoundError(source)
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    train_visible = image_ids(source / "train" / "visible")
    train_thermal = image_ids(source / "train" / "thermal")
    train_annotations = annotation_ids(source / "train" / "annotation")
    val_visible = image_ids(source / "val" / "visible")
    val_thermal = image_ids(source / "val" / "thermal")
    val_annotations = annotation_ids(source / "val" / "annotation")

    if not (train_visible == train_thermal == train_annotations):
        raise RuntimeError("train visible/thermal/annotation ids do not match")
    if not (val_visible == val_thermal == val_annotations):
        raise RuntimeError("val visible/thermal/annotation ids do not match")

    ids = sorted(train_visible)
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_fraction))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    test_ids = sorted(val_visible)

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    stats = {}

    for modality in ("visible", "thermal"):
        for split, ids_for_split in split_ids.items():
            source_split = "val" if split == "test" else "train"
            for sample_id in ids_for_split:
                image_src = source / source_split / modality / f"{sample_id}.jpg"
                annotation_src = source / source_split / "annotation" / f"{sample_id}.xml"
                if not image_src.is_file() or not annotation_src.is_file():
                    raise FileNotFoundError(f"Missing {modality} pair: {image_src} / {annotation_src}")

                link_image(
                    image_src,
                    out / modality / "images" / split / image_src.name,
                )
                boxes, skipped = xml_to_yolo(
                    annotation_src,
                    image_src,
                    out / modality / "labels" / split / f"{sample_id}.txt",
                )
                key = f"{modality}/{split}"
                stats[f"{key}/images"] = stats.get(f"{key}/images", 0) + 1
                stats[f"{key}/boxes"] = stats.get(f"{key}/boxes", 0) + boxes
                if skipped:
                    stats[f"{key}/skipped"] = stats.get(f"{key}/skipped", 0) + skipped

        write_yaml(out / "yaml" / f"rgbtdroneperson_{modality}.yaml", out / modality)

    summary = {
        "source": str(source),
        "output": str(out),
        "class_names": CLASS_NAMES,
        "split_sizes": {split: len(ids_for_split) for split, ids_for_split in split_ids.items()},
        "protocol": {
            "train": "90% of original train",
            "val": "10% deterministic holdout from original train",
            "test": "original val split",
        },
        "seed": args.seed,
        "stats": stats,
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"YAML: {out / 'yaml' / 'rgbtdroneperson_visible.yaml'}")
    print(f"YAML: {out / 'yaml' / 'rgbtdroneperson_thermal.yaml'}")


if __name__ == "__main__":
    main()
