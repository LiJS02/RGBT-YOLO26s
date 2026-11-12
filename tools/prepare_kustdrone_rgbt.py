#!/usr/bin/env python3
"""Prepare paired KUSTDrone RGB-T data for the two-stream YOLO model.

Fusion training uses the dataset's MERGE annotations, which are the union of
the visible and thermal annotations. Visible and infrared images are copied
into two roots with identical relative names so the dual-stream dataset can
pair them by file name.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

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
        default=Path.home() / ".ljs" / "data" / "KUSTDrone_RGBT",
    )
    parser.add_argument("--val-day", type=int, default=762)
    parser.add_argument("--val-night", type=int, default=221)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_id(stem: str) -> str:
    return stem.split("_", 1)[1] if "_" in stem else stem


def index_ids(directory: Path, suffix: str) -> set[str]:
    return {
        canonical_id(path.stem)
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    }


def split_val_ids(ids: set[str], count: int, seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted(ids)
    if count > len(ordered):
        raise ValueError(f"Requested {count} validation samples but only {len(ordered)} exist")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[count:]), sorted(ordered[:count])


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


def xml_to_yolo(xml_path: Path, image_path: Path, output_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size

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

        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), skipped


def write_rgbt_yaml(path: Path, visible_root: Path, ir_root: Path, test_path: str) -> None:
    lines = [
        f"path: {visible_root}",
        f"ir_path: {ir_root}",
        "train: images/train",
        "val: images/val",
        f"test: {test_path}",
        "",
        "channels: 6",
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
        raise FileNotFoundError(source)

    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    train_day, val_day = split_val_ids(
        index_ids(source / "train" / "Day" / "images" / "VISIBLE", ".jpg"),
        args.val_day,
        args.seed,
    )
    train_night, val_night = split_val_ids(
        index_ids(source / "train" / "Night" / "images" / "VISIBLE", ".jpg"),
        args.val_night,
        args.seed + 1,
    )
    test_day = sorted(index_ids(source / "test" / "Day" / "images" / "VISIBLE", ".jpg"))
    test_night = sorted(index_ids(source / "test" / "Night" / "images" / "VISIBLE", ".jpg"))

    splits = {
        "train": {"Day": train_day, "Night": train_night},
        "val": {"Day": val_day, "Night": val_night},
        "test": {"Day": test_day, "Night": test_night},
    }

    corrected = 0
    stats: dict[str, int] = {}
    for split in ("train", "val", "test"):
        source_split = "test" if split == "test" else "train"
        for phase, ids in splits[split].items():
            for sample_id in ids:
                visible_src = source / source_split / phase / "images" / "VISIBLE" / f"visible_{sample_id}.jpg"
                ir_src = source / source_split / phase / "images" / "FIR" / f"fir_{sample_id}.jpg"
                merge_xml = source / source_split / phase / "label" / "MERGE" / f"{sample_id}.xml"
                if not visible_src.is_file() or not ir_src.is_file() or not merge_xml.is_file():
                    raise FileNotFoundError(
                        f"Missing RGB-T pair for {split}/{phase}/{sample_id}: "
                        f"{visible_src} / {ir_src} / {merge_xml}"
                    )

                visible_dst = out / "visible" / "images" / split / phase / f"{sample_id}.jpg"
                ir_dst = out / "ir" / "images" / split / phase / f"{sample_id}.jpg"
                visible_label = out / "visible" / "labels" / split / phase / f"{sample_id}.txt"
                ir_label = out / "ir" / "labels" / split / phase / f"{sample_id}.txt"

                corrected += int(link_or_fix_image(visible_src, visible_dst))
                corrected += int(link_or_fix_image(ir_src, ir_dst))
                boxes, skipped = xml_to_yolo(merge_xml, visible_src, visible_label)
                ir_label.parent.mkdir(parents=True, exist_ok=True)
                ir_label.write_text(visible_label.read_text(encoding="utf-8"), encoding="utf-8")
                key = f"{split}/{phase}"
                stats[f"{key}/images"] = stats.get(f"{key}/images", 0) + 1
                stats[f"{key}/boxes"] = stats.get(f"{key}/boxes", 0) + boxes
                if skipped:
                    stats[f"{key}/skipped"] = stats.get(f"{key}/skipped", 0) + skipped

    yaml_dir = out / "yaml"
    write_rgbt_yaml(yaml_dir / "kustdrone_rgbt_all.yaml", out / "visible", out / "ir", "images/test")
    write_rgbt_yaml(yaml_dir / "kustdrone_rgbt_day.yaml", out / "visible", out / "ir", "images/test/Day")
    write_rgbt_yaml(yaml_dir / "kustdrone_rgbt_night.yaml", out / "visible", out / "ir", "images/test/Night")

    summary = {
        "source": str(source),
        "output": str(out),
        "class_names": CLASS_NAMES,
        "splits": {split: {phase: len(ids) for phase, ids in values.items()} for split, values in splits.items()},
        "stats": stats,
        "exif_corrected_images": corrected,
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared paired KUSTDrone RGB-T dataset at: {out}")
    print(json.dumps(summary["splits"], indent=2))
    for path in sorted(yaml_dir.glob("*.yaml")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
