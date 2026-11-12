#!/usr/bin/env python3
"""Prepare RGBTDronePerson for tiny-person YOLO training.

Paper-compatible evaluation classes: person, rider, crowd.
Uncertain boxes are omitted from training labels. The official train/val split
is retained, and the original val split is also used as test to match the
reference implementation's protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps

CLASS_NAMES = {0: "person", 1: "rider", 2: "crowd"}
NAME_TO_ID = {name: index for index, name in CLASS_NAMES.items()}


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
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson_Tiny",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def xml_to_yolo(xml_path: Path, image_path: Path, output_path: Path) -> tuple[int, int, int]:
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size

    root = ET.parse(xml_path).getroot()
    rows = []
    skipped_uncertain = 0
    skipped_invalid = 0

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name == "uncertain":
            skipped_uncertain += 1
            continue
        if name not in NAME_TO_ID:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")

        box = obj.find("bndbox")
        if box is None:
            skipped_invalid += 1
            continue
        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue

        xmin = min(max(xmin, 0.0), float(width))
        xmax = min(max(xmax, 0.0), float(width))
        ymin = min(max(ymin, 0.0), float(height))
        ymax = min(max(ymax, 0.0), float(height))
        if xmax <= xmin or ymax <= ymin:
            skipped_invalid += 1
            continue

        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        rows.append(f"{NAME_TO_ID[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows), skipped_uncertain, skipped_invalid


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
        "test: images/val",
        "",
        "channels: 3",
        "nc: 3",
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

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    stats = {}
    for modality in ("visible", "thermal"):
        for split in ("train", "val"):
            image_dir = source / split / modality
            annotation_dir = source / split / "annotation"
            images = sorted(path for path in image_dir.glob("*.jpg") if path.is_file())
            for image_path in images:
                annotation_path = annotation_dir / f"{image_path.stem}.xml"
                if not annotation_path.is_file():
                    raise FileNotFoundError(annotation_path)
                link_image(
                    image_path,
                    out / modality / "images" / split / image_path.name,
                )
                boxes, uncertain, invalid = xml_to_yolo(
                    annotation_path,
                    image_path,
                    out / modality / "labels" / split / f"{image_path.stem}.txt",
                )
                key = f"{modality}/{split}"
                stats[f"{key}/images"] = stats.get(f"{key}/images", 0) + 1
                stats[f"{key}/boxes"] = stats.get(f"{key}/boxes", 0) + boxes
                stats[f"{key}/uncertain_skipped"] = stats.get(f"{key}/uncertain_skipped", 0) + uncertain
                stats[f"{key}/invalid_skipped"] = stats.get(f"{key}/invalid_skipped", 0) + invalid

        write_yaml(out / "yaml" / f"rgbtdroneperson_tiny_{modality}.yaml", out / modality)

    summary = {
        "source": str(source),
        "output": str(out),
        "class_names": CLASS_NAMES,
        "split_protocol": "official train/val; val is also used as test",
        "stats": stats,
        "note": "uncertain boxes are excluded from training labels",
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"YAML: {out / 'yaml' / 'rgbtdroneperson_tiny_visible.yaml'}")
    print(f"YAML: {out / 'yaml' / 'rgbtdroneperson_tiny_thermal.yaml'}")


if __name__ == "__main__":
    main()
