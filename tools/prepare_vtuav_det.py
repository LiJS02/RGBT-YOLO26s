#!/usr/bin/env python3
"""Prepare VTUAV-det for visible and thermal single-modality YOLO training.

Only category 0 (person) is retained, matching the VTUAV-det protocol in the
paper. The official train/test split is preserved; the original test split is
also used as val because the archive has no separate validation split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "VTUAV_extracted" / "VTUAV_v1.0",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "VTUAV_det_YOLO",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_yaml(path: Path, root: Path) -> None:
    lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        "",
        "channels: 3",
        "nc: 1",
        "names:",
        "  0: person",
    ]
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

    split_specs = {
        "train": (source / "train_ir.json", "train"),
        "test": (source / "val_ir.json", "test"),
    }
    stats = {}

    for split, (ann_file, image_split) in split_specs.items():
        if not ann_file.is_file():
            raise FileNotFoundError(ann_file)
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        image_by_id = {int(image["id"]): image for image in data["images"]}
        annotations_by_image = {image_id: [] for image_id in image_by_id}
        for ann in data["annotations"]:
            if int(ann["category_id"]) == 0:
                annotations_by_image[int(ann["image_id"])].append(ann)

        for modality in ("rgb", "ir"):
            root = out / ("visible" if modality == "rgb" else "thermal")
            for image_id, image in image_by_id.items():
                file_name = image["file_name"]
                image_src = source / image_split / modality / file_name
                if not image_src.is_file():
                    raise FileNotFoundError(image_src)
                link(image_src, root / "images" / split / file_name)

                width = float(image["width"])
                height = float(image["height"])
                rows = []
                for ann in annotations_by_image[image_id]:
                    x, y, w, h = map(float, ann["bbox"])
                    x1 = max(0.0, min(x, width))
                    y1 = max(0.0, min(y, height))
                    x2 = max(0.0, min(x + w, width))
                    y2 = max(0.0, min(y + h, height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    xc = (x1 + x2) / 2.0 / width
                    yc = (y1 + y2) / 2.0 / height
                    bw = (x2 - x1) / width
                    bh = (y2 - y1) / height
                    rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                label_path = root / "labels" / split / f"{Path(file_name).stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

                key = f"{modality}/{split}"
                stats[f"{key}/images"] = stats.get(f"{key}/images", 0) + 1
                stats[f"{key}/boxes"] = stats.get(f"{key}/boxes", 0) + len(rows)

        write_yaml(out / "yaml" / "vtuav_det_visible.yaml", out / "visible")
        write_yaml(out / "yaml" / "vtuav_det_thermal.yaml", out / "thermal")

    summary = {
        "source": str(source),
        "output": str(out),
        "class_names": {0: "person"},
        "split_protocol": "official train/test; test is also used as val",
        "stats": stats,
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"YAML: {out / 'yaml' / 'vtuav_det_visible.yaml'}")
    print(f"YAML: {out / 'yaml' / 'vtuav_det_thermal.yaml'}")


if __name__ == "__main__":
    main()
