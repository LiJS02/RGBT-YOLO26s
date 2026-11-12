#!/usr/bin/env python3
"""Prepare DVTOD for independent visible and infrared YOLO training.

The public archive contains only original train/val splits. To avoid selecting
the model and reporting the final score on the same data, this script:
  - splits a deterministic subset of original train into the new val set;
  - uses the remaining original train as training data;
  - reserves the original val split as the independent test set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

CLASS_NAMES = {0: "person", 1: "car", 2: "bicycle"}
MODALITY_DIRS = {"visible": "vis1", "ir": "ir1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "DVTOD" / "DVTOD dataset",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "DVTOD_YOLO",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paired_ids(image_dir: Path, label_dir: Path) -> set[str]:
    images = {path.stem for path in image_dir.glob("*.jpg") if path.is_file()}
    labels = {path.stem for path in label_dir.glob("*.txt") if path.is_file()}
    if images != labels:
        missing_labels = sorted(images - labels)
        missing_images = sorted(labels - images)
        raise RuntimeError(
            f"Image/label mismatch in {image_dir.parent}: "
            f"missing labels={missing_labels[:5]}, missing images={missing_images[:5]}"
        )
    return images


def symlink(src: Path, dst: Path) -> None:
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

    visible_train = paired_ids(
        source / "vis1" / "images" / "train", source / "vis1" / "labels" / "train"
    )
    infrared_train = paired_ids(
        source / "ir1" / "images" / "train", source / "ir1" / "labels" / "train"
    )
    visible_test = paired_ids(
        source / "vis1" / "images" / "val", source / "vis1" / "labels" / "val"
    )
    infrared_test = paired_ids(
        source / "ir1" / "images" / "val", source / "ir1" / "labels" / "val"
    )

    common_train = visible_train & infrared_train
    common_test = visible_test & infrared_test
    if not common_train or not common_test:
        raise RuntimeError("No common paired train/test ids found")

    train_ids = sorted(common_train)
    random.Random(args.seed).shuffle(train_ids)
    n_val = max(1, round(len(train_ids) * args.val_fraction))
    val_ids = sorted(train_ids[:n_val])
    new_train_ids = sorted(train_ids[n_val:])
    test_ids = sorted(common_test)

    split_ids = {"train": new_train_ids, "val": val_ids, "test": test_ids}
    for modality, source_name in MODALITY_DIRS.items():
        for split, ids in split_ids.items():
            source_split = "val" if split == "test" else "train"
            for sample_id in ids:
                image_src = source / source_name / "images" / source_split / f"{sample_id}.jpg"
                label_src = source / source_name / "labels" / source_split / f"{sample_id}.txt"
                if not image_src.is_file() or not label_src.is_file():
                    raise FileNotFoundError(f"Missing {modality} pair: {image_src} / {label_src}")
                symlink(
                    image_src,
                    out / modality / "images" / split / image_src.name,
                )
                symlink(
                    label_src,
                    out / modality / "labels" / split / label_src.name,
                )
        write_yaml(out / "yaml" / f"dvtod_{modality}.yaml", out / modality)

    summary = {
        "source": str(source),
        "output": str(out),
        "class_names": CLASS_NAMES,
        "split_sizes": {split: len(ids) for split, ids in split_ids.items()},
        "protocol": {
            "train": "remaining 90% of original train",
            "val": "10% deterministic holdout from original train",
            "test": "original val split",
        },
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"YAML: {out / 'yaml' / 'dvtod_visible.yaml'}")
    print(f"YAML: {out / 'yaml' / 'dvtod_ir.yaml'}")


if __name__ == "__main__":
    main()
