#!/usr/bin/env python3
"""Prepare M3FD_Detection for separate Ultralytics visible/IR baselines.

Expected extracted source layout:
  M3FD_Detection/
    vi/00000.png
    ir/00000.png
    labels/00000.txt
    Annotation/00000.xml

This script creates one YOLO dataset tree per modality and a fixed 80/10/10
split shared by both modalities. Images and labels are symlinked, so no image
copies are made.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

CLASS_NAMES = {
    0: "People",
    1: "Car",
    2: "Bus",
    3: "Lamp",
    4: "Motorcycle",
    5: "Truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "datasets" / "M3FD_Detection",
        help="Extracted M3FD_Detection directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "datasets" / "M3FD_YOLO",
        help="Output root for the two YOLO dataset trees",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()
    elif dst.exists():
        raise RuntimeError(f"Refusing to overwrite non-symlink file: {dst}")
    dst.symlink_to(src)


def write_dataset_yaml(path: Path, dataset_path: Path) -> None:
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")

    required = [source / "vi", source / "ir", source / "labels"]
    missing = [str(p) for p in required if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing source directories. Extract the zip first:\n  " + "\n  ".join(missing)
        )

    vi_ids = {p.stem for p in (source / "vi").glob("*.png")}
    ir_ids = {p.stem for p in (source / "ir").glob("*.png")}
    label_ids = {p.stem for p in (source / "labels").glob("*.txt")}
    ids = sorted(vi_ids & ir_ids & label_ids)

    if not ids:
        raise RuntimeError("No paired vi/ir/label samples found")

    unpaired = (vi_ids ^ ir_ids) | (vi_ids ^ label_ids)
    if unpaired:
        print(f"Warning: {len(unpaired)} unpaired samples were ignored")

    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_train = int(len(ids) * args.train_ratio)
    n_val = int(len(ids) * args.val_ratio)
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, split_ids in splits.items():
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(split_ids) + "\n", encoding="utf-8"
        )

    for modality in ("vi", "ir"):
        dataset_root = out / modality
        for split, split_ids in splits.items():
            for sample_id in split_ids:
                symlink(
                    source / modality / f"{sample_id}.png",
                    dataset_root / "images" / split / f"{sample_id}.png",
                )
                symlink(
                    source / "labels" / f"{sample_id}.txt",
                    dataset_root / "labels" / split / f"{sample_id}.txt",
                )
        write_dataset_yaml(out / f"m3fd_{modality}.yaml", dataset_root)

    print(f"Prepared {len(ids)} paired samples at {out}")
    for split, split_ids in splits.items():
        print(f"  {split}: {len(split_ids)}")
    print(f"  visible YAML: {out / 'm3fd_vi.yaml'}")
    print(f"  IR YAML:      {out / 'm3fd_ir.yaml'}")
    print("Class order: " + ", ".join(f"{k}={v}" for k, v in CLASS_NAMES.items()))


if __name__ == "__main__":
    main()
