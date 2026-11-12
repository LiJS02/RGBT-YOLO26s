#!/usr/bin/env python3
"""YOLO26s P2 training wrapper for RGBTDronePerson tiny-person detection.

The training recipe is intentionally conservative for tiny objects:
  - P2 detection head (YOLO26s-P2)
  - no mosaic / mixup / random scale / color jitter
  - horizontal flip only
  - fixed 640x512-compatible input size
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("visible", "thermal", "both"), default="both")
    parser.add_argument(
        "--data-visible",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson_Tiny" / "yaml" / "rgbtdroneperson_tiny_visible.yaml",
    )
    parser.add_argument(
        "--data-thermal",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson_Tiny" / "yaml" / "rgbtdroneperson_tiny_thermal.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolo26s-p2.yaml",
        help="YOLO26s P2 architecture YAML or a checkpoint.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.home() / ".ljs" / "runs" / "rgbtdroneperson_tiny",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def data_for(args: argparse.Namespace, modality: str) -> Path:
    return (args.data_visible if modality == "visible" else args.data_thermal).expanduser().resolve()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    modalities = ("visible", "thermal") if args.modality == "both" else (args.modality,)

    for modality in modalities:
        data_yaml = data_for(args, modality)
        if not data_yaml.is_file():
            raise FileNotFoundError(data_yaml)

        print(f"\nTraining {modality} with {args.model}")
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            project=str(project),
            name=f"{modality}_yolo26s_p2",
            exist_ok=args.exist_ok,
            save_period=10,
            optimizer="SGD",
            lr0=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            mosaic=0.0,
            close_mosaic=0,
            scale=0.0,
            translate=0.0,
            degrees=0.0,
            shear=0.0,
            perspective=0.0,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            erasing=0.0,
            auto_augment=None,
            rect=False,
            cache=False,
            amp=True,
            val=True,
            plots=True,
        )


if __name__ == "__main__":
    main()
