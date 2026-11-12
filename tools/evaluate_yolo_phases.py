#!/usr/bin/env python3
"""Evaluate a YOLO checkpoint on KUSTDrone All, Day, and Night test subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--yaml-root",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "KUSTDrone_YOLO" / "yaml",
    )
    parser.add_argument("--modality", choices=("visible", "fir"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.home() / ".ljs" / "runs" / "kustdrone_eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    yaml_root = args.yaml_root.expanduser().resolve()
    project = args.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    rows = []

    for subset in ("all", "day", "night"):
        data_yaml = yaml_root / f"kustdrone_{args.modality}_{subset}.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(data_yaml)

        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(project),
            name=f"{args.modality}_{subset}",
            plots=True,
            exist_ok=True,
        )
        row = {
            "modality": args.modality,
            "subset": subset,
            "weights": str(weights),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        rows.append(row)
        print(
            f"{args.modality} {subset}: "
            f"mAP50={row['mAP50']:.6f}, mAP50-95={row['mAP50_95']:.6f}"
        )

    csv_path = project / f"{args.modality}_day_night_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = project / f"{args.modality}_day_night_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
