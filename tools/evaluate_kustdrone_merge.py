#!/usr/bin/env python3
"""Evaluate KUSTDrone single-modality checkpoints on common MERGE test labels.

Training remains modality-specific:
  visible model -> VISIBLE training labels
  infrared model -> FIR training labels

Evaluation uses the paired KUSTDrone_RGBT tree created by
prepare_kustdrone_rgbt.py. That tree contains merged annotations under both
visible/labels and ir/labels, so all models are scored against the same target
set. This script only evaluates models; it does not retrain them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

CLASS_NAMES = {
    0: "people",
    1: "car",
    2: "truck",
    3: "motorcycle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-weights", type=Path, required=True)
    parser.add_argument("--fir-weights", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "KUSTDrone_RGBT",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.home() / ".ljs" / "runs" / "kustdrone_merge_eval",
    )
    return parser.parse_args()


def ensure_merged_dataset(dataset_root: Path) -> None:
    required = [
        dataset_root / "visible" / "images" / "test",
        dataset_root / "visible" / "labels" / "test",
        dataset_root / "ir" / "images" / "test",
        dataset_root / "ir" / "labels" / "test",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "MERGE evaluation dataset is incomplete. Run prepare_kustdrone_rgbt.py first:\n  "
            + "\n  ".join(missing)
        )


def write_eval_yaml(
    path: Path,
    dataset_root: Path,
    modality_dir: str,
    test_path: str,
) -> Path:
    config = {
        "path": str((dataset_root / modality_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": test_path,
        "channels": 3,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def build_eval_yamls(dataset_root: Path) -> dict[str, dict[str, Path]]:
    yaml_root = dataset_root / "yaml_eval_merge"
    result: dict[str, dict[str, Path]] = {}
    for modality_dir in ("visible", "ir"):
        result[modality_dir] = {
            "all": write_eval_yaml(
                yaml_root / f"{modality_dir}_all.yaml", dataset_root, modality_dir, "images/test"
            ),
            "day": write_eval_yaml(
                yaml_root / f"{modality_dir}_day.yaml",
                dataset_root,
                modality_dir,
                "images/test/Day",
            ),
            "night": write_eval_yaml(
                yaml_root / f"{modality_dir}_night.yaml",
                dataset_root,
                modality_dir,
                "images/test/Night",
            ),
        }
    return result


def as_float(value: Any) -> float:
    return float(value)


def evaluate_model(
    modality: str,
    weights: Path,
    yaml_paths: dict[str, Path],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not weights.is_file():
        raise FileNotFoundError(weights)

    print(f"\nEvaluating {modality}: {weights}")
    model = YOLO(str(weights))
    rows = []

    for subset in ("all", "day", "night"):
        metrics = model.val(
            data=str(yaml_paths[subset]),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(args.project.expanduser().resolve()),
            name=f"{modality}_{subset}_merge",
            plots=True,
            exist_ok=True,
        )
        row = {
            "modality": modality,
            "subset": subset,
            "test_gt": "MERGE",
            "weights": str(weights),
            "precision": as_float(metrics.box.mp),
            "recall": as_float(metrics.box.mr),
            "mAP50": as_float(metrics.box.map50),
            "mAP50_95": as_float(metrics.box.map),
        }
        rows.append(row)
        print(
            f"{modality:7s} {subset:5s} "
            f"P={row['precision']:.6f} R={row['recall']:.6f} "
            f"mAP50={row['mAP50']:.6f} mAP50-95={row['mAP50_95']:.6f}"
        )

    return rows


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    project = args.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    ensure_merged_dataset(dataset_root)
    yaml_paths = build_eval_yamls(dataset_root)

    rows = []
    rows.extend(
        evaluate_model(
            "visible",
            args.visible_weights.expanduser().resolve(),
            yaml_paths["visible"],
            args,
        )
    )
    rows.extend(
        evaluate_model(
            "fir",
            args.fir_weights.expanduser().resolve(),
            yaml_paths["ir"],
            args,
        )
    )

    csv_path = project / "kustdrone_single_modal_merge_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = project / "kustdrone_single_modal_merge_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nCombined CSV: {csv_path}")
    print(f"Combined JSON: {json_path}")


if __name__ == "__main__":
    main()
