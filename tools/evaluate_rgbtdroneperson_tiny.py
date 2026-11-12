#!/usr/bin/env python3
"""Evaluate YOLO26s on RGBTDronePerson with the paper's tiny-object protocol.

Metrics follow the supplied QFDet/mmdet code:
  IoU thresholds: 0.25, 0.50, 0.75
  all:       area [1, 1e10]
  tiny:      area [1, 20^2]
  tiny1:     area [1, 8^2]
  tiny2:     area [8^2, 12^2]
  tiny3:     area [12^2, 20^2]
  small:     area [20^2, 32^2]
  reasonable:area [32^2, 1e10]

Classes person/rider/crowd are evaluated. Uncertain annotations are represented
as ignored crowd regions so detections overlapping them are not false positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

CLASS_NAMES = {0: "person", 1: "rider", 2: "crowd"}
AREA_RANGES = [
    [1.0, 1e10],
    [1.0, 400.0],
    [1.0, 64.0],
    [64.0, 144.0],
    [144.0, 400.0],
    [400.0, 1024.0],
    [1024.0, 1e10],
]
AREA_LABELS = ["all", "tiny", "tiny1", "tiny2", "tiny3", "small", "reasonable"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--modality", choices=("visible", "thermal"), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "RGBTDronePerson" / "val_thermal.json",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "runs" / "rgbtdroneperson_tiny_eval",
    )
    return parser.parse_args()


def build_gt(annotation_file: Path, output_file: Path) -> tuple[dict, dict[str, int]]:
    raw = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = raw["images"]
    image_by_name = {Path(image["file_name"]).name: int(image["id"]) for image in images}

    annotations = []
    next_id = 1
    for ann in raw["annotations"]:
        category_id = int(ann["category_id"])
        if category_id in CLASS_NAMES:
            item = dict(ann)
            item["id"] = next_id
            item["category_id"] = category_id
            item["iscrowd"] = 0
            annotations.append(item)
            next_id += 1
        elif category_id == 3:
            # Ignore uncertain regions for all three evaluated classes. COCO's
            # crowd matching uses intersection-over-detection, equivalent to
            # the reference evaluator's IOD-based ignore matching.
            for target_category in CLASS_NAMES:
                item = dict(ann)
                item["id"] = next_id
                item["category_id"] = target_category
                item["iscrowd"] = 1
                item["ignore"] = 1
                annotations.append(item)
                next_id += 1

    gt = {
        "images": images,
        "categories": [
            {"id": class_id, "name": name, "supercategory": "none"}
            for class_id, name in CLASS_NAMES.items()
        ],
        "annotations": annotations,
    }
    output_file.write_text(json.dumps(gt), encoding="utf-8")
    return gt, image_by_name


def export_predictions(
    model: YOLO,
    image_dir: Path,
    image_by_name: dict[str, int],
    args: argparse.Namespace,
    output_file: Path,
) -> tuple[list[dict[str, Any]], float]:
    predictions = []
    started = time.perf_counter()
    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for result in results:
        file_name = Path(result.path).name
        if file_name not in image_by_name:
            continue
        image_id = image_by_name[file_name]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, scores, classes):
            if class_id not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

    elapsed = time.perf_counter() - started
    output_file.write_text(json.dumps(predictions), encoding="utf-8")
    fps = len(image_by_name) / elapsed if elapsed > 0 else 0.0
    return predictions, fps


def summarize_ap(precision: np.ndarray, iou_index: int, area_index: int) -> float:
    values = precision[iou_index, :, :, area_index, -1]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


def model_complexity(model: YOLO, imgsz: int) -> tuple[int, float | None]:
    torch_model = model.model
    total_params = sum(parameter.numel() for parameter in torch_model.parameters())
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(torch_model, imgsz=imgsz))
    except Exception as exc:
        print(f"Warning: unable to calculate GFLOPs: {exc}")
    return total_params, gflops


def main() -> None:
    args = parse_args()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = args.source.expanduser().resolve()
    image_dir = source / "val" / args.modality
    annotation_file = args.ann_file.expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)

    gt_file = out / f"gt_{args.modality}_tiny.json"
    pred_file = out / f"pred_{args.modality}_tiny.json"
    _, image_by_name = build_gt(annotation_file, gt_file)

    model = YOLO(str(args.weights.expanduser().resolve()))
    predictions, fps = export_predictions(model, image_dir, image_by_name, args, pred_file)
    if not predictions:
        raise RuntimeError("No predictions were produced")

    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(pred_file))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = np.array([0.25, 0.5, 0.75])
    evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.params.areaRng = AREA_RANGES
    evaluator.params.areaRngLbl = AREA_LABELS
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]
    all_index = AREA_LABELS.index("all")
    tiny_index = AREA_LABELS.index("tiny")
    tiny1_index = AREA_LABELS.index("tiny1")
    tiny2_index = AREA_LABELS.index("tiny2")
    tiny3_index = AREA_LABELS.index("tiny3")
    small_index = AREA_LABELS.index("small")
    reasonable_index = AREA_LABELS.index("reasonable")

    total_params, gflops = model_complexity(model, args.imgsz)
    raw_ap = {
        "mAPall25": summarize_ap(precision, 0, all_index),
        "mAPall50": summarize_ap(precision, 1, all_index),
        "mAPall75": summarize_ap(precision, 2, all_index),
        "mAPtiny50": summarize_ap(precision, 1, tiny_index),
        "mAPtiny1_50": summarize_ap(precision, 1, tiny1_index),
        "mAPtiny2_50": summarize_ap(precision, 1, tiny2_index),
        "mAPtiny3_50": summarize_ap(precision, 1, tiny3_index),
        "mAPsmall50": summarize_ap(precision, 1, small_index),
        "mAPreasonable50": summarize_ap(precision, 1, reasonable_index),
    }
    metrics = {
        "model": str(args.weights),
        "modality": args.modality,
        "evaluation_split": "RGBTDronePerson val (official test proxy)",
        "classes": list(CLASS_NAMES.values()),
        "iou_thresholds": [0.25, 0.5, 0.75],
        "max_detections": 1000,
        "metric_unit": "percent",
    }
    metrics.update({name: value * 100.0 for name, value in raw_ap.items()})
    metrics.update({"params": total_params, "params_M": total_params / 1_000_000, "params_MB_fp32": total_params * 4 / (1024**2), "GFLOPs": gflops, "FPS_A100": fps, "raw_fraction_metrics": raw_ap})

    metrics_file = out / f"metrics_{args.modality}_tiny.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_file = out / f"metrics_{args.modality}_tiny.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps(metrics, indent=2))
    print(f"GT:        {gt_file}")
    print(f"Pred:      {pred_file}")
    print(f"Metrics:   {metrics_file}")
    print(f"CSV:       {csv_file}")


if __name__ == "__main__":
    main()
