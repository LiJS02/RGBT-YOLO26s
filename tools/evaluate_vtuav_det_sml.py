#!/usr/bin/env python3
"""Evaluate single-modality YOLO checkpoints on VTUAV-det with COCO S/M/L AP.

Reported values:
  mAP50 and mAP50-95 for all objects,
  small objects (< 32^2 area),
  medium objects (32^2 to 96^2),
  large objects (> 96^2).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--modality", choices=("visible", "thermal"), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "VTUAV_extracted" / "VTUAV_v1.0",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "VTUAV_extracted" / "VTUAV_v1.0" / "val_ir.json",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "runs" / "vtuav_det_eval",
    )
    return parser.parse_args()


def build_gt(annotation_file: Path, output_file: Path) -> dict[str, int]:
    raw = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = raw["images"]
    image_by_name = {Path(image["file_name"]).name: int(image["id"]) for image in images}
    annotations = []
    for ann in raw["annotations"]:
        if int(ann["category_id"]) != 0:
            continue
        item = dict(ann)
        item["category_id"] = 0
        item["iscrowd"] = 0
        annotations.append(item)

    gt = {
        "images": images,
        "categories": [{"id": 0, "name": "person", "supercategory": "none"}],
        "annotations": annotations,
    }
    output_file.write_text(json.dumps(gt), encoding="utf-8")
    return image_by_name


def export_predictions(
    model: YOLO,
    image_dir: Path,
    image_by_name: dict[str, int],
    args: argparse.Namespace,
    output_file: Path,
) -> tuple[list[dict[str, Any]], float]:
    predictions = []
    start = time.perf_counter()
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
        image_id = image_by_name.get(file_name)
        if image_id is None or result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, classes):
            if class_id != 0:
                continue
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    elapsed = time.perf_counter() - start
    output_file.write_text(json.dumps(predictions), encoding="utf-8")
    return predictions, len(image_by_name) / elapsed if elapsed > 0 else 0.0


def metric_at(precision: np.ndarray, iou_index: int | None, area_index: int) -> float:
    # precision shape: [IoU, recall, class, area, max_det]
    if iou_index is None:
        values = precision[:, :, :, area_index, -1]
    else:
        values = precision[iou_index, :, :, area_index, -1]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


def model_complexity(model: YOLO, imgsz: int) -> tuple[int, float | None]:
    parameters = sum(p.numel() for p in model.model.parameters())
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(model.model, imgsz=imgsz))
    except Exception as exc:
        print(f"Warning: unable to calculate GFLOPs: {exc}")
    return parameters, gflops


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    annotation_file = args.ann_file.expanduser().resolve()
    image_dir = source / "test" / ("rgb" if args.modality == "visible" else "ir")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)

    gt_file = out / f"gt_vtuav_{args.modality}.json"
    pred_file = out / f"pred_vtuav_{args.modality}.json"
    image_by_name = build_gt(annotation_file, gt_file)

    model = YOLO(str(args.weights.expanduser().resolve()))
    predictions, fps = export_predictions(model, image_dir, image_by_name, args, pred_file)
    if not predictions:
        raise RuntimeError("No predictions produced")

    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(pred_file))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = np.linspace(0.5, 0.95, 10)
    evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.params.areaRng = [
        [0.0, 1e10],
        [0.0, 32.0**2],
        [32.0**2, 96.0**2],
        [96.0**2, 1e10],
    ]
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]

    raw = {}
    for area_name, area_index in (("all", 0), ("small", 1), ("medium", 2), ("large", 3)):
        raw[f"mAP50_{area_name}"] = metric_at(precision, 0, area_index)
        raw[f"mAP50_95_{area_name}"] = metric_at(precision, None, area_index)

    params, gflops = model_complexity(model, args.imgsz)
    metrics = {
        "model": str(args.weights),
        "modality": args.modality,
        "test_split": "official VTUAV-det test",
        "class": "person",
        "iou_thresholds": "0.50:0.05:0.95",
        "max_detection": args.max_det,
        "metric_unit": "percent",
    }
    metrics.update({key: value * 100.0 for key, value in raw.items()})
    metrics.update(
        {
            "params": params,
            "params_M": params / 1_000_000,
            "GFLOPs": gflops,
            "FPS_A100": fps,
            "raw_fraction_metrics": raw,
        }
    )

    metrics_json = out / f"metrics_vtuav_{args.modality}.json"
    metrics_csv = out / f"metrics_vtuav_{args.modality}.csv"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with metrics_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps(metrics, indent=2))
    print(f"GT:      {gt_file}")
    print(f"Pred:    {pred_file}")
    print(f"Metrics: {metrics_json}")
    print(f"CSV:     {metrics_csv}")


if __name__ == "__main__":
    main()
