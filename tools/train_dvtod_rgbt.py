#!/usr/bin/env python3
"""Train the P3-add RGB-T YOLO26s baseline on DVTOD common-FOV.

The model uses independent visible and infrared YOLO26 P1-P3 stems, adds the two P3 feature maps elementwise, then
processes the fused feature with one shared P4/P5 backbone and the original YOLO26 neck and detection head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from train_yolo26s_single_modal_rgbt import resolve_device, train_one, write_summary_csv  # noqa: E402


def repo_root() -> Path:
    """Find the repository root from a script under tools/."""
    return REPO_ROOT


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Train the P3-add visible + infrared YOLO26s baseline on DVTOD common-FOV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "ultralytics" / "cfg" / "datasets" / "DVTOD_commonfov_rgbt.yaml",
        help="Combined visible + infrared dataset YAML.",
    )
    parser.add_argument("--model", default=None, help="Model YAML or checkpoint; defaults to yolo26s-rgbt.yaml.")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=root / "weights",
        help="Directory searched for a single-modality yolo26s.pt checkpoint.",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading a missing checkpoint.")
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=None,
        help=(
            "Checkpoint used for both P1-P3 stems, the shared P4/P5 backbone, and the head "
            "when specific ones are absent."
        ),
    )
    parser.add_argument(
        "--pretrained-vis-ckpt",
        type=Path,
        default=None,
        help="Visible single-modality checkpoint used to initialize the visible P1-P3 stem.",
    )
    parser.add_argument(
        "--pretrained-ir-ckpt",
        type=Path,
        default=None,
        help="Infrared single-modality checkpoint used to initialize the infrared P1-P3 stem.",
    )
    parser.add_argument(
        "--head-init",
        choices=("mean", "visible", "ir"),
        default="visible",
        help="Checkpoint used for the shared P4/P5 backbone and detection head.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=root / "runs" / "dvtod_commonfov_rgbt",
        help="Training output root.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', '0', or '0,1'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate an existing checkpoint.")
    parser.add_argument("--weights", type=Path, default=None, help="Checkpoint for --skip-train.")
    parser.add_argument("--exist-ok", action="store_true")

    args = parser.parse_args()
    args.data_rgbt = args.data.expanduser().resolve()
    args.dataset_root = args.data_rgbt.parent
    args.data_vi = None
    args.data_ir = None
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    row = train_one(args, "rgbt", device)
    csv_path = write_summary_csv([row], args.project, "dvtod_commonfov_rgbt_metrics.csv")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
