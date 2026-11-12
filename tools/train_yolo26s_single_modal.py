#!/usr/bin/env python3
"""Train and test YOLO26s on M3FD visible, infrared, and RGB-T (dual-stream) images.

Features:
- Sequential visible/infrared single-modality training.
- Two-stream visible + infrared training with independent P1-P3 stems and elementwise P3 fusion.
- RGB-T streams seeded from the single-modality checkpoint unless --pretrained-ckpt overrides it.
- Automatic CPU/CUDA device selection unless --device is supplied.
- Explicit --data-vi, --data-ir, and --data-rgbt paths to prevent modality mix-ups.
- Test-set mAP50, mAP50-95, parameter count, and GFLOPs reporting.
- JSON and CSV summaries.
- No unexpected pretrained-weight download unless --allow-download is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # running from a repo checkout without an editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ultralytics import YOLO


def repo_root() -> Path:
    """Find the Ultralytics repository root from a script under tools/."""
    script = Path(__file__).resolve()
    for parent in (script.parent, *script.parents):
        if (parent / "ultralytics").is_dir():
            return parent
    return script.parent.parent


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Train YOLO26s single-modality baselines on M3FD."
    )
    parser.add_argument(
        "--modality",
        choices=("vi", "ir", "rgbt", "both"),
        default="vi",
        help="Train visible (vi), infrared (ir), both sequentially, or the dual-stream RGB-T model (rgbt).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / "datasets" / "M3FD_YOLO",
        help="Directory containing m3fd_vi.yaml and m3fd_ir.yaml.",
    )
    parser.add_argument(
        "--data-vi",
        type=Path,
        default=None,
        help="Visible dataset YAML. Overrides --dataset-root/m3fd_vi.yaml.",
    )
    parser.add_argument(
        "--data-ir",
        type=Path,
        default=None,
        help="Infrared dataset YAML. Overrides --dataset-root/m3fd_ir.yaml.",
    )
    parser.add_argument(
        "--data-rgbt",
        type=Path,
        default=None,
        help="RGB-T dataset YAML. Overrides --dataset-root/m3fd_rgbt.yaml and the packaged M3FD_rgbt.yaml.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit model path/name. If omitted, search REPO/weights/yolo26s.pt.",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=root / "weights",
        help="Directory used to find yolo26s.pt when --model is omitted.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Ultralytics to download yolo26s.pt when no local model is found.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=root / "runs" / "m3fd_single_modal",
        help="Training output root.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', '0', or '0,1'. Auto prefers CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--skip-train", action="store_true", help="Only run test evaluation.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint for single-modality --skip-train. Not allowed with both.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=None,
        help="Single-modality checkpoint that seeds both RGB-T streams. Defaults to the local yolo26s.pt.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def run_name_for(modality: str) -> str:
    """Return the run directory name for a modality."""
    prefix = {"vi": "vis", "ir": "ir", "rgbt": "rgbt"}[modality]
    return f"{prefix}_yolo26s"


def data_yaml_for(args: argparse.Namespace, modality: str) -> Path:
    if modality == "vi":
        path = args.data_vi or args.dataset_root / "m3fd_vi.yaml"
    elif modality == "ir":
        path = args.data_ir or args.dataset_root / "m3fd_ir.yaml"
    elif modality == "rgbt":
        path = args.data_rgbt or args.dataset_root / "m3fd_rgbt.yaml"
        if not path.expanduser().is_file():
            path = repo_root() / "ultralytics" / "cfg" / "datasets" / "M3FD_rgbt.yaml"
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return path.expanduser().resolve()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    print("CUDA is unavailable; using CPU. Reduce --batch/--imgsz for CPU runs.")
    return "cpu"


def checkpoint_candidates(args: argparse.Namespace) -> list[Path]:
    """Return the local search paths for the single-modality yolo26s.pt checkpoint."""
    root = repo_root()
    return [
        args.weights_root.expanduser() / "yolo26s.pt",
        root / "weights" / "yolo26s.pt",
        root / "yolo26s.pt",
        Path.cwd() / "yolo26s.pt",
    ]


def find_checkpoint(args: argparse.Namespace) -> Path | None:
    """Return the first existing local yolo26s.pt checkpoint, or None."""
    for candidate in checkpoint_candidates(args):
        if candidate.is_file():
            return candidate.resolve()
    return None


def branch_layout(rows: list) -> tuple[int, int, int, int]:
    """Return the visible start, infrared start, early-branch length, and shared-tail start."""
    visible_slice = next(i for i, row in enumerate(rows) if row[2] == "ChannelSlice" and row[3] == [0, 3])
    infrared_slice = next(i for i, row in enumerate(rows) if row[2] == "ChannelSlice" and row[3] == [3, 6])
    fusion = next(i for i, row in enumerate(rows) if row[2] == "Add")
    visible_start = visible_slice + 1
    infrared_start = infrared_slice + 1
    visible_end, infrared_end = rows[fusion][0]
    early_layers = visible_end - visible_start + 1
    if infrared_end - infrared_start + 1 != early_layers:
        raise RuntimeError("Visible and infrared P3 branches must have the same number of layers")
    return visible_start, infrared_start, early_layers, fusion + 1


def dataset_class_names(data_yaml: Path) -> dict | None:
    """Return the {index: name} mapping of a dataset YAML, or None when it cannot be read."""
    try:
        from ultralytics.utils import YAML

        names = YAML.load(data_yaml).get("names")
    except Exception:
        return None
    if isinstance(names, list):
        return dict(enumerate(names))
    return names if isinstance(names, dict) else None


def detect_cls_row_map(source_names: dict | None, target_names: dict | None) -> list[int]:
    """Map each target class index onto the source class index sharing its name, or -1 when there is no match.

    Args:
        source_names (dict | None): Pretrained model class names, e.g. COCO.
        target_names (dict | None): Dataset class names to remap onto.

    Returns:
        (list[int]): One source index per target class, -1 for classes missing from the source.
    """
    if not isinstance(source_names, dict) or not isinstance(target_names, dict):
        return []
    lookup = {str(name).strip().lower(): key for key, name in source_names.items()}
    return [lookup.get(str(target_names.get(i)).strip().lower(), -1) for i in range(len(target_names))]


def detect_cls_keys(modules, nc: int) -> set[str]:
    """Return the state-dict keys of the detection head's classification convs.

    These tensors are the only head tensors whose leading dimension follows the class count.
    """
    from ultralytics.nn.modules.head import Detect

    return {
        f"model.{index}.{attr}.{i}.{len(seq) - 1}.{p}"
        for index, module in enumerate(modules)
        if isinstance(module, Detect)
        for attr in ("cv3", "one2one_cv3")
        for i, seq in enumerate(getattr(module, attr, ()))
        if getattr(seq[-1], "out_channels", None) == nc
        for p in ("weight", "bias")
    }


def init_streams_from_checkpoint(model: YOLO, checkpoint: Path, names: dict | None = None, verbose: bool = True) -> int:
    """Seed both RGB-T streams and the shared head from a single-modality checkpoint.

    The two-stream backbone duplicates only the P1-P3 stem, keeps one shared P4/P5 tail, and adds the two P3 features,
    so source layers are re-indexed onto the two stems and the shared tail instead of being loaded by state-dict name.
    The detection head's classification convs change shape with the class count, so instead of being skipped their rows
    are copied by class name, matching the ``cls_remap`` behaviour a single-modality fine-tune gets from
    ``BaseModel.load``. Without this the dual-stream model would start with a fully random classifier while the
    single-modality baseline starts with several classes already trained, making the two runs incomparable.

    Args:
        model (YOLO): Two-stream model built from the RGB-T model YAML.
        checkpoint (Path): Single-modality checkpoint to copy weights from.
        names (dict, optional): Dataset {index: name} mapping used to remap the classification rows.
        verbose (bool): Print the number of tensors seeded into each part of the model.

    Returns:
        (int): Number of transferred tensors.

    Raises:
        RuntimeError: If an early or shared backbone tensor cannot be placed, which would leave part of the model at a
            random init while the rest starts from the checkpoint.
    """
    source = YOLO(str(checkpoint)).model
    source_backbone = len(source.yaml["backbone"])
    visible_start, infrared_start, early_layers, shared_start = branch_layout(model.model.yaml["backbone"])
    head_offset = len(model.model.yaml["backbone"]) - source_backbone  # head rows shift past the shared backbone

    target_state = model.model.state_dict()
    cls_keys = detect_cls_keys(model.model.model, model.model.yaml["nc"])
    row_map = detect_cls_row_map(getattr(source, "names", None), names)
    rows = [i for i, src in enumerate(row_map) if src >= 0]
    updates = {}
    counts = {"visible": 0, "infrared": 0, "shared": 0, "head": 0}
    missed = {"visible": 0, "infrared": 0, "shared": 0, "head": 0}
    remapped = 0
    for key, value in source.state_dict().items():
        layer, _, rest = key.partition(".")[2].partition(".")
        index = int(layer)
        if index < early_layers:
            targets = (("visible", visible_start + index), ("infrared", infrared_start + index))
        elif index < source_backbone:
            targets = (("shared", shared_start + index - early_layers),)
        else:
            targets = (("head", index + head_offset),)
        for name, target in targets:
            target_key = f"model.{target}.{rest}"
            target_value = target_state.get(target_key)
            if target_value is not None and target_value.shape == value.shape:
                updates[target_key] = value
                counts[name] += 1
            elif (
                rows
                and target_key in cls_keys
                and target_value is not None
                and target_value.shape[1:] == value.shape[1:]
            ):
                tensor = target_value.clone()  # the class rows realign, e.g. COCO 'car' -> M3FD 'Car'
                tensor[rows] = value[torch.tensor([row_map[i] for i in rows])]
                updates[target_key] = tensor
                counts[name] += 1
                remapped += 1
            else:
                missed[name] += 1
    model.model.load_state_dict(updates, strict=False)
    if missed["visible"] or missed["infrared"] or missed["shared"]:
        raise RuntimeError(
            f"{checkpoint} does not match the two-stream backbone: {missed['visible']} visible, "
            f"{missed['infrared']} infrared, and {missed['shared']} shared tensors would keep their random init."
        )
    if verbose:
        if remapped and rows:
            print(f"Remapped {len(rows)}/{len(row_map)} classification rows from pretrained weights by class name")
        print(
            f"Seeded {counts['visible']} visible + {counts['infrared']} infrared P1-P3 tensors, "
            f"{counts['shared']} shared P4/P5 tensors, and {counts['head']} head tensors from {checkpoint} "
            f"({missed['head']} head tensors keep their default init)"
        )
    return len(updates)


def resolve_model(args: argparse.Namespace) -> str:
    root = repo_root()
    if args.model:
        requested = Path(args.model).expanduser()
        candidates = [
            requested,
            Path.cwd() / requested,
            root / requested,
            root / "weights" / requested.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        if requested.suffix.lower() in {".yaml", ".yml"}:
            return args.model
        if args.allow_download:
            return args.model
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            f"Place yolo26s.pt in {root / 'weights'} or pass --allow-download."
        )

    found = find_checkpoint(args)
    if found:
        return str(found)
    if args.allow_download:
        return "yolo26s.pt"
    raise FileNotFoundError(
        "yolo26s.pt was not found. Expected one of:\n  "
        + "\n  ".join(str(p) for p in checkpoint_candidates(args))
        + "\nDownload it explicitly or rerun with --allow-download."
    )


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_complexity(model: YOLO, imgsz: int) -> tuple[int, float | None]:
    torch_model = model.model
    total_params = sum(p.numel() for p in torch_model.parameters())

    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = as_float(get_flops(torch_model, imgsz=imgsz))
    except Exception as first_error:
        try:
            info = torch_model.info(verbose=False)
            if isinstance(info, (tuple, list)) and len(info) >= 4:
                gflops = as_float(info[-1])
            else:
                gflops = as_float(getattr(info, "flops", None))
        except Exception:
            print(f"Warning: unable to calculate GFLOPs: {first_error}")

    return total_params, gflops


def per_class_metrics(metrics: Any) -> dict[str, float | None]:
    names = getattr(metrics, "names", None)
    maps = getattr(metrics.box, "maps", None)
    if not names or maps is None:
        return {}

    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names, key=lambda x: int(x))]
    else:
        ordered = list(names)

    values = list(maps)
    return {
        str(name): as_float(values[i]) if i < len(values) else None
        for i, name in enumerate(ordered)
    }


def find_latest_checkpoint(project: Path, run_name: str) -> Path:
    project = project.expanduser().resolve()
    exact = project / run_name / "weights" / "best.pt"
    if exact.is_file():
        return exact

    candidates = sorted(
        project.glob(f"{run_name}*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found for run '{run_name}' under {project}"
        )
    if len(candidates) > 1:
        print(f"Multiple checkpoints found; using newest: {candidates[0]}")
    return candidates[0]


def train_one(args: argparse.Namespace, modality: str, device: str) -> dict[str, Any]:
    data_yaml = data_yaml_for(args, modality)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    name = run_name_for(modality)
    project = args.project.expanduser().resolve()
    summary_dir = project / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    if args.skip_train:
        resolved_model = str(args.model) if args.model else "N/A"
        if args.weights is not None:
            best_weights = args.weights.expanduser().resolve()
        else:
            best_weights = find_latest_checkpoint(project, name)
        if not best_weights.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {best_weights}")
    else:
        if modality == "rgbt":
            resolved_model = str(args.model) if args.model else "yolo26s-rgbt.yaml"
            model = YOLO(resolved_model)
            checkpoint = args.pretrained_ckpt or find_checkpoint(args)
            if checkpoint is None:
                raise FileNotFoundError(
                    "yolo26s.pt is required to seed the RGB-T streams. Expected one of:\n  "
                    + "\n  ".join(str(p) for p in checkpoint_candidates(args))
                    + "\nPass --pretrained-ckpt or download it explicitly."
                )
            names = dataset_class_names(data_yaml)
            init_streams_from_checkpoint(model, Path(checkpoint).expanduser().resolve(), names=names)
            # Model.train() only hands the trainer this module when self.ckpt is set; otherwise the trainer rebuilds a
            # model from the YAML with random weights and every seeded tensor above is silently discarded. A non-empty
            # ckpt makes train() rebuild from the same YAML and then reload this module's weights by name and shape.
            model.ckpt = {"model": model.model}
        else:
            resolved_model = resolve_model(args)
            model = YOLO(resolved_model)
        model.train(
            data=str(data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            device=device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            project=str(project),
            name=name,
            exist_ok=args.exist_ok,
            save_period=args.save_period,
            plots=True,
            val=True,
            pretrained=True,
            fraction=args.fraction,
            verbose=True,
        )
        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError(f"Training finished without best.pt: {best_weights}")

    train_seconds = time.time() - start

    eval_model = YOLO(str(best_weights))
    total_params, gflops = model_complexity(eval_model, args.imgsz)
    metrics = eval_model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project),
        name=f"{name}_{args.split}",
        plots=True,
        exist_ok=args.exist_ok,
    )

    map50 = as_float(metrics.box.map50)
    map50_95 = as_float(metrics.box.map)

    row: dict[str, Any] = {
        "modality": modality,
        "dataset_yaml": str(data_yaml),
        "model": resolved_model,
        "best_weights": str(best_weights),
        "test_split": args.split,
        "imgsz": args.imgsz,
        "device": device,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "params": total_params,
        "params_M": round(total_params / 1_000_000, 4),
        "GFLOPs": round(gflops, 4) if gflops is not None else None,
        "train_seconds": round(train_seconds, 2),
        "per_class_mAP50_95": per_class_metrics(metrics),
    }

    json_path = summary_dir / f"{name}_{args.split}_metrics.json"
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Final test metrics: {name} ({args.split})")
    print("=" * 72)
    print(f"mAP50:      {map50:.6f}" if map50 is not None else "mAP50:      N/A")
    print(f"mAP50-95:   {map50_95:.6f}" if map50_95 is not None else "mAP50-95:   N/A")
    print(f"Params:     {row['params_M']} M ({total_params:,})")
    print(f"GFLOPs:     {row['GFLOPs']}")
    print(f"Device:     {device}")
    print(f"Checkpoint: {best_weights}")
    print(f"JSON:       {json_path}")
    print("=" * 72 + "\n")
    return row


def write_summary_csv(rows: list[dict[str, Any]], project: Path, filename: str = "single_modal_metrics.csv") -> Path:
    summary_dir = project.expanduser().resolve() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / filename
    columns = [
        "modality",
        "dataset_yaml",
        "model",
        "best_weights",
        "test_split",
        "imgsz",
        "device",
        "mAP50",
        "mAP50_95",
        "params",
        "params_M",
        "GFLOPs",
        "train_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    return csv_path


def validate_args(args: argparse.Namespace) -> None:
    if args.skip_train and args.modality == "both" and args.weights is not None:
        raise ValueError(
            "--weights cannot be used with '--modality both --skip-train'. "
            "Omit it so each modality uses its own newest best.pt."
        )
    if args.modality == "ir" and args.data_vi is not None:
        raise ValueError("--data-vi cannot be used with --modality ir")
    if args.modality == "vi" and args.data_ir is not None:
        raise ValueError("--data-ir cannot be used with --modality vi")
    if args.modality == "rgbt" and (args.data_vi is not None or args.data_ir is not None):
        raise ValueError("--data-vi and --data-ir cannot be used with --modality rgbt")
    if args.data_rgbt is not None and args.modality != "rgbt":
        raise ValueError("--data-rgbt can only be used with --modality rgbt")


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    modalities = ("vi", "ir") if args.modality == "both" else (args.modality,)
    rows = []
    for modality in modalities:
        rows.append(train_one(args, modality, device))
        gc.collect()
        try:
            import torch

            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    csv_file = "rgbt_metrics.csv" if args.modality == "rgbt" else "single_modal_metrics.csv"
    csv_path = write_summary_csv(rows, args.project, csv_file)
    print(f"Combined CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
