#!/usr/bin/env python3
"""Train and test YOLO26s on M3FD visible, infrared, and RGB-T (dual-stream) images.

Features:
- Sequential visible/infrared single-modality training.
- Two-stream visible + infrared training with channel-concat 1x1 fusion after the backbones.
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
        help="Fallback checkpoint used for both RGB-T streams when modality-specific checkpoints are not supplied.",
    )
    parser.add_argument(
        "--pretrained-vis-ckpt",
        type=Path,
        default=None,
        help="Visible single-modality checkpoint used to initialize the visible RGB-T stream.",
    )
    parser.add_argument(
        "--pretrained-ir-ckpt",
        type=Path,
        default=None,
        help="Infrared single-modality checkpoint used to initialize the infrared RGB-T stream.",
    )
    parser.add_argument(
        "--fusion-init",
        choices=("mean", "visible", "ir"),
        default="visible",
        help="Initial fusion behavior. 'visible' preserves the visible model at step 0.",
    )
    parser.add_argument(
        "--head-init",
        choices=("mean", "visible", "ir"),
        default="visible",
        help="Which single-modality head initializes the shared detector head.",
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


def stream_layer_indices(rows: list) -> tuple[int, int]:
    """Return the first visible and first infrared layer index of a two-stream backbone definition."""
    visible = next(i for i, row in enumerate(rows) if row[2] == "ChannelSlice" and row[3] == [0, 3])
    infrared = next(i for i, row in enumerate(rows) if row[2] == "ChannelSlice" and row[3] == [3, 6])
    return visible + 1, infrared + 1



def _source_layer_index_and_suffix(key: str) -> tuple[int, str]:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model":
        raise ValueError(f"Unexpected state-dict key: {key}")
    return int(parts[1]), parts[2]


def _merge_head_tensor(current: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(current):
        return (current + incoming) / 2
    return torch.maximum(current, incoming)


def init_streams_from_checkpoints(
    model: YOLO,
    visible_checkpoint: Path,
    infrared_checkpoint: Path,
    head_init: str = "visible",
    verbose: bool = True,
) -> int:
    """Seed each RGB-T stream from its own modality-specific checkpoint.

    Backbone weights come from the matching single-modality model. Compatible
    shared-head tensors are averaged across both checkpoints. If the
    checkpoints use the same six M3FD classes, the classification head is
    initialized as well instead of being left random.
    """
    visible_source = YOLO(str(visible_checkpoint)).model.cpu().float()
    infrared_source = YOLO(str(infrared_checkpoint)).model.cpu().float()

    visible_backbone = len(visible_source.yaml["backbone"])
    infrared_backbone = len(infrared_source.yaml["backbone"])
    if visible_backbone != infrared_backbone:
        raise RuntimeError(
            "Visible and infrared checkpoints use different backbone lengths: "
            f"{visible_backbone} versus {infrared_backbone}"
        )

    visible_start, infrared_start = stream_layer_indices(model.model.yaml["backbone"])
    head_offset = len(model.model.yaml["backbone"]) - visible_backbone
    target_state = model.model.state_dict()
    updates: dict[str, torch.Tensor] = {}
    counts = {"visible": 0, "infrared": 0, "head": 0}
    missed = {"visible": 0, "infrared": 0, "head": 0}

    def copy_backbone(source: torch.nn.Module, target_start: int, label: str) -> None:
        for key, value in source.state_dict().items():
            index, suffix = _source_layer_index_and_suffix(key)
            if index >= visible_backbone:
                continue
            target_key = f"model.{target_start + index}.{suffix}"
            if target_state.get(target_key) is not None and target_state[target_key].shape == value.shape:
                updates[target_key] = value
                counts[label] += 1
            else:
                missed[label] += 1

    copy_backbone(visible_source, visible_start, "visible")
    copy_backbone(infrared_source, infrared_start, "infrared")

    if missed["visible"] or missed["infrared"]:
        raise RuntimeError(
            "A backbone checkpoint does not match the two-stream model: "
            f"{missed['visible']} visible and {missed['infrared']} infrared tensors "
            "would keep their random init."
        )

    if head_init == "mean":
        head_sources = (visible_source, infrared_source)
    elif head_init == "ir":
        head_sources = (infrared_source,)
    else:
        head_sources = (visible_source,)

    for source in head_sources:
        for key, value in source.state_dict().items():
            index, suffix = _source_layer_index_and_suffix(key)
            if index < visible_backbone:
                continue
            target_key = f"model.{index + head_offset}.{suffix}"
            target_tensor = target_state.get(target_key)
            if target_tensor is None or target_tensor.shape != value.shape:
                missed["head"] += 1
                continue
            if target_key in updates:
                updates[target_key] = _merge_head_tensor(updates[target_key], value)
            else:
                updates[target_key] = value
            counts["head"] += 1

    model.model.load_state_dict(updates, strict=False)
    if verbose:
        print(
            f"Seeded {counts['visible']} visible backbone tensors from {visible_checkpoint} and "
            f"{counts['infrared']} infrared backbone tensors from {infrared_checkpoint}; "
            f"loaded/averaged {counts['head']} shared-head tensors "
            f"({missed['head']} incompatible head tensors keep their default init)."
        )
    return len(updates)


def fusion_layer_indices(rows: list) -> list[int]:
    """Return the indices of the 1x1 Conv layers that fuse two concatenated streams in a backbone definition."""
    sources = [i - 1 if row[0] == -1 else row[0] for i, row in enumerate(rows)]  # -1 is the preceding row
    concat_rows = {i for i, row in enumerate(rows) if row[2] == "Concat" and isinstance(row[0], list)}
    return [i for i, row in enumerate(rows) if row[2] == "Conv" and sources[i] in concat_rows and row[3][1] == 1]


def init_fusion_layers(model: YOLO, fusion_init: str = "visible", verbose: bool = True) -> int:
    """Initialize each 1x1 fusion Conv as the channel-aligned mean of its two input streams.

    The fusion convs are the only new parameterized layers between the COCO-pretrained backbones and the
    COCO-pretrained neck, so leaving them random scrambles the channel layout the neck and head expect and wastes their
    pretrained weights. Setting each kernel to 0.5 * [I | I] makes a fused map the mean of the two streams, so the
    pretrained neck keeps working from step 0 and training only has to learn the fusion residual.

    Args:
        model (YOLO): Two-stream model whose fusion convs are initialized in place.
        verbose (bool): Print the initialized layer indices.

    Returns:
        (int): Number of initialized fusion layers.
    """
    rows = model.model.yaml["backbone"]  # fusion rows sit between the backbones and the neck, i.e. the last backbone rows
    indices = fusion_layer_indices(rows)
    for i in indices:
        conv = model.model.model[i].conv
        if conv.in_channels != 2 * conv.out_channels:  # a fusion concat joins two equally sized streams
            raise ValueError(
                f"Fusion layer {i} expects two equally sized streams, got {conv.in_channels} -> {conv.out_channels}"
            )
        with torch.no_grad():
            conv.weight.zero_()
            identity = torch.eye(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
            if fusion_init == "mean":
                conv.weight[:, : conv.out_channels, 0, 0] = identity * 0.5
                conv.weight[:, conv.out_channels :, 0, 0] = identity * 0.5
            elif fusion_init == "ir":
                conv.weight[:, conv.out_channels :, 0, 0] = identity
            else:
                conv.weight[:, : conv.out_channels, 0, 0] = identity
            if conv.bias is not None:
                conv.bias.zero_()
    if verbose:
        print(f"Initialized {len(indices)} fusion layers {indices} with '{fusion_init}' passthrough")
    return len(indices)


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

            visible_checkpoint = args.pretrained_vis_ckpt or args.pretrained_ckpt
            infrared_checkpoint = args.pretrained_ir_ckpt or args.pretrained_ckpt

            if visible_checkpoint is None:
                try:
                    visible_checkpoint = find_latest_checkpoint(project, "vis_yolo26s")
                except FileNotFoundError:
                    visible_checkpoint = find_checkpoint(args)
            if infrared_checkpoint is None:
                try:
                    infrared_checkpoint = find_latest_checkpoint(project, "ir_yolo26s")
                except FileNotFoundError:
                    infrared_checkpoint = visible_checkpoint

            if visible_checkpoint is None or infrared_checkpoint is None:
                raise FileNotFoundError(
                    "No checkpoint is available for the RGB-T streams. Pass "
                    "--pretrained-vis-ckpt and --pretrained-ir-ckpt, pass --pretrained-ckpt, "
                    "or place yolo26s.pt in one of:\n  "
                    + "\n  ".join(str(p) for p in checkpoint_candidates(args))
                )

            init_streams_from_checkpoints(
                model,
                Path(visible_checkpoint).expanduser().resolve(),
                Path(infrared_checkpoint).expanduser().resolve(),
                head_init=args.head_init,
            )
            init_fusion_layers(model, fusion_init=args.fusion_init)

            # Model.train() rebuilds a model loaded from YAML and would discard
            # the in-memory initialization above. Save and reload a checkpoint
            # so Ultralytics starts training from the initialized weights.
            init_dir = project / "initialized"
            init_dir.mkdir(parents=True, exist_ok=True)
            init_checkpoint = init_dir / f"{name}_init.pt"
            model.save(str(init_checkpoint))
            print(f"Saved initialized RGB-T checkpoint: {init_checkpoint}")
            model = YOLO(str(init_checkpoint))
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
