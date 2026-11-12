#!/usr/bin/env python3
"""Train and test YOLO26s on visible, infrared, and RGB-T (dual-stream) images.

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
from copy import deepcopy
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
    parser = argparse.ArgumentParser(description="Train YOLO26s visible, infrared, or RGB-T models.")
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


def build_rgbt_model(model_name: str, names: dict | None, config_dir: Path) -> YOLO:
    """Build a two-stream model whose detection head matches the dataset class count."""
    model = YOLO(model_name)
    if not names or model.model.yaml["nc"] == len(names):
        return model

    from ultralytics.utils import YAML

    config = deepcopy(model.model.yaml)
    config["nc"] = len(names)
    config["names"] = dict(names)
    config.pop("yaml_file", None)
    scale = config.get("scale") or "s"
    config_path = config_dir / f"yolo26{scale}-rgbt-nc{len(names)}.yaml"
    YAML.save(config_path, config)
    return YOLO(str(config_path))


def detect_cls_row_map(source_names: dict | None, target_names: dict | None) -> list[int]:
    """Map each target class index onto the source class index sharing its name, or -1 when there is no match."""
    if not isinstance(source_names, dict) or not isinstance(target_names, dict):
        return []
    lookup = {str(name).strip().lower(): key for key, name in source_names.items()}
    return [lookup.get(str(target_names.get(i)).strip().lower(), -1) for i in range(len(target_names))]


def detect_cls_keys(modules, nc: int) -> set[str]:
    """Return the state-dict keys of the detection head's classification convs."""
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


def _source_layer_index_and_suffix(key: str) -> tuple[int, str]:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model":
        raise ValueError(f"Unexpected state-dict key: {key}")
    return int(parts[1]), parts[2]


def _merge_tensor(current: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(current):
        return (current + incoming) / 2
    return torch.maximum(current, incoming)


def _aligned_tensor(
    value: torch.Tensor,
    target_value: torch.Tensor | None,
    target_key: str,
    cls_keys: set[str],
    row_map: list[int],
    rows: list[int],
) -> torch.Tensor | None:
    """Return a source tensor aligned to the target, remapping classification rows by class name when needed."""
    if target_value is None:
        return None
    if target_value.shape == value.shape:
        return value
    if rows and target_key in cls_keys and target_value.shape[1:] == value.shape[1:]:
        tensor = target_value.clone()
        tensor[rows] = value[torch.tensor([row_map[index] for index in rows])]
        return tensor
    return None


def init_streams_from_checkpoints(
    model: YOLO,
    visible_checkpoint: Path,
    infrared_checkpoint: Path,
    head_init: str = "visible",
    names: dict | None = None,
    verbose: bool = True,
) -> int:
    """Seed the two P1-P3 stems and the shared P4/P5 backbone plus head from modality-specific checkpoints.

    Only source backbone layers before P3 are duplicated into the visible and infrared branches. The remaining source
    layers form the shared P4/P5 backbone and are copied once, followed by one shared detection head.
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

    visible_start, infrared_start, early_layers, shared_start = branch_layout(model.model.yaml["backbone"])
    head_offset = len(model.model.yaml["backbone"]) - visible_backbone
    target_state = model.model.state_dict()
    cls_keys = detect_cls_keys(model.model.model, model.model.yaml["nc"])
    updates: dict[str, torch.Tensor] = {}
    counts = {"visible": 0, "infrared": 0, "shared": 0, "head": 0}
    missed = {"visible": 0, "infrared": 0, "shared": 0, "head": 0}

    def copy_early(source: torch.nn.Module, target_start: int, label: str) -> None:
        for key, value in source.state_dict().items():
            index, suffix = _source_layer_index_and_suffix(key)
            if index >= early_layers:
                continue
            target_key = f"model.{target_start + index}.{suffix}"
            target_value = target_state.get(target_key)
            if target_value is not None and target_value.shape == value.shape:
                updates[target_key] = value
                counts[label] += 1
            else:
                missed[label] += 1

    copy_early(visible_source, visible_start, "visible")
    copy_early(infrared_source, infrared_start, "infrared")

    if head_init == "mean":
        shared_sources = (visible_source, infrared_source)
    elif head_init == "ir":
        shared_sources = (infrared_source,)
    else:
        shared_sources = (visible_source,)

    remapped = 0
    for source in shared_sources:
        row_map = detect_cls_row_map(getattr(source, "names", None), names)
        rows = [index for index, source_index in enumerate(row_map) if source_index >= 0]
        for key, value in source.state_dict().items():
            index, suffix = _source_layer_index_and_suffix(key)
            if index < early_layers:
                continue
            label = "shared" if index < visible_backbone else "head"
            target_index = shared_start + index - early_layers if label == "shared" else index + head_offset
            target_key = f"model.{target_index}.{suffix}"
            target_value = target_state.get(target_key)
            tensor = _aligned_tensor(value, target_value, target_key, cls_keys, row_map, rows)
            if tensor is None:
                missed[label] += 1
                continue
            if target_key in updates:
                updates[target_key] = _merge_tensor(updates[target_key], tensor)
            else:
                updates[target_key] = tensor
            counts[label] += 1
            remapped += int(target_value.shape != value.shape)

    model.model.load_state_dict(updates, strict=False)
    if missed["visible"] or missed["infrared"] or missed["shared"]:
        raise RuntimeError(
            "A checkpoint does not match the two-stream backbone: "
            f"{missed['visible']} visible, {missed['infrared']} infrared, and {missed['shared']} shared tensors "
            "would keep their random init."
        )
    if verbose:
        if remapped:
            print(f"Remapped classification rows from pretrained weights for {remapped} head tensors")
        print(
            f"Seeded {counts['visible']} visible and {counts['infrared']} infrared P1-P3 tensors, "
            f"{counts['shared']} shared P4/P5 tensors, and {counts['head']} head tensors "
            f"({missed['head']} incompatible head tensors keep their default init)."
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
            names = dataset_class_names(data_yaml)
            init_dir = project / "initialized"
            init_dir.mkdir(parents=True, exist_ok=True)
            model = build_rgbt_model(resolved_model, names, init_dir)

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
                names=names,
            )
            if names:
                model.model.names = names

            # Model.train() rebuilds a model loaded from YAML and would discard
            # the in-memory initialization above. Save and reload a checkpoint
            # so Ultralytics starts training from the initialized weights.
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
