#!/usr/bin/env python3
"""DVTOD common-FOV preprocessing for Linux/A100.

Visible images are cropped with one fixed crop box and resized to the infrared
resolution. Infrared images and YOLO labels are copied unchanged. No per-frame
registration is performed, so residual DVTOD misalignment is preserved.

The output keeps the original structure:
  OUT/vis1/images/{train,val}
  OUT/vis1/labels/{train,val}
  OUT/ir1/images/{train,val}
  OUT/ir1/labels/{train,val}
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop DVTOD visible images to the common IR field of view."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "DVTOD_extracted" / "DVTOD dataset",
        help="Original DVTOD directory containing vis1/ and ir1/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".ljs" / "data" / "DVTOD_commonfov",
        help="Output directory with the same vis1/ir1 structure.",
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--vis-size", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--ir-size", type=int, nargs=2, default=[640, 512])
    parser.add_argument("--target-size", type=int, nargs=2, default=[640, 512])
    parser.add_argument("--vis-dfov", type=float, default=84.0)
    parser.add_argument("--ir-dfov", type=float, default=40.6)
    parser.add_argument("--vis-fov-type", choices=("dfov", "hfov", "vfov"), default="dfov")
    parser.add_argument("--ir-fov-type", choices=("dfov", "hfov", "vfov"), default="dfov")
    parser.add_argument("--vis-hfov", type=float, default=None)
    parser.add_argument("--vis-vfov", type=float, default=None)
    parser.add_argument("--ir-hfov", type=float, default=None)
    parser.add_argument("--ir-vfov", type=float, default=None)
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument(
        "--crop-box",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Use the exact fixed crop box from the earlier local experiment.",
    )
    parser.add_argument("--enforce-aspect", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument(
        "--preview-stems",
        default="",
        help="Comma- or space-separated sample stems, for example 1,500,79.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path.home() / ".ljs" / "runs" / "preprocess_preview",
    )
    return parser.parse_args()


def fov_to_focal(width: float, height: float, fov_deg: float, fov_type: str) -> float:
    if fov_type == "hfov":
        return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    if fov_type == "vfov":
        return (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return (math.hypot(width, height) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def compute_crop_box(args: argparse.Namespace) -> tuple[tuple[float, float, float, float], dict]:
    vw, vh = map(float, args.vis_size)
    iw, ih = map(float, args.ir_size)
    tw, th = args.target_size

    if args.crop_box is not None:
        x0, y0, x1, y1 = args.crop_box
        box = (x0, y0, x1, y1)
        info = {
            "source": "manual --crop-box",
            "crop_w": x1 - x0,
            "crop_h": y1 - y0,
            "crop_aspect": (x1 - x0) / (y1 - y0),
        }
    else:
        f_vis = fov_to_focal(vw, vh, args.vis_dfov, args.vis_fov_type)
        fx_vis = (vw / 2.0) / math.tan(math.radians(args.vis_hfov) / 2.0) if args.vis_hfov else f_vis
        fy_vis = (vh / 2.0) / math.tan(math.radians(args.vis_vfov) / 2.0) if args.vis_vfov else f_vis
        f_ir = fov_to_focal(iw, ih, args.ir_dfov, args.ir_fov_type)

        hfov_ir = math.radians(args.ir_hfov) if args.ir_hfov else 2.0 * math.atan((iw / 2.0) / f_ir)
        vfov_ir = math.radians(args.ir_vfov) if args.ir_vfov else 2.0 * math.atan((ih / 2.0) / f_ir)
        crop_w = 2.0 * fx_vis * math.tan(hfov_ir / 2.0)
        crop_h = 2.0 * fy_vis * math.tan(vfov_ir / 2.0)

        cx = vw / 2.0 + args.offset_x
        cy = vh / 2.0 + args.offset_y
        box = (
            max(0.0, cx - crop_w / 2.0),
            max(0.0, cy - crop_h / 2.0),
            min(vw, cx + crop_w / 2.0),
            min(vh, cy + crop_h / 2.0),
        )
        info = {
            "source": "FOV-based co-axial pinhole estimate",
            "f_vis": f_vis,
            "f_ir": f_ir,
            "hfov_ir_deg": math.degrees(hfov_ir),
            "vfov_ir_deg": math.degrees(vfov_ir),
            "crop_w": box[2] - box[0],
            "crop_h": box[3] - box[1],
            "crop_aspect": (box[2] - box[0]) / (box[3] - box[1]),
        }

    if args.enforce_aspect:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cw, ch = x1 - x0, y1 - y0
        aspect = tw / th
        if cw / ch > aspect:
            cw = ch * aspect
        else:
            ch = cw / aspect
        box = (cx - cw / 2.0, cy - ch / 2.0, cx + cw / 2.0, cy + ch / 2.0)
        info.update(
            {
                "enforce_aspect": True,
                "crop_w": cw,
                "crop_h": ch,
                "crop_aspect": cw / ch,
            }
        )

    return box, info


def first_existing_image(directory: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None



CLASS_NAMES = {0: "person", 1: "car", 2: "bicycle"}
CLASS_COLORS = {0: (255, 64, 64), 1: (60, 200, 60), 2: (60, 140, 255)}


def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for filename in candidates:
        if Path(filename).is_file():
            return ImageFont.truetype(filename, size)
    return ImageFont.load_default()


def read_labels(path: Path):
    labels = []
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        labels.append((int(float(parts[0])), *map(float, parts[1:5])))
    return labels


def draw_boxes(image: Image.Image, labels) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    label_font = get_font(18)
    for class_id, xc, yc, bw, bh in labels:
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        color = CLASS_COLORS.get(class_id, (255, 255, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text(
            (x1 + 2, max(0, y1 - 20)),
            CLASS_NAMES.get(class_id, str(class_id)),
            fill=color,
            font=label_font,
        )
    return image


def preview_pair(root: Path, split: str, stem: str, box, out_dir: Path, target) -> Path | None:
    visible_path = root / "vis1" / "images" / split / f"{stem}.jpg"
    infrared_path = root / "ir1" / "images" / split / f"{stem}.jpg"
    label_path = root / "ir1" / "labels" / split / f"{stem}.txt"
    if not visible_path.is_file() or not infrared_path.is_file():
        return None

    visible = Image.open(visible_path).convert("RGB")
    infrared = Image.open(infrared_path).convert("RGB")
    cropped = visible.crop(tuple(int(round(v)) for v in box)).resize(
        target, Image.Resampling.LANCZOS
    )
    infrared = infrared.resize(target, Image.Resampling.LANCZOS)
    labels = read_labels(label_path)

    original_small = visible.resize(
        (640, round(visible.height * 640 / visible.width)),
        Image.Resampling.LANCZOS,
    )
    original_draw = ImageDraw.Draw(original_small)
    sx = original_small.width / visible.width
    sy = original_small.height / visible.height
    original_draw.rectangle(
        (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy),
        outline=(0, 255, 255),
        width=3,
    )

    panels = [
        original_small,
        draw_boxes(cropped, labels),
        draw_boxes(infrared, labels),
        Image.blend(cropped, infrared, 0.5),
    ]
    titles = [
        "original VIS + crop",
        "cropped VIS + labels",
        "IR + labels",
        "VIS/IR blend",
    ]
    header = 30
    canvas = Image.new("RGB", (640 * 4, target[1] + header), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(18)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * 640
        canvas.paste(panel, (x, header))
        draw.text((x + 5, 6), title, fill=(255, 255, 255), font=title_font)

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{stem}_{split}_fov_preview.jpg"
    canvas.save(output, quality=94)
    return output



def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    target_size = tuple(args.target_size)

    if not root.is_dir():
        raise FileNotFoundError(root)

    box, crop_info = compute_crop_box(args)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid crop box: {box}")

    print(f"Source: {root}")
    print(f"Output: {out}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print("Crop box: %.3f %.3f %.3f %.3f" % box)
    print(f"Crop source: {crop_info['source']}")
    if args.dry_run:
        return

    if args.preview > 0:
        preview_dir = args.preview_dir.expanduser().resolve()
        requested = [
            item
            for item in args.preview_stems.replace(",", " ").split()
            if item
        ]
        selected = []
        for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
            available = {
                path.stem
                for path in (root / "vis1" / "images" / split).glob("*.jpg")
                if path.is_file()
            }
            for stem in requested or sorted(available):
                if stem in available:
                    selected.append((split, stem))
                if args.preview and len(selected) >= args.preview:
                    break
            if args.preview and len(selected) >= args.preview:
                break
        for split, stem in selected:
            output = preview_pair(root, split, stem, box, preview_dir, target_size)
            if output:
                print(f"preview: {output}")
        return

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "DVTOD",
        "operation": "fixed common-FOV visible crop + resize; IR and labels unchanged",
        "no_per_frame_registration": True,
        "source": str(root),
        "output": str(out),
        "crop_box_visible": list(box),
        "crop_info": crop_info,
        "target_size": list(target_size),
        "splits": {},
    }

    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        visible_image_dir = root / "vis1" / "images" / split
        visible_label_dir = root / "vis1" / "labels" / split
        infrared_image_dir = root / "ir1" / "images" / split
        infrared_label_dir = root / "ir1" / "labels" / split

        if not visible_image_dir.is_dir() or not infrared_image_dir.is_dir():
            raise FileNotFoundError(f"Missing split directories for {split}")

        visible_images = sorted(path for path in visible_image_dir.iterdir() if path.is_file())
        copied = 0
        missing_ir = 0
        missing_labels = 0

        for index, visible_src in enumerate(visible_images, 1):
            stem = visible_src.stem
            infrared_src = first_existing_image(infrared_image_dir, stem)
            if infrared_src is None:
                missing_ir += 1
                continue

            visible_dst = out / "vis1" / "images" / split / f"{stem}.jpg"
            infrared_dst = out / "ir1" / "images" / split / infrared_src.name
            visible_label_src = visible_label_dir / f"{stem}.txt"
            infrared_label_src = infrared_label_dir / f"{stem}.txt"
            visible_label_dst = out / "vis1" / "labels" / split / f"{stem}.txt"
            infrared_label_dst = out / "ir1" / "labels" / split / f"{stem}.txt"

            visible_dst.parent.mkdir(parents=True, exist_ok=True)
            infrared_dst.parent.mkdir(parents=True, exist_ok=True)
            visible_label_dst.parent.mkdir(parents=True, exist_ok=True)
            infrared_label_dst.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(visible_src) as image:
                cropped = image.convert("RGB").crop(box).resize(target_size, Image.Resampling.LANCZOS)
                cropped.save(visible_dst, quality=args.jpeg_quality)

            shutil.copy2(infrared_src, infrared_dst)

            if visible_label_src.is_file():
                shutil.copy2(visible_label_src, visible_label_dst)
            else:
                visible_label_dst.write_text("", encoding="utf-8")
                missing_labels += 1
            if infrared_label_src.is_file():
                shutil.copy2(infrared_label_src, infrared_label_dst)
            else:
                infrared_label_dst.write_text("", encoding="utf-8")
                missing_labels += 1

            copied += 1
            if index % 200 == 0:
                print(f"{split}: {index}/{len(visible_images)}")

        report["splits"][split] = {
            "visible_images": len(visible_images),
            "written_pairs": copied,
            "missing_infrared": missing_ir,
            "missing_label_files": missing_labels,
        }
        print(f"{split} complete: {copied} pairs")

    (out / "preprocess_meta.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Metadata: {out / 'preprocess_meta.json'}")


if __name__ == "__main__":
    main()
