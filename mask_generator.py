"""
mask_generator.py — Batch image segmentation mask generator using Ultralytics YOLO11-seg.

Outputs binary masks (white=valid, black=excluded) compatible with COLMAP, LichtFeld Studio,
RealityScan, and Nerfstudio (with --nerfstudio flag).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# YOLO backend
# ---------------------------------------------------------------------------

def load_yolo_model(size: str, device: str):
    from ultralytics import YOLO

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"yolo11{size}-seg.pt"

    model = YOLO(str(model_path))
    model.to(device)
    return model


def run_yolo(model, image_path: Path, class_list: set, conf: float) -> np.ndarray:
    """Run YOLO inference; return binary mask (255=detected object, 0=background)."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    W, H = img.size

    results = model(str(image_path), conf=conf, verbose=False)
    combined = np.zeros((H, W), dtype=np.uint8)

    for r in results:
        if r.masks is None:
            continue
        for i, cls_tensor in enumerate(r.boxes.cls):
            cls_name = model.names[int(cls_tensor)]
            if cls_name not in class_list:
                continue
            mask_data = r.masks.data[i].cpu().numpy()
            # YOLO masks may be smaller than the image; resize to match
            if mask_data.shape != (H, W):
                mask_data = cv2.resize(
                    mask_data, (W, H), interpolation=cv2.INTER_NEAREST
                )
            combined = np.maximum(combined, (mask_data > 0.5).astype(np.uint8) * 255)

    return combined


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
    )
    return cv2.dilate(mask, kernel)


def build_output_path(src: Path, out_dir: Path, naming: str) -> Path:
    """
    colmap:  image.jpg  → out_dir/image.jpg  (same filename as source)
    simple:  image.jpg  → out_dir/image.png
    """
    if naming == "colmap":
        return out_dir / src.name
    return out_dir / (src.stem + ".png")


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    image_path: Path,
    out_dir: Path,
    model,
    class_list: set,
    conf: float,
    dilate: int,
    nerfstudio: bool,
    naming: str,
    dry_run: bool,
    alpha_dir: Path | None = None,
) -> dict:
    t0 = time.perf_counter()

    if dry_run:
        out_path = build_output_path(image_path, out_dir, naming)
        return {"source": str(image_path), "output": str(out_path), "dry_run": True}

    orig = Image.open(image_path).convert("RGB")
    W, H = orig.size

    mask = run_yolo(model, image_path, class_list, conf)

    detected = bool(np.any(mask > 0))

    if detected:
        if dilate > 0:
            mask = dilate_mask(mask, dilate)
        # YOLO mask: 255=object, 0=background
        # Default (COLMAP): white=valid, black=excluded → invert
        # Nerfstudio: white=excluded, black=valid → keep as-is
        if not nerfstudio:
            mask = 255 - mask
    else:
        fill = 0 if nerfstudio else 255
        mask = np.full((H, W), fill, dtype=np.uint8)

    out_path = build_output_path(image_path, out_dir, naming)
    Image.fromarray(mask, mode="L").save(out_path, format="PNG")

    if alpha_dir is not None:
        # Alpha = mask value; nerfstudio convention inverts meaning so re-invert for alpha
        alpha_data = (255 - mask) if nerfstudio else mask
        rgba = orig.copy()
        rgba.putalpha(Image.fromarray(alpha_data, mode="L"))
        alpha_path = alpha_dir / (image_path.stem + ".png")
        rgba.save(alpha_path, format="PNG")

    elapsed = time.perf_counter() - t0
    return {
        "source": str(image_path),
        "output": str(out_path),
        "detected": bool(detected),
        "elapsed_s": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch segmentation mask generator (YOLO11-seg backend)."
    )
    parser.add_argument("input_dir", help="Folder of source images")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Folder to write mask images (default: <input_dir_name>_mask sibling folder)",
    )
    parser.add_argument(
        "--classes",
        default="person",
        help='Comma-separated COCO class names to mask (default: "person")',
    )
    parser.add_argument(
        "--model-size",
        default="l",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO model size (default: l)",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.40,
        help="YOLO confidence threshold (default: 0.40)",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=0,
        help="Dilate masks by N pixels (default: 0)",
    )
    parser.add_argument(
        "--naming",
        default="colmap",
        choices=["colmap", "simple"],
        help="Output file naming: colmap=same filename as source, simple=image.png (default: colmap)",
    )
    parser.add_argument(
        "--nerfstudio",
        action="store_true",
        help="Flip convention: white=excluded, black=valid (Nerfstudio/3DGS)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | mps (default: auto-detect)",
    )
    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Also output original images with mask as alpha channel into <input_dir>_alpha folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched files and output paths; do not write masks",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input_dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        out_dir = input_dir.parent / ".mask"
    else:
        out_dir = Path(args.output_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_dir: Path | None = None
    if args.alpha:
        alpha_dir = input_dir.parent / f"{input_dir.name}_alpha"
        alpha_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not images:
        print(f"No supported images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    class_list = {c.strip() for c in args.classes.split(",") if c.strip()}
    device = args.device or detect_device()

    print(f"Input:    {input_dir}")
    print(f"Output:   {out_dir}")
    print(f"Images:   {len(images)}")
    print(f"Classes:  {sorted(class_list)}")
    print(f"Model:    yolo11{args.model_size}-seg  |  conf={args.yolo_conf}")
    print(f"Device:   {device}")
    print(f"Naming:   {args.naming}{'  (nerfstudio inversion)' if args.nerfstudio else ''}")
    if alpha_dir:
        print(f"Alpha:    {alpha_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be written")
    print()

    if args.dry_run:
        for img_path in images:
            out_path = build_output_path(img_path, out_dir, args.naming)
            print(f"  {img_path.name}  →  {out_path.name}")
        print(f"\n{len(images)} files matched.")
        return

    print("Loading YOLO model...", flush=True)
    model = load_yolo_model(args.model_size, device)
    print("Processing images...", flush=True)

    # Verify requested classes exist in model vocabulary
    valid_names = set(model.names.values())
    unknown = class_list - valid_names
    if unknown:
        print(
            f"WARNING: class(es) not in COCO vocabulary and will never match: {sorted(unknown)}",
            file=sys.stderr,
        )

    try:
        from tqdm import tqdm
        iterator = tqdm(images, unit="img")
    except ImportError:
        iterator = images

    results = []
    detected_count = 0
    t_batch = time.perf_counter()

    for img_path in iterator:
        rec = process_image(
            img_path,
            out_dir,
            model,
            class_list,
            args.yolo_conf,
            args.dilate,
            args.nerfstudio,
            args.naming,
            args.dry_run,
            alpha_dir=alpha_dir,
        )
        results.append(rec)
        if rec.get("detected"):
            detected_count += 1

    elapsed_total = time.perf_counter() - t_batch

    # Write manifest
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "alpha_dir": str(alpha_dir) if alpha_dir else None,
        "classes": sorted(class_list),
        "model": f"yolo11{args.model_size}-seg",
        "yolo_conf": args.yolo_conf,
        "naming": args.naming,
        "nerfstudio": args.nerfstudio,
        "dilate": args.dilate,
        "device": device,
        "total_images": len(images),
        "images_with_detections": detected_count,
        "elapsed_s": round(elapsed_total, 2),
        "images_per_sec": round(len(images) / elapsed_total, 2) if elapsed_total > 0 else 0,
        "results": results,
    }
    manifest_path = out_dir / "mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"\nDone: {len(images)} images in {elapsed_total:.1f}s "
        f"({manifest['images_per_sec']} img/s) — "
        f"{detected_count}/{len(images)} had detections"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
