# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python toolkit for preprocessing images for 3D Gaussian Splatting (GSPLAT) pipelines, consisting of two utilities:

- **`batch_equirect_to_persp.py`** — batch-converts equirectangular (360°) panoramic images into multiple perspective-projected crops.
- **`mask_generator.py`** — batch-generates segmentation masks using YOLO11-seg, compatible with COLMAP, LichtFeld Studio, RealityScan, and Nerfstudio.

## Installation & System-Wide Commands

The package is installed via pipx and exposes two commands globally:

```bash
pipx install .          # install / reinstall from local source
pipx upgrade gsplat-tools
```

| Command | Script | Entry point |
|---------|--------|-------------|
| `equirect-to-persp` | `batch_equirect_to_persp.py` | `batch_equirect_to_persp:main` |
| `mask-gen` | `mask_generator.py` | `mask_generator:main` |

## Setup & Running

**Install dependencies (dev/direct):**
```bash
pip install numpy pillow opencv-python ultralytics tqdm
```

**Basic usage:**
```bash
equirect-to-persp <input_dir> <output_dir> [options]
mask-gen <input_dir> [output_dir] [options]
```

**Common invocations — equirect-to-persp:**
```bash
# Adaptive sampling, 8 workers
equirect-to-persp input output --fov 80 --workers 8

# Latitude-aware sampling (fewer views near poles), per-image subfolders
equirect-to-persp input output --fov 80 --latitude-aware --per-image-folders

# Manual angle control
equirect-to-persp input output --yaws "0,60,120,180,240,300" --pitches "-45,0,45"

# Include explicit north/south pole views
equirect-to-persp input output --fov 80 --include-poles

# Cubemap — 6 faces at default 1920×1920
equirect-to-persp input output --cubemap

# Cubemap — custom resolution, per-image subfolders
equirect-to-persp input output --cubemap --cubemap-size 2048 --per-image-folders
```

**Common invocations — mask-gen:**
```bash
# Default: mask persons, output to sibling .mask/ folder (COLMAP convention)
mask-gen images/

# Mask persons and cars, larger model, dilate edges by 8px
mask-gen images/ masks/ --classes "person,car" --model-size x --dilate 8

# Nerfstudio/3DGS convention (white=excluded), also output RGBA images
mask-gen images/ --nerfstudio --alpha

# Preview matched files without writing
mask-gen images/ --dry-run
```

**No build step, no test suite, no linter configured.**

## Architecture

### mask_generator.py

- **`detect_device()`** — auto-selects `cuda` / `mps` / `cpu` via torch.
- **`load_yolo_model(size, device)`** — loads `yolo11{size}-seg.pt` from `~/.gsplat-tools/models/` (override with `GSPLAT_MODELS_DIR` env var); downloads on first use via Ultralytics.
- **`run_yolo(model, image_path, class_list, conf)`** — runs inference, unions per-class masks, resizes to source resolution.
- **`dilate_mask(mask, pixels)`** — optional morphological dilation with elliptical kernel.
- **`build_output_path(src, out_dir, naming)`** — `colmap` mode preserves source filename; `simple` mode uses `stem.png`.
- **`process_image(...)`** — loads image, runs YOLO, inverts mask for COLMAP convention (white=valid), optionally writes RGBA alpha image.
- **`main()`** — CLI entry point; writes `mask_manifest.json` to output dir.

Mask convention: YOLO detects objects as white (255). COLMAP/default inverts this (white=valid background, black=excluded object). `--nerfstudio` keeps YOLO convention (white=excluded).

### batch_equirect_to_persp.py

The entire implementation lives in `batch_equirect_to_persp.py`. Key layers:

### 1. Rotation & Projection (`rot_y`, `rot_x`, `equirect_to_perspective`)
- `rot_y(yaw)` / `rot_x(pitch)` produce 3×3 rotation matrices.
- `equirect_to_perspective` composes them as `R = rot_y(yaw) @ rot_x(pitch)`, maps a perspective pixel grid to 3D ray directions, converts to spherical coordinates, then bilinearly samples from the equirectangular source.

### 2. Angle Sampling Strategies (`build_pitch_angles`, `build_adaptive_angles`, `build_latitude_aware_grid`)
- **Adaptive (default):** Uniform yaw/pitch grid; step size derived from `fov * (1 - max_overlap)`.
- **Latitude-aware (`--latitude-aware`):** Scales yaw step by `1/cos(latitude)` so polar regions get fewer samples (meridian convergence compensation).
- **Manual:** User supplies `--yaws` and `--pitches` directly.
- **Pole views (`--include-poles`):** Appends explicit `pitch=±90°` entries.

### 3. Cubemap Mode (`CUBEMAP_FACES`, `process_image_cubemap`, `worker_cubemap`)
- `CUBEMAP_FACES` — list of 6 face descriptors (name, yaw, pitch) for front/back/right/left/up/down; all use FOV=90°.
- `CUBEMAP_SIZE` — default face resolution constant (1920); overridden at runtime by `--cubemap-size`.
- `process_image_cubemap` — iterates `CUBEMAP_FACES`, calls `equirect_to_perspective` for each, writes files named `{stem}_{face}.{ext}` (or just `{face}.{ext}` with `--per-image-folders`), and saves a `manifest.json`.
- `worker_cubemap` — thin multiprocessing wrapper; task tuple includes `size` so each worker uses the correct resolution.

### 4. Per-image Worker (`process_image`, `worker`)
- `process_image` loads one equirectangular image, iterates every (yaw, pitch) pair, calls `equirect_to_perspective`, and saves each crop.
- Writes a per-image `manifest.json` alongside the crops with view metadata (yaw, pitch, FOV, dimensions, source path).
- `worker` is a thin multiprocessing wrapper around `process_image`.

### 5. Orchestration (`main`)
- Parses CLI args, resolves the angle grid, distributes images across a `multiprocessing.Pool` via `imap_unordered`.
- Writes a `batch_manifest.json` in the output root summarising all images and total view count.
- Falls back to single-process loop when `--workers 1`.

## Output Structure

**equirect-to-persp (perspective mode):**
```
output_dir/
├── batch_manifest.json          # Summary of entire batch
├── image1/                      # (with --per-image-folders)
│   ├── view_000_yaw000_pit-45.png
│   ├── manifest.json
│   └── ...
└── image1_view_000_yaw000_pit-45.png   # (without --per-image-folders)
```

**equirect-to-persp (cubemap mode — `--cubemap`):**
```
output_dir/
├── batch_manifest.json
├── image1_front.jpg             # (without --per-image-folders)
├── image1_back.jpg
├── image1_right.jpg
├── image1_left.jpg
├── image1_up.jpg
└── image1_down.jpg

output_dir/image1/               # (with --per-image-folders)
├── manifest.json
├── front.jpg
├── back.jpg
├── right.jpg
├── left.jpg
├── up.jpg
└── down.jpg
```

**mask-gen:**
```
<input_dir>_mask/   (or specified output_dir, default: sibling .mask/)
├── mask_manifest.json
├── image1.jpg      # colmap naming: same as source
└── image1.png      # simple naming: stem + .png

<input_dir>_alpha/  (with --alpha)
└── image1.png      # RGBA, alpha channel = mask
```

## Key CLI Flags — equirect-to-persp

| Flag | Default | Notes |
|------|---------|-------|
| `--fov` | 80 | Degrees |
| `--width` / `--height` | 1024 / same as width | Output crop size |
| `--max-overlap` | 0.30 | Controls grid density |
| `--latitude-aware` | off | Recommended for full-sphere coverage |
| `--include-poles` | off | Adds nadir/zenith crops |
| `--per-image-folders` | off | One subfolder per source image |
| `--format` | png | `png`, `jpg`, or `jpeg` |
| `--workers` | CPU count − 1 | Set to 1 to disable multiprocessing |
| `--cubemap` | off | Produce 6 cubemap faces instead of perspective crops; ignores `--fov`, `--width`, `--height`, `--yaws`, `--pitches` |
| `--cubemap-size` | 1920 | Face resolution in pixels (square) |

## Key CLI Flags — mask-gen

| Flag | Default | Notes |
|------|---------|-------|
| `--classes` | `person` | Comma-separated COCO class names |
| `--model-size` | `l` | `n`, `s`, `m`, `l`, `x` |
| `--yolo-conf` | 0.40 | Detection confidence threshold |
| `--dilate` | 0 | Expand mask edges by N pixels |
| `--naming` | `colmap` | `colmap` (keep source name) or `simple` (stem.png) |
| `--nerfstudio` | off | Flip convention: white=excluded, black=valid |
| `--device` | auto | `cuda`, `cpu`, or `mps` |
| `--alpha` | off | Also write RGBA images to `<input>_alpha/` |
| `--dry-run` | off | Preview matched files without writing |

Supported input formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.webp`.

Model weights are cached in `~/.gsplat-tools/models/` (override: `GSPLAT_MODELS_DIR` env var).
