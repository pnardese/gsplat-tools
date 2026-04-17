# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-file Python utility that batch-converts equirectangular (360°) panoramic images into multiple perspective-projected crops. Commonly used as a preprocessing step for 3D Gaussian Splatting (GSPLAT) pipelines.

## Setup & Running

**Install dependencies:**
```bash
pip install numpy pillow
```

**Basic usage:**
```bash
python batch_equirect_to_persp.py <input_dir> <output_dir> [options]
```

**Common invocations:**
```bash
# Adaptive sampling, 8 workers
python batch_equirect_to_persp.py input output --fov 80 --workers 8

# Latitude-aware sampling (fewer views near poles), per-image subfolders
python batch_equirect_to_persp.py input output --fov 80 --latitude-aware --per-image-folders

# Manual angle control
python batch_equirect_to_persp.py input output --yaws "0,60,120,180,240,300" --pitches "-45,0,45"

# Include explicit north/south pole views
python batch_equirect_to_persp.py input output --fov 80 --include-poles
```

**No build step, no test suite, no linter configured.**

## Architecture

The entire implementation lives in `batch_equirect_to_persp.py`. Key layers:

### 1. Rotation & Projection (`rot_y`, `rot_x`, `equirect_to_perspective`)
- `rot_y(yaw)` / `rot_x(pitch)` produce 3×3 rotation matrices.
- `equirect_to_perspective` composes them as `R = rot_y(yaw) @ rot_x(pitch)`, maps a perspective pixel grid to 3D ray directions, converts to spherical coordinates, then bilinearly samples from the equirectangular source.

### 2. Angle Sampling Strategies (`build_pitch_angles`, `build_adaptive_angles`, `build_latitude_aware_grid`)
- **Adaptive (default):** Uniform yaw/pitch grid; step size derived from `fov * (1 - max_overlap)`.
- **Latitude-aware (`--latitude-aware`):** Scales yaw step by `1/cos(latitude)` so polar regions get fewer samples (meridian convergence compensation).
- **Manual:** User supplies `--yaws` and `--pitches` directly.
- **Pole views (`--include-poles`):** Appends explicit `pitch=±90°` entries.

### 3. Per-image Worker (`process_image`, `worker`)
- `process_image` loads one equirectangular image, iterates every (yaw, pitch) pair, calls `equirect_to_perspective`, and saves each crop.
- Writes a per-image `manifest.json` alongside the crops with view metadata (yaw, pitch, FOV, dimensions, source path).
- `worker` is a thin multiprocessing wrapper around `process_image`.

### 4. Orchestration (`main`)
- Parses CLI args, resolves the angle grid, distributes images across a `multiprocessing.Pool` via `imap_unordered`.
- Writes a `batch_manifest.json` in the output root summarising all images and total view count.
- Falls back to single-process loop when `--workers 1`.

## Output Structure

```
output_dir/
├── batch_manifest.json          # Summary of entire batch
├── image1/                      # (with --per-image-folders)
│   ├── view_000_yaw000_pit-45.png
│   ├── manifest.json
│   └── ...
└── image1_view_000_yaw000_pit-45.png   # (without --per-image-folders)
```

## Key CLI Flags

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

Supported input formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.webp`.
