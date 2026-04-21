# split360 — 360° Panorama Tools for GSPLAT Pipelines

Two utilities for preprocessing equirectangular panoramas into perspective crops and segmentation masks, ready for use in COLMAP, LichtFeld Studio, RealityScan, Nerfstudio, and 3D Gaussian Splatting workflows.

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pillow ultralytics opencv-python tqdm
```

---

## batch_equirect_to_persp.py

Converts equirectangular (360°) images into multiple perspective-projected crops.

**Basic usage:**
```bash
python batch_equirect_to_persp.py <input_dir> <output_dir> [options]
```

**Defaults:**
- Output format: JPG (quality 95)
- Adaptive sampling from FOV with max 30% overlap
- Flat output folder (all views in one directory)

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--fov` | 80 | Horizontal field of view in degrees |
| `--width` / `--height` | 1024 / same as width | Output crop size |
| `--max-overlap` | 0.30 | Max overlap between adjacent views |
| `--latitude-aware` | off | Fewer yaw views near poles (recommended) |
| `--include-poles` | off | Add nadir/zenith crops |
| `--per-image-folders` | off | One subfolder per source image |
| `--format` | jpg | `jpg`, `jpeg`, or `png` |
| `--workers` | CPU count − 1 | Set to 1 to disable multiprocessing |
| `--yaws` | auto | Comma-separated yaw angles (manual override) |
| `--pitches` | auto | Comma-separated pitch angles (manual override) |
| `--cubemap` | off | Convert to 6 cubemap faces (FOV=90°); ignores `--fov`, `--width`, `--height`, `--yaws`, `--pitches` |
| `--cubemap-size` | 1920 | Output resolution in pixels for each cubemap face |

**Examples:**

```bash
# Adaptive sampling, 8 workers
python batch_equirect_to_persp.py input output --fov 80 --workers 8

# Latitude-aware spacing (fewer views near poles)
python batch_equirect_to_persp.py input output --fov 80 --latitude-aware

# Per-image subfolders with pole views
python batch_equirect_to_persp.py input output --fov 80 --latitude-aware --include-poles --per-image-folders

# Output PNG instead of JPG
python batch_equirect_to_persp.py input output --fov 80 --format png

# Manual angle control
python batch_equirect_to_persp.py input output --yaws "0,60,120,180,240,300" --pitches "-45,0,45"

# Cubemap — 6 faces at default 1920×1920
python batch_equirect_to_persp.py input output --cubemap

# Cubemap — custom resolution, per-image subfolders
python batch_equirect_to_persp.py input output --cubemap --cubemap-size 2048 --per-image-folders

# Disable multiprocessing
python batch_equirect_to_persp.py input output --workers 1
```

**Output (perspective mode):**
```
output_dir/
├── batch_manifest.json
├── image1_view_000_yaw000_pit-45.jpg
└── ...
```
With `--per-image-folders`:
```
output_dir/
├── batch_manifest.json
└── image1/
    ├── manifest.json
    ├── view_000_yaw000_pit-45.jpg
    └── ...
```

**Output (cubemap mode):**
```
output_dir/
├── batch_manifest.json
├── image1_front.jpg
├── image1_back.jpg
├── image1_right.jpg
├── image1_left.jpg
├── image1_up.jpg
└── image1_down.jpg
```
With `--per-image-folders`:
```
output_dir/
├── batch_manifest.json
└── image1/
    ├── manifest.json
    ├── front.jpg
    ├── back.jpg
    ├── right.jpg
    ├── left.jpg
    ├── up.jpg
    └── down.jpg
```

---

## mask_generator.py

Batch segmentation mask generator using Ultralytics YOLO11-seg. Detects objects by class name and writes binary masks for use in photogrammetry reconstruction.

**Mask convention (default — COLMAP / LichtFeld Studio / RealityScan):**
- White (255) = valid region, used in reconstruction
- Black (0) = excluded region (the masked object)

Use `--nerfstudio` to flip: white=excluded, black=valid.

**Basic usage:**
```bash
python mask_generator.py <input_dir> [output_dir] [options]
```

Output folder defaults to `<input_dir_name>_mask` alongside the input folder.

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--classes` | `person` | Comma-separated COCO class names to mask |
| `--model-size` | `l` | YOLO model size: `n`, `s`, `m`, `l`, `x` |
| `--yolo-conf` | 0.40 | Detection confidence threshold |
| `--dilate` | 0 | Dilate masks by N pixels (safety margin) |
| `--naming` | `colmap` | `colmap` = same filename as source, `simple` = `image.png` |
| `--nerfstudio` | off | Flip mask convention for Nerfstudio/3DGS |
| `--device` | auto | `cuda`, `cpu`, or `mps` |
| `--dry-run` | off | Preview output paths without writing files |

**Examples:**

```bash
# Mask people (default)
python mask_generator.py ./images

# Mask people and cars, with dilation
python mask_generator.py ./images --classes "person, car" --dilate 10

# Custom output folder, simple naming (RealityScan)
python mask_generator.py ./images ./masks --naming simple

# Nerfstudio convention
python mask_generator.py ./images --nerfstudio --naming simple

# Preview without writing
python mask_generator.py ./images --dry-run
```

**Output:**
```
images_mask/
├── mask_manifest.json
├── frame_0001.jpg    (--naming colmap, default — same filename as source, PNG data)
└── ...
```

Masks are always written as PNG data regardless of the output filename extension.

**Supported COCO classes:** see `coco_classes.txt` for the full list of 80 classes.

**File naming by target application:**

| Application | Naming flag | Convention |
|-------------|-------------|------------|
| LichtFeld Studio | `--naming colmap` (default) | same filename as source |
| RealityScan | `--naming colmap` (default) | same filename as source |
| COLMAP | `--naming colmap` (default) | same filename as source |
| Nerfstudio | `--naming simple --nerfstudio` | `image.png` |

**RealityScan folder naming convention:**

RealityScan requires specific folder names to recognize images and masks:

| Folder | Purpose |
|--------|---------|
| `.geometry` | Images to be aligned (source perspective crops) |
| `.mask` | Corresponding masks (same filenames as images) |

Example structure for RealityScan:
```
project/
├── .geometry/
│   ├── frame_0001.jpg
│   └── ...
└── .mask/
    ├── frame_0001.jpg    (PNG data, same filename as image)
    └── ...
```

**Model weights** are cached in `models/` next to the script and reused on subsequent runs.

---

## Supported Input Formats

`.jpg` `.jpeg` `.png` `.tif` `.tiff` `.bmp` `.webp`
