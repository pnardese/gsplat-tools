#!/usr/bin/env python
import os
import math
import json
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image


def rot_y(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def rot_x(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=np.float64)


def equirect_to_perspective(img, fov_deg=80, yaw_deg=0, pitch_deg=0, out_w=1024, out_h=None):
    if out_h is None:
        out_h = out_w

    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]

    H, W = arr.shape[:2]
    C = arr.shape[2]

    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    xs = (np.arange(out_w) + 0.5) / out_w * 2.0 - 1.0
    ys = (np.arange(out_h) + 0.5) / out_h * 2.0 - 1.0
    xv, yv = np.meshgrid(xs, ys)

    aspect = out_w / out_h
    tan_half = math.tan(fov / 2.0)
    x = xv * tan_half * aspect
    y = -yv * tan_half
    z = np.ones_like(x)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    R = rot_y(yaw) @ rot_x(pitch)
    dirs = dirs @ R.T

    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))

    u = (lon / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H

    u0 = np.floor(u).astype(np.int32) % W
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H - 1)
    u1 = (u0 + 1) % W
    v1 = np.clip(v0 + 1, 0, H - 1)

    du = (u - np.floor(u))[..., None]
    dv = (v - np.floor(v))[..., None]

    c00 = arr[v0, u0]
    c10 = arr[v0, u1]
    c01 = arr[v1, u0]
    c11 = arr[v1, u1]

    out = (c00 * (1 - du) * (1 - dv) +
           c10 * du * (1 - dv) +
           c01 * (1 - du) * dv +
           c11 * du * dv)

    out = np.clip(out, 0, 255).astype(np.uint8)
    if C == 1:
        out = out[..., 0]
    return Image.fromarray(out)


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def build_pitch_angles(fov_deg, max_overlap=0.30, include_poles=False):
    if not (0 <= max_overlap < 1):
        raise ValueError('max_overlap must be in [0, 1)')
    step = max(1.0, fov_deg * (1.0 - max_overlap))
    pitch_count = max(1, math.ceil(180.0 / step))
    if include_poles:
        if pitch_count == 1:
            pitches = [0]
        else:
            pitch_step = 180.0 / (pitch_count - 1)
            pitches = [int(round(-90.0 + i * pitch_step)) for i in range(pitch_count)]
    else:
        pitch_step = 180.0 / pitch_count
        pitches = [int(round(-90.0 + pitch_step * (i + 0.5))) for i in range(pitch_count)]
    return sorted(set(max(-90, min(90, p)) for p in pitches))


def build_adaptive_angles(fov_deg, max_overlap=0.30, include_poles=False):
    step = max(1.0, fov_deg * (1.0 - max_overlap))
    yaw_count = max(1, math.ceil(360.0 / step))
    yaw_step = 360.0 / yaw_count
    yaws = sorted(set(int(round(i * yaw_step)) % 360 for i in range(yaw_count)))
    pitches = build_pitch_angles(fov_deg, max_overlap=max_overlap, include_poles=include_poles)
    return yaws, pitches


def build_latitude_aware_grid(fov_deg, max_overlap=0.30, include_poles=False):
    pitches = build_pitch_angles(fov_deg, max_overlap=max_overlap, include_poles=include_poles)
    step = max(1.0, fov_deg * (1.0 - max_overlap))
    grid = []
    for pitch in pitches:
        cos_lat = max(0.1736481777, math.cos(math.radians(abs(pitch))))
        effective_step = min(360.0, step / cos_lat)
        yaw_count = max(1, math.ceil(360.0 / effective_step))
        yaw_step = 360.0 / yaw_count
        yaws = sorted(set(int(round(i * yaw_step)) % 360 for i in range(yaw_count)))
        grid.append({'pitch': pitch, 'yaws': yaws})
    return grid


def process_image(input_path, output_root, fov_deg, out_w, out_h, yaws, pitches, fmt, latitude_aware=False, max_overlap=0.30, include_poles=False, per_image_folders=False):
    img = Image.open(input_path).convert('RGB')
    stem = input_path.stem
    out_dir = output_root / stem if per_image_folders else output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    idx = 0
    if latitude_aware:
        angle_sets = build_latitude_aware_grid(fov_deg, max_overlap=max_overlap, include_poles=include_poles)
    else:
        angle_sets = [{'pitch': pitch, 'yaws': yaws} for pitch in pitches]

    for band in angle_sets:
        pitch = band['pitch']
        for yaw in band['yaws']:
            view = equirect_to_perspective(
                img,
                fov_deg=fov_deg,
                yaw_deg=yaw,
                pitch_deg=pitch,
                out_w=out_w,
                out_h=out_h,
            )
            suffix = 'jpg' if fmt.lower() in ('jpg', 'jpeg') else 'png'
            prefix = f'{stem}_' if not per_image_folders else ''
            name = (prefix + f'view_{idx:03d}_yaw{yaw:+04d}_pit{pitch:+03d}.{suffix}').replace('+', 'p').replace('-', 'm')
            out_path = out_dir / name
            save_kwargs = {'quality': 95} if suffix == 'jpg' else {}
            view.save(out_path, **save_kwargs)
            manifest.append({
                'file': str(out_path),
                'source': str(input_path),
                'yaw_deg': yaw,
                'pitch_deg': pitch,
                'fov_deg': fov_deg,
                'width': out_w,
                'height': out_h,
            })
            idx += 1

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return len(manifest)


def worker(task):
    file, output_dir, fov, width, out_h, yaws, pitches, fmt, latitude_aware, max_overlap, include_poles, per_image_folders = task
    count = process_image(file, output_dir, fov, width, out_h, yaws, pitches, fmt, latitude_aware=latitude_aware, max_overlap=max_overlap, include_poles=include_poles, per_image_folders=per_image_folders)
    return {'source': str(file), 'views_written': count}


def main():
    parser = argparse.ArgumentParser(description='Batch convert equirectangular images into perspective crops.')
    parser.add_argument('input_dir', help='Folder containing input equirectangular images')
    parser.add_argument('output_dir', help='Folder to write perspective views into')
    parser.add_argument('--fov', type=float, default=80.0, help='Horizontal field of view in degrees (default: 80)')
    parser.add_argument('--width', type=int, default=1024, help='Output image width (default: 1024)')
    parser.add_argument('--height', type=int, default=None, help='Output image height (default: same as width)')
    parser.add_argument('--yaws', type=str, default=None, help='Comma-separated yaw angles in degrees; if omitted, compute minimal set from FOV and overlap')
    parser.add_argument('--pitches', type=str, default=None, help='Comma-separated pitch angles in degrees; if omitted, compute minimal set from FOV and overlap')
    parser.add_argument('--max-overlap', type=float, default=0.30, help='Maximum overlap fraction between adjacent views when auto-generating angles (default: 0.30)')
    parser.add_argument('--include-poles', action='store_true', help='Include views centered on the poles when auto-generating pitch angles')
    parser.add_argument('--latitude-aware', action='store_true', help='Use fewer yaw views toward the poles based on latitude')
    parser.add_argument('--per-image-folders', action='store_true', help='Write each source image into its own folder under the output directory')
    parser.add_argument('--format', type=str, default='jpg', choices=['png', 'jpg', 'jpeg'], help='Output format (default: jpg)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) - 1), help='Number of parallel worker processes (default: CPU count minus one)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f'Input folder does not exist or is not a directory: {input_dir}')

    out_h = args.height if args.height is not None else args.width
    if args.yaws and args.pitches:
        yaws = parse_int_list(args.yaws)
        pitches = parse_int_list(args.pitches)
    elif args.yaws or args.pitches:
        raise SystemExit('Provide both --yaws and --pitches, or neither to use adaptive sampling.')
    else:
        yaws, pitches = build_adaptive_angles(args.fov, max_overlap=args.max_overlap, include_poles=args.include_poles)

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not files:
        raise SystemExit(f'No supported image files found in {input_dir}')

    tasks = [(file, output_dir, args.fov, args.width, out_h, yaws, pitches, args.format, args.latitude_aware, args.max_overlap, args.include_poles, args.per_image_folders) for file in files]
    summary = []
    total_views = 0

    if args.workers == 1:
        for task in tasks:
            result = worker(task)
            summary.append(result)
            total_views += result['views_written']
            print(f"Processed {Path(result['source']).name}: {result['views_written']} views")
    else:
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(worker, tasks):
                summary.append(result)
                total_views += result['views_written']
                print(f"Processed {Path(result['source']).name}: {result['views_written']} views")

    summary.sort(key=lambda x: x['source'])

    with open(output_dir / 'batch_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'fov_deg': args.fov,
            'width': args.width,
            'height': out_h,
            'yaws': yaws,
            'pitches': pitches,
            'format': args.format,
            'max_overlap': args.max_overlap,
            'include_poles': args.include_poles,
            'latitude_aware': args.latitude_aware,
            'per_image_folders': args.per_image_folders,
            'files': summary,
            'total_input_images': len(files),
            'total_views_written': total_views,
        }, f, indent=2)

    print(f'Done. {len(files)} source images -> {total_views} perspective views')


if __name__ == '__main__':
    main()
