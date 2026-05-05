#!/usr/bin/env python3
"""
tools/visualize_cst.py — Visualize CST-AntiUAV bounding box annotations
========================================================================
Usage:
    # Visualize 10 random frames from test split
    python tools/visualize_cst.py --split test --n 10

    # Visualize specific scene
    python tools/visualize_cst.py --split train --scene building_1 --n 20

    # Visualize specific category
    python tools/visualize_cst.py --split val --category water --n 15

    # Visualize with stride
    python tools/visualize_cst.py --split test --stride 1 --n 10

    # Save to custom output directory
    python tools/visualize_cst.py --split test --output debug_vis/
"""

import argparse
import sys
import random
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CST-AntiUAV annotations")
    parser.add_argument("--root",     default="/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/CST-AntiUAV/CST-AntiUAV")
    parser.add_argument("--split",    default="test",   choices=["train", "val", "test"])
    parser.add_argument("--scene",    default=None,     help="Specific scene name e.g. building_1")
    parser.add_argument("--category", default=None,     help="Specific category e.g. water, jungle")
    parser.add_argument("--n",        type=int, default=10, help="Number of frames to visualize")
    parser.add_argument("--stride",   type=int, default=1,  help="Frame stride")
    parser.add_argument("--output",   default="outputs/cst_vis", help="Output directory")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--sequential", action="store_true",
                        help="Show sequential frames instead of random")
    parser.add_argument("--img_size", type=int, default=640)
    return parser.parse_args()


def draw_box_on_image(img_bgr, box_x1y1x2y2, color, label="", thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box_x1y1x2y2]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                    font, scale, (255, 255, 255), 1)
    return img_bgr


def visualize_sample(sample, img_size, output_dir: Path, idx: int):
    """
    Load image, draw GT box, save.
    Returns dict with diagnostic info.
    """
    img_path = sample["img_path"]
    ann      = sample["annotation"]
    scene    = sample["scene"]
    category = sample["category"]
    frame_id = sample["frame_id"]

    # Load original image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [ERROR] Cannot read: {img_path}")
        return None

    orig_h, orig_w = img.shape[:2]

    # Convert to BGR for visualization
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── Draw GT box on ORIGINAL image (before any scaling)
    info = {
        "img_path":  str(img_path),
        "scene":     scene,
        "category":  category,
        "frame_id":  frame_id,
        "orig_size": (orig_w, orig_h),
        "ann_raw":   ann,
        "has_box":   ann is not None,
    }

    img_orig = img_bgr.copy()
    if ann is not None:
        x, y, w, h = ann
        x1_orig = int(max(0, x))
        y1_orig = int(max(0, y))
        x2_orig = int(min(orig_w, x + w))
        y2_orig = int(min(orig_h, y + h))

        box_area = w * h
        info["box_xywh"]   = (round(x,2), round(y,2), round(w,2), round(h,2))
        info["box_area"]   = round(box_area, 1)
        info["box_in_frame"] = (x1_orig < orig_w and y1_orig < orig_h and
                                 x2_orig > 0 and y2_orig > 0)

        label_orig = f"w={w:.1f} h={h:.1f} a={box_area:.0f}px²"
        draw_box_on_image(img_orig, [x1_orig, y1_orig, x2_orig, y2_orig],
                          (0, 255, 0), label_orig)

        # Mark center point
        cx, cy = int(x + w/2), int(y + h/2)
        cv2.circle(img_orig, (cx, cy), 3, (0, 255, 0), -1)
    else:
        info["box_xywh"]    = None
        info["box_area"]    = 0
        info["box_in_frame"] = False
        cv2.putText(img_orig, "NO ANNOTATION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ── Also show scaled version (what model sees)
    scale_x = img_size / orig_w
    scale_y = img_size / orig_h

    # Apply same upsampling as dataset loader
    img_up   = cv2.resize(img, (orig_w * 2, orig_h * 2), interpolation=cv2.INTER_CUBIC)
    img_scaled = cv2.resize(img_up, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_scaled_bgr = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)

    if ann is not None:
        x1_s = max(0.0, x * scale_x)
        y1_s = max(0.0, y * scale_y)
        x2_s = min(float(img_size), (x + w) * scale_x)
        y2_s = min(float(img_size), (y + h) * scale_y)

        info["box_scaled_x1y1x2y2"] = (round(x1_s,1), round(y1_s,1),
                                        round(x2_s,1), round(y2_s,1))
        info["box_scaled_wh"]       = (round(x2_s-x1_s,1), round(y2_s-y1_s,1))
        info["box_valid_after_scale"] = (x2_s > x1_s + 1 and y2_s > y1_s + 1)

        label_s = f"scaled: {x2_s-x1_s:.1f}x{y2_s-y1_s:.1f}px"
        draw_box_on_image(img_scaled_bgr, [x1_s, y1_s, x2_s, y2_s],
                          (0, 255, 0), label_s)
        cx_s, cy_s = int((x1_s+x2_s)/2), int((y1_s+y2_s)/2)
        cv2.circle(img_scaled_bgr, (cx_s, cy_s), 3, (0, 255, 0), -1)

        # Draw a zoomed inset of the box region
        pad = 30
        x1z = max(0, int(x1_s) - pad)
        y1z = max(0, int(y1_s) - pad)
        x2z = min(img_size, int(x2_s) + pad)
        y2z = min(img_size, int(y2_s) + pad)
        if x2z > x1z and y2z > y1z:
            zoomed = img_scaled_bgr[y1z:y2z, x1z:x2z]
            zoomed = cv2.resize(zoomed, (200, 200), interpolation=cv2.INTER_NEAREST)
            # Paste into top-right corner
            img_scaled_bgr[0:200, img_size-200:img_size] = zoomed
            cv2.rectangle(img_scaled_bgr, (img_size-200, 0),
                          (img_size, 200), (255, 255, 0), 2)
            cv2.putText(img_scaled_bgr, "ZOOM", (img_size-195, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ── Add info text overlay
    def add_text(img, lines, start_y=20):
        for i, line in enumerate(lines):
            cv2.putText(img, line, (5, start_y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    lines_orig = [
        f"Scene: {scene}",
        f"Frame: {frame_id}  |  Orig: {orig_w}x{orig_h}",
        f"Ann: {info.get('box_xywh', 'None')}",
        f"Area: {info.get('box_area', 0):.1f}px²",
        f"In frame: {info.get('box_in_frame', False)}",
    ]
    add_text(img_orig, lines_orig)

    lines_scaled = [
        f"Scaled to {img_size}x{img_size}",
        f"Box: {info.get('box_scaled_wh', 'None')}px",
        f"Valid: {info.get('box_valid_after_scale', False)}",
        f"scale_x={scale_x:.3f} scale_y={scale_y:.3f}",
    ]
    add_text(img_scaled_bgr, lines_scaled)

    # ── Combine side by side
    # Resize orig to img_size for consistent layout
    img_orig_resized = cv2.resize(img_orig, (img_size, img_size))
    combined = np.hstack([img_orig_resized, img_scaled_bgr])

    # Save
    fname = f"{idx:04d}_{category}_{scene}_f{frame_id}.jpg"
    save_path = output_dir / fname
    cv2.imwrite(str(save_path), combined)

    return info


def main():
    args = parse_args()

    # Import dataset
    from data.cst_dataset import CSTAntiUAVDataset

    print(f"\nLoading CST-AntiUAV [{args.split}] from {args.root}")
    ds = CSTAntiUAVDataset(
        root=args.root,
        split=args.split,
        img_size=args.img_size,
        frame_stride=args.stride,
        skip_absent=False,   # include absent frames for diagnosis
    )

    print(f"Total samples: {len(ds)}")
    print(f"Scenes: {len(ds.get_scene_names())}")
    print(f"Categories: {ds.get_category_names()}")

    # Select indices based on filters
    if args.scene:
        if args.scene not in ds.scene_to_indices:
            print(f"ERROR: scene '{args.scene}' not found.")
            print(f"Available: {ds.get_scene_names()[:10]}")
            sys.exit(1)
        pool = ds.scene_to_indices[args.scene]
        print(f"Scene '{args.scene}': {len(pool)} frames")

    elif args.category:
        if args.category not in ds.category_to_indices:
            print(f"ERROR: category '{args.category}' not found.")
            print(f"Available: {ds.get_category_names()}")
            sys.exit(1)
        pool = ds.category_to_indices[args.category]
        print(f"Category '{args.category}': {len(pool)} frames")

    else:
        pool = list(range(len(ds)))

    # Sample indices
    random.seed(args.seed)
    n = min(args.n, len(pool))
    if args.sequential:
        indices = pool[:n]
    else:
        indices = random.sample(pool, n)
    indices = sorted(indices)

    # Output directory
    output_dir = Path(args.output) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {n} visualizations to {output_dir}/\n")

    # Visualize
    all_info = []
    for viz_idx, ds_idx in enumerate(indices):
        sample = ds.samples[ds_idx]
        info   = visualize_sample(sample, args.img_size, output_dir, viz_idx)
        if info is None:
            continue
        all_info.append(info)

        # Print per-frame diagnostic
        ann = info.get("box_xywh")
        if ann:
            print(
                f"  [{viz_idx:3d}] {info['scene']:<25} "
                f"frame={info['frame_id']:5d} | "
                f"orig={info['orig_size'][0]}x{info['orig_size'][1]} | "
                f"xywh=({ann[0]:.1f},{ann[1]:.1f},{ann[2]:.1f},{ann[3]:.1f}) | "
                f"area={info['box_area']:.0f}px² | "
                f"scaled_wh={info.get('box_scaled_wh','?')} | "
                f"valid={info.get('box_valid_after_scale', False)}"
            )
        else:
            print(f"  [{viz_idx:3d}] {info['scene']:<25} "
                  f"frame={info['frame_id']:5d} | NO ANNOTATION")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_info)} frames):")
    areas = [i["box_area"] for i in all_info if i["box_area"] > 0]
    if areas:
        print(f"  Box area — min={min(areas):.1f}  "
              f"mean={sum(areas)/len(areas):.1f}  "
              f"max={max(areas):.1f}  px²")
    no_box    = sum(1 for i in all_info if not i["has_box"])
    not_valid = sum(1 for i in all_info
                    if i["has_box"] and not i.get("box_valid_after_scale", True))
    out_frame = sum(1 for i in all_info
                    if i["has_box"] and not i.get("box_in_frame", True))
    print(f"  No annotation    : {no_box}/{len(all_info)}")
    print(f"  Box out of frame : {out_frame}/{len(all_info)}")
    print(f"  Invalid after scale: {not_valid}/{len(all_info)}")
    print(f"\nImages saved to: {output_dir}/")


if __name__ == "__main__":
    main()