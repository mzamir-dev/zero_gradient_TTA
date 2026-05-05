"""
Anti-UAV410 Dataset Loader
Structure:
  root/
    train/  val/  test/
      <video_name>/          e.g. 01_1667_0001-1500/
        <frame>.jpg          e.g. 01_1667_0001-1500_0001.jpg (or 0001.jpg)
    annos/
      train/  val/  test/
        <video_name>.txt     e.g. 01_1667_0001-1500.txt
          content: one line per frame → "x,y,w,h"  (absolute pixel coords)
          absent/occluded frames: "0,0,0,0" or empty line
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_att_file(att_path: Path) -> List[int]:
    """
    Parse attribute file.
    Returns list of int flags per frame:
    0 = visible/easy
    1 = partially occluded or small  
    2 = absent/fully occluded
    """
    if not att_path.exists():
        return []
    
    with open(att_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
    
    if not content:
        return []
    
    try:
        flags = [int(v.strip()) for v in content.replace("\n", ",").split(",")
                if v.strip()]
        return flags
    except ValueError:
        return []

class AntiUAV410Dataset(Dataset):
    """
    Anti-UAV410 dataset.
    Annotations: x,y,w,h per line per frame (absolute pixel, top-left origin).
    Returns single-frame samples suitable for the detection pipeline.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        transforms=None,
        class_offset: int = 0,
        skip_absent: bool = False,   # if True, skip frames with no UAV
        frame_stride=10
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.class_offset = class_offset
        self.skip_absent = skip_absent

        self.frame_dir = self.root / split
        self.anno_dir = self.root / "annos" / split
        self.frame_stride = frame_stride

        if not self.frame_dir.exists():
            raise FileNotFoundError(f"Frame dir not found: {self.frame_dir}")
        if not self.anno_dir.exists():
            raise FileNotFoundError(f"Anno dir not found: {self.anno_dir}")

        self.samples = self._collect_samples()
    
        visible = sum(1 for s in self.samples if s["visibility"] == 1)
        print(
            f"[AntiUAV410/{split}] {len(self.samples)} frames loaded "
            f"({visible} visible, {len(self.samples)-visible} absent)"
        )

    def _parse_anno_file(self, anno_path: Path) -> List[Optional[Tuple]]:
        """
        Parse annotation file → list of (x, y, w, h) or None per frame.
        """
        annotations = []
        with open(anno_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    annotations.append(None)
                    continue
                # Support comma and space delimiters
                parts = line.replace(",", " ").split()
                if len(parts) < 4:
                    annotations.append(None)
                    continue
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    if w <= 0 or h <= 0 or (x == 0 and y == 0 and w == 0 and h == 0):
                        annotations.append(None)   # absent
                    else:
                        annotations.append((x, y, w, h))
                except ValueError:
                    annotations.append(None)
        return annotations

    def _collect_samples(self) -> List[Dict]:
        """
        Walk all video directories, match frames to annotation lines.
        Returns flat list of per-frame sample dicts.
        """
        samples = []
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # Find all video directories in split
        video_dirs = sorted([
            d for d in self.frame_dir.iterdir() if d.is_dir()
        ])

        for video_dir in video_dirs:
            video_name = video_dir.name

            # Find matching annotation file
            anno_path = self.anno_dir / f"{video_name}.txt"
            if not anno_path.exists():
                # Try direct match without extension assumption
                candidates = list(self.anno_dir.glob(f"{video_name}*"))
                if candidates:
                    anno_path = candidates[0]
                else:
                    # No annotation → skip this video
                    continue

            annotations = self._parse_anno_file(anno_path)

            # Collect sorted frame images
            frames = sorted(
                [f for f in video_dir.iterdir() if f.suffix.lower() in img_exts],
                key=lambda p: p.name
            )

            n = min(len(frames), len(annotations))

            stride = self.frame_stride if hasattr(self, 'frame_stride') else 1

            annotations = self._parse_anno_file(anno_path)

            # Read att flags if available
            att_path = self.anno_dir / "att" / f"{video_name}.txt"
            att_flags = _parse_att_file(att_path) if att_path.exists() else []
            for i in range(0, n, stride):
                ann      = annotations[i]
                
                # Use att flag if available, otherwise derive from annotation
                if att_flags and i < len(att_flags):
                    att_flag   = att_flags[i]
                    visibility = 0 if att_flag == 2 else 1
                else:
                    att_flag   = -1
                    visibility = 1 if ann is not None else 0

                if self.skip_absent and visibility == 0:
                    continue

                samples.append({
                    "img_path":   frames[i],
                    "annotation": ann,
                    "visibility": visibility,
                    "att_flag":   att_flag,
                    "video":      video_name,
                    "frame_id":   i,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        ann = sample["annotation"]

        # ── Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        orig_h, orig_w = img.shape[:2]

        # Scale factors for bbox
        scale_x = self.img_size / orig_w if orig_w > 0 else 1.0
        scale_y = self.img_size / orig_h if orig_h > 0 else 1.0

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # ── Build target
        boxes = []
        labels = []

        if ann is not None:
            x, y, w, h = ann
            # Scale to resized coords
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            # Clip to image bounds
            x1 = max(0.0, min(x1, self.img_size - 1))
            y1 = max(0.0, min(y1, self.img_size - 1))
            x2 = max(0.0, min(x2, self.img_size))
            y2 = max(0.0, min(y2, self.img_size))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(0 + self.class_offset)   # single class: UAV

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes":      boxes_tensor,
            "labels":     labels_tensor,
            "visibility": sample["visibility"],
            "att_flag":   sample.get("att_flag", -1),
            "img_path":   str(img_path),
            "orig_size":  (orig_h, orig_w),
            "video":      sample["video"],
            "frame_id":   sample["frame_id"],
            "dataset":    "antiuav410",
            }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
