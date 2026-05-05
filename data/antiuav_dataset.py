"""
Anti-UAV Dataset Loader — YOLO format
Structure:
  root/
    images/
      train/  val/  test/
        *.jpg
    labels/
      train/  val/  test/
        *.txt   (class cx cy w h  — normalized)
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class AntiUAVDataset(Dataset):
    """
    Anti-UAV dataset in YOLO format.
    Returns single-frame detections for the detection pipeline.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        transforms=None,
        class_offset: int = 0,   # add to class IDs if mixing datasets
        frame_stride=10
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.class_offset = class_offset

        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        self.frame_stride = frame_stride

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {self.lbl_dir}")

        self.samples = self._collect_samples()
        
        print(f"[AntiUAV/{split}] {len(self.samples)} frames loaded from {self.img_dir}")

    def _collect_samples(self):
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        samples = []
        all_files = sorted([f for f in self.img_dir.iterdir()
                            if f.suffix.lower() in img_exts])
        for idx in range(0, len(all_files), getattr(self, 'frame_stride', 10)):
            img_path = all_files[idx]
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            samples.append((img_path, lbl_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # ── Load image (thermal: grayscale → 1-channel float)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return blank frame on corrupt image — logged separately
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        orig_h, orig_w = img.shape[:2]

        # Resize to target size
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # ── Load labels
        boxes = []   # [x1, y1, x2, y2]  in pixel coords after resize
        labels = []  # class IDs

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0])) + self.class_offset
                        cx, cy, bw, bh = (
                            float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4])
                        )
                        # Convert YOLO normalized [cx,cy,w,h] → pixel [x1,y1,x2,y2]
                        x1 = (cx - bw / 2) * self.img_size
                        y1 = (cy - bh / 2) * self.img_size
                        x2 = (cx + bw / 2) * self.img_size
                        y2 = (cy + bh / 2) * self.img_size
                        # Clip
                        x1 = max(0.0, min(x1, self.img_size - 1))
                        y1 = max(0.0, min(y1, self.img_size - 1))
                        x2 = max(0.0, min(x2, self.img_size))
                        y2 = max(0.0, min(y2, self.img_size))
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)
                    except (ValueError, IndexError):
                        continue

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_tensor,      # [N, 4] x1y1x2y2 pixel
            "labels": labels_tensor,    # [N]
            "img_path": str(img_path),
            "orig_size": (orig_h, orig_w),
            "dataset": "antiuav",
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    """Custom collate: images stacked, targets kept as list (variable boxes)."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
