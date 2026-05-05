"""
JTDUAV / MM-AntiUAV Dataset Loader
====================================
Structure:
  root/
    images/
      val/   (or train/ test/)
        MultiUAV-002_00001.jpg
        MultiUAV-002_00002.jpg
        MultiUAV-003_00001.jpg
        ...
    labels/
      val/
        MultiUAV-002_00001.txt   ← YOLO format, multiple boxes per frame
        ...

Annotation format (YOLO normalized):
  class cx cy w h
  0 0.057547 0.177881 0.043969 0.047285
  0 0.070156 0.332100 0.039094 0.041230
  ...

Key differences from Anti-UAV:
  - Multiple drones per frame (swarm detection)
  - YOLO normalized coordinates
  - Video name embedded in filename: MultiUAV-002_00001.jpg
    → video = MultiUAV-002, frame_id = 00001
  - No att file, no absent frames — every frame has at least 1 drone
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_video_frame(filename: str) -> Tuple[str, int]:
    """
    Extract video name and frame id from filename.
    MultiUAV-002_00001.jpg → ("MultiUAV-002", 1)
    """
    stem = Path(filename).stem
    # Split on last underscore
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        video_name = parts[0]
        try:
            frame_id = int(parts[1])
            return video_name, frame_id
        except ValueError:
            pass
    return stem, 0


class JTDUAVDataset(Dataset):
    """
    JTDUAV / MM-AntiUAV swarm detection dataset.
    YOLO format annotations with multiple drones per frame.
    """

    def __init__(
        self,
        root: str,
        split: str = "val",
        img_size: int = 640,
        transforms=None,
        class_offset: int = 0,
        frame_stride: int = 1,
    ):
        self.root         = Path(root)
        self.split        = split
        self.img_size     = img_size
        self.transforms   = transforms
        self.class_offset = class_offset
        self.frame_stride = frame_stride

        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {self.lbl_dir}")

        self.samples: List[Dict]                     = []
        self.video_to_indices: Dict[str, List[int]]  = {}

        self._collect_samples()
        self._print_summary()

    def _collect_samples(self):
        all_imgs = sorted(
            [f for f in self.img_dir.iterdir()
             if f.suffix.lower() in IMG_EXTS],
            key=lambda p: p.name,
        )

        # Group by video for stride application
        video_files: Dict[str, List[Path]] = {}
        for img_path in all_imgs:
            video_name, _ = _parse_video_frame(img_path.name)
            video_files.setdefault(video_name, []).append(img_path)

        for video_name, frames in sorted(video_files.items()):
            frames = sorted(frames, key=lambda p: p.name)
            video_indices = []

            for i in range(0, len(frames), self.frame_stride):
                img_path  = frames[i]
                lbl_path  = self.lbl_dir / (img_path.stem + ".txt")
                _, frame_id = _parse_video_frame(img_path.name)

                idx = len(self.samples)
                self.samples.append({
                    "img_path":  img_path,
                    "lbl_path":  lbl_path,
                    "video":     video_name,
                    "frame_id":  frame_id,
                })
                video_indices.append(idx)

            if video_indices:
                self.video_to_indices[video_name] = video_indices

    def _print_summary(self):
        total_boxes = 0
        for s in self.samples:
            lbl = s["lbl_path"]
            if lbl.exists():
                with open(lbl) as f:
                    total_boxes += sum(
                        1 for line in f if line.strip()
                    )
        avg_boxes = total_boxes / max(len(self.samples), 1)
        print(
            f"[JTDUAV/{self.split}] {len(self.samples)} frames | "
            f"{len(self.video_to_indices)} videos | "
            f"avg {avg_boxes:.1f} drones/frame"
        )

    def get_video_indices(self) -> Dict[str, List[int]]:
        return self.video_to_indices

    def get_video_names(self) -> List[str]:
        return sorted(self.video_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = sample["img_path"]
        lbl_path = sample["lbl_path"]

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Load YOLO labels — multiple boxes per frame
        boxes, labels = [], []
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
                        cx  = float(parts[1])
                        cy  = float(parts[2])
                        bw  = float(parts[3])
                        bh  = float(parts[4])
                        # Convert YOLO normalized → pixel x1y1x2y2
                        x1 = max(0.0, (cx - bw / 2) * self.img_size)
                        y1 = max(0.0, (cy - bh / 2) * self.img_size)
                        x2 = min(float(self.img_size), (cx + bw / 2) * self.img_size)
                        y2 = min(float(self.img_size), (cy + bh / 2) * self.img_size)
                        if x2 > x1 + 1 and y2 > y1 + 1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)
                    except (ValueError, IndexError):
                        continue

        target = {
            "boxes":    torch.tensor(boxes, dtype=torch.float32)
                        if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long)
                        if labels else torch.zeros((0,), dtype=torch.long),
            "img_path": str(img_path),
            "orig_size": (orig_h, orig_w),
            "video":    sample["video"],
            "frame_id": sample["frame_id"],
            "dataset":  "jtduav",
            "num_drones": len(boxes),
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
