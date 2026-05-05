"""
AntiUAV600 Dataset Loader
==========================
Structure:
  root/
    01_1751_0250-1750/
      000001.jpg
      000001.txt    ← YOLO annotation same directory as frame
      000002.jpg
      000002.txt
      ...
    wg2022_ir_034_split_01/
      000001.jpg
      000001.txt
      ...
    new21_train_newfix/
    3700000000002_142320_2/
    ...  (50 random-name sequence directories)

Annotation format (YOLO normalized, single UAV):
  0 0.54609375 0.53125 0.0859375 0.05859375

Note: annotation file has same stem as image file, lives in same directory.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class AntiUAV600Dataset(Dataset):
    """
    AntiUAV600 dataset loader.
    Sequences are top-level directories with mixed frames + annotations.
    Supports dynamic frame stride.
    """

    def __init__(
        self,
        root: str,
        img_size: int = 640,
        transforms=None,
        class_offset: int = 0,
        frame_stride: int = 1,
        skip_missing_labels: bool = True,
    ):
        self.root                = Path(root)
        self.img_size            = img_size
        self.transforms          = transforms
        self.class_offset        = class_offset
        self.frame_stride        = frame_stride
        self.skip_missing_labels = skip_missing_labels

        if not self.root.exists():
            raise FileNotFoundError(f"Root not found: {self.root}")

        self.samples: List[Dict]                     = []
        self.seq_to_indices: Dict[str, List[int]]    = {}

        self._collect_samples()
        self._print_summary()

    def _collect_samples(self):
        # Each subdirectory is a sequence
        seq_dirs = sorted([
            d for d in self.root.iterdir() if d.is_dir()
        ])

        for seq_dir in seq_dirs:
            seq_name = seq_dir.name

            # Collect sorted frame images
            frames = sorted(
                [f for f in seq_dir.iterdir()
                 if f.suffix.lower() in IMG_EXTS],
                key=lambda p: p.name,
            )

            if not frames:
                continue

            seq_indices = []

            for i in range(0, len(frames), self.frame_stride):
                img_path = frames[i]
                # Annotation has same stem, same directory
                lbl_path = img_path.with_suffix(".txt")

                if self.skip_missing_labels and not lbl_path.exists():
                    continue

                idx = len(self.samples)
                self.samples.append({
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "sequence": seq_name,
                    "frame_id": i,
                })
                seq_indices.append(idx)

            if seq_indices:
                self.seq_to_indices[seq_name] = seq_indices

    def _print_summary(self):
        print(
            f"[AntiUAV600] {len(self.samples)} frames | "
            f"{len(self.seq_to_indices)} sequences | "
            f"stride={self.frame_stride}"
        )
        for seq, indices in sorted(self.seq_to_indices.items()):
            print(f"    {seq:<40} {len(indices):5d} frames")

    def get_sequence_indices(self) -> Dict[str, List[int]]:
        return self.seq_to_indices

    def get_sequence_names(self) -> List[str]:
        return sorted(self.seq_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = sample["img_path"]
        lbl_path = sample["lbl_path"]

        # Load thermal frame as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # Load YOLO annotation
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
                        x1  = max(0.0, (cx - bw / 2) * self.img_size)
                        y1  = max(0.0, (cy - bh / 2) * self.img_size)
                        x2  = min(float(self.img_size),
                                  (cx + bw / 2) * self.img_size)
                        y2  = min(float(self.img_size),
                                  (cy + bh / 2) * self.img_size)
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
            "img_path":  str(img_path),
            "orig_size": (orig_h, orig_w),
            "sequence":  sample["sequence"],
            "frame_id":  sample["frame_id"],
            "dataset":   "antiuav600",
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
