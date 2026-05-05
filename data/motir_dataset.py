"""
MOT_IR_sequences Dataset Loader
=================================
Structure:
  root/
    001/
      000001.jpg
      groundtruth_01.txt
      groundtruth_02.txt    (if 2 UAVs)
      groundtruth_03.txt    (if 3 UAVs)
    002/ ... 015/

Annotation per groundtruth_XX.txt (one line per frame):
  000001.jpg,162,96,183,102,0,0,0,0,0,0,0,0,1,3.5,0.984
  Fields: filename, x1, y1, x2, y2, <MOT flags...>

  Absent frame:
  000001.jpg,0,0,0,0,1,0,...   → box is (0,0,0,0) → skip this UAV this frame

Multi-UAV handling:
  - Parse ALL groundtruth_XX.txt files per sequence
  - Merge valid boxes from all UAVs per frame
  - Frame with no valid UAV boxes → skip (if skip_all_absent=True)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_groundtruth_file(
    gt_path: Path,
) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
    """
    Parse one groundtruth_XX.txt.
    Returns: {frame_filename: (x1,y1,x2,y2)} or None if absent.
    """
    result = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            frame_name = parts[0].strip()
            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
            except ValueError:
                result[frame_name] = None
                continue
            # Absent: zero box or degenerate
            if (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                result[frame_name] = None
            elif x2 <= x1 or y2 <= y1:
                result[frame_name] = None
            else:
                result[frame_name] = (x1, y1, x2, y2)
    return result


def _read_img_size(img_path: Path) -> Tuple[int, int]:
    """Return (width, height) from image file."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 640, 512
    h, w = img.shape[:2]
    return w, h


class MOTIRDataset(Dataset):
    """
    MOT_IR multi-UAV thermal dataset (sequences 001-015).
    Up to 3 UAVs per sequence, merged into one target per frame.
    """

    def __init__(
        self,
        root: str,
        img_size: int = 640,
        transforms=None,
        class_offset: int = 0,
        frame_stride: int = 1,
        skip_all_absent: bool = True,
    ):
        self.root            = Path(root)
        self.img_size        = img_size
        self.transforms      = transforms
        self.class_offset    = class_offset
        self.frame_stride    = frame_stride
        self.skip_all_absent = skip_all_absent

        if not self.root.exists():
            raise FileNotFoundError(f"Root not found: {self.root}")

        self.samples: List[Dict]                      = []
        self.seq_to_indices: Dict[str, List[int]]     = {}
        self.seq_img_size: Dict[str, Tuple[int, int]] = {}

        self._collect_samples()
        self._print_summary()

    def _collect_samples(self):
        seq_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

        for seq_dir in seq_dirs:
            seq_name = seq_dir.name

            # Find all groundtruth files for this sequence
            gt_files = sorted(seq_dir.glob("groundtruth_*.txt"))
            if not gt_files:
                print(f"  [WARNING] No groundtruth files in {seq_dir.name}")
                continue

            num_uavs = len(gt_files)

            # Parse all UAV annotation files
            # uav_annos[frame_name] = [box_uav1, box_uav2, ...]
            uav_annos: Dict[str, List] = defaultdict(list)
            for gt_file in gt_files:
                frame_boxes = _parse_groundtruth_file(gt_file)
                for frame_name, box in frame_boxes.items():
                    uav_annos[frame_name].append(box)

            # Collect sorted frame images
            frames = sorted(
                [f for f in seq_dir.iterdir()
                 if f.suffix.lower() in IMG_EXTS],
                key=lambda p: p.name,
            )
            if not frames:
                continue

            # Read actual image size from first valid frame
            orig_w, orig_h = _read_img_size(frames[0])
            self.seq_img_size[seq_name] = (orig_w, orig_h)

            seq_indices = []

            for i in range(0, len(frames), self.frame_stride):
                frame_path = frames[i]
                frame_name = frame_path.name

                # Merge valid boxes from all UAVs for this frame
                all_boxes = uav_annos.get(frame_name, [None] * num_uavs)
                valid_boxes = [b for b in all_boxes if b is not None]

                if self.skip_all_absent and len(valid_boxes) == 0:
                    continue

                idx = len(self.samples)
                self.samples.append({
                    "img_path":   frame_path,
                    "boxes":      valid_boxes,
                    "num_uavs":   len(valid_boxes),
                    "sequence":   seq_name,
                    "frame_id":   i,
                    "frame_name": frame_name,
                    "orig_w":     orig_w,
                    "orig_h":     orig_h,
                })
                seq_indices.append(idx)

            if seq_indices:
                self.seq_to_indices[seq_name] = seq_indices

    def _print_summary(self):
        total_boxes = sum(s["num_uavs"] for s in self.samples)
        avg = total_boxes / max(len(self.samples), 1)
        print(
            f"[MOT_IR] {len(self.samples)} frames | "
            f"{len(self.seq_to_indices)} sequences | "
            f"stride={self.frame_stride} | "
            f"avg {avg:.2f} UAVs/frame"
        )
        for seq, indices in sorted(self.seq_to_indices.items()):
            w, h = self.seq_img_size.get(seq, (0, 0))
            gt_count = len(list(
                (self.root / seq).glob("groundtruth_*.txt")
            ))
            print(
                f"    seq {seq}  "
                f"{len(indices):5d} frames | "
                f"img={w}x{h} | "
                f"UAVs={gt_count}"
            )

    def get_sequence_indices(self) -> Dict[str, List[int]]:
        return self.seq_to_indices

    def get_sequence_names(self) -> List[str]:
        return sorted(self.seq_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        img_path  = sample["img_path"]
        abs_boxes = sample["boxes"]
        orig_w    = sample["orig_w"]
        orig_h    = sample["orig_h"]

        # Load thermal frame
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            # Use real size in case it differs from stored
            orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Scale boxes from original pixel space to model input size
        scale_x = self.img_size / orig_w if orig_w > 0 else 1.0
        scale_y = self.img_size / orig_h if orig_h > 0 else 1.0

        boxes, labels = [], []
        for (x1, y1, x2, y2) in abs_boxes:
            sx1 = max(0.0,              x1 * scale_x)
            sy1 = max(0.0,              y1 * scale_y)
            sx2 = min(float(self.img_size), x2 * scale_x)
            sy2 = min(float(self.img_size), y2 * scale_y)
            if sx2 > sx1 + 1 and sy2 > sy1 + 1:
                boxes.append([sx1, sy1, sx2, sy2])
                labels.append(0 + self.class_offset)

        target = {
            "boxes":    torch.tensor(boxes, dtype=torch.float32)
                        if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long)
                        if labels else torch.zeros((0,), dtype=torch.long),
            "img_path":   str(img_path),
            "orig_size":  (orig_h, orig_w),
            "sequence":   sample["sequence"],
            "frame_id":   sample["frame_id"],
            "frame_name": sample["frame_name"],
            "num_uavs":   sample["num_uavs"],
            "dataset":    "mot_ir",
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
