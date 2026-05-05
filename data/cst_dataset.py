"""
CST-Anti-UAV Dataset Loader
============================
Exact directory structure:
  root/
    train/  val/  test/
      building_66/          ← scene directory (random name)
        frame_0001.jpg
        frame_0002.jpg
        ...
      cn_mountains_30/
      cn_sky_11/
      jungle_10/
      urban-areas_24/
      water_20/
    annos/
      building_66.txt       ← one .txt per scene, name matches directory
      cn_mountains_30.txt
      ...
        content: one line per frame → "x,y,w,h" (absolute pixel, float)
        empty lines or zero boxes → absent frame → SKIPPED

Scene category extracted from name prefix:
  building_66      → building
  cn_mountains_30  → cn_mountains
  urban-areas_24   → urban-areas
  water_20         → water
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _extract_scene_category(scene_name: str) -> str:
    """Strip trailing numeric ID to get category."""
    parts = scene_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scene_name


def _parse_anno_file(
    anno_path: Path,
) -> List[Optional[Tuple[float, float, float, float]]]:
    """
    Parse CST annotation file.
    Returns list of (x, y, w, h) or None per frame.
    Skips empty lines and zero/invalid boxes.
    """
    annotations = []
    with open(anno_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                annotations.append(None)
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                annotations.append(None)
                continue
            try:
                x, y, w, h = (float(parts[0]), float(parts[1]),
                               float(parts[2]), float(parts[3]))
            except ValueError:
                annotations.append(None)
                continue
            # Skip zero or negative boxes
            if w <= 0 or h <= 0:
                annotations.append(None)
                continue
            if x == 0.0 and y == 0.0 and w == 0.0 and h == 0.0:
                annotations.append(None)
                continue
            annotations.append((x, y, w, h))
    return annotations


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

class CSTAntiUAVDataset(Dataset):
    """
    CST-Anti-UAV dataset.
    Skips empty frames and zero boxes automatically.
    Indexes by scene name and scene category for per-scene evaluation.
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        img_size: int = 640,
        transforms=None,
        class_offset: int = 0,
        frame_stride: int = 1,
        skip_absent: bool = True,
    ):
        self.root         = Path(root)
        self.split        = split
        self.img_size     = img_size
        self.transforms   = transforms
        self.class_offset = class_offset
        self.frame_stride = frame_stride
        self.skip_absent  = skip_absent

        self.split_dir = self.root / split
        self.anno_dir  = self.root / "annos"

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")
        if not self.anno_dir.exists():
            raise FileNotFoundError(f"Anno dir not found: {self.anno_dir}")

        self.samples: List[Dict]                      = []
        self.scene_to_indices: Dict[str, List[int]]   = {}
        self.category_to_indices: Dict[str, List[int]] = {}

        self._collect_samples()
        self._print_summary()


    def _collect_samples(self):
        scene_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
        skipped_no_anno = 0

        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            category   = _extract_scene_category(scene_name)

            # Locate annotation file
            anno_path = self.anno_dir / f"{scene_name}.txt"
            if not anno_path.exists():
                alt = self.anno_dir / self.split / f"{scene_name}.txt"
                if alt.exists():
                    anno_path = alt
                else:
                    skipped_no_anno += 1
                    continue

            annotations = _parse_anno_file(anno_path)

            frames = sorted(
                [f for f in scene_dir.iterdir()
                 if f.suffix.lower() in IMG_EXTS],
                key=lambda p: p.name,
            )

            n = min(len(frames), len(annotations))

            # Check for non-sequential naming that could cause misalignment
            if len(frames) > 0 and len(annotations) > 0:
                if len(frames) != len(annotations):
                    # Try to align by frame number extracted from filename
                    frame_nums = []
                    for f in frames:
                        num_str = ''.join(filter(str.isdigit, f.stem))
                        if num_str:
                            frame_nums.append((int(num_str), f))
                    
                    if frame_nums:
                        # Build index: frame_number → annotation_line
                        # Assumes annotation line i corresponds to frame number i+1
                        aligned_samples = []
                        for frame_num, frame_path in frame_nums:
                            anno_idx = frame_num - 1   # 000001.jpg → annotation line 0
                            if 0 <= anno_idx < len(annotations):
                                aligned_samples.append((frame_path, annotations[anno_idx]))
                        
                        # Use aligned samples instead of zip
                        for frame_path, ann in aligned_samples[::self.frame_stride]:
                            visibility = 1 if ann is not None else 0
                            if self.skip_absent and visibility == 0:
                                continue
                            idx = len(self.samples)
                            frame_num = int(''.join(filter(str.isdigit, frame_path.stem)))
                            self.samples.append({
                                "img_path":   frame_path,
                                "annotation": ann,
                                "visibility": visibility,
                                "scene":      scene_name,
                                "category":   category,
                                "frame_id":   frame_num,
                            })
                            scene_indices.append(idx)
                        
                        if scene_indices:
                            self.scene_to_indices[scene_name] = scene_indices
                            self.category_to_indices.setdefault(category, []).extend(scene_indices)
                        continue   # skip the normal loop below

            scene_indices = []

            # Find att file
            att_path = self.anno_dir / "att" / f"{scene_name}.txt"
            if not att_path.exists():
                # Try split-specific
                att_path = self.anno_dir / self.split / "att" / f"{scene_name}.txt"

            att_flags = _parse_att_file(att_path)

            for i in range(0, n, self.frame_stride):
                ann      = annotations[i]
                
                if att_flags and i < len(att_flags):
                    att_flag = att_flags[i]
                    # att=2 means absent — override annotation even if box is nonzero
                    if att_flag == 2:
                        ann        = None   # discard placeholder box
                        visibility = 0
                    else:
                        visibility = 1 if ann is not None else 0
                else:
                    att_flag   = -1
                    visibility = 1 if ann is not None else 0

                # Additional check: discard boxes that are fully outside frame boundary
                if ann is not None:
                    x, y, w, h = ann
                    # We need orig image size — read from first frame if not cached
                    if not hasattr(self, '_scene_img_size'):
                        probe = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
                        if probe is not None:
                            self._scene_img_size = probe.shape[:2]  # (h, w)
                        else:
                            self._scene_img_size = (512, 640)
                    orig_h_s, orig_w_s = self._scene_img_size
                    if x >= orig_w_s or y >= orig_h_s or (x + w) <= 0 or (y + h) <= 0:
                        ann        = None
                        visibility = 0

                if self.skip_absent and visibility == 0:
                    continue

                idx = len(self.samples)
                self.samples.append({
                    "img_path":   frames[i],
                    "annotation": ann,
                    "visibility": visibility,
                    "att_flag":   att_flag,
                    "scene":      scene_name,
                    "category":   category,
                    "frame_id":   i,
                })
                scene_indices.append(idx)

            if scene_indices:
                self.scene_to_indices[scene_name] = scene_indices
                self.category_to_indices.setdefault(category, []).extend(
                    scene_indices
                )

        if skipped_no_anno > 0:
            print(f"  [CST/{self.split}] WARNING: {skipped_no_anno} scenes "
                  f"skipped — no annotation file found")

    def _print_summary(self):
        visible = sum(1 for s in self.samples if s["visibility"] == 1)
        cats    = sorted(self.category_to_indices.keys())
        print(
            f"[CST-AntiUAV/{self.split}] {len(self.samples)} frames | "
            f"{len(self.scene_to_indices)} scenes | "
            f"{len(cats)} categories | "
            f"visible={visible}"
        )
        for cat in cats:
            n_scenes = len({
                self.samples[i]["scene"]
                for i in self.category_to_indices[cat]
            })
            n_frames = len(self.category_to_indices[cat])
            print(f"    {cat:<25} {n_scenes:3d} scenes  {n_frames:6d} frames")

    def get_scene_indices(self) -> Dict[str, List[int]]:
        return self.scene_to_indices

    def get_category_indices(self) -> Dict[str, List[int]]:
        return self.category_to_indices

    def get_scene_names(self) -> List[str]:
        return sorted(self.scene_to_indices.keys())

    def get_category_names(self) -> List[str]:
        return sorted(self.category_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        sample   = self.samples[idx]
        img_path = sample["img_path"]
        ann      = sample["annotation"]

        TARGET_SIZE = self.img_size

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        orig_h, orig_w = img.shape[:2]

        # Step 1: upsample 2x with cubic to preserve tiny object signal
        up_h = int(orig_h * 2.0)
        up_w = int(orig_w * 2.0)
        img  = cv2.resize(img, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

        # Step 2: downsample to target size with area interpolation
        img  = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # Scale factors map original coordinates → TARGET_SIZE
        scale_x = TARGET_SIZE / orig_w if orig_w > 0 else 1.0
        scale_y = TARGET_SIZE / orig_h if orig_h > 0 else 1.0

        boxes, labels = [], []
        if ann is not None:
            x, y, w, h = ann
            x1 = max(0.0, x * scale_x)
            y1 = max(0.0, y * scale_y)
            x2 = min(float(TARGET_SIZE), (x + w) * scale_x)
            y2 = min(float(TARGET_SIZE), (y + h) * scale_y)
            if x2 > x1 + 1 and y2 > y1 + 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(0 + self.class_offset)

        target = {
            "boxes":    torch.tensor(boxes, dtype=torch.float32)
                        if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long)
                        if labels else torch.zeros((0,), dtype=torch.long),
            "visibility":  sample["visibility"],
            "att_flag":   sample.get("att_flag", -1), 
            "img_path":    str(img_path),
            "orig_size":   (orig_h, orig_w),
            "scene":       sample["scene"],
            "category":    sample["category"],
            "frame_id":    sample["frame_id"],
            "dataset":     "cst_antiuav",
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
