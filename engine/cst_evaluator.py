"""
CST-Anti-UAV Evaluator
=======================
Tests the trained model on CST-Anti-UAV test split.
Stores results per scene, per category, and overall in a single JSON file.

Usage:
    python tools/cst_test.py --config configs/base.yaml
    python tools/cst_test.py --config configs/base.yaml --split val
    python tools/cst_test.py --config configs/base.yaml --no_tta
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, List

from data.cst_dataset import CSTAntiUAVDataset, collate_fn
from engine.evaluator import Evaluator
from utils.metrics import DetectionMetrics


class CSTEvaluator:
    """
    Evaluates model on CST-Anti-UAV.
    Produces per-scene, per-category, and overall metrics.
    All results saved to a single JSON file.
    """

    def __init__(
        self,
        model,
        cfg: dict,
        device: torch.device,
        split: str = "test",
        use_tta: bool = False,
        output_path: str = None,
    ):
        self.model       = model
        self.cfg         = cfg
        self.device      = device
        self.split       = split
        self.use_tta     = use_tta
        self.num_classes = cfg["num_known_classes"]

        eval_cfg = cfg.get("eval", {})
        self.iou_threshold  = eval_cfg.get("iou_threshold",  0.5)
        self.conf_threshold = eval_cfg.get("conf_threshold", 0.25)
        self.nms_threshold  = eval_cfg.get("nms_threshold",  0.45)

        # Output JSON path
        if output_path is None:
            log_dir    = Path(cfg["experiment"]["log_dir"])
            exp_name   = cfg["experiment"]["name"]
            suffix     = "tta" if use_tta else "no_tta"
            output_path = str(log_dir / f"{exp_name}_cst_{split}_{suffix}.json")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # CST dataset root from config
        cst_cfg = cfg.get("cst_dataset", {})
        self.cst_root = cst_cfg.get(
            "root",
            "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/CST-AntiUAV"
        )
        self.frame_stride = cst_cfg.get("frame_stride", 1)
        self.batch_size   = cfg["train"]["batch_size"]
        self.num_workers  = cfg["train"]["num_workers"]

    # ──────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Full evaluation pipeline.
        Returns combined results dict and saves to JSON.
        """
        print(f"\n{'='*65}")
        print(f"  CST-Anti-UAV Evaluation  |  split={self.split}  "
              f"|  TTA={'ON' if self.use_tta else 'OFF'}")
        print(f"{'='*65}")

        # Load full dataset
        dataset = CSTAntiUAVDataset(
            root=self.cst_root,
            split=self.split,
            img_size=self.cfg["train"]["img_size"],
            frame_stride=self.frame_stride,
            skip_absent=True,
        )

        if len(dataset) == 0:
            print("  ERROR: No samples loaded. Check dataset path and structure.")
            return {}

        # Enable TTA if requested
        if self.use_tta:
            self.model.enable_tta()
        else:
            self.model.disable_tta()

        results = {
            "experiment":   self.cfg["experiment"]["name"],
            "split":        self.split,
            "tta_enabled":  self.use_tta,
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {
                "total_frames":  len(dataset),
                "num_scenes":    len(dataset.get_scene_names()),
                "num_categories": len(dataset.get_category_names()),
                "scenes":        dataset.get_scene_names(),
                "categories":    dataset.get_category_names(),
            },
            "per_scene":    {},
            "per_category": {},
            "overall":      {},
        }

        # # ── Per-scene evaluation
        # print(f"\n  Evaluating {len(dataset.get_scene_names())} scenes...")
        # scene_metrics_list = []

        # for scene_name, indices in dataset.get_scene_indices().items():
        #     if self.use_tta:
        #         self.model.reset_tta()   # fresh adapter per scene

        #     scene_metrics = self._evaluate_subset(
        #         dataset, indices,
        #         desc=f"  {scene_name}"
        #     )
        #     results["per_scene"][scene_name] = scene_metrics
        #     scene_metrics_list.append(scene_metrics)

        #     category = dataset.samples[indices[0]]["category"]
        #     print(
        #         f"    {scene_name:<30} "
        #         f"mAP={scene_metrics.get('mAP@0.5', 0):.4f}  "
        #         f"P={scene_metrics.get('precision', 0):.4f}  "
        #         f"R={scene_metrics.get('recall', 0):.4f}  "
        #         f"F1={scene_metrics.get('F1', 0):.4f}  "
        #         f"frames={len(indices)}"
        #     )

        # ── Per-category aggregation
        print(f"\n  Aggregating by category...")
        for category, indices in dataset.get_category_indices().items():
            cat_metrics = self._evaluate_subset(
                dataset, indices,
                desc=f"  [{category}]"
            )
            results["per_category"][category] = cat_metrics
            print(
                f"    {category:<25} "
                f"mAP={cat_metrics.get('mAP@0.5', 0):.4f}  "
                f"P={cat_metrics.get('precision', 0):.4f}  "
                f"R={cat_metrics.get('recall', 0):.4f}  "
                f"F1={cat_metrics.get('F1', 0):.4f}  "
                f"frames={len(indices)}"
            )
        
        # ── Difficulty breakdown using att flags (add HERE, after category loop)
        print(f"\n  Difficulty breakdown (att flags)...")
        easy_indices = [i for i in range(len(dataset))
                        if dataset.samples[i].get("att_flag", -1) == 0]
        hard_indices = [i for i in range(len(dataset))
                        if dataset.samples[i].get("att_flag", -1) == 1]
        absent_indices = [i for i in range(len(dataset))
                        if dataset.samples[i].get("att_flag", -1) == 2]

        if easy_indices:
            easy_metrics = self._evaluate_subset(
                dataset, easy_indices, desc="  [easy att=0]"
            )
            results["difficulty"] = results.get("difficulty", {})
            results["difficulty"]["easy"] = easy_metrics
            print(f"    easy  (att=0): mAP={easy_metrics.get('mAP@0.5',0):.4f}  "
                f"frames={len(easy_indices)}")

        if hard_indices:
            hard_metrics = self._evaluate_subset(
                dataset, hard_indices, desc="  [hard att=1]"
            )
            results["difficulty"] = results.get("difficulty", {})
            results["difficulty"]["hard"] = hard_metrics
            print(f"    hard  (att=1): mAP={hard_metrics.get('mAP@0.5',0):.4f}  "
                f"frames={len(hard_indices)}")

        if absent_indices:
            # Absent frames — model should NOT fire, measures false positive rate
            absent_metrics = self._evaluate_subset(
                dataset, absent_indices, desc="  [absent att=2]"
            )
            results["difficulty"] = results.get("difficulty", {})
            results["difficulty"]["absent"] = absent_metrics
            print(f"    absent(att=2): FPR={absent_metrics.get('false_positive_rate',0):.4f}  "
                f"frames={len(absent_indices)}")

        # ── Overall evaluation (all frames at once)
        print(f"\n  Overall evaluation on all {len(dataset)} frames...")
        all_indices   = list(range(len(dataset)))
        overall_metrics = self._evaluate_subset(
            dataset, all_indices, desc="  [Overall]"
        )
        results["overall"] = overall_metrics

        # ── Summary print
        self._print_summary(results)

        # ── Save to JSON
        self._save(results)

        if self.use_tta:
            self.model.disable_tta()

        return results

    # ──────────────────────────────────────────────────────────────
    # Subset evaluation
    # ──────────────────────────────────────────────────────────────

    # @torch.no_grad()
    def _evaluate_subset(
    self,
    dataset: CSTAntiUAVDataset,
    indices: List[int],
    desc: str = "",
    ) -> Dict:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        base_evaluator = Evaluator(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            device=self.device,
        )

        metrics_acc = DetectionMetrics(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
        )

        self.model.eval()

        for images, targets in tqdm(loader, desc=desc, ncols=80, leave=False):
            images  = images.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()}
                for t in targets
            ]

            # TTA requires gradients for adapter update, no_grad for baseline
            if self.use_tta:
                # with torch.enable_grad():
                raw_output = self.model(images, targets=None)
            else:
                with torch.no_grad():
                    raw_output = self.model(images, targets=None)

            pred_boxes_b, pred_scores_b, pred_labels_b, unk_pred_b = \
                base_evaluator._decode_predictions(raw_output, images.shape[-2:])

            gt_boxes_b  = [t["boxes"].cpu().numpy()  for t in targets]
            gt_labels_b = [t["labels"].cpu().numpy() for t in targets]
            unk_gt_b    = [
                (t["labels"].cpu().numpy() >= self.num_classes).astype(int)
                for t in targets
            ]

            metrics_acc.update(
                pred_boxes_b, pred_scores_b, pred_labels_b,
                gt_boxes_b, gt_labels_b,
                unknown_pred_flags_batch=unk_pred_b,
                unknown_gt_flags_batch=unk_gt_b,
            )

        result = metrics_acc.compute()
        result["num_frames"] = len(indices)
        
        return result

    # ──────────────────────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────────────────────

    def _print_summary(self, results: Dict):
        overall = results.get("overall", {})
        print(f"\n{'='*65}")
        print(f"  OVERALL RESULTS — CST-Anti-UAV [{self.split}]")
        print(f"{'='*65}")
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            if k in overall:
                print(f"  {k:<30}: {overall[k]:.4f}")

        # Best and worst scenes
        scene_maps = {
            name: m.get("mAP@0.5", 0)
            for name, m in results["per_scene"].items()
        }
        if scene_maps:
            best  = max(scene_maps, key=scene_maps.get)
            worst = min(scene_maps, key=scene_maps.get)
            print(f"\n  Best scene:  {best:<30} mAP={scene_maps[best]:.4f}")
            print(f"  Worst scene: {worst:<30} mAP={scene_maps[worst]:.4f}")

    def _save(self, results: Dict):
        """Save all results to a single JSON file."""

        def _convert(obj):
            """Make numpy types JSON-serializable."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        with open(self.output_path, "w") as f:
            json.dump(_convert(results), f, indent=2)

        print(f"\n  Results saved → {self.output_path}")
