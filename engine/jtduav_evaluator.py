"""
JTDUAV Evaluator
=================
Tests trained model on JTDUAV swarm detection dataset.
Computes TTA vs no-TTA comparison with all required metrics:
  - mAP@0.5, Precision, Recall, F1
  - Unknown Recall, False Positive Rate
  - Forgetting Rate (performance drop on Anti-UAV/Anti-UAV410 after TTA)
  - Adaptation Speed (frames to reach 90% of final mAP)
Saves everything to a single JSON file.
"""

import json
import time
import copy
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, List, Optional

from data.jtduav_dataset import JTDUAVDataset, collate_fn
from engine.evaluator import Evaluator
from utils.metrics import DetectionMetrics, compute_forgetting_rate


class JTDUAVEvaluator:

    def __init__(
        self,
        model,
        cfg: dict,
        device: torch.device,
        split: str = "val",
        output_path: str = None,
    ):
        self.model       = model
        self.cfg         = cfg
        self.device      = device
        self.split       = split
        self.num_classes = cfg["num_known_classes"]

        eval_cfg = cfg.get("eval", {})
        self.iou_threshold  = eval_cfg.get("iou_threshold",  0.5)
        self.conf_threshold = cfg.get("jtduav_dataset", {}).get("conf_threshold", 0.05)
        self.nms_threshold  = cfg.get("jtduav_dataset", {}).get("nms_threshold",  0.35)
        self.max_dets       = cfg.get("jtduav_dataset", {}).get("max_dets", 100)

        jtduav_cfg   = cfg.get("jtduav_dataset", {})
        self.root    = jtduav_cfg.get(
            "root",
            "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/MM-AntiUAV-Yolo"
        )
        self.frame_stride = jtduav_cfg.get("frame_stride", 1)
        self.batch_size   = cfg["train"]["batch_size"]
        self.num_workers  = cfg["train"]["num_workers"]

        if output_path is None:
            log_dir    = Path(cfg["experiment"]["log_dir"])
            exp_name   = cfg["experiment"]["name"]
            output_path = str(log_dir / f"{exp_name}_jtduav_{split}_results.json")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Main comparison run
    # ──────────────────────────────────────────────────────────────

    def run_comparison(
        self,
        baseline_ap: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """
        Run both no-TTA and TTA evaluation and compare.
        baseline_ap: per-class AP from original training dataset
                     used to compute forgetting rate.
        """
        print(f"\n{'='*65}")
        print(f"  JTDUAV Evaluation  |  split={self.split}")
        print(f"{'='*65}")

        dataset = JTDUAVDataset(
            root=self.root,
            split=self.split,
            img_size=self.cfg["train"]["img_size"],
            frame_stride=self.frame_stride,
        )

        if len(dataset) == 0:
            print("  ERROR: No samples loaded.")
            return {}

        results = {
            "experiment":    self.cfg["experiment"]["name"],
            "split":         self.split,
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {
                "total_frames": len(dataset),
                "num_videos":   len(dataset.get_video_names()),
                "videos":       dataset.get_video_names(),
            },
            "no_tta":  {},
            "tta":     {},
            "comparison": {},
        }

        # ── 1. No-TTA baseline
        print("\n  [1/2] Evaluating WITHOUT TTA...")
        self.model.disable_tta()
        no_tta_metrics, no_tta_per_video = self._evaluate_full(
            dataset, use_tta=False
        )
        results["no_tta"] = {
            "overall":   no_tta_metrics,
            "per_video": no_tta_per_video,
        }
        self._print_metrics("No-TTA", no_tta_metrics)

        # ── 2. TTA
        print("\n  [2/2] Evaluating WITH TTA...")
        self.model.enable_tta()
        tta_metrics, tta_per_video = self._evaluate_full(
            dataset, use_tta=True
        )
        results["tta"] = {
            "overall":   tta_metrics,
            "per_video": tta_per_video,
        }
        self._print_metrics("TTA", tta_metrics)

        # ── 3. Forgetting Rate
        if baseline_ap is not None:
            current_ap = {
                cls_id: no_tta_metrics.get(f"AP_class{cls_id}", 0.0)
                for cls_id in range(self.num_classes)
            }
            fr = compute_forgetting_rate(baseline_ap, current_ap)
            results["forgetting_rate"] = fr
            print(f"\n  Forgetting Rate: {fr['forgetting_rate_overall']:.4f}")

        # ── 4. Adaptation Speed
        adaptation_speed_no_tta = self._compute_adaptation_speed(
            dataset, use_tta=False
        )
        adaptation_speed_tta = self._compute_adaptation_speed(
            dataset, use_tta=True
        )
        results["adaptation_speed"] = {
            "no_tta_frames": adaptation_speed_no_tta,
            "tta_frames":    adaptation_speed_tta,
        }
        print(f"\n  Adaptation Speed:")
        print(f"    No-TTA: {adaptation_speed_no_tta} frames to 90% mAP")
        print(f"    TTA:    {adaptation_speed_tta} frames to 90% mAP")

        # ── 5. Comparison table
        comparison = {}
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            v_base = no_tta_metrics.get(k, 0.0)
            v_tta  = tta_metrics.get(k, 0.0)
            comparison[k] = {
                "no_tta": v_base,
                "tta":    v_tta,
                "delta":  round(v_tta - v_base, 4),
            }
        results["comparison"] = comparison

        self._print_comparison(comparison)
        self._save(results)

        self.model.disable_tta()
        return results

    # ──────────────────────────────────────────────────────────────
    # Full dataset evaluation
    # ──────────────────────────────────────────────────────────────

    def _evaluate_full(
        self,
        dataset: JTDUAVDataset,
        use_tta: bool,
    ):
        """Evaluate on full dataset + per-video breakdown."""
        loader = DataLoader(
            dataset,
            batch_size=1 if use_tta else self.batch_size,
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

        for images, targets in tqdm(
            loader,
            desc=f"    {'TTA' if use_tta else 'No-TTA'}",
            ncols=80
        ):
            images  = images.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]

            if use_tta:
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

        overall = metrics_acc.compute()

        # Per-video breakdown
        per_video = {}
        for video_name, indices in dataset.get_video_indices().items():
            if use_tta and hasattr(self.model, 'reset_tta'):
                self.model.reset_tta()
            vm = self._evaluate_subset(dataset, indices, use_tta=use_tta,
                                       desc=f"    {video_name}")
            per_video[video_name] = vm

        return overall, per_video

    # ──────────────────────────────────────────────────────────────
    # Subset evaluation
    # ──────────────────────────────────────────────────────────────

    def _evaluate_subset(
        self,
        dataset: JTDUAVDataset,
        indices: List[int],
        use_tta: bool,
        desc: str = "",
    ) -> Dict:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=1 if use_tta else self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
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

        for images, targets in loader:
            images  = images.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]

            if use_tta:
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
    # Adaptation Speed
    # ──────────────────────────────────────────────────────────────

    def _compute_adaptation_speed(
        self,
        dataset: JTDUAVDataset,
        use_tta: bool,
        target_fraction: float = 0.9,
        max_frames: int = 200,
    ) -> int:
        """
        Measure how many frames needed to reach 90% of final mAP.
        Returns frame count or -1 if never reached.
        """
        if use_tta:
            self.model.enable_tta()
            self.model.reset_tta()
        else:
            self.model.disable_tta()

        base_evaluator = Evaluator(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            device=self.device,
        )

        # First pass: compute final mAP on first max_frames * 3 frames
        n_eval = min(max_frames * 3, len(dataset))
        indices = list(range(n_eval))
        final_metrics = self._evaluate_subset(
            dataset, indices, use_tta=use_tta,
            desc=f"    AdaptSpeed {'TTA' if use_tta else 'No-TTA'}"
        )
        final_map = final_metrics.get("mAP@0.5", 0.0)
        target    = final_map * target_fraction

        if final_map == 0:
            return -1

        # Second pass: find when target is first reached
        metrics_acc = DetectionMetrics(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
        )

        if use_tta:
            self.model.enable_tta()
            self.model.reset_tta()

        self.model.eval()
        for frame_idx in range(min(max_frames, len(dataset))):
            img_t, tgt = dataset[frame_idx]
            img_input = img_t.unsqueeze(0).to(self.device)
            tgt_dev   = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in tgt.items()
            }

            if use_tta:
                raw_output = self.model(img_input, targets=None)
            else:
                with torch.no_grad():
                    raw_output = self.model(img_input, targets=None)

            pred_boxes_b, pred_scores_b, pred_labels_b, unk_pred_b = \
                base_evaluator._decode_predictions(raw_output, img_input.shape[-2:])

            metrics_acc.update(
                pred_boxes_b, pred_scores_b, pred_labels_b,
                [tgt_dev["boxes"].cpu().numpy()],
                [tgt_dev["labels"].cpu().numpy()],
            )

            if frame_idx >= 5:   # need at least 5 frames for stable estimate
                interim = metrics_acc.compute()
                if interim.get("mAP@0.5", 0.0) >= target:
                    self.model.disable_tta()
                    return frame_idx + 1

        self.model.disable_tta()
        return -1

    # ──────────────────────────────────────────────────────────────
    # Output helpers
    # ──────────────────────────────────────────────────────────────

    def _print_metrics(self, label: str, metrics: Dict):
        print(f"\n  [{label}]")
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            if k in metrics:
                print(f"    {k:<30}: {metrics[k]:.4f}")

    def _print_comparison(self, comparison: Dict):
        print(f"\n{'='*65}")
        print(f"  COMPARISON: No-TTA vs TTA — JTDUAV")
        print(f"{'='*65}")
        print(f"  {'Metric':<30} {'No-TTA':>10} {'TTA':>10} {'Delta':>10}")
        print(f"  {'-'*62}")
        for k, v in comparison.items():
            sign = "+" if v["delta"] >= 0 else ""
            print(f"  {k:<30} {v['no_tta']:>10.4f} {v['tta']:>10.4f} "
                  f"{sign}{v['delta']:>9.4f}")

    def _save(self, results: Dict):
        def _convert(obj):
            if isinstance(obj, (np.integer,)):   return int(obj)
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, np.ndarray):      return obj.tolist()
            if isinstance(obj, dict):            return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):            return [_convert(v) for v in obj]
            return obj

        with open(self.output_path, "w") as f:
            json.dump(_convert(results), f, indent=2)
        print(f"\n  Results saved → {self.output_path}")
