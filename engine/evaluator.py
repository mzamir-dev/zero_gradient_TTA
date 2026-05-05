"""
Evaluator — Detection metrics computation
Handles: NMS, box decoding, mAP@0.5, Precision, Recall, F1,
         Unknown Recall, False Positive Rate, Forgetting Rate
"""

import torch
import torch.nn.functional as F
from torchvision.ops import nms, box_iou
from tqdm import tqdm
from typing import Dict, List, Optional

from utils.metrics import DetectionMetrics, compute_forgetting_rate


class Evaluator:
    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        device: torch.device = None,
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device or torch.device("cpu")

    # ──────────────────────────────────────────────────────────────
    # Main evaluation loop
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data_loader,
        desc: str = "Evaluating",
    ) -> Dict:
        model.eval()
        metrics = DetectionMetrics(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold,
        )

        for images, targets in tqdm(data_loader, desc=f"  [{desc}]", ncols=80):
            images = images.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]

            # Forward pass
            raw_output = model(images, targets=None)

            # Decode predictions
            pred_boxes_batch, pred_scores_batch, pred_labels_batch, \
                unknown_pred_batch = self._decode_predictions(
                    raw_output, images.shape[-2:]
                )

            # Ground truth
            gt_boxes_batch = [t["boxes"].cpu().numpy() for t in targets]
            gt_labels_batch = [t["labels"].cpu().numpy() for t in targets]

            # Unknown GT flags: currently no GT unknowns during base training
            # (CST-AntiUAV TTA evaluator handles this separately)
            unknown_gt_batch = [
                (t["labels"].cpu().numpy() >= self.num_classes).astype(int)
                for t in targets
            ]

            metrics.update(
                pred_boxes_batch, pred_scores_batch, pred_labels_batch,
                gt_boxes_batch, gt_labels_batch,
                unknown_pred_flags_batch=unknown_pred_batch,
                unknown_gt_flags_batch=unknown_gt_batch,
            )

        return metrics.compute()

    # ──────────────────────────────────────────────────────────────
    # Prediction decoding
    # ──────────────────────────────────────────────────────────────

    def _decode_predictions(self, raw_output, img_shape):
        """
        Handles FCOS output format: boxes already decoded, no delta decoding needed.
        """
        all_boxes   = raw_output.get("boxes",   [])
        all_scores  = raw_output.get("scores",  [])
        all_labels  = raw_output.get("labels",  [])
        is_unknown  = raw_output.get("is_unknown", None)

        pred_boxes_batch   = []
        pred_scores_batch  = []
        pred_labels_batch  = []
        unknown_pred_batch = []

        for i in range(len(all_boxes)):
            boxes  = all_boxes[i]   # [N, 4]
            scores = all_scores[i]  # [N]
            labels = all_labels[i]  # [N]

            # Confidence filter
            if scores.numel() > 0:
                keep = scores >= self.conf_threshold
                boxes  = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # NMS
            if boxes.shape[0] > 0:
                from torchvision.ops import nms
                keep_idx = nms(boxes, scores, self.nms_threshold)
                boxes  = boxes[keep_idx]
                scores = scores[keep_idx]
                labels = labels[keep_idx]

            pred_boxes_batch.append(boxes.detach().cpu().numpy())
            pred_scores_batch.append(scores.detach().cpu().numpy())
            pred_labels_batch.append(labels.detach().cpu().numpy())

            # Unknown flags per detection
            if is_unknown is not None and is_unknown.numel() > i:
                unk = torch.zeros(boxes.shape[0], dtype=torch.bool,
                                device=boxes.device)
                unknown_pred_batch.append(unk.cpu().numpy())
            else:
                import numpy as np
                unknown_pred_batch.append(
                    np.zeros(boxes.shape[0], dtype=bool)
                )

        return pred_boxes_batch, pred_scores_batch, pred_labels_batch, unknown_pred_batch

    @staticmethod
    def _decode_boxes(proposals, deltas, weights=(1., 1., 1., 1.)):
        """Decode box deltas back to x1y1x2y2 coordinates."""
        px1, py1, px2, py2 = proposals.unbind(1)
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        pw = (px2 - px1).clamp(min=1)
        ph = (py2 - py1).clamp(min=1)

        dx, dy, dw, dh = deltas.unbind(1)

        gcx = pcx + dx / weights[0] * pw
        gcy = pcy + dy / weights[1] * ph
        gw = pw * torch.exp(dw / weights[2])
        gh = ph * torch.exp(dh / weights[3])

        x1 = gcx - gw / 2
        y1 = gcy - gh / 2
        x2 = gcx + gw / 2
        y2 = gcy + gh / 2

        return torch.stack([x1, y1, x2, y2], dim=1)

    # ──────────────────────────────────────────────────────────────
    # Forgetting rate measurement
    # ──────────────────────────────────────────────────────────────

    def measure_forgetting_rate(
        self,
        model,
        data_loader,
        ap_before: Dict[int, float],
    ) -> Dict:
        """
        Re-evaluate on a previously seen dataset after incremental learning.
        Returns forgetting rate metrics.
        """
        metrics = DetectionMetrics(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
        )

        with torch.no_grad():
            model.eval()
            for images, targets in tqdm(data_loader, desc="  [Forgetting Rate]", ncols=80):
                images = images.to(self.device)
                targets = [
                    {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()}
                    for t in targets
                ]
                raw_output = model(images, targets=None)
                pred_boxes_batch, pred_scores_batch, pred_labels_batch, _ = \
                    self._decode_predictions(raw_output, images.shape[-2:])
                gt_boxes_batch = [t["boxes"].cpu().numpy() for t in targets]
                gt_labels_batch = [t["labels"].cpu().numpy() for t in targets]
                metrics.update(
                    pred_boxes_batch, pred_scores_batch, pred_labels_batch,
                    gt_boxes_batch, gt_labels_batch,
                )

        ap_after = metrics.per_class_ap()
        fr = compute_forgetting_rate(ap_before, ap_after)
        return {**fr, "ap_after": ap_after}
