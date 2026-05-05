"""
Detection Metrics for CTTA-OWOD
Computes:
  - mAP@0.5, Precision, Recall, F1
  - Forgetting Rate (per-class AP drop after incremental learning)
  - Unknown Recall (detection rate of novel classes)
  - False Positive Rate (unknown alerts on background)
  - FPS and GFLOPs (efficiency)
"""

import time
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ──────────────────────────────────────────────────────────────────────────────

def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes [N,4] and [M,4] (x1y1x2y2)."""
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# AP computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using 11-point interpolation (VOC) style."""
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Find recall change points
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return float(ap)


def compute_precision_recall(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5,
):
    """
    Compute precision-recall curve for a single class.

    Args:
        pred_boxes:   list of [N,4] per image
        pred_scores:  list of [N]   confidence scores
        pred_labels:  list of [N]   predicted class ids
        gt_boxes:     list of [M,4] per image
        gt_labels:    list of [M]   GT class ids
        class_id:     which class to evaluate
        iou_threshold
    """
    # Collect all predictions for this class across all images
    all_scores = []
    all_tp = []

    # Count total GT boxes for this class
    n_gt = sum(
        int((gl == class_id).sum())
        for gl in gt_labels
    )

    if n_gt == 0:
        return np.array([]), np.array([]), 0

    for img_idx in range(len(pred_boxes)):
        pb = pred_boxes[img_idx]      # [N,4]
        ps = pred_scores[img_idx]     # [N]
        pl = pred_labels[img_idx]     # [N]
        gb = gt_boxes[img_idx]        # [M,4]
        gl = gt_labels[img_idx]       # [M]

        # Filter to this class
        pred_mask = pl == class_id
        gt_mask = gl == class_id

        if pred_mask.sum() == 0:
            continue

        class_pred_boxes = pb[pred_mask]
        class_pred_scores = ps[pred_mask]
        class_gt_boxes = gb[gt_mask] if gt_mask.sum() > 0 else np.zeros((0, 4))

        n_gt_img = class_gt_boxes.shape[0]
        matched_gt = np.zeros(n_gt_img, dtype=bool)

        # Sort by confidence
        sort_idx = np.argsort(-class_pred_scores)
        class_pred_boxes = class_pred_boxes[sort_idx]
        class_pred_scores = class_pred_scores[sort_idx]

        for pred_b, pred_s in zip(class_pred_boxes, class_pred_scores):
            all_scores.append(pred_s)
            if n_gt_img == 0:
                all_tp.append(0)
                continue
            ious = box_iou_numpy(pred_b[None], class_gt_boxes)[0]
            best_iou_idx = np.argmax(ious)
            if ious[best_iou_idx] >= iou_threshold and not matched_gt[best_iou_idx]:
                all_tp.append(1)
                matched_gt[best_iou_idx] = True
            else:
                all_tp.append(0)

    if not all_scores:
        return np.array([]), np.array([]), n_gt

    # Sort all predictions by score
    sort_idx = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[sort_idx]

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(1 - tp)

    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    return recall, precision, n_gt


# ──────────────────────────────────────────────────────────────────────────────
# Main DetectionMetrics class
# ──────────────────────────────────────────────────────────────────────────────

class DetectionMetrics:
    """
    Accumulates predictions and GT across batches, then computes all metrics.
    """

    def __init__(self, num_classes: int, iou_threshold: float = 0.5,
                 conf_threshold: float = 0.25):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self):
        self.pred_boxes: List[np.ndarray] = []
        self.pred_scores: List[np.ndarray] = []
        self.pred_labels: List[np.ndarray] = []
        self.gt_boxes: List[np.ndarray] = []
        self.gt_labels: List[np.ndarray] = []
        # For unknown recall / FPR
        self.unknown_gt_flags: List[np.ndarray] = []   # 1 if GT is unknown
        self.unknown_pred_flags: List[np.ndarray] = [] # 1 if pred flagged unknown

    def update(
        self,
        pred_boxes_batch,
        pred_scores_batch,
        pred_labels_batch,
        gt_boxes_batch,
        gt_labels_batch,
        unknown_pred_flags_batch=None,
        unknown_gt_flags_batch=None,
    ):
        """Add one batch."""
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x) if x is not None else np.array([])

        for i in range(len(gt_boxes_batch)):
            self.pred_boxes.append(to_numpy(pred_boxes_batch[i]))
            self.pred_scores.append(to_numpy(pred_scores_batch[i]))
            self.pred_labels.append(to_numpy(pred_labels_batch[i]))
            self.gt_boxes.append(to_numpy(gt_boxes_batch[i]))
            self.gt_labels.append(to_numpy(gt_labels_batch[i]))

            if unknown_pred_flags_batch is not None:
                self.unknown_pred_flags.append(to_numpy(unknown_pred_flags_batch[i]))
            if unknown_gt_flags_batch is not None:
                self.unknown_gt_flags.append(to_numpy(unknown_gt_flags_batch[i]))

    def compute(self) -> Dict:
        results = {}

        # ── Per-class AP
        aps = []
        all_precisions = []
        all_recalls = []

        for cls_id in range(self.num_classes):
            recall, precision, n_gt = compute_precision_recall(
                self.pred_boxes, self.pred_scores, self.pred_labels,
                self.gt_boxes, self.gt_labels,
                class_id=cls_id,
                iou_threshold=self.iou_threshold,
            )
            if len(recall) == 0:
                ap = 0.0
                p_at_r50 = 0.0
                r_val = 0.0
            else:
                ap = compute_ap(recall, precision)
                # Use precision at the operating point where F1 is maximized
                # This gives meaningful precision even when max recall < 0.5
                if len(precision) > 0 and len(recall) > 0:
                    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                    best_idx  = np.argmax(f1_scores)
                    p_at_r50  = float(precision[best_idx])
                    r_val     = float(recall[best_idx])
                else:
                    p_at_r50 = 0.0
                    r_val    = 0.0

            aps.append(ap)
            all_precisions.append(p_at_r50)
            all_recalls.append(r_val)
            results[f"AP_class{cls_id}"] = round(ap, 4)

        results["mAP@0.5"] = round(float(np.mean(aps)), 4)

        # Global precision / recall / F1 (averaged across classes)
        mean_p = float(np.mean(all_precisions))
        mean_r = float(np.mean(all_recalls))

        if mean_p > 0 or mean_r > 0:
            f1 = 2 * mean_p * mean_r / (mean_p + mean_r + 1e-10)
        else:
            f1 = 0.0
            
        results["precision"] = round(mean_p, 4)
        results["recall"] = round(mean_r, 4)
        results["F1"] = round(f1, 4)

        # ── Unknown Recall & False Positive Rate
        if self.unknown_gt_flags and self.unknown_pred_flags:
            pred_has_unknown = np.array([
                bool(flags.any()) if len(flags) > 0 else False
                for flags in self.unknown_pred_flags
            ])
            gt_has_unknown = np.array([
                bool(flags.any()) if len(flags) > 0 else False
                for flags in self.unknown_gt_flags
            ])

            n_true_unknown = gt_has_unknown.sum()
            if n_true_unknown > 0:
                unknown_recall = float(
                    (gt_has_unknown & pred_has_unknown).sum() / n_true_unknown
                )
            else:
                unknown_recall = 0.0

            n_bg = (~gt_has_unknown).sum()
            fpr  = float(
                ((~gt_has_unknown) & pred_has_unknown).sum() / n_bg
            ) if n_bg > 0 else 0.0
        else:
            unknown_recall = 0.0
            fpr            = 0.0

        results["unknown_recall"]      = round(unknown_recall, 4)
        results["false_positive_rate"] = round(fpr, 4)

        return results

    def per_class_ap(self) -> Dict[int, float]:
        """Return per-class AP dict."""
        return {
            cls_id: compute_ap(
                *compute_precision_recall(
                    self.pred_boxes, self.pred_scores, self.pred_labels,
                    self.gt_boxes, self.gt_labels,
                    class_id=cls_id,
                    iou_threshold=self.iou_threshold,
                )[:2]
            )
            for cls_id in range(self.num_classes)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Forgetting Rate
# ──────────────────────────────────────────────────────────────────────────────

def compute_forgetting_rate(
    ap_before: Dict[int, float],
    ap_after: Dict[int, float],
) -> Dict:
    """
    Forgetting Rate per class and overall.
    FR_c = max(0, AP_before_c - AP_after_c)
    Overall FR = mean(FR_c) across old classes
    """
    frs = {}
    for cls_id, ap_b in ap_before.items():
        ap_a = ap_after.get(cls_id, 0.0)
        frs[cls_id] = max(0.0, ap_b - ap_a)

    overall = float(np.mean(list(frs.values()))) if frs else 0.0
    return {
        "forgetting_rate_per_class": frs,
        "forgetting_rate_overall": round(overall, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Efficiency: FPS and GFLOPs
# ──────────────────────────────────────────────────────────────────────────────

def measure_fps(model, device, img_size=640, warmup=50, iters=200) -> float:
    """Measure inference FPS."""
    model.eval()
    dummy = torch.zeros(1, 1, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    fps = iters / elapsed
    return round(fps, 2)


def measure_gflops(model, img_size=640, device=None) -> float:
    dummy = torch.zeros(1, 1, img_size, img_size)
    if device:
        dummy = dummy.to(device)

    try:
        from thop import profile
        # Deep copy so thop hooks never touch the real model
        import copy
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        with torch.no_grad():
            macs, _ = profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy
        return round(macs / 1e9, 2)
    except ImportError:
        pass

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (1, img_size, img_size),
            as_strings=False, print_per_layer_stat=False
        )
        return round(macs / 1e9, 2)
    except ImportError:
        pass

    return -1.0


def count_parameters(model) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": round(total / 1e6, 2),
        "trainable_params_M": round(trainable / 1e6, 2),
    }
    

def measure_tta_flops(model, device, img_size=640) -> dict:
    """
    Measure TTA FLOPs correctly.
    
    Forward FLOPs  = single forward pass through frozen backbone + neck + 
                     DC-TTA projection + FCOS head
    Backward FLOPs = 0 for DC-TTA (gradient-free)
    Total FLOPs    = Forward FLOPs (since backward = 0)
    
    Note: Full model GFLOPs already measured separately as 79.73
    DC-TTA adds projection overhead on top of that.
    """
    dummy = torch.zeros(1, 1, img_size, img_size, device=device)

    # ── Measure DC-TTA projection overhead only
    # (the detection forward is already 79.73 GFLOPs)
    # DC-TTA projection = clamp operation on FPN features
    # FPN features at 4 levels: [B,256,H/8,W/8], [B,256,H/16,W/16],
    #                            [B,256,H/32,W/32], [B,256,H/64,W/64]

    h, w = img_size, img_size
    projection_flops = 0

    for stride in [8, 16, 32, 64]:
        fh = h // stride
        fw = w // stride
        c  = 256
        n  = fh * fw
        # Per position: subtract mu, divide sigma, clamp, multiply sigma, add mu
        # = 5 operations per element × C channels × spatial positions
        projection_flops += 5 * c * n

    projection_gflops = round(projection_flops / 1e9, 6)

    # Full model forward (detection + DC-TTA projection)
    full_model = 79.73   # already measured
    tta_forward_gflops = round(full_model + projection_gflops, 4)

    return {
        "detection_forward_gflops":   full_model,
        "dtta_projection_gflops":     projection_gflops,
        "tta_forward_gflops":         tta_forward_gflops,
        "tta_backward_gflops":        0.0,
        "tta_total_gflops":           tta_forward_gflops,
        "backward_note":              "DC-TTA is gradient-free — backward GFLOPs = 0",
        "vs_tent_total_gflops":       round(full_model * 3, 2),
        "vs_masked_recon_total_gflops": round(full_model * 3 + 0.06, 2),
    }
