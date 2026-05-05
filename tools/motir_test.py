"""
MOT_IR TTA Evaluator + Test Script
====================================
Usage:
    python tools/motir_test.py --config configs/base.yaml \
        --root /path/to/MOT_IR_sequences

    python tools/motir_test.py --config configs/base.yaml \
        --root /path/to/MOT_IR_sequences --stride 5

    python tools/motir_test.py --config configs/base.yaml \
        --root /path/to/MOT_IR_sequences --tta_only

    python tools/motir_test.py --config configs/base.yaml \
        --root /path/to/MOT_IR_sequences --compute_forgetting
"""

import argparse
import json
import sys
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.motir_dataset import MOTIRDataset, collate_fn
from engine.evaluator import Evaluator
from utils.metrics import DetectionMetrics, compute_forgetting_rate
from models.ctta_owod_model import CTTAOWODModel
from utils.checkpoint import CheckpointManager
from utils.metrics import measure_fps, measure_gflops, count_parameters


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class MOTIREvaluator:

    def __init__(self, model, cfg, device, root, frame_stride,
                 output_path=None):
        self.model        = model
        self.cfg          = cfg
        self.device       = device
        self.root         = root
        self.frame_stride = frame_stride
        self.num_classes  = cfg["num_known_classes"]

        eval_cfg = cfg.get("eval", {})
        self.iou_threshold  = eval_cfg.get("iou_threshold",  0.5)
        self.conf_threshold = cfg.get("motir_dataset", {}).get("conf_threshold", 0.05)
        self.nms_threshold  = cfg.get("motir_dataset", {}).get("nms_threshold",  0.45)
        self.batch_size     = cfg["train"]["batch_size"]
        self.num_workers    = cfg["train"]["num_workers"]

        if output_path is None:
            log_dir    = Path(cfg["experiment"]["log_dir"])
            exp_name   = cfg["experiment"]["name"]
            output_path = str(
                log_dir / f"{exp_name}_motir_s{frame_stride}.json"
            )
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_comparison(self, baseline_ap=None, tta_only=False) -> Dict:
        print(f"\n{'='*65}")
        print(f"  MOT_IR TTA Evaluation  |  stride={self.frame_stride}")
        print(f"{'='*65}")

        dataset = MOTIRDataset(
            root=self.root,
            img_size=self.cfg["train"]["img_size"],
            frame_stride=self.frame_stride,
        )
        if len(dataset) == 0:
            print("  ERROR: No samples loaded.")
            return {}

        results = {
            "experiment":   self.cfg["experiment"]["name"],
            "frame_stride": self.frame_stride,
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {
                "total_frames":  len(dataset),
                "num_sequences": len(dataset.get_sequence_names()),
                "sequences":     dataset.get_sequence_names(),
            },
            "no_tta": {}, "tta": {}, "comparison": {},
        }

        # Load existing no-TTA if tta_only
        if tta_only and self.output_path.exists():
            with open(self.output_path) as f:
                existing = json.load(f)
            results["no_tta"] = existing.get("no_tta", {})
            no_tta_overall = results["no_tta"].get("overall", {})
            print("\n  Using existing no-TTA results:")
            for k in ["mAP@0.5", "precision", "recall", "F1"]:
                print(f"    {k}: {no_tta_overall.get(k, 0.0):.4f}")
        elif not tta_only:
            print("\n  [1/2] Evaluating WITHOUT TTA...")
            self.model.disable_tta()
            no_tta_overall, no_tta_per_seq = self._evaluate_full(
                dataset, use_tta=False
            )
            results["no_tta"] = {
                "overall": no_tta_overall, "per_sequence": no_tta_per_seq
            }
            self._print_metrics("No-TTA", no_tta_overall)
            self._save(results)

        print("\n  [2/2] Evaluating WITH TTA...")
        self.model.enable_tta()
        tta_overall, tta_per_seq = self._evaluate_full(dataset, use_tta=True)
        results["tta"] = {
            "overall": tta_overall, "per_sequence": tta_per_seq
        }
        self._print_metrics("TTA", tta_overall)

        # Forgetting Rate
        if baseline_ap is not None:
            no_tta_overall = results["no_tta"].get("overall", {})
            current_ap = {
                cls_id: no_tta_overall.get(f"AP_class{cls_id}", 0.0)
                for cls_id in range(self.num_classes)
            }
            fr = compute_forgetting_rate(baseline_ap, current_ap)
            results["forgetting_rate"] = fr
            print(f"\n  Forgetting Rate: {fr['forgetting_rate_overall']:.4f}")

        # Adaptation Speed
        print("\n  Computing adaptation speed...")
        adapt_no_tta = self._adaptation_speed(dataset, use_tta=False)
        adapt_tta    = self._adaptation_speed(dataset, use_tta=True)
        results["adaptation_speed"] = {
            "no_tta_frames": adapt_no_tta,
            "tta_frames":    adapt_tta,
        }
        print(f"    No-TTA: {adapt_no_tta} frames  |  TTA: {adapt_tta} frames")

        # Comparison
        no_tta_overall = results["no_tta"].get("overall", {})
        comparison = {}
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            v_base = no_tta_overall.get(k, 0.0)
            v_tta  = tta_overall.get(k, 0.0)
            comparison[k] = {
                "no_tta": v_base, "tta": v_tta,
                "delta": round(v_tta - v_base, 4)
            }
        results["comparison"] = comparison
        self._print_comparison(comparison)
        self._save(results)

        self.model.disable_tta()
        return results

    def _evaluate_full(self, dataset, use_tta):
        loader = DataLoader(
            dataset,
            batch_size=1 if use_tta else self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        base_eval = Evaluator(
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
        label = "TTA" if use_tta else "No-TTA"

        for images, targets in tqdm(loader, desc=f"    [{label}]", ncols=80):
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
                base_eval._decode_predictions(raw_output, images.shape[-2:])
            gt_boxes_b  = [t["boxes"].cpu().numpy() for t in targets]
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

        # Per-sequence breakdown
        per_seq = {}
        print(f"\n  Per-sequence [{label}]:")
        for seq_name, indices in dataset.get_sequence_indices().items():
            if use_tta:
                self.model.reset_tta()
            sm = self._evaluate_subset(dataset, indices, use_tta)
            per_seq[seq_name] = sm
            print(
                f"    seq {seq_name}  "
                f"mAP={sm.get('mAP@0.5',0):.4f}  "
                f"P={sm.get('precision',0):.4f}  "
                f"R={sm.get('recall',0):.4f}  "
                f"F1={sm.get('F1',0):.4f}  "
                f"frames={len(indices)}"
            )

        return overall, per_seq

    def _evaluate_subset(self, dataset, indices, use_tta) -> Dict:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=1 if use_tta else self.batch_size,
            shuffle=False, num_workers=0, collate_fn=collate_fn,
        )
        base_eval = Evaluator(
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
                base_eval._decode_predictions(raw_output, images.shape[-2:])
            gt_boxes_b  = [t["boxes"].cpu().numpy() for t in targets]
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

    def _adaptation_speed(self, dataset, use_tta,
                          target_fraction=0.9, max_frames=300) -> int:
        if use_tta:
            self.model.enable_tta()
            self.model.reset_tta()
        else:
            self.model.disable_tta()

        n_eval    = min(max_frames * 2, len(dataset))
        final_m   = self._evaluate_subset(dataset, list(range(n_eval)), use_tta)
        final_map = final_m.get("mAP@0.5", 0.0)
        target    = final_map * target_fraction

        if final_map == 0:
            return -1

        base_eval = Evaluator(
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

        if use_tta:
            self.model.enable_tta()
            self.model.reset_tta()
        self.model.eval()

        for frame_idx in range(min(max_frames, len(dataset))):
            img_t, tgt = dataset[frame_idx]
            img_input  = img_t.unsqueeze(0).to(self.device)
            tgt_dev    = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in tgt.items()
            }
            if use_tta:
                raw_output = self.model(img_input, targets=None)
            else:
                with torch.no_grad():
                    raw_output = self.model(img_input, targets=None)
            pred_boxes_b, pred_scores_b, pred_labels_b, _ = \
                base_eval._decode_predictions(raw_output, img_input.shape[-2:])
            metrics_acc.update(
                pred_boxes_b, pred_scores_b, pred_labels_b,
                [tgt_dev["boxes"].cpu().numpy()],
                [tgt_dev["labels"].cpu().numpy()],
            )
            if frame_idx >= 5:
                interim = metrics_acc.compute()
                if interim.get("mAP@0.5", 0.0) >= target:
                    self.model.disable_tta()
                    return frame_idx + 1

        self.model.disable_tta()
        return -1

    def _print_metrics(self, label, metrics):
        print(f"\n  [{label}]")
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            if k in metrics:
                print(f"    {k:<30}: {metrics[k]:.4f}")

    def _print_comparison(self, comparison):
        print(f"\n{'='*65}")
        print(f"  COMPARISON: No-TTA vs TTA — MOT_IR")
        print(f"{'='*65}")
        print(f"  {'Metric':<30} {'No-TTA':>10} {'TTA':>10} {'Delta':>10}")
        print(f"  {'-'*62}")
        for k, v in comparison.items():
            sign = "+" if v["delta"] >= 0 else ""
            print(f"  {k:<30} {v['no_tta']:>10.4f} "
                  f"{v['tta']:>10.4f} {sign}{v['delta']:>9.4f}")

    def _save(self, results):
        def _convert(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, dict):           return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):           return [_convert(v) for v in obj]
            return obj
        with open(self.output_path, "w") as f:
            json.dump(_convert(results), f, indent=2)
        print(f"\n  Results saved → {self.output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="MOT_IR TTA Evaluation")
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--root",       required=True,
                        help="Path to MOT_IR_sequences root directory")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stride",     type=int, default=1)
    parser.add_argument("--tta_only",   action="store_true")
    parser.add_argument("--compute_forgetting", action="store_true")
    parser.add_argument("--output",     default=None)
    return parser.parse_args()


def get_baseline_ap(model, cfg, device):
    from data import AntiUAVDataset, build_val_transforms
    from data.antiuav_dataset import collate_fn as antiuav_collate
    from torch.utils.data import DataLoader

    print("\n  Computing baseline AP on Anti-UAV val...")
    ds_cfg = next(
        (d for d in cfg.get("val_datasets", []) if d["name"] == "antiuav"), None
    )
    if ds_cfg is None:
        return None

    ds = AntiUAVDataset(
        root=ds_cfg["root"], split="val",
        transforms=build_val_transforms(),
        frame_stride=ds_cfg.get("frame_stride", 10),
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=2,
                        collate_fn=antiuav_collate)
    evaluator = Evaluator(
        num_classes=cfg["num_known_classes"],
        iou_threshold=cfg["eval"]["iou_threshold"],
        conf_threshold=cfg["eval"]["conf_threshold"],
        nms_threshold=cfg["eval"]["nms_threshold"],
        device=device,
    )
    metrics = evaluator.evaluate(model, loader, desc="  AntiUAV baseline")
    ap = {i: metrics.get(f"AP_class{i}", 0.0)
          for i in range(cfg["num_known_classes"])}
    print(f"  Baseline AP: {ap}")
    return ap


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.setdefault("motir_dataset", {
        "conf_threshold": 0.05,
        "nms_threshold":  0.45,
    })

    print("\n" + "="*65)
    print("  CTTA-OWOD — MOT_IR TTA Evaluation")
    print(f"  Root         : {args.root}")
    print(f"  Frame stride : {args.stride}")
    print(f"  Device       : {device}")
    print("="*65)

    model = CTTAOWODModel(cfg).to(device)
    ckpt_manager = CheckpointManager(cfg["experiment"]["output_dir"])
    ckpt_path    = args.checkpoint or str(
        Path(cfg["experiment"]["output_dir"]) / "best_model.pth"
    )
    ckpt_manager.load(model, ckpt_path, device=str(device))
    print(f"  Loaded: {ckpt_path}")

    from utils.metrics import measure_tta_flops
    import torch.nn as nn

    params    = count_parameters(model)
    gflops    = measure_gflops(model, img_size=cfg["train"]["img_size"], device=device)
    fps       = measure_fps(model, device, img_size=cfg["train"]["img_size"],
                            warmup=20, iters=100)
    tta_flops = measure_tta_flops(model, device, img_size=cfg["train"]["img_size"])

    print(f"\n  Model Efficiency:")
    print(f"    Parameters (total)    : {params['total_params_M']}M")
    print(f"    Parameters (trainable): {params['trainable_params_M']}M")
    print(f"    GFLOPs (full model)   : {gflops}")
    print(f"    FPS                   : {fps}")
    print(f"\n  TTA Efficiency (per frame):")
    print(f"    Detection forward     : {tta_flops['detection_forward_gflops']} GFLOPs")
    print(f"    DC-TTA projection     : {tta_flops['dtta_projection_gflops']} GFLOPs  (clamp only)")
    print(f"    TTA total forward     : {tta_flops['tta_forward_gflops']} GFLOPs")
    print(f"    TTA backward          : {tta_flops['tta_backward_gflops']} GFLOPs  "
        f"({tta_flops['backward_note']})")
    print(f"    ─── Comparison ───")
    print(f"    TENT (entropy min)    : {tta_flops['vs_tent_total_gflops']} GFLOPs  (3× forward)")
    print(f"    Masked reconstruction : {tta_flops['vs_masked_recon_total_gflops']} GFLOPs  (3× forward + adapter)")
    print(f"    DC-TTA (ours)         : {tta_flops['tta_total_gflops']} GFLOPs  (1× forward only)")

    baseline_ap = None
    if args.compute_forgetting:
        baseline_ap = get_baseline_ap(model, cfg, device)

    output_path = args.output
    if output_path is None:
        log_dir    = Path(cfg["experiment"]["log_dir"])
        exp_name   = cfg["experiment"]["name"]
        output_path = str(log_dir / f"{exp_name}_motir_s{args.stride}.json")

    evaluator = MOTIREvaluator(
        model=model, cfg=cfg, device=device,
        root=args.root, frame_stride=args.stride,
        output_path=output_path,
    )

    results = evaluator.run_comparison(
        baseline_ap=baseline_ap, tta_only=args.tta_only
    )

    results["efficiency"] = {
        "fps":                    fps,
        "gflops":                 gflops,
        "total_params_M":         params["total_params_M"],
        "trainable_params_M":     params["trainable_params_M"],
        "tta_forward_gflops":     tta_flops["tta_forward_gflops"],
        "tta_backward_gflops":    tta_flops["tta_backward_gflops"],
        "tta_total_gflops":       tta_flops["tta_total_gflops"],
        "tta_adapter_only_gflops": tta_flops["tta_adapter_only_gflops"],
        "backward_note":          tta_flops["backward_note"],
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Final results saved → {output_path}")


if __name__ == "__main__":
    main()
