#!/usr/bin/env python3
"""
tools/jtduav_test.py — Test on JTDUAV swarm detection dataset
==============================================================
Usage:
    # Full comparison: no-TTA vs TTA
    python tools/jtduav_test.py --config configs/base.yaml

    # With forgetting rate (pass AP from training datasets)
    python tools/jtduav_test.py --config configs/base.yaml --compute_forgetting

    # Specific split
    python tools/jtduav_test.py --config configs/base.yaml --split val

    # Custom checkpoint
    python tools/jtduav_test.py --config configs/base.yaml \
                                 --checkpoint outputs/best_model.pth
"""

import argparse
import sys
import json
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ctta_owod_model import CTTAOWODModel
from engine.jtduav_evaluator import JTDUAVEvaluator
from engine.evaluator import Evaluator
from utils.checkpoint import CheckpointManager
from utils.metrics import measure_fps, measure_gflops, count_parameters


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Test on JTDUAV dataset")
    parser.add_argument("--config",      default="configs/base.yaml")
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--split",       default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--compute_forgetting", action="store_true",
                        help="Compute forgetting rate vs Anti-UAV baseline")
    parser.add_argument("--output",      default=None)
    parser.add_argument("--tta_only", action="store_true",
                    help="Run TTA evaluation only, skip no-TTA baseline")
    return parser.parse_args()


def get_baseline_ap(model, cfg, device) -> dict:
    """
    Compute per-class AP on Anti-UAV val set as forgetting rate baseline.
    """
    from data import AntiUAVDataset, build_val_transforms
    from data.antiuav_dataset import collate_fn as antiuav_collate
    from torch.utils.data import DataLoader

    print("\n  Computing baseline AP on Anti-UAV val for forgetting rate...")

    ds_cfg = next(
        (d for d in cfg.get("val_datasets", []) if d["name"] == "antiuav"),
        None
    )
    if ds_cfg is None:
        print("  WARNING: Anti-UAV val not found in config, skipping forgetting rate")
        return None

    ds = AntiUAVDataset(
        root=ds_cfg["root"], split="val",
        transforms=build_val_transforms(),
        frame_stride=ds_cfg.get("frame_stride", 10),
    )
    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=2,
        collate_fn=antiuav_collate,
    )

    evaluator = Evaluator(
        num_classes=cfg["num_known_classes"],
        iou_threshold=cfg["eval"]["iou_threshold"],
        conf_threshold=cfg["eval"]["conf_threshold"],
        nms_threshold=cfg["eval"]["nms_threshold"],
        device=device,
    )
    metrics = evaluator.evaluate(model, loader, desc="  AntiUAV baseline")
    ap = {
        cls_id: metrics.get(f"AP_class{cls_id}", 0.0)
        for cls_id in range(cfg["num_known_classes"])
    }
    print(f"  Baseline AP: {ap}")
    return ap


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure jtduav_dataset section exists
    cfg.setdefault("jtduav_dataset", {})
    cfg["jtduav_dataset"].setdefault(
        "root",
        "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/MM-AntiUAV-Yolo"
    )
    cfg["jtduav_dataset"].setdefault("frame_stride", 1)
    cfg["jtduav_dataset"].setdefault("conf_threshold", 0.05)
    cfg["jtduav_dataset"].setdefault("nms_threshold", 0.35)
    cfg["jtduav_dataset"].setdefault("max_dets", 100)

    print("\n" + "="*65)
    print("  CTTA-OWOD — JTDUAV Swarm Detection Evaluation")
    print(f"  Split    : {args.split}")
    print(f"  Device   : {device}")
    print(f"  Root     : {cfg['jtduav_dataset']['root']}")
    print("="*65)

    # Load model
    model = CTTAOWODModel(cfg).to(device)
    ckpt_manager = CheckpointManager(cfg["experiment"]["output_dir"])
    ckpt_path    = args.checkpoint or str(
        Path(cfg["experiment"]["output_dir"]) / "best_model.pth"
    )
    ckpt_manager.load(model, ckpt_path, device=str(device))
    model.tta_adapter.initialize_stats_from_checkpoint()
    print(f"  Loaded: {ckpt_path}")

    # Efficiency
    params = count_parameters(model)
    gflops = measure_gflops(model, img_size=cfg["train"]["img_size"], device=device)
    fps    = measure_fps(model, device, img_size=cfg["train"]["img_size"],
                         warmup=20, iters=100)
    print(f"\n  Parameters : {params['total_params_M']}M total | "
          f"{params['trainable_params_M']}M trainable")
    print(f"  GFLOPs     : {gflops}")
    print(f"  FPS        : {fps}")

    # Baseline AP for forgetting rate
    baseline_ap = None
    if args.compute_forgetting:
        baseline_ap = get_baseline_ap(model, cfg, device)

    # Output path
    output_path = args.output
    if output_path is None:
        log_dir    = Path(cfg["experiment"]["log_dir"])
        exp_name   = cfg["experiment"]["name"]
        output_path = str(log_dir / f"{exp_name}_jtduav_{args.split}.json")

    # Run evaluation
    evaluator = JTDUAVEvaluator(
        model=model,
        cfg=cfg,
        device=device,
        split=args.split,
        output_path=output_path,
    )

        # ── TTA only mode — skip no-TTA baseline, use existing results
    if args.tta_only:
        import json as json_mod
        existing = Path(output_path)
        prev_no_tta = {}
        if existing.exists():
            with open(existing) as f:
                prev = json_mod.load(f)
            prev_no_tta = prev.get("no_tta", {}).get("overall", {})
            print("\n  Using existing no-TTA results:")
            for k in ["mAP@0.5", "precision", "recall", "F1"]:
                print(f"    {k}: {prev_no_tta.get(k, 0.0):.4f}")
        else:
            print("\n  WARNING: No existing results found at {output_path}")
            print("  Run without --tta_only first to generate no-TTA baseline.")

        from data.jtduav_dataset import JTDUAVDataset
        dataset = JTDUAVDataset(
            root=cfg["jtduav_dataset"]["root"],
            split=args.split,
            img_size=cfg["train"]["img_size"],
            frame_stride=cfg["jtduav_dataset"].get("frame_stride", 1),
        )
        model.enable_tta()
        tta_metrics, tta_per_video = evaluator._evaluate_full(
            dataset, use_tta=True
        )

        print("\n  [TTA Results]")
        print(f"  {'Metric':<30} {'No-TTA':>10} {'TTA':>10} {'Delta':>10}")
        print(f"  {'-'*62}")
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            no_tta_val = prev_no_tta.get(k, 0.0)
            tta_val    = tta_metrics.get(k, 0.0)
            delta      = tta_val - no_tta_val
            sign       = "+" if delta >= 0 else ""
            print(f"  {k:<30} {no_tta_val:>10.4f} {tta_val:>10.4f} "
                  f"{sign}{delta:>9.4f}")

        # Update and save results JSON
        if existing.exists():
            with open(existing) as f:
                full_results = json_mod.load(f)
        else:
            full_results = {}

        full_results["tta"] = {
            "overall":   tta_metrics,
            "per_video": tta_per_video,
        }
        full_results["efficiency"] = {
            "fps":                fps,
            "gflops":             gflops,
            "total_params_M":     params["total_params_M"],
            "trainable_params_M": params["trainable_params_M"],
        }
        with open(output_path, "w") as f:
            json_mod.dump(full_results, f, indent=2)
        print(f"\n  Results saved → {output_path}")
        return   # ← exit here, skip run_comparison below

    results = evaluator.run_comparison(baseline_ap=baseline_ap)

    # Add efficiency to results and re-save
    results["efficiency"] = {
        "fps":              fps,
        "gflops":           gflops,
        "total_params_M":   params["total_params_M"],
        "trainable_params_M": params["trainable_params_M"],
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Final results saved → {output_path}")


if __name__ == "__main__":
    main()
