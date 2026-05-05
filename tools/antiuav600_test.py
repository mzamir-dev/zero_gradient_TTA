#!/usr/bin/env python3
"""
tools/antiuav600_test.py — TTA Evaluation on AntiUAV600-validation
====================================================================
Usage:
    # Full comparison: no-TTA vs TTA
    python tools/antiuav600_test.py --config configs/base.yaml \
        --root /path/to/AntiUAV600-validation

    # Different strides
    python tools/antiuav600_test.py --config configs/base.yaml \
        --root /path/to/AntiUAV600-validation --stride 5

    # TTA only (no-TTA already computed)
    python tools/antiuav600_test.py --config configs/base.yaml \
        --root /path/to/AntiUAV600-validation --tta_only

    # With forgetting rate
    python tools/antiuav600_test.py --config configs/base.yaml \
        --root /path/to/AntiUAV600-validation --compute_forgetting
"""

import argparse
import sys
import json
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ctta_owod_model import CTTAOWODModel
from engine.antiuav600_evaluator import AntiUAV600Evaluator
from engine.evaluator import Evaluator
from utils.checkpoint import CheckpointManager
from utils.metrics import measure_fps, measure_gflops, count_parameters


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AntiUAV600 TTA Evaluation"
    )
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--root",       required=True,
                        help="Path to AntiUAV600-validation root directory")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stride",     type=int, default=1,
                        help="Frame stride (1=all frames, 5=every 5th, etc.)")
    parser.add_argument("--tta_only",   action="store_true",
                        help="Run TTA only, use existing JSON for no-TTA")
    parser.add_argument("--compute_forgetting", action="store_true",
                        help="Compute forgetting rate vs Anti-UAV val baseline")
    parser.add_argument("--output",     default=None,
                        help="Override output JSON path")
    return parser.parse_args()


def get_baseline_ap(model, cfg, device) -> dict:
    from data import AntiUAVDataset, build_val_transforms
    from data.antiuav_dataset import collate_fn as antiuav_collate
    from torch.utils.data import DataLoader

    print("\n  Computing baseline AP on Anti-UAV val...")
    ds_cfg = next(
        (d for d in cfg.get("val_datasets", []) if d["name"] == "antiuav"),
        None
    )
    if ds_cfg is None:
        print("  WARNING: antiuav val not found in config")
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

    # Inject dataset config
    cfg["antiuav600_dataset"] = {
        "root":           args.root,
        "frame_stride":   args.stride,
        "conf_threshold": 0.05,
        "nms_threshold":  0.45,
    }

    print("\n" + "="*65)
    print("  CTTA-OWOD — AntiUAV600 TTA Evaluation")
    print(f"  Root         : {args.root}")
    print(f"  Frame stride : {args.stride}")
    print(f"  Device       : {device}")
    print("="*65)

    # Load model
    model = CTTAOWODModel(cfg).to(device)
    ckpt_manager = CheckpointManager(cfg["experiment"]["output_dir"])
    ckpt_path    = args.checkpoint or str(
        Path(cfg["experiment"]["output_dir"]) / "best_model.pth"
    )
    ckpt_manager.load(model, ckpt_path, device=str(device))
    print(f"  Loaded: {ckpt_path}")

    # Efficiency
    params = count_parameters(model)
    gflops = measure_gflops(
        model, img_size=cfg["train"]["img_size"], device=device
    )
    fps = measure_fps(
        model, device, img_size=cfg["train"]["img_size"],
        warmup=20, iters=100,
    )
    print(f"\n  Parameters : {params['total_params_M']}M | "
          f"GFLOPs: {gflops} | FPS: {fps}")

    # Forgetting rate baseline
    baseline_ap = None
    if args.compute_forgetting:
        baseline_ap = get_baseline_ap(model, cfg, device)

    # Output path
    output_path = args.output
    if output_path is None:
        log_dir    = Path(cfg["experiment"]["log_dir"])
        exp_name   = cfg["experiment"]["name"]
        output_path = str(
            log_dir / f"{exp_name}_antiuav600_s{args.stride}.json"
        )

    # Run
    evaluator = AntiUAV600Evaluator(
        model=model,
        cfg=cfg,
        device=device,
        output_path=output_path,
    )

    results = evaluator.run_comparison(
        baseline_ap=baseline_ap,
        tta_only=args.tta_only,
    )

    results["efficiency"] = {
        "fps":                fps,
        "gflops":             gflops,
        "total_params_M":     params["total_params_M"],
        "trainable_params_M": params["trainable_params_M"],
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Final results saved → {output_path}")


if __name__ == "__main__":
    main()
