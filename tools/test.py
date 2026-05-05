#!/usr/bin/env python3
"""
tools/test.py — Test on Anti-UAV, Anti-UAV410 (or any dataset)

Usage:
    # Test on Anti-UAV test split
    python tools/test.py --config configs/base.yaml --dataset antiuav --split test

    # Test on Anti-UAV410 test split
    python tools/test.py --config configs/base.yaml --dataset antiuav410 --split test

    # Use specific checkpoint
    python tools/test.py --config configs/base.yaml --dataset antiuav --split test \
                         --checkpoint outputs/best_model.pth
"""

import argparse
import sys
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import AntiUAVDataset, AntiUAV410Dataset, build_val_transforms
from data.antiuav_dataset import collate_fn
from models.ctta_owod_model import CTTAOWODModel
from engine.evaluator import Evaluator
from utils.logger import TrainingLogger
from utils.metrics import measure_fps, measure_gflops, count_parameters
from utils.checkpoint import CheckpointManager


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Test CTTA-OWOD")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", required=True,
                        choices=["antiuav", "antiuav410"],
                        help="Which dataset to test on")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (default: outputs/best_model.pth)")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("  CTTA-OWOD Testing")
    print(f"  Dataset: {args.dataset} / {args.split}")
    print(f"  Device: {device}")
    print("="*60 + "\n")

    # ── Dataset
    val_tf = build_val_transforms()
    if args.dataset == "antiuav":
        # Find root from config
        ds_root = next(
            d["root"] for d in cfg["train_datasets"] if d["name"] == "antiuav"
        )
        dataset = AntiUAVDataset(root=ds_root, split=args.split, transforms=val_tf)
    else:
        ds_root = next(
            d["root"] for d in cfg["train_datasets"] if d["name"] == "antiuav410"
        )
        dataset = AntiUAV410Dataset(root=ds_root, split=args.split, transforms=val_tf)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"[Test] {len(dataset)} frames")

    # ── Model
    model = CTTAOWODModel(cfg).to(device)
    ckpt_manager = CheckpointManager(cfg["experiment"]["output_dir"])

    ckpt_path = args.checkpoint or str(
        Path(cfg["experiment"]["output_dir"]) / "best_model.pth"
    )
    ckpt_manager.load(model, ckpt_path, device=str(device))

    # ── Efficiency stats
    params = count_parameters(model)
    gflops = measure_gflops(model, img_size=cfg["train"]["img_size"], device=device)
    fps = measure_fps(model, device, img_size=cfg["train"]["img_size"],
                      warmup=cfg["efficiency"]["fps_warmup_iters"],
                      iters=cfg["efficiency"]["fps_eval_iters"])

    print(f"\nEfficiency:")
    print(f"  Parameters : {params['total_params_M']}M total | "
          f"{params['trainable_params_M']}M trainable")
    print(f"  GFLOPs     : {gflops}")
    print(f"  FPS        : {fps}")

    # ── Evaluate
    evaluator = Evaluator(
        num_classes=cfg["num_known_classes"],
        iou_threshold=cfg["eval"]["iou_threshold"],
        conf_threshold=cfg["eval"]["conf_threshold"],
        nms_threshold=cfg["eval"]["nms_threshold"],
        device=device,
    )

    metrics = evaluator.evaluate(
        model, loader,
        desc=f"{args.dataset}/{args.split}"
    )

    # ── Print results
    print(f"\n{'='*50}")
    print(f"Results — {args.dataset} / {args.split}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    # ── Log results
    logger = TrainingLogger(
        log_dir=cfg["experiment"]["log_dir"],
        experiment_name=cfg["experiment"]["name"],
        config=cfg,
    )
    logger.log_test(
        dataset_name=f"{args.dataset}_{args.split}",
        metrics={
            **metrics,
            "fps": fps,
            "gflops": gflops,
            "total_params_M": params["total_params_M"],
            "trainable_params_M": params["trainable_params_M"],
        }
    )
    print(f"\n[Test] Results saved to {logger.log_path}")


if __name__ == "__main__":
    main()
