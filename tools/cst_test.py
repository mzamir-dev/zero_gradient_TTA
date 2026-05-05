#!/usr/bin/env python3
"""
tools/cst_test.py — Test trained model on CST-Anti-UAV
========================================================
Usage:
    # Test on CST test split (no TTA — baseline)
    python tools/cst_test.py --config configs/base.yaml

    # Test on CST test split WITH TTA
    python tools/cst_test.py --config configs/base.yaml --tta

    # Test on val split
    python tools/cst_test.py --config configs/base.yaml --split val

    # Use specific checkpoint
    python tools/cst_test.py --config configs/base.yaml \
                              --checkpoint outputs/best_model.pth

    # Run both with and without TTA for comparison
    python tools/cst_test.py --config configs/base.yaml --compare
"""

import argparse
import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ctta_owod_model import CTTAOWODModel
from engine.cst_evaluator import CSTEvaluator
from utils.checkpoint import CheckpointManager
from utils.metrics import measure_fps, measure_gflops, count_parameters


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Test on CST-Anti-UAV")
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: outputs/best_model.pth)")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--tta",        action="store_true",
                        help="Enable Test-Time Adaptation")
    parser.add_argument("--compare",    action="store_true",
                        help="Run both no-TTA and TTA for comparison")
    parser.add_argument("--cst_root",   default=None,
                        help="Override CST-Anti-UAV root path")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Frame stride for CST dataset (1=all frames)")
    parser.add_argument("--output",     default=None,
                        help="Override output JSON path")
    return parser.parse_args()


def run_evaluation(model, cfg, device, split, use_tta, output_path=None):
    evaluator = CSTEvaluator(
        model=model,
        cfg=cfg,
        device=device,
        split=split,
        use_tta=use_tta,
        output_path=output_path,
    )
    return evaluator.run()


# Visualize 5 CST predictions to see what model outputs
import cv2
import numpy as np
from data.cst_dataset import CSTAntiUAVDataset

def visualize_predictions(model, cfg, device, n=5, indices=None):
    from data.cst_dataset import CSTAntiUAVDataset
    ds = CSTAntiUAVDataset(
        root=cfg["cst_dataset"]["root"],
        split="test", skip_absent=True
    )
    model.eval()

    if indices is None:
        step = max(1, len(ds) // n)
        indices = list(range(0, min(n * step, len(ds)), step))

    for i in indices:
        img_tensor, target = ds[i]
        img_input = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_input)
        
        boxes  = out["boxes"][0]
        scores = out["scores"][0]
        
        img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # Draw GT box
        if target["boxes"].shape[0] > 0:
            b = target["boxes"][0].numpy().astype(int)
            cv2.rectangle(img_bgr, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)
        
        # Draw all predictions above 0.01 confidence
        for j, (box, score) in enumerate(zip(boxes, scores)):
            if score > 0.01:
                b = box.cpu().numpy().astype(int)
                cv2.rectangle(img_bgr, (b[0],b[1]), (b[2],b[3]), (0,0,255), 1)
                cv2.putText(img_bgr, f"{score:.2f}", (b[0],b[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        save_path = f"outputs/cst_debug_{i}.jpg"
        cv2.imwrite(save_path, img_bgr)
        print(f"  Saved {save_path} | GT boxes: {target['boxes'].shape[0]} | "
              f"Preds above 0.01: {(scores > 0.01).sum()}")


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply overrides
    if args.cst_root:
        cfg.setdefault("cst_dataset", {})["root"] = args.cst_root
    if args.frame_stride != 1:
        cfg.setdefault("cst_dataset", {})["frame_stride"] = args.frame_stride

    # Ensure cst_dataset section exists with defaults
    cfg.setdefault("cst_dataset", {}).setdefault(
        "root",
        "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/CST-AntiUAV"
    )
    cfg["cst_dataset"].setdefault("frame_stride", args.frame_stride)

    print("\n" + "="*65)
    print("  CTTA-OWOD — CST-Anti-UAV Evaluation")
    print(f"  Split    : {args.split}")
    print(f"  TTA      : {'ON' if args.tta else 'OFF'}")
    print(f"  Device   : {device}")
    print(f"  CST root : {cfg['cst_dataset']['root']}")
    print("="*65)

    # ── Load model
    model = CTTAOWODModel(cfg).to(device)
    ckpt_manager = CheckpointManager(cfg["experiment"]["output_dir"])
    ckpt_path = args.checkpoint or str(
        Path(cfg["experiment"]["output_dir"]) / "best_model.pth"
    )
    ckpt_manager.load(model, ckpt_path, device=str(device))
    model.tta_adapter.initialize_stats_from_checkpoint()

    # #visualize_predictions(model, cfg, device)  # Debug: visualize some predictions before evaluation
    # visualize_predictions(model, cfg, device, indices=[0, 50, 344, 640, 740])  # Visualize specific indices for debugging
    # import sys; 
    # sys.exit(0) 

    stats_ready = model.tta_adapter.stats_initialized.item()
    print(f"  [TTA Mode] {'Full gradient alignment' if stats_ready else 'BN statistics accumulation (safe mode)'}")

    # ── Efficiency stats
    params = count_parameters(model)
    gflops = measure_gflops(
        model, img_size=cfg["train"]["img_size"], device=device
    )
    fps = measure_fps(
        model, device,
        img_size=cfg["train"]["img_size"],
        warmup=cfg["efficiency"]["fps_warmup_iters"],
        iters=cfg["efficiency"]["fps_eval_iters"],
    )
    print(f"\n  Parameters : {params['total_params_M']}M total | "
          f"{params['trainable_params_M']}M trainable")
    print(f"  GFLOPs     : {gflops}")
    print(f"  FPS        : {fps}")

    if args.compare:
        # ── Run both no-TTA and TTA
        log_dir  = Path(cfg["experiment"]["log_dir"])
        exp_name = cfg["experiment"]["name"]

        print("\n  [1/2] Running WITHOUT TTA (baseline)...")
        out_no_tta = str(
            log_dir / f"{exp_name}_cst_{args.split}_no_tta.json"
        )
        results_no_tta = run_evaluation(
            model, cfg, device, args.split,
            use_tta=False, output_path=out_no_tta
        )

        print("\n  [2/2] Running WITH TTA...")
        out_tta = str(
            log_dir / f"{exp_name}_cst_{args.split}_tta.json"
        )
        results_tta = run_evaluation(
            model, cfg, device, args.split,
            use_tta=True, output_path=out_tta
        )

        # Print side-by-side comparison
        print("\n" + "="*65)
        print("  COMPARISON: No-TTA vs TTA")
        print("="*65)
        print(f"  {'Metric':<30} {'No-TTA':>10} {'TTA':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        for k in ["mAP@0.5", "precision", "recall", "F1",
                  "unknown_recall", "false_positive_rate"]:
            v_base = results_no_tta.get("overall", {}).get(k, 0.0)
            v_tta  = results_tta.get("overall", {}).get(k, 0.0)
            delta  = v_tta - v_base
            sign   = "+" if delta >= 0 else ""
            print(f"  {k:<30} {v_base:>10.4f} {v_tta:>10.4f} "
                  f"{sign}{delta:>9.4f}")

    else:
        # ── Single run
        output_path = args.output
        run_evaluation(
            model, cfg, device, args.split,
            use_tta=args.tta, output_path=output_path
        )


if __name__ == "__main__":
    main()
