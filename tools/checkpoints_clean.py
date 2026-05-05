#!/usr/bin/env python3
"""
tools/calibrate_dtta.py
========================
Populate DC-TTA statistics from training data.
Run once after loading old checkpoint — takes ~5 minutes.
Saves updated checkpoint with statistics.

Usage:
    python tools/calibrate_dtta.py --config configs/base.yaml
"""

import sys, torch, yaml
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ctta_owod_model import CTTAOWODModel
from data import AntiUAVDataset, AntiUAV410Dataset, build_val_transforms
from data.antiuav_dataset import collate_fn
from data.cst_dataset import CSTAntiUAVDataset
from utils.checkpoint import CheckpointManager


def calibrate(cfg_path: str, checkpoint: str = None, n_batches: int = 200):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CTTAOWODModel(cfg).to(device)

    ckpt_manager = CheckpointManager(cfg["experiment"]["output_dir"])
    ckpt_path    = checkpoint or str(
        Path(cfg["experiment"]["output_dir"]) / "best_model.pth"
    )
    ckpt_manager.load(model, ckpt_path, device=str(device))
    print(f"Loaded: {ckpt_path}")

    val_tf = build_val_transforms()
    model.eval()

    total_batches = 0

    for ds_cfg in cfg["train_datasets"]:
        print(f"\nCalibrating from {ds_cfg['name']}...")
        fmt    = ds_cfg.get("format", "yolo")
        root   = ds_cfg["root"]
        stride = 20   # coarse stride — just need representative statistics

        if fmt == "yolo":
            ds = AntiUAVDataset(root=root, split="train",
                                transforms=val_tf, frame_stride=stride)
        elif fmt == "antiuav410":
            ds = AntiUAV410Dataset(root=root, split="train",
                                   transforms=val_tf, frame_stride=stride)
        elif fmt == "cst":
            ds = CSTAntiUAVDataset(root=root, split="train",
                                   transforms=val_tf, frame_stride=stride,
                                   skip_absent=True)
        else:
            continue

        loader = DataLoader(ds, batch_size=8, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

        batch_count = 0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                feats  = model.backbone(images)
                fpn    = model.neck(feats)

                # DC-TTA statistics
                for level_idx, (k, feat) in enumerate(sorted(fpn.items())):
                    model.tta_adapter.update_train_statistics(
                        feat.float(), level=level_idx
                    )

                # RES class statistics
                global_feat  = fpn["0"].mean(dim=[2, 3])   # [B, C]
                known_labels = torch.zeros(
                    global_feat.shape[0], dtype=torch.long, device=device
                )
                model.energy_detector.update_class_statistics(
                    global_feat.float(), known_labels
                )

                batch_count += 1
                if batch_count >= n_batches // len(cfg["train_datasets"]):
                    break

        print(f"  {batch_count} batches processed")

    # Print statistics summary
    print("\nDC-TTA Statistics Summary:")
    for i, proj in enumerate(model.tta_adapter.projectors):
        if proj.stats_initialized.item():
            print(f"  FPN level {i}: "
                  f"mu_norm={proj.mu.norm():.3f}  "
                  f"sigma_mean={proj.sigma.mean():.3f}  "
                  f"sigma_std={proj.sigma.std():.3f}")

    # Save calibrated checkpoint
    cal_path = str(Path(ckpt_path).with_stem(
        Path(ckpt_path).stem + "_dtta_calibrated"
    ))
    import torch as _torch
    state = _torch.load(ckpt_path, map_location="cpu")
    state["model"] = model.state_dict()
    _torch.save(state, cal_path)
    print(f"\nCalibrated checkpoint saved → {cal_path}")
    print("Use --checkpoint outputs/best_model_dtta_calibrated.pth for evaluation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n_batches",  type=int, default=200)
    args = parser.parse_args()
    calibrate(args.config, args.checkpoint, args.n_batches)