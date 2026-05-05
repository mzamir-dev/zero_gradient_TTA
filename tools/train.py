#!/usr/bin/env python3
"""
tools/train.py — Train CTTA-OWOD on Anti-UAV + Anti-UAV410

Usage:
    python tools/train.py --config configs/base.yaml
    python tools/train.py --config configs/base.yaml --resume outputs/checkpoint_epoch0010.pth
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTTA-OWOD")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir

    set_seed(cfg["experiment"]["seed"])

    print("\n" + "="*60)
    print("  CTTA-OWOD Training")
    print(f"  Config: {args.config}")
    print(f"  Experiment: {cfg['experiment']['name']}")
    print(f"  Epochs: {cfg['train']['epochs']}")
    print(f"  Batch size: {cfg['train']['batch_size']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60 + "\n")

    trainer = Trainer(cfg)

    # Resume if requested
    if args.resume:
        start_epoch, _ = trainer.ckpt_manager.load(
            trainer.model, args.resume,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=str(trainer.device),
        )
    trainer.start_epoch = start_epoch + 1  # continue from next epoch
    print(f"[Resume] Continuing from epoch {trainer.start_epoch}")
    
    trainer.train()


if __name__ == "__main__":
    main()
