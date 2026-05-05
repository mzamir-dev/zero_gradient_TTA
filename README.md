# CTTA-OWOD: Continual Test-Time Adaptation for Open-World UAV Detection

## Project Structure
```
ctta_owod/
├── configs/                    # YAML config files
│   ├── base.yaml               # Base config (shared across experiments)
│   ├── antiuav.yaml            # Anti-UAV dataset config
│   ├── antiuav410.yaml         # Anti-UAV410 dataset config
│   └── cst_antiuav.yaml        # CST-Anti-UAV config (TTA evaluation)
│
├── data/                       # Dataset loaders
│   ├── __init__.py
│   ├── antiuav_dataset.py      # Anti-UAV (YOLO format) loader
│   ├── antiuav410_dataset.py   # Anti-UAV410 (x,y,w,h per line) loader
│   ├── cst_dataset.py          # CST-Anti-UAV loader
│   └── transforms.py           # Thermal-specific augmentations
│
├── models/                     # Model components
│   ├── __init__.py
│   ├── backbone/
│   │   ├── __init__.py
│   │   └── resnet_thermal.py   # Frozen backbone with thermal adaptation
│   ├── neck/
│   │   ├── __init__.py
│   │   └── fpn.py              # Feature Pyramid Network
│   ├── head/
│   │   ├── __init__.py
│   │   └── detection_head.py   # Detection head with energy scoring
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── tta_adapter.py      # Test-Time Adaptation module
│   │   ├── energy_detector.py  # Adaptive energy-based open-world detection
│   │   ├── continual_learner.py# Continual learning with thermal diffusion memory
│   │   └── prompt_pool.py      # Class prompt pool
│   └── ctta_owod_model.py      # Full integrated model
│
├── engine/                     # Training & evaluation engines
│   ├── __init__.py
│   ├── trainer.py              # Main trainer (Anti-UAV + Anti-UAV410)
│   ├── evaluator.py            # Detection metrics (mAP, FR, UR, etc.)
│   ├── tta_evaluator.py        # Test-time adaptation evaluator
│   └── losses.py               # All loss functions
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logger.py               # JSON training log writer
│   ├── metrics.py              # mAP, Forgetting Rate, Unknown Recall, FPS, GFLOPs
│   ├── checkpoint.py           # Save/load checkpoints
│   └── visualize.py            # Detection visualization
│
├── tools/                      # Entry-point scripts
│   ├── train.py                # Train on Anti-UAV + Anti-UAV410
│   ├── test.py                 # Test on any dataset
│   ├── tta_test.py             # TTA cross-dataset evaluation
│   └── audit_datasets.py       # Dataset audit (Phase 1)
│
├── scripts/                    # Shell scripts for convenience
│   ├── train_antiuav.sh
│   ├── test_antiuav.sh
│   └── tta_cst.sh
│
├── logs/                       # JSON training logs (auto-created)
└── outputs/                    # Checkpoints & results (auto-created)
```

## Training Pipeline

### Phase 1: Train on Anti-UAV + Anti-UAV410
```bash
python tools/train.py --config configs/base.yaml
```

### Phase 2: Test on Anti-UAV and Anti-UAV410 test sets
```bash
python tools/test.py --config configs/base.yaml --dataset antiuav --split test
python tools/test.py --config configs/base.yaml --dataset antiuav410 --split test
```

### Phase 3: TTA evaluation on CST-Anti-UAV
```bash
python tools/tta_test.py --config configs/cst_antiuav.yaml
```
