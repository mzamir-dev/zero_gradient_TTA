# Zero gradient Test-Time Adaptation for Open-World Anti-UAV Detection

## Training Pipeline

### Phase 1: Train on Anti-UAV + Anti-UAV410
```bash
python tools/train.py --config configs/base.yaml
```

### Phase 2: TTA evaluation TDUAV
```bash
python tools/jtduav_test.py --config configs/base.yaml --tta_only
```
```bash
python tools/antiuav600_test.py --config configs/base.yaml --tta_only
```
```bash
python tools/tta_cst.py --config configs/base.yaml --tta_only
```
```bash
python tools/motir_test.py --config configs/base.yaml --tta_only
```
