# Zero gradient Test-Time Adaptation for Open-World Anti-UAV Detection
## Installation

[**Python>=3.10.0**](https://www.python.org/) is required 

```bash
conda create -n "environment_name" python=3.12
conda activate "environment_name" 
pip install -r requirements.txt
```

## Data Preparation

We currently support [**AntiUAV-MUAV15**](https://github.com/Shihan0325/Anti-MUAV15), [**TDUAV**](https://huggingface.co/datasets/yifwang/MM-AntiUAV/tree/main), [**CST-AntiUAV**](https://github.com/PCwenyue/CST-Anti-UAV/blob/main/README.md), and [**Anti-UAV600**](https://github.com/xuefeng-zhu5/EDTC) datasets. Follow the instructions below to prepare datasets.

* **Dataset - preprocessing**: Simply downlaod the dataset and paste the dataset path in the config/base.yaml file.
  Make sure the path for dataset videos and annotations are correct. You can change the dataset bounding box format from the [**data**](https://github.com/Muhammad-Zamir/zero_gradient_TTA/tree/main/data)
  For TD-UAV dataset extract the frames from the videos and convert dataset into yolo format,
## Implementation

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
### Contact dettails:
If you face any problem while running this code you can reach my via my personal email. Email address and others contact details are mentioned in my public github profile page. 
