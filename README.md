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

* **Dataset - preprocessing**: Simply download the dataset and specify its path in the [**config**](https://github.com/Muhammad-Zamir/zero_gradient_TTA/blob/main/configs/base.ya) file.
  Ensure that the paths to the dataset videos and annotation files are correct. You can modify the bounding box format for the dataset from the [**data**](https://github.com/Muhammad-Zamir/zero_gradient_TTA/tree/main/data) directory.
For the TD-UAV dataset, first extract the video frames and then convert the dataset into YOLO format.”## Implementation

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
If you encounter any issues while running this code, feel free to contact me via my personal email. My email address and other contact details are available on my public GitHub profile page.
