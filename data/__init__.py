from .antiuav_dataset import AntiUAVDataset
from .antiuav410_dataset import AntiUAV410Dataset
from .cst_dataset import CSTAntiUAVDataset
from .transforms import build_train_transforms, build_val_transforms

__all__ = [
    "AntiUAVDataset",
    "AntiUAV410Dataset",
    "CSTAntiUAVDataset",
    "build_train_transforms",
    "build_val_transforms",
]


def build_dataset(cfg_entry, split, transforms=None):
    fmt    = cfg_entry.get("format", "yolo")
    root   = cfg_entry["root"]
    stride = cfg_entry.get("frame_stride", 10)

    if fmt == "yolo":
        return AntiUAVDataset(
            root=root, split=split,
            transforms=transforms, frame_stride=stride
        )
    elif fmt == "antiuav410":
        return AntiUAV410Dataset(
            root=root, split=split,
            transforms=transforms, frame_stride=stride
        )
    elif fmt == "cst":
        return CSTAntiUAVDataset(
            root=root, split=split,
            transforms=transforms,
            frame_stride=stride,
            skip_absent=True,
        )
    else:
        raise ValueError(f"Unknown dataset format: {fmt}")