"""
Thermal-Specific Augmentations for Anti-UAV Detection
All transforms operate on 1-channel (grayscale) thermal images.
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = torch.flip(img, dims=[-1])
            if target["boxes"].numel() > 0:
                W = img.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, 0] = W - target["boxes"][:, 2]
                boxes[:, 2] = W - target["boxes"][:, 0]
                target["boxes"] = boxes
        return img, target


class RandomVerticalFlip:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = torch.flip(img, dims=[-2])
            if target["boxes"].numel() > 0:
                H = img.shape[-2]
                boxes = target["boxes"].clone()
                boxes[:, 1] = H - target["boxes"][:, 3]
                boxes[:, 3] = H - target["boxes"][:, 1]
                target["boxes"] = boxes
        return img, target


class ThermalNoise:
    """Add Gaussian noise to simulate thermal sensor noise."""
    def __init__(self, std_range=(0.01, 0.05), p=0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(img) * std
            img = torch.clamp(img + noise, 0.0, 1.0)
        return img, target


class ThermalContrastJitter:
    """Randomly adjust thermal contrast (simulates emissivity variation)."""
    def __init__(self, factor_range=(0.7, 1.3), p=0.5):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            mean = img.mean()
            img = torch.clamp((img - mean) * factor + mean, 0.0, 1.0)
        return img, target


class ThermalBlur:
    """Gaussian blur to simulate atmospheric scattering (fog, rain)."""
    def __init__(self, kernel_sizes=(3, 5), p=0.3):
        self.kernel_sizes = kernel_sizes
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            # Use F.avg_pool2d as a cheap blur approximation
            img = img.unsqueeze(0)  # [1, 1, H, W]
            img = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
            img = img.squeeze(0)
        return img, target


class RandomCrop:
    """Random crop while keeping all boxes inside."""
    def __init__(self, scale=(0.7, 1.0), p=0.3):
        self.scale = scale
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target

        _, H, W = img.shape
        scale = random.uniform(*self.scale)
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        img = img[:, top:top+crop_h, left:left+crop_w]
        img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear",
                            align_corners=False).squeeze(0)

        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] = (boxes[:, 0] - left).clamp(min=0)
            boxes[:, 1] = (boxes[:, 1] - top).clamp(min=0)
            boxes[:, 2] = (boxes[:, 2] - left).clamp(max=crop_w)
            boxes[:, 3] = (boxes[:, 3] - top).clamp(max=crop_h)

            # Scale back to original size
            scale_x = W / crop_w
            scale_y = H / crop_h
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y

            # Remove boxes that became degenerate
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            target["boxes"] = boxes[keep]
            target["labels"] = target["labels"][keep]

        return img, target


class RandomErasing:
    """Randomly erase a region (simulates partial occlusion of UAV)."""
    def __init__(self, scale=(0.02, 0.1), p=0.2):
        self.scale = scale
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            _, H, W = img.shape
            area = H * W
            erase_area = random.uniform(*self.scale) * area
            aspect = random.uniform(0.3, 3.0)
            eh = int(np.sqrt(erase_area / aspect))
            ew = int(np.sqrt(erase_area * aspect))
            eh = min(eh, H)
            ew = min(ew, W)
            top = random.randint(0, H - eh)
            left = random.randint(0, W - ew)
            img[:, top:top+eh, left:left+ew] = random.random()  # thermal noise fill
        return img, target


def build_train_transforms():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        ThermalContrastJitter(factor_range=(0.7, 1.3), p=0.5),
        ThermalNoise(std_range=(0.005, 0.03), p=0.4),
        ThermalBlur(p=0.2),
        RandomCrop(scale=(0.75, 1.0), p=0.3),
        RandomErasing(p=0.2),
    ])


def build_val_transforms():
    """No augmentation for validation/test."""
    return Compose([])
