"""
CTTA-OWOD with FCOS detection head — correct for tiny UAV objects.
FCOS is anchor-free: predicts (l,r,t,b) distances from each point to box edges.
No IoU matching issues, no anchor size tuning needed.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork

from .modules.tta_adapter import TTAAdapter
from .modules.energy_detector import RelativeEnergySeparation
from .modules.continual_learner import ContinualLearner


# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────

class ThermalBackbone(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True, frozen=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        # Replace first conv for 1-channel thermal input
        old_conv = base.conv1
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        base.conv1 = new_conv

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # stride 4,   256ch
        self.layer2 = base.layer2   # stride 8,   512ch
        self.layer3 = base.layer3   # stride 16, 1024ch
        self.layer4 = base.layer4   # stride 32, 2048ch
        self.out_channels = {"layer2": 512, "layer3": 1024, "layer4": 2048}

        if frozen:
            self._freeze_except_last_n(2)

    def _freeze_except_last_n(self, n):
        layers = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(layers):
            requires = (i >= len(layers) - n)
            for p in layer.parameters():
                p.requires_grad = requires

    def forward(self, x):
        x  = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"layer2": c3, "layer3": c4, "layer4": c5}


# ──────────────────────────────────────────────────────────────────────────────
# FPN Neck
# ──────────────────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        from collections import OrderedDict
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, features):
        from collections import OrderedDict
        ordered = OrderedDict([
            ("0", features["layer2"]),
            ("1", features["layer3"]),
            ("2", features["layer4"]),
        ])
        return self.fpn(ordered)   # returns {"0","1","2","pool"}


# ──────────────────────────────────────────────────────────────────────────────
# FCOS Head — anchor-free, perfect for tiny objects
# ──────────────────────────────────────────────────────────────────────────────

class FCOSHead(nn.Module):
    """
    Per-pixel prediction head.
    At each FPN level pixel predicts:
      - cls_logits:  [B, num_classes, H, W]
      - bbox_pred:   [B, 4, H, W]  (l, r, t, b distances, log-scale)
      - centerness:  [B, 1, H, W]  (how centered the pixel is in GT box)
    """
    def __init__(self, in_channels=256, num_classes=1, num_convs=4):
        super().__init__()
        self.num_classes = num_classes

        cls_tower, bbox_tower = [], []
        for _ in range(num_convs):
            cls_tower  += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                           nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
            bbox_tower += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                           nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]

        self.cls_tower  = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_logits  = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred   = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness  = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Learnable scale per FPN level
        self.scales = nn.Parameter(torch.ones(4))   # 4 FPN levels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Focal-loss style bias init for cls head
        import math
        prior = 0.01
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, features):
        """
        features: OrderedDict {"0","1","2","pool"} from FPN
        Returns lists (one per FPN level):
          cls_logits_list, bbox_pred_list, centerness_list
        """
        cls_logits_list, bbox_pred_list, centerness_list = [], [], []

        for i, (_, feat) in enumerate(features.items()):
            cls_x    = self.cls_tower(feat)
            bbox_x   = self.bbox_tower(feat)

            cls_logits_list.append(self.cls_logits(cls_x))
            # exp + scale ensures positive distances; clamp for stability
            scale = self.scales[min(i, len(self.scales)-1)].abs() + 1e-6
            bbox_pred_list.append(
                torch.exp(self.bbox_pred(bbox_x) * scale).clamp(max=1000)
            )
            centerness_list.append(self.centerness(bbox_x))

        return cls_logits_list, bbox_pred_list, centerness_list


# ──────────────────────────────────────────────────────────────────────────────
# FCOS Loss
# ──────────────────────────────────────────────────────────────────────────────

def fcos_loss(cls_logits_list, bbox_pred_list, centerness_list,
              targets, fpn_strides, img_size=640, num_classes=1):

    device = cls_logits_list[0].device
    total_cls_loss  = torch.tensor(0.0, device=device)
    total_bbox_loss = torch.tensor(0.0, device=device)
    total_ctr_loss  = torch.tensor(0.0, device=device)
    total_pos       = 0

    B = cls_logits_list[0].shape[0]

    for lvl_idx, stride in enumerate(fpn_strides):
        cls_lvl = cls_logits_list[lvl_idx]   # [B, C, H, W]
        bbox_lvl = bbox_pred_list[lvl_idx]   # [B, 4, H, W]
        ctr_lvl  = centerness_list[lvl_idx]  # [B, 1, H, W]

        _, C, H, W = cls_lvl.shape

        # Build pixel grid
        ys = (torch.arange(H, device=device).float() + 0.5) * stride
        xs = (torch.arange(W, device=device).float() + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # [H*W, 2]
        n_points = points.shape[0]

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)   # [N, 4]
            n_gt     = gt_boxes.shape[0]

            cls_target  = torch.zeros(n_points, dtype=torch.long,  device=device)
            bbox_target = torch.zeros(n_points, 4,                  device=device)
            ctr_target  = torch.zeros(n_points,                     device=device)
            pos_mask    = torch.zeros(n_points, dtype=torch.bool,   device=device)

            if n_gt > 0:
                px = points[:, 0].unsqueeze(1)  # [P, 1]
                py = points[:, 1].unsqueeze(1)

                x1 = gt_boxes[:, 0].unsqueeze(0)  # [1, N]
                y1 = gt_boxes[:, 1].unsqueeze(0)
                x2 = gt_boxes[:, 2].unsqueeze(0)
                y2 = gt_boxes[:, 3].unsqueeze(0)

                l = px - x1   # [P, N]
                r = x2 - px
                t = py - y1
                b_dist = y2 - py

                in_box = (l > 0) & (r > 0) & (t > 0) & (b_dist > 0)

                max_dist = torch.stack([l, r, t, b_dist], dim=2).max(dim=2).values
                level_limits = [[0, 64], [64, 128], [128, 256], [256, 1e8]]
                lo, hi = level_limits[min(lvl_idx, 3)]
                in_level = (max_dist >= lo) & (max_dist <= hi)

                valid = in_box & in_level  # [P, N]

                if valid.any():
                    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                               (gt_boxes[:, 3] - gt_boxes[:, 1])
                    areas_exp = gt_areas.unsqueeze(0).expand(n_points, -1).clone()
                    areas_exp[~valid] = 1e8

                    min_area, assigned_gt = areas_exp.min(dim=1)
                    pos_mask = min_area < 1e8

                    if pos_mask.any():
                        ag      = assigned_gt[pos_mask]
                        px_pos  = points[pos_mask, 0]
                        py_pos  = points[pos_mask, 1]

                        bbox_target[pos_mask, 0] = px_pos - gt_boxes[ag, 0]
                        bbox_target[pos_mask, 1] = gt_boxes[ag, 2] - px_pos
                        bbox_target[pos_mask, 2] = py_pos - gt_boxes[ag, 1]
                        bbox_target[pos_mask, 3] = gt_boxes[ag, 3] - py_pos

                        l_ = bbox_target[pos_mask, 0]
                        r_ = bbox_target[pos_mask, 1]
                        t_ = bbox_target[pos_mask, 2]
                        b_ = bbox_target[pos_mask, 3]
                        ctr_target[pos_mask] = torch.sqrt(
                            (torch.min(l_, r_) / (torch.max(l_, r_) + 1e-6)) *
                            (torch.min(t_, b_) / (torch.max(t_, b_) + 1e-6))
                        ).clamp(0, 1)

                        cls_target[pos_mask] = 1

            # Classification loss (Focal)
            cls_pred_flat = cls_lvl[b].permute(1, 2, 0).reshape(-1, C)  # [P, C]
            cls_tgt_onehot = F.one_hot(
                cls_target.clamp(0), num_classes + 1
            )[:, 1:].float()

            total_cls_loss = total_cls_loss + sigmoid_focal_loss(
                cls_pred_flat, cls_tgt_onehot
            )
            total_pos += pos_mask.sum().item()

            if pos_mask.any():
                pred_flat = bbox_lvl[b].permute(1, 2, 0).reshape(-1, 4)
                total_bbox_loss = total_bbox_loss + iou_loss(
                    pred_flat[pos_mask], bbox_target[pos_mask]
                )
                ctr_pred = ctr_lvl[b].reshape(-1)[pos_mask]
                total_ctr_loss = total_ctr_loss + F.binary_cross_entropy_with_logits(
                    ctr_pred, ctr_target[pos_mask]
                )

    norm = max(total_pos, 1)
    return {
        "cls":   total_cls_loss / norm,
        "bbox":  total_bbox_loss / norm,
        "ctr":   total_ctr_loss  / norm,
        "total": total_cls_loss / norm + total_bbox_loss / norm + total_ctr_loss / norm,
    }


def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    p   = torch.sigmoid(pred)
    ce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt  = target * p + (1 - target) * (1 - p)
    fl  = ce * ((1 - pt) ** gamma)
    fl  = alpha * target * fl + (1 - alpha) * (1 - target) * fl
    return fl.sum()


def iou_loss(pred, target, eps=1e-6):
    """GIoU loss between predicted and target (l,r,t,b) distances."""
    pred_area   = (pred[:, 0]  + pred[:, 1])  * (pred[:, 2]  + pred[:, 3])
    tgt_area    = (target[:, 0] + target[:, 1]) * (target[:, 2] + target[:, 3])
    inter_w     = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 1], target[:, 1])
    inter_h     = torch.min(pred[:, 2], target[:, 2]) + torch.min(pred[:, 3], target[:, 3])
    inter_area  = (inter_w * inter_h).clamp(min=0)
    union_area  = pred_area + tgt_area - inter_area + eps
    iou         = inter_area / union_area
    return (1 - iou).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Full CTTA-OWOD Model
# ──────────────────────────────────────────────────────────────────────────────

class CTTAOWODModel(nn.Module):

    FPN_STRIDES = [8, 16, 32, 64]   # matches FPN levels P3-P6

    def __init__(self, cfg):
        super().__init__()
        mcfg        = cfg["model"]
        feature_dim = mcfg["feature_dim"]
        num_classes = cfg["num_known_classes"]

        # Backbone
        self.backbone = ThermalBackbone(
            arch=mcfg["backbone"],
            pretrained=mcfg["pretrained"],
            frozen=mcfg["frozen_backbone"],
        )

        # FPN
        in_ch = list(self.backbone.out_channels.values())
        self.neck = FPNNeck(in_ch, out_channels=feature_dim)

        # TTA adapter
        tta = mcfg["tta"]
        self.tta_adapter = TTAAdapter(
            feature_dim=feature_dim,
            adapter_dim=tta["adapter_dim"],
            ema_decay=tta["ema_decay"],
            lr=tta["lr"],
            entropy_weight=tta["entropy_weight"],
            consistency_weight=tta["consistency_weight"],
            temporal_weight=tta["temporal_weight"],
        )

        # FCOS detection head
        self.fcos_head = FCOSHead(
            in_channels=feature_dim,
            num_classes=num_classes,
            num_convs=4,
        )

        # EMA teacher
        self._build_ema_teacher()

        # Adaptive energy detector
        ecfg = mcfg["energy"]
        self.energy_detector = RelativeEnergySeparation(
        num_known_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dim=ecfg["hidden_dim"],
        num_unknown_prototypes=ecfg["num_unknown_prototypes"],
        adaptive_k=ecfg["adaptive_k"],
        warmup_frames=ecfg["warmup_frames"],
        momentum=ecfg["momentum"],
        )

        # Continual learner
        ccfg = mcfg["continual"]
        self.continual_learner = ContinualLearner(
            feature_dim=feature_dim,
            num_initial_classes=num_classes,
            prompt_dim=ccfg["prompt_dim"],
            memory_bank_size=ccfg["memory_bank_size"],
            replay_epochs=ccfg["replay_epochs"],
            replay_lr=ccfg["replay_lr"],
            min_unknown_samples=ccfg["min_unknown_samples"],
            synthetic_samples_per_class=ccfg["synthetic_samples_per_class"],
        )

        self._tta_enabled = False
        self.num_classes  = num_classes
        self.img_size     = cfg["train"]["img_size"]

    def _build_ema_teacher(self):
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_neck     = copy.deepcopy(self.neck)
        for p in list(self.teacher_backbone.parameters()) + \
                 list(self.teacher_neck.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def update_ema_teacher(self, decay=0.999):
        for t, s in zip(self.teacher_backbone.parameters(),
                        self.backbone.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)
        for t, s in zip(self.teacher_neck.parameters(),
                        self.neck.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)

    def enable_tta(self):
        self._tta_enabled = True
        self.tta_adapter.build_optimizer()
        self.tta_adapter.reset()
        print("[CTTAOWODModel] TTA enabled — adapter reset to identity")

    def disable_tta(self):
        self._tta_enabled = False

    def reset_tta(self):
        self.tta_adapter.reset()
        print("[CTTAOWODModel] TTA adapter reset for new scene")

    def extract_features(self, images, use_tta=False):
        if use_tta and self._tta_enabled:
            # Teacher: always no_grad
            with torch.no_grad():
                t_feats = self.teacher_backbone(images)
                t_fpn   = self.teacher_neck(t_feats)

            # Student: backbone no_grad (frozen), adapter has its own grad graph
            with torch.no_grad():
                s_feats = self.backbone(images)
                s_fpn   = self.neck(s_feats)

            # Adapt each FPN level independently
            adapted_fpn = {}

            # for k in s_fpn:
            #     adapted_fpn[k] = self.tta_adapter.adapt_step(
            #         s_fpn[k], t_fpn[k]
            #     )


            # DC-TTA:
            for level_idx, k in enumerate(sorted(s_fpn.keys())):
                # DC-TTA: gradient-free projection per FPN level
                adapted_fpn[k] = self.tta_adapter.adapt_step(
                    s_fpn[k],
                    t_fpn[k],
                    level=level_idx,
                )


            return adapted_fpn
        else:
            with torch.no_grad():
                feats = self.backbone(images)
                fpn   = self.neck(feats)
            return fpn

    def forward(self, images, targets=None):
        B, _, H, W = images.shape
        fpn_feats = self.extract_features(images, use_tta=self._tta_enabled)

        cls_logits_list, bbox_pred_list, centerness_list = \
            self.fcos_head(fpn_feats)

        # Energy scoring on global average-pooled features
        global_feat = fpn_feats["0"].mean(dim=[2, 3])  # [B, C]
        ow_logits, energy, is_unknown = self.energy_detector(
            global_feat, update_stats=(not self.training)
        )

        if self.training and targets is not None:
            loss_dict = fcos_loss(
                cls_logits_list, bbox_pred_list, centerness_list,
                targets,
                fpn_strides=self.FPN_STRIDES,
                img_size=self.img_size,
                num_classes=self.num_classes,
            )

            # Energy/RES regularization
            known_labels = torch.zeros(B, dtype=torch.long, device=images.device)
            from engine.losses import energy_reg_loss
            e_loss = energy_reg_loss(energy, known_labels,
                                    num_known=self.energy_detector.num_known)
            loss_dict["energy_reg"] = e_loss
            loss_dict["total"]      = loss_dict["total"] + 0.1 * e_loss

            # Update RES class statistics
            with torch.no_grad():
                self.energy_detector.update_class_statistics(
                    global_feat.detach().float(), known_labels
                )

            # Update DC-TTA statistics per FPN level
            with torch.no_grad():
                for level_idx, k in enumerate(sorted(fpn_feats.keys())):
                    self.tta_adapter.update_train_statistics(
                        fpn_feats[k].detach().float(), level=level_idx
                    )

            return loss_dict

        else:
            return self._decode_inference(
                cls_logits_list, bbox_pred_list, centerness_list,
                energy, is_unknown, images.shape
            )

    def _decode_inference(self, cls_logits_list, bbox_pred_list,
                          centerness_list, energy, is_unknown, img_shape):
        B = img_shape[0]
        device = cls_logits_list[0].device
        all_boxes, all_scores, all_labels = [], [], []

        for b in range(B):
            boxes_b, scores_b, labels_b = [], [], []

            for lvl, (cls_l, bbox_l, ctr_l, stride) in enumerate(
                zip(cls_logits_list, bbox_pred_list,
                    centerness_list, self.FPN_STRIDES)
            ):
                _, C, H, W = cls_l.shape
                cls_p = torch.sigmoid(cls_l[b]).permute(1, 2, 0).reshape(-1, C)
                ctr_p = torch.sigmoid(ctr_l[b]).reshape(-1)
                scores_lvl = (cls_p * ctr_p.unsqueeze(1)).max(dim=1).values
                labels_lvl = cls_p.argmax(dim=1)

                ys = (torch.arange(H, device=device).float() + 0.5) * stride
                xs = (torch.arange(W, device=device).float() + 0.5) * stride
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

                bbox_p = bbox_pred_list[lvl][b].permute(1, 2, 0).reshape(-1, 4)
                x1 = (points[:, 0] - bbox_p[:, 0]).clamp(0, img_shape[-1])
                y1 = (points[:, 1] - bbox_p[:, 2]).clamp(0, img_shape[-2])
                x2 = (points[:, 0] + bbox_p[:, 1]).clamp(0, img_shape[-1])
                y2 = (points[:, 1] + bbox_p[:, 3]).clamp(0, img_shape[-2])
                boxes_lvl = torch.stack([x1, y1, x2, y2], dim=1)

                keep = scores_lvl > 0.05
                boxes_b.append(boxes_lvl[keep])
                scores_b.append(scores_lvl[keep])
                labels_b.append(labels_lvl[keep])

            if boxes_b:
                all_boxes.append(torch.cat(boxes_b))
                all_scores.append(torch.cat(scores_b))
                all_labels.append(torch.cat(labels_b))
            else:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))

        return {
            "boxes":      all_boxes,
            "scores":     all_scores,
            "labels":     all_labels,
            "energy":     energy,
            "is_unknown": is_unknown,
            "proposals":  all_boxes,   # alias for evaluator compatibility
            "cls_logits": torch.cat([s.unsqueeze(1) for s in all_scores], dim=0)
                          if all_scores[0].numel() > 0
                          else torch.zeros((0, 1), device=device),
            "bbox_deltas": torch.zeros((0, 4), device=device),
        }

    def maybe_add_new_class(self):
        if self.continual_learner.should_trigger_new_class():
            new_id = self.continual_learner.add_new_class(
                classifier=None,
                energy_detector=self.energy_detector,
            )
            return new_id
        return None

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}