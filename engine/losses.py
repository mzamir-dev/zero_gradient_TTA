"""
Loss functions for CTTA-OWOD detection pipeline.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def detection_loss(cls_logits, bbox_deltas, proposals, targets,
                   bg_iou_thresh=0.1, fg_iou_thresh=0.4,
                   num_samples=256, pos_fraction=0.25):

    device = cls_logits.device
    all_cls_targets = []
    all_reg_targets = []
    all_pos_mask = []
    prop_offset = 0

    for i, tgt in enumerate(targets):
        gt_boxes = tgt["boxes"].to(device)
        gt_labels = tgt["labels"].to(device)
        n_props = proposals[i].shape[0] if proposals[i].numel() > 0 else 0

        if n_props == 0:
            prop_offset += 0
            continue

        props = proposals[i].to(device)

        if gt_boxes.numel() == 0:
            cls_targets = torch.zeros(n_props, dtype=torch.long, device=device)
            reg_targets = torch.zeros(n_props, 4, device=device)
            pos_mask = torch.zeros(n_props, dtype=torch.bool, device=device)
        else:
            iou_mat = box_iou(props, gt_boxes)
            max_iou, matched_gt = iou_mat.max(dim=1)

            # DEBUG — remove after confirming correct
            print(f"    [DEBUG] img={i} gt={gt_boxes.shape[0]} "
                  f"props={n_props} "
                  f"max_iou={iou_mat.max():.4f} "
                  f"mean_iou={iou_mat.mean():.4f} "
                  f"fg_count={(max_iou >= fg_iou_thresh).sum()}")

            cls_targets = torch.zeros(n_props, dtype=torch.long, device=device)
            pos_mask = max_iou >= fg_iou_thresh
            cls_targets[pos_mask] = gt_labels[matched_gt[pos_mask]] + 1
            ignore_mask = (max_iou >= bg_iou_thresh) & (~pos_mask)
            cls_targets[ignore_mask] = -1

            matched_boxes = gt_boxes[matched_gt]
            reg_targets = encode_boxes(props, matched_boxes)
            reg_targets[~pos_mask] = 0

        all_cls_targets.append(cls_targets)
        all_reg_targets.append(reg_targets)
        all_pos_mask.append(pos_mask)
        prop_offset += n_props

    if not all_cls_targets:
        return cls_logits.sum() * 0.0

    cls_tgts = torch.cat(all_cls_targets, dim=0)
    reg_tgts = torch.cat(all_reg_targets, dim=0)
    pos_msk = torch.cat(all_pos_mask, dim=0)

    valid = cls_tgts >= 0
    if valid.sum() > 0:
        cls_loss = F.cross_entropy(cls_logits[valid], cls_tgts[valid])
    else:
        cls_loss = cls_logits.sum() * 0.0

    if pos_msk.sum() > 0 and bbox_deltas.shape[0] > pos_msk.shape[0]:
        bbox_deltas_pos = bbox_deltas[:pos_msk.shape[0]][pos_msk]
        reg_loss = F.smooth_l1_loss(bbox_deltas_pos, reg_tgts[pos_msk])
    elif pos_msk.sum() > 0:
        reg_loss = F.smooth_l1_loss(bbox_deltas[pos_msk], reg_tgts[pos_msk])
    else:
        reg_loss = bbox_deltas.sum() * 0.0

    return cls_loss + reg_loss


def energy_reg_loss(energy, labels, num_known, margin=10.0):
    """
    Energy compactness loss:
      Known proposals → low energy (below -margin)
      Forces a clear separation between known and background in energy space.
    """
    # All current proposals are known classes (training phase)
    known_mask = labels < num_known
    if known_mask.sum() == 0:
        return energy.sum() * 0.0

    loss = F.relu(margin + energy[known_mask]).mean()
    return loss


def encode_boxes(proposals, gt_boxes, weights=(1., 1., 1., 1.)):
    """Encode GT boxes as deltas relative to proposals."""
    ex_x1, ex_y1, ex_x2, ex_y2 = proposals.unbind(1)
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes.unbind(1)

    ex_cx = (ex_x1 + ex_x2) / 2
    ex_cy = (ex_y1 + ex_y2) / 2
    ex_w = (ex_x2 - ex_x1).clamp(min=1)
    ex_h = (ex_y2 - ex_y1).clamp(min=1)

    gt_cx = (gt_x1 + gt_x2) / 2
    gt_cy = (gt_y1 + gt_y2) / 2
    gt_w = (gt_x2 - gt_x1).clamp(min=1)
    gt_h = (gt_y2 - gt_y1).clamp(min=1)

    dx = weights[0] * (gt_cx - ex_cx) / ex_w
    dy = weights[1] * (gt_cy - ex_cy) / ex_h
    dw = weights[2] * torch.log(gt_w / ex_w)
    dh = weights[3] * torch.log(gt_h / ex_h)

    return torch.stack([dx, dy, dw, dh], dim=1)
