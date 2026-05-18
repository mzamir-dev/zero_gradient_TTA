"""
Relative Energy Separation (RES) — Novel Unknown Detection
===========================================================
Replaces global energy thresholding with Mahalanobis distance
from per-class feature manifolds.

Core insight:
  Known object  → feature is close to at least one class cluster
  Unknown object → feature is far from ALL known class clusters

Two-stage decision:
  1. d_min = min_k( Mahalanobis(f, mu_k, cov_inv_k) )
     → if d_min > threshold → unknown (far from all classes)

  2. margin = d_second_min - d_min
     → if margin < margin_threshold → ambiguous → also unknown
     → prevents confident assignment to wrong class

This is fundamentally better than energy thresholding because:
  - Energy is a global uncertainty measure, unstable under domain shift
  - RES measures distance to known structure, stable by construction
  - DC-TTA constrains features to training space first, then RES
    checks class membership within that space — two complementary checks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class RelativeEnergySeparation(nn.Module):
    """
    RES: Unknown detection via Mahalanobis distance from class manifolds.

    During training:
      - Accumulates per-class feature means and covariances via EMA

    During inference:
      - Computes Mahalanobis distance from each known class center
      - Flags unknown if min distance > threshold OR margin is too small
      - Adapts threshold using running statistics (no manual tuning)
    """

    def __init__(
        self,
        num_known_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,         # kept for interface compatibility
        num_unknown_prototypes: int = 10,
        adaptive_k: float = 0.5,       # k-sigma for adaptive threshold
        warmup_frames: int = 200,
        momentum: float = 0.99,
        use_margin: bool = True,        # enable margin-based ambiguity detection
        margin_threshold: float = 0.5,  # relative margin below which = ambiguous
        use_mahalanobis: bool = True,   # False = Euclidean distance (faster)
        cov_reg: float = 1e-4,          # regularization for covariance inversion
    ):
        super().__init__()
        self.num_known          = num_known_classes
        self.feature_dim        = feature_dim
        self.adaptive_k         = adaptive_k
        self.warmup_frames      = warmup_frames
        self.momentum           = momentum
        self.use_margin         = use_margin
        self.margin_threshold   = margin_threshold
        self.use_mahalanobis    = use_mahalanobis
        self.cov_reg            = cov_reg

        # ── Per-class statistics (updated during training)
        self.register_buffer(
            "class_means",
            torch.zeros(num_known_classes, feature_dim)
        )
        # Diagonal covariance approximation (full cov too expensive for real-time)
        self.register_buffer(
            "class_vars",
            torch.ones(num_known_classes, feature_dim)
        )
        self.register_buffer(
            "class_counts",
            torch.zeros(num_known_classes)
        )
        self.register_buffer(
            "class_initialized",
            torch.zeros(num_known_classes, dtype=torch.bool)
        )

        # ── Adaptive threshold statistics
        # Running mean/var of d_min values on known samples
        self.register_buffer("running_mean_dist", torch.tensor(0.0))
        self.register_buffer("running_var_dist",  torch.tensor(1.0))
        self.register_buffer("frame_count",       torch.tensor(0))

        # ── Unknown prototype bank (for new class discovery)
        self.register_buffer(
            "unknown_prototypes",
            F.normalize(torch.randn(num_unknown_prototypes, feature_dim), dim=-1)
        )
        self.register_buffer(
            "unknown_counts",
            torch.zeros(num_unknown_prototypes)
        )

        # ── Classifier head (kept for interface compatibility with old code)
        self.classifier = nn.Linear(feature_dim, num_known_classes)
        # Energy network (kept for interface compatibility — outputs RES score)
        self.energy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Training phase: accumulate class statistics
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_class_statistics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        EMA update of per-class mean and variance.
        features: [N, feature_dim]
        labels:   [N] class indices (0 to num_known-1)
        """
        for cls_id in range(self.num_known):
            mask = labels == cls_id
            if mask.sum() == 0:
                continue

            cls_feats = features[mask].float()     # [K, D]
            cls_mean  = cls_feats.mean(dim=0)      # [D]
            cls_var   = cls_feats.var(dim=0, unbiased=False).clamp(min=1e-6)

            m = self.momentum
            if not self.class_initialized[cls_id]:
                self.class_means[cls_id].copy_(cls_mean)
                self.class_vars[cls_id].copy_(cls_var)
                self.class_initialized[cls_id] = True
            else:
                self.class_means[cls_id].mul_(m).add_(cls_mean, alpha=1-m)
                self.class_vars[cls_id].mul_(m).add_(cls_var,   alpha=1-m)

            self.class_counts[cls_id] += mask.sum().float()

    # ──────────────────────────────────────────────────────────────────────
    # Core RES computation
    # ──────────────────────────────────────────────────────────────────────

    def _mahalanobis_distance(
        self,
        features: torch.Tensor,
        class_id: int,
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance from features to class_id center.
        Uses diagonal covariance approximation for efficiency.
        features: [N, D]
        Returns: [N] distances
        """
        mu    = self.class_means[class_id].to(features.device)   # [D]
        var   = self.class_vars[class_id].to(features.device)     # [D]

        diff  = features - mu.unsqueeze(0)                         # [N, D]

        if self.use_mahalanobis:
            # Diagonal Mahalanobis: sum((diff / sigma)^2)
            cov_inv_diag = 1.0 / (var + self.cov_reg)
            dist = (diff * diff * cov_inv_diag.unsqueeze(0)).sum(dim=1)
        else:
            # Euclidean fallback
            dist = (diff * diff).sum(dim=1)

        return dist   # [N]

    def _compute_res_scores(
        self,
        features: torch.Tensor,
    ):
        """
        Compute RES scores for all features.
        features: [N, D]

        Returns:
            d_min:      [N] minimum distance to any class (RES score)
            pred_class: [N] predicted class (argmin distance)
            margin:     [N] d_second_min - d_min (ambiguity measure)
            all_dists:  [N, K] distances to all classes
        """
        initialized = self.class_initialized.to(features.device)

        if not initialized.any():
            # No class statistics yet — return zeros
            N = features.shape[0]
            device = features.device
            return (
                torch.zeros(N, device=device),
                torch.zeros(N, dtype=torch.long, device=device),
                torch.ones(N, device=device),
                torch.zeros(N, self.num_known, device=device),
            )

        # Compute distance to each initialized class
        all_dists = []
        for cls_id in range(self.num_known):
            if initialized[cls_id]:
                d = self._mahalanobis_distance(features, cls_id)
            else:
                # Uninitalized class → assign very large distance
                d = torch.full(
                    (features.shape[0],), 1e9, device=features.device
                )
            all_dists.append(d)

        all_dists = torch.stack(all_dists, dim=1)   # [N, K]

        # Sort distances to get min and second-min
        sorted_dists, _ = all_dists.sort(dim=1)
        d_min       = sorted_dists[:, 0]             # [N] closest class
        d_second    = sorted_dists[:, 1] if self.num_known > 1 \
                      else d_min * 2                  # [N] second closest

        pred_class  = all_dists.argmin(dim=1)        # [N]
        margin      = d_second - d_min               # [N] separation margin

        return d_min, pred_class, margin, all_dists

    # ──────────────────────────────────────────────────────────────────────
    # Adaptive threshold
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_running_stats(self, d_min: torch.Tensor):
        """Update running mean/var of d_min on known samples."""
        batch_mean = d_min.mean()
        batch_var  = d_min.var(unbiased=False) if d_min.numel() > 1 \
                     else torch.tensor(0.0, device=d_min.device)
        m = self.momentum
        self.running_mean_dist = m * self.running_mean_dist + (1-m) * batch_mean
        self.running_var_dist  = m * self.running_var_dist  + (1-m) * batch_var
        self.frame_count += 1

    @property
    def adaptive_threshold(self) -> float:
        std = self.running_var_dist.sqrt().item()
        return self.running_mean_dist.item() + self.adaptive_k * std

    @property
    def threshold_is_active(self) -> bool:
        return self.frame_count.item() >= self.warmup_frames

    # ──────────────────────────────────────────────────────────────────────
    # Forward — main interface
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,
        update_stats: bool = True,
    ):
        """
        RES forward pass.

        Args:
            features:     [N, feature_dim] — pooled proposal/global features
            update_stats: update running threshold statistics

        Returns:
            class_logits: [N, num_known]  — classification scores
            energy:       [N]             — RES score (d_min, replaces energy)
            is_unknown:   [N] bool        — True if unknown
        """
        # Standard classification logits (for detection loss)
        class_logits = self.classifier(features)    # [N, K]

        # RES: compute distances to all class manifolds
        d_min, pred_class, margin, all_dists = \
            self._compute_res_scores(features)

        # Use d_min as the "energy" analog — high distance = unknown
        # This replaces the old energy network output
        energy = d_min

        # Update running statistics on known samples
        if update_stats and not self.training:
            self._update_running_stats(d_min.detach())

        # Unknown decision — two criteria
        if self.threshold_is_active:
            threshold = self.adaptive_threshold

            # Criterion 1: far from ALL known classes
            dist_unknown = d_min > threshold

            # Criterion 2: margin too small → ambiguous between classes
            if self.use_margin:
                # Relative margin: small relative to d_min = ambiguous
                relative_margin = margin / (d_min + 1e-6)
                margin_unknown  = relative_margin < self.margin_threshold
                is_unknown = dist_unknown | margin_unknown
            else:
                is_unknown = dist_unknown
        else:
            # During warmup: never flag unknown
            is_unknown = torch.zeros(
                features.shape[0], dtype=torch.bool, device=features.device
            )

        return class_logits, energy, is_unknown

    # ──────────────────────────────────────────────────────────────────────
    # Interface compatibility methods
    # ──────────────────────────────────────────────────────────────────────

    def energy_loss(
        self,
        features_known: torch.Tensor,
        features_unknown: torch.Tensor = None,
        margin: float = 10.0,
    ) -> torch.Tensor:
        """
        RES training loss: known features should be close to their class centers.
        Replaces energy compactness loss.
        """
        # For known features: d_min should be small
        d_min_known, _, _, _ = self._compute_res_scores(features_known)
        loss = d_min_known.mean()   # minimize distance to nearest class

        if features_unknown is not None and features_unknown.numel() > 0:
            # For unknown: d_min should be large
            d_min_unk, _, _, _ = self._compute_res_scores(features_unknown)
            loss = loss + F.relu(margin - d_min_unk).mean()

        return loss

    @torch.no_grad()
    def update_unknown_prototypes(
        self,
        features: torch.Tensor,
        energy: torch.Tensor,
    ):
        """Update unknown prototype bank with high-distance features."""
        if not self.threshold_is_active:
            return

        unknown_mask = energy > self.adaptive_threshold
        if unknown_mask.sum() == 0:
            return

        unknown_feats = F.normalize(
            features[unknown_mask].detach(), dim=-1
        )
        sims    = torch.mm(unknown_feats, self.unknown_prototypes.t())
        nearest = sims.argmax(dim=1)

        for feat, proto_idx in zip(unknown_feats, nearest):
            self.unknown_prototypes[proto_idx] = F.normalize(
                0.9 * self.unknown_prototypes[proto_idx] + 0.1 * feat,
                dim=-1
            )
            self.unknown_counts[proto_idx] += 1

    def expand_classifier(self, new_num_classes: int):
        """Expand classifier and class statistics for new class."""
        old_weight = self.classifier.weight.data.clone()
        old_bias   = self.classifier.bias.data.clone()
        old_n      = self.num_known

        self.classifier = nn.Linear(
            old_weight.shape[1], new_num_classes
        ).to(old_weight.device)

        with torch.no_grad():
            self.classifier.weight[:old_n] = old_weight
            self.classifier.bias[:old_n]   = old_bias
            nn.init.normal_(self.classifier.weight[old_n:], std=0.01)
            nn.init.zeros_(self.classifier.bias[old_n:])

        # Expand class statistics buffers
        new_means  = torch.zeros(new_num_classes, self.feature_dim,
                                  device=self.class_means.device)
        new_vars   = torch.ones( new_num_classes, self.feature_dim,
                                  device=self.class_vars.device)
        new_counts = torch.zeros(new_num_classes,
                                  device=self.class_counts.device)
        new_init   = torch.zeros(new_num_classes, dtype=torch.bool,
                                  device=self.class_initialized.device)

        new_means[:old_n]  = self.class_means
        new_vars[:old_n]   = self.class_vars
        new_counts[:old_n] = self.class_counts
        new_init[:old_n]   = self.class_initialized

        self.class_means       = new_means
        self.class_vars        = new_vars
        self.class_counts      = new_counts
        self.class_initialized = new_init
        self.num_known         = new_num_classes

    def get_threshold_info(self) -> dict:
        return {
            "adaptive_threshold":    self.adaptive_threshold,
            "running_mean_dist":     self.running_mean_dist.item(),
            "running_std_dist":      self.running_var_dist.sqrt().item(),
            "frame_count":           self.frame_count.item(),
            "threshold_active":      self.threshold_is_active,
            "classes_initialized":   self.class_initialized.sum().item(),
            "use_margin":            self.use_margin,
            "margin_threshold":      self.margin_threshold,
        }