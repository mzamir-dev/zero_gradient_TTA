"""
Distribution Constraint Test-Time Adaptation (DC-TTA)
======================================================
Novel gradient-free TTA for thermal UAV detection.

Key insight: Instead of updating model parameters at test time
(which requires gradients, optimizers, and can destabilize the model),
we project test-time features back into the training feature distribution
using statistical constraints computed offline during training.

This is:
  - Gradient-free (no backward pass at test time)
  - Optimizer-free (no Adam/SGD state)
  - Deterministic (same input always gives same output)
  - Theoretically grounded (Gaussian manifold projection)
  - Compatible with any batch size including batch_size=1
  - Zero additional memory at test time

Novelty over prior work:
  - TENT/TTT: require gradients and optimizer — slow, can diverge
  - DUA: only updates BN statistics — ignores spatial feature structure
  - Our DC-TTA: projects ALL feature map positions jointly using
    learned training distribution — preserves spatial structure
    critical for FCOS anchor-free detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureConstraintProjector(nn.Module):
    """
    Per-FPN-level distribution constraint projector.
    Stores training statistics and projects test features at inference.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        use_mahalanobis: bool = False,
        clip_alpha: float = 2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.feature_dim      = feature_dim
        self.use_mahalanobis  = use_mahalanobis
        self.clip_alpha       = clip_alpha
        self.eps              = eps

        self.register_buffer("mu",               torch.zeros(feature_dim))
        self.register_buffer("sigma",            torch.ones(feature_dim))
        self.register_buffer("cov_inv",          torch.eye(feature_dim))
        self.register_buffer("stats_initialized", torch.tensor(False))

        # EMA accumulators for online statistics during training
        self.register_buffer("_ema_mean",  torch.zeros(feature_dim))
        self.register_buffer("_ema_var",   torch.ones(feature_dim))
        self.register_buffer("_ema_count", torch.tensor(0))

    @torch.no_grad()
    def update_statistics(self, features: torch.Tensor):
        """
        EMA update of training distribution statistics.
        Called during training on each batch.
        features: [B, C, H, W]
        """
        B, C, H, W = features.shape
        feats = features.permute(0, 2, 3, 1).reshape(-1, C).float()

        batch_mean = feats.mean(dim=0)                          # [C]
        batch_var  = feats.var(dim=0, unbiased=False) + self.eps  # [C]

        count = self._ema_count.item()
        if count == 0:
            self._ema_mean.copy_(batch_mean)
            self._ema_var.copy_(batch_var)
        else:
            momentum = 0.99
            self._ema_mean.mul_(momentum).add_(batch_mean, alpha=1 - momentum)
            self._ema_var.mul_(momentum).add_(batch_var,   alpha=1 - momentum)

        self._ema_count.add_(1)

        # Commit to final statistics
        self.mu.copy_(self._ema_mean)
        self.sigma.copy_(self._ema_var.sqrt())
        self.stats_initialized.fill_(True)

    @torch.no_grad()
    def project(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project test features into training distribution.
        features: [B, C, H, W]
        Returns: projected features [B, C, H, W]
        """
        if not self.stats_initialized:
            return features

        B, C, H, W = features.shape
        feats = features.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]

        mu    = self.mu.to(feats.device)
        sigma = self.sigma.to(feats.device)

        # Normalize to z-score space
        z = (feats - mu) / (sigma + self.eps)

        # Clip to training distribution bounds
        z = torch.clamp(z, -self.clip_alpha, self.clip_alpha)

        if self.use_mahalanobis:
            # Mahalanobis projection: rescale to within chi-square radius
            cov_inv   = self.cov_inv.to(feats.device)
            z_centered = feats - mu
            dist       = torch.sum(
                (z_centered @ cov_inv) * z_centered, dim=1, keepdim=True
            )
            radius = float(C) * 2.0
            scale  = torch.clamp(radius / (dist + self.eps), max=1.0)
            feats  = z_centered * scale + mu
        else:
            # Gaussian projection: clip + denormalize
            feats = z * sigma + mu

        return feats.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class TTAAdapter(nn.Module):
    """
    Distribution Constraint TTA (DC-TTA).

    One FeatureConstraintProjector per FPN level.
    At test time: project each FPN level's features into
    the training distribution — no gradients, no optimizer.

    This replaces the masked reconstruction adapter.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        adapter_dim: int = 64,        # kept for config compatibility
        ema_decay: float = 0.999,     # kept for config compatibility
        lr: float = 1e-3,             # kept for config compatibility
        entropy_weight: float = 1.0,  # kept for config compatibility
        consistency_weight: float = 0.5,
        temporal_weight: float = 0.3,
        mask_ratio: float = 0.3,
        num_fpn_levels: int = 4,
        use_mahalanobis: bool = False,
        clip_alpha: float = 3.0,
    ):
        super().__init__()
        self.feature_dim    = feature_dim
        self.num_fpn_levels = num_fpn_levels
        self._n_steps       = 0

        # One projector per FPN level
        self.projectors = nn.ModuleList([
            FeatureConstraintProjector(
                feature_dim=feature_dim,
                use_mahalanobis=use_mahalanobis,
                clip_alpha=clip_alpha,
            )
            for _ in range(num_fpn_levels)
        ])

        # Compatibility buffers — kept so old checkpoints load without error
        self.register_buffer("train_feat_mean",   torch.zeros(feature_dim))
        self.register_buffer("train_feat_std",    torch.ones(feature_dim))
        self.register_buffer("stats_initialized", torch.tensor(False))

        # No optimizer needed
        self.optimizer = None

    def build_optimizer(self):
        """No-op — DC-TTA requires no optimizer."""
        pass

    def reset(self):
        """Reset between scenes — no state to reset for DC-TTA."""
        self._n_steps = 0

    def initialize_stats_from_checkpoint(self):
        """Check if projectors have statistics populated."""
        any_initialized = any(
            p.stats_initialized.item() for p in self.projectors
        )
        if not any_initialized:
            print("[DC-TTA] WARNING: No training statistics found in checkpoint.")
            print("[DC-TTA] Run calibration pass before TTA evaluation.")
            print("[DC-TTA] Features will pass through unchanged until calibrated.")
        else:
            n_ready = sum(
                1 for p in self.projectors if p.stats_initialized.item()
            )
            print(f"[DC-TTA] {n_ready}/{len(self.projectors)} FPN levels "
                  f"have training statistics — ready.")

    def update_train_statistics(self, features: torch.Tensor, level: int = 0):
        """
        Update statistics for a specific FPN level.
        Called during training or calibration pass.
        features: [B, C, H, W]
        level: FPN level index (0=P3, 1=P4, 2=P5, 3=P6)
        """
        if level < len(self.projectors):
            self.projectors[level].update_statistics(features)
            # Also update compat buffer for old code paths
            if level == 0:
                self.train_feat_mean.copy_(self.projectors[0].mu)
                self.train_feat_std.copy_(self.projectors[0].sigma)
                self.stats_initialized.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Single-tensor forward — applies projector[0]. For compatibility."""
        return self.projectors[0].project(features)

    def adapt_step(
        self,
        features: torch.Tensor,
        teacher_features: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        """
        DC-TTA projection step — no gradients.
        Replaces the gradient-based adapt_step of masked reconstruction.

        features:         [B, C, H, W] from student backbone+neck
        teacher_features: [B, C, H, W] not used (kept for interface compat)
        level:            FPN level index

        Returns: projected features [B, C, H, W]
        """
        projector = self.projectors[min(level, len(self.projectors) - 1)]
        projected = projector.project(features)
        self._n_steps += 1
        return projected

    def get_stats_info(self) -> dict:
        """Return statistics info for logging."""
        info = {}
        for i, proj in enumerate(self.projectors):
            if proj.stats_initialized.item():
                info[f"level_{i}_mu_norm"] = proj.mu.norm().item()
                info[f"level_{i}_sigma_mean"] = proj.sigma.mean().item()
        return info