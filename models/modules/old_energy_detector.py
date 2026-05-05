"""
Adaptive Energy-Based Open-World Detector
Fixes the static threshold=-20 weakness.

Key improvement: threshold = running_mean(E) + k * running_std(E)
The threshold adapts online as the feature distribution shifts during TTA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveEnergyDetector(nn.Module):
    """
    Energy-based unknown object detection with adaptive threshold.

    During training on known classes: energy scores are low (known dist.)
    At test time: high energy → likely unknown UAV type

    Threshold adaptation:
        E_thresh(t) = μ_E(t) + k * σ_E(t)
    where μ, σ are updated via EMA on every inference batch.
    This handles domain shift without manual recalibration.
    """

    def __init__(
        self,
        num_known_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_unknown_prototypes: int = 10,
        adaptive_k: float = 2.0,       # k-sigma rule
        warmup_frames: int = 200,       # collect stats before using threshold
        momentum: float = 0.99,         # EMA momentum for running stats
    ):
        super().__init__()
        self.num_known = num_known_classes
        self.adaptive_k = adaptive_k
        self.warmup_frames = warmup_frames
        self.momentum = momentum

        # Energy network
        self.energy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Known class classifier head
        self.classifier = nn.Linear(feature_dim, num_known_classes)

        # Unknown prototype bank (updated online, not trained by backprop)
        self.register_buffer(
            "unknown_prototypes",
            F.normalize(torch.randn(num_unknown_prototypes, feature_dim), dim=-1)
        )
        self.register_buffer("unknown_counts", torch.zeros(num_unknown_prototypes))

        # ── Adaptive threshold statistics (running EMA)
        self.register_buffer("running_mean_energy", torch.tensor(0.0))
        self.register_buffer("running_var_energy", torch.tensor(1.0))
        self.register_buffer("frame_count", torch.tensor(0))

    # ──────────────────────────────────────────────────────────────
    # Running statistics update
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_running_stats(self, energy: torch.Tensor):
        """EMA update of running mean and variance of energy scores."""
        batch_mean = energy.mean()
        batch_var = energy.var(unbiased=False) if energy.numel() > 1 else torch.tensor(0.0, device=energy.device)

        m = self.momentum
        self.running_mean_energy = m * self.running_mean_energy + (1 - m) * batch_mean
        self.running_var_energy = m * self.running_var_energy + (1 - m) * batch_var
        self.frame_count += 1

    @property
    def adaptive_threshold(self) -> float:
        """Current adaptive threshold value."""
        std = self.running_var_energy.sqrt().item()
        return self.running_mean_energy.item() + self.adaptive_k * std

    @property
    def threshold_is_active(self) -> bool:
        """Only use adaptive threshold after warmup."""
        return self.frame_count.item() >= self.warmup_frames

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────

    def forward(self, features: torch.Tensor, update_stats: bool = True):
        """
        Args:
            features: [N, feature_dim]  — RoI-pooled proposal features
            update_stats: update running mean/var (False during validation)

        Returns:
            class_logits:  [N, num_known]
            energy:        [N]           scalar energy per proposal
            is_unknown:    [N] bool      True if energy > adaptive threshold
        """
        # Classification logits
        class_logits = self.classifier(features)  # [N, num_known]

        # Energy score  (higher = more uncertain/unknown)
        energy = self.energy_net(features).squeeze(-1)  # [N]

        # Update running stats (during inference / TTA)
        if update_stats and not self.training:
            self._update_running_stats(energy.detach())

        # Unknown detection
        if self.threshold_is_active:
            threshold = self.adaptive_threshold
        else:
            # During warmup: use a generous fixed threshold (miss nothing)
            threshold = float("inf")

        is_unknown = energy > threshold

        return class_logits, energy, is_unknown

    # ──────────────────────────────────────────────────────────────
    # Energy loss (used during training to separate known/unknown)
    # ──────────────────────────────────────────────────────────────

    def energy_loss(
        self,
        features_known: torch.Tensor,
        features_unknown: torch.Tensor = None,
        margin: float = 10.0,
    ) -> torch.Tensor:
        """
        Compactness loss:
          - Known features should have LOW energy
          - Unknown features (if provided) should have HIGH energy
        Uses hinge loss: max(0, margin + E_known) + max(0, margin - E_unknown)
        """
        _, energy_known, _ = self(features_known, update_stats=False)
        loss = F.relu(margin + energy_known).mean()

        if features_unknown is not None and features_unknown.numel() > 0:
            _, energy_unknown, _ = self(features_unknown, update_stats=False)
            loss = loss + F.relu(margin - energy_unknown).mean()

        return loss

    # ──────────────────────────────────────────────────────────────
    # Online unknown prototype update
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_unknown_prototypes(
        self,
        features: torch.Tensor,
        energy: torch.Tensor,
    ):
        """
        Update unknown prototype bank with high-energy features.
        Uses momentum update to nearest prototype.
        """
        if not self.threshold_is_active:
            return

        unknown_mask = energy > self.adaptive_threshold
        if unknown_mask.sum() == 0:
            return

        unknown_feats = F.normalize(features[unknown_mask].detach(), dim=-1)

        # Assign each unknown feature to nearest prototype
        sims = torch.mm(unknown_feats, self.unknown_prototypes.t())  # [M, K]
        nearest = sims.argmax(dim=1)                                  # [M]

        for feat, proto_idx in zip(unknown_feats, nearest):
            self.unknown_prototypes[proto_idx] = F.normalize(
                0.9 * self.unknown_prototypes[proto_idx] + 0.1 * feat,
                dim=-1
            )
            self.unknown_counts[proto_idx] += 1

    # ──────────────────────────────────────────────────────────────
    # Expand classifier for new class (continual learning)
    # ──────────────────────────────────────────────────────────────

    def expand_classifier(self, new_num_classes: int):
        """Add new class to classifier head, preserving old weights."""
        old_weight = self.classifier.weight.data.clone()
        old_bias = self.classifier.bias.data.clone()
        old_n = self.num_known

        self.classifier = nn.Linear(
            old_weight.shape[1], new_num_classes
        ).to(old_weight.device)

        # Copy old weights
        with torch.no_grad():
            self.classifier.weight[:old_n] = old_weight
            self.classifier.bias[:old_n] = old_bias
            # Init new class weights with small random values
            nn.init.normal_(self.classifier.weight[old_n:], std=0.01)
            nn.init.zeros_(self.classifier.bias[old_n:])

        self.num_known = new_num_classes

    def get_threshold_info(self) -> dict:
        """Return current threshold statistics for logging."""
        return {
            "adaptive_threshold": self.adaptive_threshold,
            "running_mean_energy": self.running_mean_energy.item(),
            "running_std_energy": self.running_var_energy.sqrt().item(),
            "frame_count": self.frame_count.item(),
            "threshold_active": self.threshold_is_active,
        }
