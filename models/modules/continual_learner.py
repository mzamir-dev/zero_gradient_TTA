"""
Continual Learner — Thermal-Aware Feature-Space Memory

Fixes the core weakness of pixel-space diffusion for thermal imagery:
  - Stable Diffusion is trained on RGB images and CANNOT generate valid thermal frames
  - Instead, we store class-conditional feature statistics (mean + covariance)
    and synthesize replay features directly in feature space
  - This is both more accurate (thermal features captured from real data)
    and more efficient (no diffusion model required)

Memory strategy:
  - For each known class: maintain a Gaussian mixture in feature space
  - At replay time: sample from the mixture → synthesize feature vectors
  - These synthetic features are used to train the classifier head
    (backbone frozen, no pixel generation needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import copy


class FeatureMemoryBank:
    """
    Per-class feature statistics for replay.
    Stores: reservoir sample of real features (bounded by bank_size)
    and running Gaussian statistics for efficient synthesis.
    """

    def __init__(self, feature_dim: int, bank_size: int = 200):
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        # class_id → tensor [K, feature_dim]
        self.reservoir: Dict[int, torch.Tensor] = {}
        # class_id → (mean [D], cov_diag [D])
        self.statistics: Dict[int, dict] = {}
        self._counts: Dict[int, int] = {}

    @torch.no_grad()
    def update(self, class_id: int, features: torch.Tensor):
        """
        Reservoir sampling update for class_id.
        features: [N, D]
        """
        features = features.detach().cpu()
        n_new = features.shape[0]

        if class_id not in self.reservoir:
            self.reservoir[class_id] = features[:self.bank_size]
            self._counts[class_id] = n_new
        else:
            existing = self.reservoir[class_id]
            combined = torch.cat([existing, features], dim=0)
            if combined.shape[0] > self.bank_size:
                # Reservoir: keep random subset
                idx = torch.randperm(combined.shape[0])[:self.bank_size]
                combined = combined[idx]
            self.reservoir[class_id] = combined
            self._counts[class_id] = self._counts.get(class_id, 0) + n_new

        # Update Gaussian statistics
        bank = self.reservoir[class_id]
        self.statistics[class_id] = {
            "mean": bank.mean(0),
            "std": bank.std(0).clamp(min=1e-6),
        }

    @torch.no_grad()
    def sample(self, class_id: int, n: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Sample n synthetic feature vectors for class_id.
        Uses real stored features + Gaussian noise for diversity.
        """
        if class_id not in self.reservoir:
            return None

        bank = self.reservoir[class_id].to(device)
        stats = self.statistics[class_id]
        mean = stats["mean"].to(device)
        std = stats["std"].to(device)

        # Draw random features from bank (with replacement)
        idx = torch.randint(0, bank.shape[0], (n,))
        base = bank[idx]

        # Add small Gaussian noise (intra-class variation)
        noise = torch.randn_like(base) * std.unsqueeze(0) * 0.1
        synthetic = base + noise

        return synthetic

    def known_classes(self) -> List[int]:
        return list(self.reservoir.keys())

    def count(self, class_id: int) -> int:
        return self._counts.get(class_id, 0)


class ContinualLearner(nn.Module):
    """
    Manages incremental class learning for the detection pipeline.

    Design:
      - Backbone frozen throughout
      - Classifier head expanded for new classes
      - Class-specific prompt tokens condition the RoI features
      - Feature-space replay prevents forgetting old classes
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_initial_classes: int = 1,
        prompt_dim: int = 256,
        memory_bank_size: int = 200,
        replay_epochs: int = 10,
        replay_lr: float = 1e-4,
        min_unknown_samples: int = 5,
        synthetic_samples_per_class: int = 50,
        device: str = "cuda",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_initial_classes
        self.device = device
        self.replay_epochs = replay_epochs
        self.replay_lr = replay_lr
        self.min_unknown_samples = min_unknown_samples
        self.synthetic_per_class = synthetic_samples_per_class

        # Class prompt pool: one learnable vector per class
        # Conditions RoI features for class-specific discrimination
        self.prompt_pool = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_dim) * 0.01)
            for _ in range(num_initial_classes)
        ])

        # Feature memory bank (no pixel storage needed)
        self.memory = FeatureMemoryBank(feature_dim, memory_bank_size)

        # Unknown candidate buffer
        self._unknown_buffer: List[torch.Tensor] = []

    # ──────────────────────────────────────────────────────────────
    # Prompt application
    # ──────────────────────────────────────────────────────────────

    def apply_prompt(self, features: torch.Tensor, class_id: int) -> torch.Tensor:
        """
        Add class prompt to features.
        features: [N, D]
        Returns: [N, D]
        """
        if class_id < len(self.prompt_pool):
            prompt = self.prompt_pool[class_id].to(features.device)
            return features + prompt
        return features

    # ──────────────────────────────────────────────────────────────
    # Memory population (called during training)
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def store_features(self, class_id: int, features: torch.Tensor):
        """Store features for a known class into the memory bank."""
        self.memory.update(class_id, features)

    # ──────────────────────────────────────────────────────────────
    # Unknown candidate collection
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def buffer_unknown(self, features: torch.Tensor):
        """Collect unknown candidate features for new class discovery."""
        self._unknown_buffer.append(features.detach().cpu())

    def unknown_buffer_size(self) -> int:
        if not self._unknown_buffer:
            return 0
        return sum(f.shape[0] for f in self._unknown_buffer)

    def should_trigger_new_class(self) -> bool:
        return self.unknown_buffer_size() >= self.min_unknown_samples

    # ──────────────────────────────────────────────────────────────
    # Incremental class addition
    # ──────────────────────────────────────────────────────────────

    def add_new_class(
        self,
        classifier: nn.Linear,
        energy_detector,
        new_class_features: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Add a new class:
        1. Expand classifier head
        2. Add new prompt token
        3. Initialize memory bank with new class features
        4. Run replay training to prevent forgetting

        Returns: new class id
        """
        new_class_id = self.num_classes
        self.num_classes += 1

        # 1. Expand classifier
        energy_detector.expand_classifier(self.num_classes)

        # 2. Add new prompt token
        new_prompt = nn.Parameter(
            torch.randn(1, self.feature_dim, device=self.device) * 0.01
        )
        self.prompt_pool.append(new_prompt)

        # 3. Store new class features in memory
        if new_class_features is not None:
            self.memory.update(new_class_id, new_class_features.cpu())
        elif self._unknown_buffer:
            all_unknown = torch.cat(self._unknown_buffer, dim=0)
            self.memory.update(new_class_id, all_unknown)

        self._unknown_buffer.clear()

        print(f"[ContinualLearner] Added class {new_class_id}. "
              f"Total classes: {self.num_classes}")

        return new_class_id

    # ──────────────────────────────────────────────────────────────
    # Feature-space replay training
    # ──────────────────────────────────────────────────────────────

    def replay_train(
        self,
        classifier: nn.Linear,
        new_class_id: int,
        device: torch.device,
    ):
        """
        Train classifier head using synthetic features from memory bank.
        This prevents catastrophic forgetting of old classes.
        Only classifier + prompts are trained; backbone stays frozen.
        """
        known = self.memory.known_classes()
        if len(known) < 2:
            return  # nothing to replay yet

        opt = torch.optim.Adam(
            list(classifier.parameters()) +
            [p for p in self.prompt_pool.parameters()],
            lr=self.replay_lr
        )

        classifier.train()
        for p in self.prompt_pool.parameters():
            p.requires_grad_(True)

        losses = []
        for epoch in range(self.replay_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for class_id in known:
                n = self.synthetic_per_class
                feats = self.memory.sample(class_id, n, device)
                if feats is None:
                    continue

                # Apply class prompt
                feats = self.apply_prompt(feats, class_id)

                labels = torch.full((n,), class_id, dtype=torch.long, device=device)
                logits = classifier(feats)
                loss = F.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                losses.append(epoch_loss / batch_count)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        print(f"[ContinualLearner] Replay training done. "
              f"Avg loss: {avg_loss:.4f} over {self.replay_epochs} epochs")

        return losses
