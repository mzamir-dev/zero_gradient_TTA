import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAAdapter(nn.Module):
    """
    Test-Time Adaptation via masked feature reconstruction.
    
    At test time on each batch:
    1. Mask 30% of spatial positions in the FPN feature map
    2. Ask the adapter to reconstruct the masked positions
    3. The reconstruction loss trains the adapter to understand
       the new domain's feature statistics
    4. The adapted (unmasked) features are used for detection
    
    This is domain-agnostic — works regardless of how different
    the test domain is from training domain.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        adapter_dim: int = 64,
        ema_decay: float = 0.999,
        lr: float = 1e-3,
        entropy_weight: float = 1.0,
        consistency_weight: float = 0.5,
        temporal_weight: float = 0.3,
        mask_ratio: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_ratio  = mask_ratio
        self.lr          = lr
        self.ema_decay   = ema_decay

        # Lightweight adapter: two conv layers
        self.adapter = nn.Sequential(
            nn.Conv2d(feature_dim, adapter_dim, 1, bias=False),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )

        # Reconstruction head: predicts masked feature values
        self.reconstructor = nn.Sequential(
            nn.Conv2d(feature_dim, adapter_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, feature_dim, 1, bias=False),
        )

        # Residual scale (starts at 0 = identity)
        self.scale = nn.Parameter(torch.zeros(1))

        self.optimizer = None
        self._n_steps  = 0

        # Buffers for compatibility with checkpoint loading
        self.register_buffer("train_feat_mean",   torch.zeros(feature_dim))
        self.register_buffer("train_feat_std",    torch.ones(feature_dim))
        self.register_buffer("stats_initialized", torch.tensor(False))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.adapter.parameters()) +
            list(self.reconstructor.parameters()) +
            [self.scale],
            lr=self.lr,
        )

    def reset(self):
        """Reset between scenes."""
        nn.init.zeros_(self.scale)
        self._n_steps = 0
        if self.optimizer is not None:
            self.optimizer.state.clear()

    def initialize_stats_from_checkpoint(self):
        self.stats_initialized.fill_(False)
        print("[TTAAdapter] Loaded without training stats — "
              "using masked reconstruction TTA.")

    def update_train_statistics(self, features: torch.Tensor):
        """Called during training — not used in reconstruction approach."""
        pass

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply adapter with residual connection."""
        delta = self.adapter(features)
        return features + torch.tanh(self.scale) * delta

    def adapt_step(
        self,
        features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        One TTA step via masked feature reconstruction.
        
        1. Create random spatial mask
        2. Corrupt input by zeroing masked positions
        3. Adapter + reconstructor predicts original features at masked positions
        4. Reconstruction loss updates adapter
        5. Return clean adapted features
        """
        if self.optimizer is None:
            self.build_optimizer()

        feat_in = features.detach()   # [B, C, H, W]
        B, C, H, W = feat_in.shape

        # ── Step 1: Create spatial mask
        n_masked = max(1, int(H * W * self.mask_ratio))
        mask = torch.zeros(B, 1, H, W, device=feat_in.device)
        for b in range(B):
            idx = torch.randperm(H * W, device=feat_in.device)[:n_masked]
            mask[b, 0].view(-1)[idx] = 1.0   # 1 = masked position

        # ── Step 2: Corrupt input
        feat_corrupted = feat_in * (1 - mask)  # zero out masked positions

        # ── Step 3: Forward through adapter + reconstructor
        self.adapter.train()
        self.reconstructor.train()
        self.optimizer.zero_grad()

        adapted      = feat_corrupted + torch.tanh(self.scale) * self.adapter(feat_corrupted)
        reconstructed = self.reconstructor(adapted)   # [B, C, H, W]

        # ── Step 4: Reconstruction loss on masked positions only
        target = feat_in   # original unmasked features are the target
        recon_loss = F.mse_loss(
            reconstructed * mask,
            target * mask
        )

        # Small consistency with teacher to prevent drift
        teacher_feat = teacher_features.detach()
        consistency  = F.mse_loss(
            adapted.mean(dim=[2, 3]),
            teacher_feat.mean(dim=[2, 3])
        )

        total = recon_loss + 0.1 * consistency
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.adapter.parameters()) +
            list(self.reconstructor.parameters()), 1.0
        )
        self.optimizer.step()

        # # ── Second gradient step for stronger adaptation
        # self.adapter.train()
        # self.reconstructor.train()
        # self.optimizer.zero_grad()
        # adapted2      = feat_in + torch.tanh(self.scale) * self.adapter(feat_in)
        # reconstructed2 = self.reconstructor(adapted2)
        # loss2 = F.mse_loss(reconstructed2 * mask, target * mask)
        # loss2.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     list(self.adapter.parameters()) +
        #     list(self.reconstructor.parameters()), 1.0
        # )
        # self.optimizer.step()

        # ── Step 5: Return clean adapted features
        self.adapter.eval()
        with torch.no_grad():
            adapted_out = feat_in + torch.tanh(self.scale) * self.adapter(feat_in)

        self._n_steps += 1
        return adapted_out