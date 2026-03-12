import torch
import torch.nn as nn


class SingleBranch(nn.Module):
    """Lightweight CNN encoder for one feature map (1 → out_ch channels)."""
    def __init__(self, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,    32,     3, padding=1), nn.BatchNorm2d(32),     nn.ReLU(inplace=True),
            nn.Conv2d(32,   32,     3, padding=1), nn.BatchNorm2d(32),     nn.ReLU(inplace=True),
            nn.Conv2d(32,   out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)   # (B, out_ch, H, W)


class MBENFusionModule(nn.Module):
    """
    Multi-Branch Encoder Network fusion module with flexible feature combinations.

    Supported feature combinations:
        ('prnu', 'illumination', 'frequency')  — all three branches active
        ('prnu', 'frequency')                  — PRNU + Frequency branches
        ('prnu', 'illumination')               — PRNU + Illumination branches
        ('frequency', 'illumination')          — Frequency + Illumination branches

    Steps:
      1. Each active feature passes through its own SingleBranch CNN encoder.
      2. The active branch outputs are fused via element-wise summation.
      3. A concat stem processes the channel-stacked input (one channel per active feature).
      4. Both paths are concatenated and projected to 3 channels for U-Net++.

    Args:
        out_ch:    Number of output channels for each SingleBranch and the concat stem.
        features:  Iterable of active feature names ('prnu', 'illumination', 'frequency').
    """

    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, out_ch=64, features=('prnu', 'illumination', 'frequency')):
        super().__init__()

        self.features        = frozenset(f.lower() for f in features)
        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]
        self.num_features    = len(self.active_features)

        if self.num_features < 2:
            raise ValueError(
                f"At least 2 features are required. Got: {self.features}"
            )

        # Create one branch per active feature (stored in a ModuleDict for proper registration)
        self.branches = nn.ModuleDict({
            feat: SingleBranch(out_ch) for feat in self.active_features
        })

        # Shallow stem for channel-wise concatenated input (C = num_features)
        self.concat_stem = nn.Sequential(
            nn.Conv2d(self.num_features, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Merge MBEN-sum + concat-stem → project to 3 channels
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,     3,      1),
        )
    
    def forward(self, feature_dict: dict, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_dict:  Dict mapping feature name → tensor (B, 1, H, W).
                           Only keys matching self.active_features are used.
            fused:         Channel-stacked tensor (B, C, H, W) where C = num_features,
                           in canonical order (prnu → illumination → frequency).

        Returns:
            Projected tensor (B, 3, H, W) ready for U-Net++.
        """
        # Element-wise sum of all active branch outputs
        branch_outputs = [self.branches[feat](feature_dict[feat]) for feat in self.active_features]
        mben_sum = branch_outputs[0]
        for out in branch_outputs[1:]:
            mben_sum = mben_sum + out

        # Channel-wise concat path
        concat_feat = self.concat_stem(fused)

        # Combine and project
        combined = torch.cat([mben_sum, concat_feat], dim=1)   # (B, out_ch*2, H, W)
        return self.fusion_proj(combined)                       # (B, 3, H, W)
