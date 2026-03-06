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
    Multi-Branch Encoder Network fusion module.

    Steps:
      1. PRNU, Illumination, Frequency each pass through their own SingleBranch.
      2. The three branch outputs are fused via element-wise summation.
      3. A concat stem processes the stacked 3-ch input (channel-wise concat path).
      4. Both paths are concatenated and projected to 3 channels for U-Net++.
    """
    def __init__(self, out_ch=64):
        super().__init__()
        self.prnu_branch  = SingleBranch(out_ch)
        self.illu_branch  = SingleBranch(out_ch)
        self.freq_branch  = SingleBranch(out_ch)

        # Shallow stem for channel-wise concatenated input
        self.concat_stem = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

        # Merge MBEN-sum + concat-stem → project to 3 channels
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,     3,      1),
        )

    def forward(self, prnu, illu, freq, fused):
        # MBEN: independent encoding + element-wise sum
        mben_sum = self.prnu_branch(prnu) + self.illu_branch(illu) + self.freq_branch(freq)

        # Channel-wise concat path
        concat_feat = self.concat_stem(fused)

        # Combine and project
        combined = torch.cat([mben_sum, concat_feat], dim=1)   # (B, out_ch*2, H, W)
        return self.fusion_proj(combined)                       # (B, 3, H, W)


