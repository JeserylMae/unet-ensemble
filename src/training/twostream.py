import torch
import torch.nn as nn

from tqdm import tqdm
from src.dataset.mben import MBENFusionModule


class MBENTwoStream(nn.Module):
    """
    Complete model: MBENFusionModule → Unet (Baseline backbone).

    Supports flexible feature combinations:
        ('prnu', 'illumination', 'frequency')  [default — all three]
        ('prnu', 'frequency')
        ('prnu', 'illumination')
        ('frequency', 'illumination')

    The `features` argument must be the same combination used in DataLoader and Dataset.

    Args:
        mben_out_ch:  Intermediate channel count for each MBEN branch / concat stem.
        features:     Iterable of active feature names.
    """

    def __init__(self, model, mben_out_ch=64, features=('prnu', 'illumination', 'frequency')):
        super().__init__()
        self.features = frozenset(f.lower() for f in features)
        self.mben = MBENFusionModule(out_ch=mben_out_ch, features=features)
        self.model = model

    def forward(self, feature_dict: dict, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_dict:  Dict mapping feature name → tensor (B, 1, H, W).
            fused:         Channel-stacked tensor (B, C, H, W).

        Returns:
            Logit mask (B, 1, H, W).
        """
        x = self.mben(feature_dict, fused)   # (B, 3, H, W)
        return self.model(x)                # (B, 1, H, W) logits