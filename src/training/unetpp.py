import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.dataset.mben import MBENFusionModule



class MBENUNetPlusPlus(nn.Module):
    """
    Complete model: MBENFusionModule → U-Net++ (EfficientNet-B4 backbone).

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

    def __init__(self, mben_out_ch=64, features=('prnu', 'illumination', 'frequency')):
        super().__init__()
        self.features = frozenset(f.lower() for f in features)
        self.mben = MBENFusionModule(out_ch=mben_out_ch, features=features)
        self.unetpp = smp.UnetPlusPlus(
            encoder_name   ='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels    =3,
            classes        =1,
            activation     =None,
        )

    def forward(self, feature_dict: dict, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_dict:  Dict mapping feature name → tensor (B, 1, H, W).
            fused:         Channel-stacked tensor (B, C, H, W).

        Returns:
            Logit mask (B, 1, H, W).
        """
        x = self.mben(feature_dict, fused)   # (B, 3, H, W)
        return self.unetpp(x)                # (B, 1, H, W) logits
