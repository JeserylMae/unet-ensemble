import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.dataset.mben import MBENFusionModule


class MBENAttentionUNet(nn.Module):

    def __init__(self, mben_out_ch=64, features=('prnu','illumination','frequency')):
        super().__init__()

        self.features = frozenset(f.lower() for f in features)

        self.mben = MBENFusionModule(
            out_ch=mben_out_ch,
            features=features
        )

        # DIFFERENT ARCHITECTURE
        self.attention_unet = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type='scse'
        )

    def forward(self, feature_dict: dict, fused: torch.Tensor) -> torch.Tensor:

        x = self.mben(feature_dict, fused)
        return self.attention_unet(x)