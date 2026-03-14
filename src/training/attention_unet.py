import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from tqdm import tqdm
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

    def forward(self, feature_dict, fused):

        x = self.mben(feature_dict, fused)

        return self.attention_unet(x)
        

class Train:
    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, device, loss, features=('prnu','illumination','frequency')):

        self.device = device
        self.loss = loss

        self.features = frozenset(f.lower() for f in features)
        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]


    def run_epoch(self, loader, model, optimizer=None, train=True):

        if train and optimizer is None:
            raise ValueError("optimizer required when train=True")

        model.train() if train else model.eval()

        total_loss = 0.0

        context = torch.enable_grad() if train else torch.no_grad()

        with context:

            for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):

                *feat_tensors, fused, masks = batch

                feature_dict = {
                    feat: feat_tensors[i].to(self.device)
                    for i, feat in enumerate(self.active_features)
                }

                fused = fused.to(self.device)
                masks = masks.to(self.device)

                preds = model(feature_dict, fused)

                loss = self.loss(preds, masks)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)