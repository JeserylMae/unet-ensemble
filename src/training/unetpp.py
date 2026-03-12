import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from tqdm import tqdm
from src.dataset.mben import MBENFusionModule


class MBENUNetPlusPlus(nn.Module):
    """Complete model: MBENFusionModule → U-Net++ (EfficientNet-B4 backbone)."""
    def __init__(self, mben_out_ch=64):
        super().__init__()
        self.mben = MBENFusionModule(out_ch=mben_out_ch)
        self.unetpp = smp.UnetPlusPlus(
            encoder_name   ='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels    =3,
            classes        =1,
            activation     =None,
        )

    def forward(self, prnu, illu, freq, fused):
        x = self.mben(prnu, illu, freq, fused)   # (B, 3, H, W)
        return self.unetpp(x)                    # (B, 1, H, W) logits
    

class Train:
    def __init__(self, device, loss):
        self.device = device
        self.loss = loss

    def run_epoch(self, loader, model, optimizer=None, train=True):
        """
        Run one epoch of training or validation.

        Args:
            loader:    DataLoader for the split.
            model:     MBENUNetPlusPlus instance.
            optimizer: Required when train=True; must be None or omitted for validation.
            train:     If True, updates model weights. If False, runs inference only.
        """

        if train and optimizer is None:
            raise ValueError("optimizer must be provided when train=True.")

        model.train() if train else model.eval()
        total_loss = 0.0
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for prnu, illu, freq, fused, masks in tqdm(
                    loader, desc='Train' if train else 'Val', leave=False):

                prnu, illu, freq = prnu.to(self.device), illu.to(self.device), freq.to(self.device)
                fused, masks     = fused.to(self.device), masks.to(self.device)

                preds = model(prnu, illu, freq, fused)   # (B, 1, H, W)
                loss  = self.loss(preds, masks)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)