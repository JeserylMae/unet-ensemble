import torch
import torch.nn as nn

from tqdm import tqdm
from src.dataset.mben import MBENFusionModule


class MBENBaseline(nn.Module):
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


class Train:
    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, device, loss, features=('prnu', 'illumination', 'frequency')):
        """
        Args:
            device:   torch.device to move tensors to.
            loss:     Loss function (e.g. smp DiceLoss).
            features: Active feature names — must match the combination used in DataLoader/Dataset.
        """
        self.device          = device
        self.loss            = loss
        self.features        = frozenset(f.lower() for f in features)
        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]

    def run_epoch(self, loader, model, optimizer=None, train=True):
        """
        Run one epoch of training or validation.

        The DataLoader yields batches as:
            (*feature_tensors_in_canonical_order, fused_t, mask_t)

        This method unpacks them dynamically based on the active feature set.

        Args:
            loader:    DataLoader for the split.
            model:     MBENBaseline instance.
            optimizer: Required when train=True; must be None or omitted for validation.
            train:     If True, updates model weights. If False, runs inference only.
        """
        if train and optimizer is None:
            raise ValueError("optimizer must be provided when train=True.")

        model.train() if train else model.eval()
        total_loss = 0.0
        context    = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in tqdm(loader, desc='Train' if train else 'Val', leave=False):
                # Unpack: first N tensors are features (canonical order), then fused, then mask
                # batch = (feat_1, ..., feat_N, fused, mask)
                *feat_tensors, fused, masks = batch

                # Build feature_dict {name: tensor} and move everything to device
                feature_dict = {
                    feat: feat_tensors[i].to(self.device)
                    for i, feat in enumerate(self.active_features)
                }
                fused = fused.to(self.device)
                masks = masks.to(self.device)

                preds = model(feature_dict, fused)   # (B, 1, H, W)
                loss  = self.loss(preds, masks)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)
