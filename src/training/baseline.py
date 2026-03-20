import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

class RGBBaseline(nn.Module):
    """
    Minimal baseline model: passes a 3-channel RGB tensor directly into
    the segmentation backbone (e.g. a U-Net).

    No feature-engineering branches, no MBEN fusion — raw RGB in,
    binary logit mask out.

    Args:
        model: Any segmentation model that accepts ``(B, 3, H, W)`` input
               and returns ``(B, 1, H, W)`` logits
               (e.g. ``smp.Unet(..., in_channels=3, classes=1)``).

    Forward:
        rgb    — ``(B, 3, H, W)`` float tensor, normalised to ``[-1, 1]``.

    Returns:
        Logit mask — ``(B, 1, H, W)``.
    """

    def __init__(self, model: nn.Module):
        super().__init__)
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)   # (B, 1, H, W)


class Train:
    """
    Training / validation runner for the RGB baseline model.

    The DataLoader is expected to yield batches of::

        (rgb_t, mask_t)

    where:
        ``rgb_t``  — ``(B, 3, H, W)`` float32 tensor.
        ``mask_t`` — ``(B, 1, H, W)`` float32 tensor, binary ``{0, 1}``.

    Args:
        device: ``torch.device`` to move tensors to.
        loss:   Loss function callable, e.g. ``smp.losses.DiceLoss``.
    """

    def __init__(self, device: torch.device, loss):
        self.device = device
        self.loss   = loss

    def run_epoch(self, loader, model, optimizer=None, train: bool = True):
        """
        Run one full epoch of training or validation.

        Args:
            loader:    PyTorch DataLoader yielding ``(rgb_t, mask_t)`` batches.
            model:     ``RGBBaseline`` instance (or any compatible model).
            optimizer: Required when ``train=True``; ignored otherwise.
            train:     ``True`` → update weights; ``False`` → inference only.

        Returns:
            * ``train=True``  → ``float`` average training loss.
            * ``train=False`` → ``dict`` with keys
              ``val_loss``, ``AUC_ROC``, ``Precision``, ``Recall``, ``MCC``.
        """
        if train and optimizer is None:
            raise ValueError("optimizer must be provided when train=True.")

        model.train() if train else model.eval()

        total_loss  = 0.0
        all_preds   = []
        all_targets = []

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for rgb, masks in tqdm(loader, desc='Train' if train else 'Val', leave=False):
                rgb   = rgb.to(self.device)    # (B, 3, H, W)
                masks = masks.to(self.device)  # (B, 1, H, W)

                preds = model(rgb)             # (B, 1, H, W) — logits
                loss  = self.loss(preds, masks)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                if not train:
                    probs = torch.sigmoid(preds)
                    all_preds.append(probs.detach().cpu().flatten())
                    all_targets.append(masks.detach().cpu().flatten())

        avg_loss = total_loss / len(loader)

        if train:
            return avg_loss

        all_preds   = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        binary_preds = (all_preds > 0.5).astype(int)

        return {
            'val_loss':  avg_loss,
            'AUC_ROC':   roc_auc_score(all_targets, all_preds),
            'Precision': precision_score(all_targets, binary_preds, zero_division=0),
            'Recall':    recall_score(all_targets, binary_preds, zero_division=0),
            'MCC':       matthews_corrcoef(all_targets, binary_preds),
        }
