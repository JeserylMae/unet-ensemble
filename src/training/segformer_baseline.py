import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SegFormerBaseline(nn.Module):
    """
    Baseline model wrapping HuggingFace SegFormer for binary segmentation.

    SegFormer is a pure-transformer segmentation architecture (no U-Net).
    It uses a hierarchical Mix Transformer (MiT) encoder and a lightweight
    MLP decoder head (SegFormerHead).

    The model is initialised from ImageNet-pretrained weights. The decoder
    head is replaced so ``num_labels=1`` for binary segmentation.

    Available backbone variants (encoder_name):
        ``'nvidia/mit-b0'``  —  3.7 M encoder params  (fastest)
        ``'nvidia/mit-b1'``  —  14 M encoder params
        ``'nvidia/mit-b2'``  —  25 M encoder params
        ``'nvidia/mit-b3'``  —  45 M encoder params
        ``'nvidia/mit-b4'``  —  64 M encoder params
        ``'nvidia/mit-b5'``  —  82 M encoder params  (most accurate)

    Args:
        encoder_name: HuggingFace model-hub identifier for the MiT backbone.
        img_size:     Spatial resolution the model will be trained on.
                      Used to up-sample SegFormer's low-res logit output
                      back to ``(img_size, img_size)``.

    Forward:
        rgb — ``(B, 3, H, W)`` float tensor, normalised to ``[-1, 1]``.

    Returns:
        Logit mask — ``(B, 1, H, W)``.
    """

    def __init__(
        self,
        encoder_name: str = 'nvidia/mit-b2',
        img_size: int = 256,
    ):
        super().__init__()
        self.img_size = img_size

        # Load pretrained SegFormer; replace head for binary segmentation
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            encoder_name,
            num_labels=1,
            ignore_mismatched_sizes=True,   # head dims change (num_labels=1)
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: ``(B, 3, H, W)`` float32 tensor in ``[-1, 1]``.

        Returns:
            Logit mask: ``(B, 1, H, W)``.
        """
        # HuggingFace SegFormer expects pixel_values in [0, 1] or normalised;
        # it outputs logits at H/4 × W/4 resolution.
        outputs = self.segformer(pixel_values=rgb)
        logits  = outputs.logits          # (B, 1, H/4, W/4)

        # Bilinear up-sample to the original spatial resolution
        logits = F.interpolate(
            logits,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False,
        )                                 # (B, 1, H, W)
        return logits


# ---------------------------------------------------------------------------
# Training / validation runner
# ---------------------------------------------------------------------------

class Train:
    """
    Training / validation runner for the SegFormer baseline model.

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
            model:     ``SegFormerBaseline`` instance.
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

        all_preds    = torch.cat(all_preds).numpy()
        all_targets  = torch.cat(all_targets).numpy()
        binary_preds = (all_preds > 0.5).astype(int)

        return {
            'val_loss':  avg_loss,
            'AUC_ROC':   roc_auc_score(all_targets, all_preds),
            'Precision': precision_score(all_targets, binary_preds, zero_division=0),
            'Recall':    recall_score(all_targets, binary_preds, zero_division=0),
            'MCC':       matthews_corrcoef(all_targets, binary_preds),
        }
