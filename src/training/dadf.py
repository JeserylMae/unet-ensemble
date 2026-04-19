import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


class MultiscaleAdapter(nn.Module):
    """
    Lightweight adapter injected into each SAM transformer block.

    Uses parallel 1×1 and 3×3 (dilated) convolutions to capture
    short-range and long-range forgery contexts with minimal parameters.

    Args:
        embed_dim : feature dimension of the SAM block output (default 1280 for ViT-H).
        mid_dim   : bottleneck width (default 256, ≈18 % of SAM params tuned).
    """

    def __init__(self, embed_dim: int = 1280, mid_dim: int = 256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(embed_dim, mid_dim, 1)
        self.conv3x3 = nn.Conv2d(embed_dim, mid_dim, 3, padding=2, dilation=2)
        self.proj    = nn.Conv2d(mid_dim * 2, embed_dim, 1)
        self.bn      = nn.BatchNorm2d(embed_dim)
        self.act     = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)  →  permute to (B, C, H, W) for conv
        B, H, W, C = x.shape
        xp   = x.permute(0, 3, 1, 2).contiguous()
        feat = torch.cat([self.conv1x1(xp), self.conv3x3(xp)], dim=1)
        out  = self.act(self.bn(self.proj(feat)))
        return x + out.permute(0, 2, 3, 1)   # residual, back to (B, H, W, C)


class ReconstructionGuidedAttention(nn.Module):
    """
    RGA module: reconstructs real-face features from the encoder output
    so that forged (out-of-distribution) regions produce high residual
    error — acting as a spatial attention map for the mask decoder.

    Args:
        embed_dim : channel dimension of the feature map (default 256).
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, 1),
        )
        self.attn_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat : (B, C, H, W) image embedding from SAM.

        Returns:
            attended_feat : (B, C, H, W) — forgery-aware feature.
            rec_loss      : scalar reconstruction loss (for real images only
                            during training; pass zeros for fake images).
        """
        z         = self.encoder(feat)
        recon     = self.decoder(z)
        rec_loss  = F.mse_loss(recon, feat.detach())
        residual  = torch.abs(feat - recon)
        attn      = torch.sigmoid(self.attn_conv(residual))
        return feat * attn, rec_loss


class DADFModel(nn.Module):
    """
    DADF wrapper around a pre-loaded SAM model.

    Injects Multiscale Adapters into every SAM transformer block and
    prepends a Reconstruction Guided Attention module before the mask
    decoder.  The SAM image encoder weights are frozen; only the
    adapters, RGA, and mask decoder are trained.

    Args:
        sam            : a pre-loaded SAM model from ``segment_anything``.
        embed_dim      : channel dim of SAM image embeddings (default 256).
        adapter_mid_dim: bottleneck width for each MultiscaleAdapter (default 256).

    Forward:
        rgb  — ``(B, 3, H, W)`` float32, normalised to ``[-1, 1]``.

    Returns:
        mask_logits — ``(B, 1, H, W)`` raw logits.
        rec_loss    — scalar reconstruction loss from RGA.
    """

    def __init__(self, sam, embed_dim: int = 256, adapter_mid_dim: int = 256):
        super().__init__()
        self.sam = sam

        # ── Inject adapters into every transformer block ───────────────────
        enc_dim = sam.image_encoder.blocks[0].norm1.normalized_shape[0]
        self.adapters = nn.ModuleList([
            MultiscaleAdapter(enc_dim, adapter_mid_dim)
            for _ in sam.image_encoder.blocks
        ])

        # ── RGA on top of image embeddings ─────────────────────────────────
        self.rga = ReconstructionGuidedAttention(embed_dim)

        # ── Simple 1×1 mask head on top of SAM mask decoder output ─────────
        self.mask_head = nn.Conv2d(1, 1, 1)

    def _encode_with_adapters(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run SAM image encoder with adapter residuals injected per block."""
        enc = self.sam.image_encoder
        x   = enc.patch_embed(rgb)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        for blk, adapter in zip(enc.blocks, self.adapters):
            x = blk(x)
            x = adapter(x)          # inject adapter residual
        x = enc.neck(x.permute(0, 3, 1, 2))
        return x                    # (B, 256, 64, 64) for ViT-H / 1024 px

    def forward(self, rgb: torch.Tensor):
        # 1. Encode image with adapter-augmented backbone
        image_embeddings = self._encode_with_adapters(rgb)  # (B, 256, 64, 64)

        # 2. RGA — forgery-aware attention + reconstruction loss
        image_embeddings, rec_loss = self.rga(image_embeddings)

        # 3. SAM prompt encoder (no-prompt: sparse=zeros, dense=zeros)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None,
        )

        # 4. SAM mask decoder
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings         = image_embeddings,
            image_pe                 = self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings  = dense_embeddings,
            multimask_output         = False,
        )  # (B, 1, 256, 256)

        # 5. Upsample to input resolution
        mask_logits = F.interpolate(
            self.mask_head(low_res_masks),
            size=rgb.shape[-2:],
            mode='bilinear', align_corners=False,
        )  # (B, 1, H, W)

        return mask_logits, rec_loss


def run_epoch(loader, model, device, optimizer=None, train: bool = True):
    """
    Run one full epoch of training or validation.

    The DataLoader is expected to yield batches of::

        (rgb_t, mask_t)

    where:
        ``rgb_t``  — ``(B, 3, H, W)`` float32 tensor.
        ``mask_t`` — ``(B, 1, H, W)`` float32 tensor, binary ``{0, 1}``.

    Args:
        loader:    PyTorch DataLoader yielding ``(rgb_t, mask_t)`` batches.
        model:     ``DADFModel`` instance.
        device:    ``torch.device`` to move tensors to.
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
            rgb   = rgb.to(device)    # (B, 3, H, W)
            masks = masks.to(device)  # (B, 1, H, W)

            logits, rec_loss = model(rgb)
            loss = combined_loss(logits, masks, rec_loss)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if not train:
                probs = torch.sigmoid(logits)
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


def combined_loss(logits, masks, rec_loss,
                  lambda_seg: float = 1.0, lambda_rec: float = 0.1):
    """
    Combines Dice + BCE segmentation loss with the RGA reconstruction loss.

    Args:
        logits     : ``(B, 1, H, W)`` raw model output.
        masks      : ``(B, 1, H, W)`` binary ground-truth mask.
        rec_loss   : scalar reconstruction loss returned by ``DADFModel``.
        lambda_seg : weight for the segmentation term (default 1.0).
        lambda_rec : weight for the reconstruction term (default 0.1).

    Returns:
        Scalar combined loss.
    """
    import segmentation_models_pytorch as smp
    
    _dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    _bce  = nn.BCEWithLogitsLoss()
    seg   = lambda_seg * (_dice(logits, masks) + _bce(logits, masks))
    return seg + lambda_rec * rec_loss
