import torch
import numpy as np

from tqdm import tqdm
from skimage.segmentation import find_boundaries


class Evaluate:
    """
    Evaluation class for MBEN-based segmentation models (UNet++ and Baseline).

    Loads a model from a Hugging Face repository using safetensors weights and
    a config.json, then runs inference over a DataLoader-provided test split to
    compute pixel-level segmentation metrics.

    Metrics
    -------
    - IoU (Intersection over Union)  : Area of Overlap / Area of Union
    - Dice Coefficient               : 2×Intersection / (Prediction + Ground Truth)
    - Pixel Accuracy                 : Number of Correct Pixels / Total Number of Pixels
    - Boundary F1 Score (BF Score)   : 2 × (Precision×Recall) / (Precision+Recall)
                                       computed on boundary pixels of predicted vs true masks.

    Note: IoU, Dice, Pixel Accuracy, and BF Score are computed per image and then
    macro-averaged across the dataset so that each image contributes equally
    regardless of its foreground ratio.

    Args
    ----
    device        : torch.device — target device for inference.
    features      : tuple of str — active feature names; must match the model checkpoint.
                                   e.g. ('prnu', 'illumination', 'frequency')
    threshold     : float — binarisation threshold applied to sigmoid probabilities (default 0.5).
    boundary_width: int   — dilation tolerance in pixels used when comparing boundary pixels
                            for the BF Score (default 2).
    """

    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, device, features=('prnu', 'illumination', 'frequency'),
                 threshold=0.5, boundary_width=2):
        self.device          = device
        self.features        = frozenset(f.lower() for f in features)
        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]
        self.threshold       = threshold
        self.boundary_width  = boundary_width

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_unetpp_from_hub(self, repo_id: str, subfolder: str = 'all_features'):
        """
        Download and reconstruct a MBENUNetPlusPlus model from a Hugging Face repo.

        Args
        ----
        repo_id   : str — HuggingFace repo id, e.g. 'hf-username/mben-unetpp'.
        subfolder : str — subdirectory inside the repo that holds model.safetensors
                          and config.json (default: 'all_features').

        Returns
        -------
        model : MBENUNetPlusPlus loaded on self.device in eval mode.
        """
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from src.training.unetpp import MBENUNetPlusPlus

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'{subfolder}/config.json',
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'{subfolder}/model.safetensors',
        )

        with open(config_path) as f:
            config = json.load(f)

        model = MBENUNetPlusPlus(
            mben_out_ch=config.get('mben_out_ch', 64),
            features=config.get('features', list(self.features)),
        ).to(self.device)

        state_dict = load_file(weights_path, device=str(self.device))
        model.load_state_dict(state_dict)
        model.eval()

        print(f'[UNet++] Loaded from {repo_id}/{subfolder}')
        return model

    def load_baseline_from_hub(self, repo_id: str, subfolder: str = 'all_features',
                                backbone_name: str = 'unet'):
        """
        Download and reconstruct a MBENBaseline model from a Hugging Face repo.

        Args
        ----
        repo_id       : str — HuggingFace repo id, e.g. 'hf-username/mben-baseline'.
        subfolder     : str — subdirectory inside the repo.
        backbone_name : str — smp model name used as the baseline backbone
                              (read from config.json; this arg is the fallback).

        Returns
        -------
        model : MBENBaseline loaded on self.device in eval mode.
        """
        import json
        import segmentation_models_pytorch as smp
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from src.training.baseline import MBENBaseline

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'{subfolder}/config.json',
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'{subfolder}/model.safetensors',
        )

        with open(config_path) as f:
            config = json.load(f)

        _backbone_name = config.get('backbone', backbone_name)
        backbone = getattr(smp, _backbone_name)(
            encoder_name   =config.get('encoder', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels    =config.get('in_channels', 3),
            classes        =config.get('classes', 1),
            activation     =None,
        )

        model = MBENBaseline(
            model      =backbone,
            mben_out_ch=config.get('mben_out_ch', 64),
            features   =config.get('features', list(self.features)),
        ).to(self.device)

        state_dict = load_file(weights_path, device=str(self.device))
        model.load_state_dict(state_dict)
        model.eval()

        print(f'[Baseline] Loaded from {repo_id}/{subfolder}')
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Inference loop
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, loader, model):
        """
        Run inference over the full dataset split and compute all four metrics.

        Metrics are computed per image and then macro-averaged so that every
        image contributes equally regardless of foreground pixel ratio.

        The DataLoader is expected to yield batches in the same format produced
        by dataset.py:
            (*feature_tensors_in_canonical_order, fused_t, mask_t)

        Args
        ----
        loader : torch.utils.data.DataLoader — wrapping a Dataset instance.
        model  : nn.Module — a loaded MBEN model in eval mode.

        Returns
        -------
        metrics : dict with keys: IoU, Dice, Pixel_Accuracy, BF_Score.
        """
        iou_scores      = []
        dice_scores     = []
        accuracy_scores = []
        bf_scores       = []

        model.eval()

        for batch in tqdm(loader, desc='Evaluating', leave=True):
            *feat_tensors, fused, masks = batch

            feature_dict = {
                feat: feat_tensors[i].to(self.device)
                for i, feat in enumerate(self.active_features)
            }
            fused = fused.to(self.device)
            masks = masks.to(self.device)

            logits = model(feature_dict, fused)         # (B, 1, H, W)
            probs  = torch.sigmoid(logits)              # (B, 1, H, W)
            preds  = (probs >= self.threshold).float()  # (B, 1, H, W) binary

            preds_np = preds.cpu().numpy().astype(np.uint8)  # (B, 1, H, W)
            masks_np = masks.cpu().numpy().astype(np.uint8)  # (B, 1, H, W)

            for b in range(preds_np.shape[0]):
                pred = preds_np[b, 0]  # (H, W)
                gt   = masks_np[b, 0]  # (H, W)

                iou_scores.append(self._iou(pred, gt))
                dice_scores.append(self._dice(pred, gt))
                accuracy_scores.append(self._pixel_accuracy(pred, gt))
                bf_scores.append(self._boundary_f1(pred, gt))

        return {
            'IoU'           : round(float(np.mean(iou_scores)),       4),
            'Dice'          : round(float(np.mean(dice_scores)),       4),
            'Pixel_Accuracy': round(float(np.mean(accuracy_scores)),  4),
            'BF_Score'      : round(float(np.mean(bf_scores)),        4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Per-image metric helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """IoU = Area of Overlap / Area of Union."""
        intersection = int(np.logical_and(pred, gt).sum())
        union        = int(np.logical_or(pred, gt).sum())
        return intersection / union if union > 0 else 1.0  # both empty → perfect match

    @staticmethod
    def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """Dice = 2×Intersection / (|Prediction| + |Ground Truth|)."""
        intersection = int(np.logical_and(pred, gt).sum())
        denominator  = int(pred.sum()) + int(gt.sum())
        return (2 * intersection) / denominator if denominator > 0 else 1.0

    @staticmethod
    def _pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """Pixel Accuracy = Number of Correct Pixels / Total Number of Pixels."""
        return float((pred == gt).sum()) / pred.size

    def _boundary_f1(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Boundary F1 Score = 2 × (Precision × Recall) / (Precision + Recall).

        Boundaries are extracted from both masks using skimage find_boundaries
        (mode='outer'). A predicted boundary pixel counts as a true positive if it
        falls within `boundary_width` pixels of any GT boundary pixel, and vice versa
        for recall.

        Edge cases:
          - Both masks have no boundary → 1.0 (both empty, perfect agreement).
          - Only one mask has no boundary → 0.0.
        """
        pred_boundary = find_boundaries(pred, mode='outer').astype(np.uint8)
        gt_boundary   = find_boundaries(gt,   mode='outer').astype(np.uint8)

        if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
            return 1.0
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            return 0.0

        pred_dilated = self._dilate(pred_boundary, self.boundary_width)
        gt_dilated   = self._dilate(gt_boundary,   self.boundary_width)

        # Precision: how many predicted boundary pixels are near a GT boundary pixel
        precision = float(np.logical_and(pred_boundary, gt_dilated).sum()) / pred_boundary.sum()
        # Recall: how many GT boundary pixels are near a predicted boundary pixel
        recall    = float(np.logical_and(gt_boundary, pred_dilated).sum()) / gt_boundary.sum()

        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def _dilate(binary_mask: np.ndarray, width: int) -> np.ndarray:
        """Dilate a binary mask using a square structuring element of side 2*width+1."""
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * width + 1, 2 * width + 1), dtype=bool)
        return binary_dilation(binary_mask, structure=struct).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # Pretty-print helper
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def print_metrics(metrics: dict, label: str = ''):
        """Print a metrics dict as a formatted table."""
        header = f'  Evaluation Metrics — {label}  ' if label else '  Evaluation Metrics  '
        width  = max(len(header), 42)
        print('─' * width)
        print(header.center(width))
        print('─' * width)
        for k, v in metrics.items():
            print(f'  {k:<16} {v:.4f}')
        print('─' * width)
