import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
)


class Evaluate:
    """
    Evaluation class for MBEN-based segmentation models (UNet++ and Baseline).

    Loads a model from a Hugging Face repository using safetensors weights and
    a config.json, then runs inference over a DataLoader-provided test split to
    compute pixel-level segmentation metrics.

    Supported metrics
    -----------------
    - AUC-ROC        : Area under the ROC curve (threshold-free, probability-based).
    - Precision      : TP / (TP + FP) at threshold 0.5.
    - Recall         : TP / (TP + FN) at threshold 0.5.
    - F1 Score       : Harmonic mean of Precision and Recall.
    - IoU            : Intersection over Union (Jaccard index) at threshold 0.5.
    - Dice           : 2 * |A ∩ B| / (|A| + |B|) at threshold 0.5.
    - Accuracy       : Pixel-level accuracy at threshold 0.5.
    - MCC            : Matthews Correlation Coefficient at threshold 0.5.

    Args
    ----
    device   : torch.device — target device for inference.
    features : tuple of str — active feature names; must match the model checkpoint.
                              e.g. ('prnu', 'illumination', 'frequency')
    threshold: float — binarisation threshold applied to sigmoid probabilities (default 0.5).
    """

    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, device, features=('prnu', 'illumination', 'frequency'), threshold=0.5):
        self.device          = device
        self.features        = frozenset(f.lower() for f in features)
        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]
        self.threshold       = threshold

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

        The backbone is re-instantiated from segmentation_models_pytorch using the
        backbone_name stored in config.json (falls back to 'unet' if absent).

        Args
        ----
        repo_id       : str — HuggingFace repo id, e.g. 'hf-username/mben-baseline'.
        subfolder     : str — subdirectory inside the repo.
        backbone_name : str — smp model name used as the baseline backbone.

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
        Run inference over the full dataset split and collect pixel-level
        probabilities and ground-truth labels.

        The DataLoader is expected to yield batches in the same format produced
        by dataset.py:
            (*feature_tensors_in_canonical_order, fused_t, mask_t)

        Args
        ----
        loader : torch.utils.data.DataLoader — wrapping a Dataset instance.
        model  : nn.Module — a loaded MBEN model in eval mode.

        Returns
        -------
        metrics : dict — evaluation metrics (see class docstring).
        """
        all_probs   = []
        all_targets = []

        model.eval()

        for batch in tqdm(loader, desc='Evaluating', leave=True):
            *feat_tensors, fused, masks = batch

            feature_dict = {
                feat: feat_tensors[i].to(self.device)
                for i, feat in enumerate(self.active_features)
            }
            fused = fused.to(self.device)
            masks = masks.to(self.device)

            logits = model(feature_dict, fused)          # (B, 1, H, W)
            probs  = torch.sigmoid(logits)               # (B, 1, H, W)

            all_probs.append(probs.cpu().flatten())
            all_targets.append(masks.cpu().flatten())

        all_probs   = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy().astype(int)

        return self._compute_metrics(all_probs, all_targets)

    # ─────────────────────────────────────────────────────────────────────────
    # Metric computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_metrics(self, probs: np.ndarray, targets: np.ndarray) -> dict:
        """
        Compute all evaluation metrics from flattened probability and target arrays.

        Args
        ----
        probs   : 1-D float array of sigmoid probabilities in [0, 1].
        targets : 1-D int array of binary ground-truth labels (0 or 1).

        Returns
        -------
        dict with keys: AUC_ROC, Precision, Recall, F1, IoU, Dice, Accuracy, MCC.
        """
        binary = (probs >= self.threshold).astype(int)

        # Threshold-free metric
        auc = roc_auc_score(targets, probs)

        # Threshold-based metrics
        precision = precision_score(targets, binary, zero_division=0)
        recall    = recall_score(targets, binary, zero_division=0)
        f1        = f1_score(targets, binary, zero_division=0)
        accuracy  = accuracy_score(targets, binary)
        mcc       = matthews_corrcoef(targets, binary)

        # IoU and Dice computed directly from TP / FP / FN counts
        tp = int(np.logical_and(binary == 1, targets == 1).sum())
        fp = int(np.logical_and(binary == 1, targets == 0).sum())
        fn = int(np.logical_and(binary == 0, targets == 1).sum())

        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        return {
            'AUC_ROC'  : round(float(auc),       4),
            'Precision': round(float(precision),  4),
            'Recall'   : round(float(recall),     4),
            'F1'       : round(float(f1),         4),
            'IoU'      : round(float(iou),        4),
            'Dice'     : round(float(dice),       4),
            'Accuracy' : round(float(accuracy),   4),
            'MCC'      : round(float(mcc),        4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Pretty-print helper
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def print_metrics(metrics: dict, label: str = ''):
        """Print a metrics dict as a formatted table."""
        header = f'  Evaluation Metrics — {label}  ' if label else '  Evaluation Metrics  '
        width  = max(len(header), 40)
        print('─' * width)
        print(header.center(width))
        print('─' * width)
        for k, v in metrics.items():
            print(f'  {k:<12} {v:.4f}')
        print('─' * width)
