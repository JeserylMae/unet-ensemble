import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries


class Metrics:    
    def feature_correlation(self, prob_map, gt):
        p = prob_map.flatten().astype(np.float64)
        g = gt.flatten().astype(np.float64)

        if g.std() == 0 and g.mean() == 0:
            return float(1.0 - p.mean())

        if g.std() == 0 and g.mean() == 1:
            return float(p.mean())

        if p.std() == 0:
            return float('nan')

        r, _ = pearsonr(p, g)
        return float(r)

    def dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Dice = 2·|P∩G| / (|P|+|G|)."""
        intersection = int(np.logical_and(pred, gt).sum())
        denominator  = int(pred.sum()) + int(gt.sum())
        return (2 * intersection) / denominator if denominator > 0 else 1.0


    def iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """IoU = |P∩G| / |P∪G|."""
        intersection = int(np.logical_and(pred, gt).sum())
        union        = int(np.logical_or(pred, gt).sum())
        return intersection / union if union > 0 else 1.0


    def pixel_accuracy(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Pixel Accuracy = correct pixels / total pixels."""
        return float((pred == gt).sum()) / pred.size


    def _dilate(self, binary_mask: np.ndarray, width: int) -> np.ndarray:
        struct = np.ones((2 * width + 1, 2 * width + 1), dtype=bool)
        return binary_dilation(binary_mask, structure=struct).astype(np.uint8)


    def bf_score(self, pred: np.ndarray, gt: np.ndarray, boundary_width: int = 2) -> float:
        """
        Boundary F1 Score using dilated boundary matching.
        Both masks empty → 1.0; only one empty → 0.0.
        """
        pred_b = find_boundaries(pred, mode='outer').astype(np.uint8)
        gt_b   = find_boundaries(gt,   mode='outer').astype(np.uint8)

        if pred_b.sum() == 0 and gt_b.sum() == 0:
            return 1.0
        if pred_b.sum() == 0 or gt_b.sum() == 0:
            return 0.0

        pred_dilated = self._dilate(pred_b, boundary_width)
        gt_dilated   = self._dilate(gt_b,   boundary_width)

        precision = float(np.logical_and(pred_b, gt_dilated).sum()) / pred_b.sum()
        recall    = float(np.logical_and(gt_b, pred_dilated).sum()) / gt_b.sum()

        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)


    def compute_all_metrics(
        self, 
        prob_map : np.ndarray,   # (H, W)  raw sigmoid probability
        pred     : np.ndarray,   # (H, W)  binarised prediction
        gt       : np.ndarray,   # (H, W)  ground-truth mask
        bw       : int = 2,
    ) -> dict:
        return {
            'Feature_Correlation': round(self.feature_correlation(prob_map, gt), 6),
            'Dice'               : round(self.dice(pred, gt),                    6),
            'IoU'                : round(self.iou(pred, gt),                     6),
            'Pixel_Accuracy'     : round(self.pixel_accuracy(pred, gt),          6),
            'BF_Score'           : round(self.bf_score(pred, gt, bw),            6),
        }

    print('Metric helpers defined.')