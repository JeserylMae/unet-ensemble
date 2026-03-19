"""
Segmentation Model Evaluator
Evaluates HuggingFace segmentation models using:
  - IoU (Intersection over Union)
  - Dice Coefficient
  - Pixel Accuracy
  - Boundary F1 Score
"""

import numpy as np
from scipy.ndimage import binary_erosion
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
from torch.utils.data import DataLoader
from typing import Union


class SegmentationEvaluator:
    """
    Evaluates a HuggingFace semantic segmentation model using standard metrics.

    Args:
        model_name (str): HuggingFace model repo ID (e.g. "nvidia/segformer-b0-finetuned-ade-512-512")
        num_classes (int): Number of segmentation classes.
        ignore_index (int): Class index to ignore during evaluation (e.g. background or void). Default: -1 (none).
        device (str): "cuda" or "cpu". Auto-detected if not specified.

    Example:
        evaluator = SegmentationEvaluator("your-username/your-model", num_classes=21)
        results = evaluator.evaluate_dataset(dataloader)
        evaluator.print_report(results)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        ignore_index: int = -1,
        device: str = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Core metric computations (static, work on numpy arrays)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = -1) -> dict:
        """
        Computes per-class IoU and mean IoU.

        IoU (per class) = TP / (TP + FP + FN)
        """
        ious = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            pred_mask = pred == cls
            target_mask = target == cls

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union == 0:
                ious.append(float("nan"))  # class not present
            else:
                ious.append(intersection / union)

        valid = [v for v in ious if not np.isnan(v)]
        return {
            "per_class_iou": ious,
            "mean_iou": float(np.mean(valid)) if valid else 0.0,
        }

    @staticmethod
    def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = -1) -> dict:
        """
        Computes per-class Dice coefficient and mean Dice.

        Dice (per class) = 2 * TP / (2 * TP + FP + FN)
        """
        dices = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            pred_mask = pred == cls
            target_mask = target == cls

            intersection = np.logical_and(pred_mask, target_mask).sum()
            denom = pred_mask.sum() + target_mask.sum()

            if denom == 0:
                dices.append(float("nan"))
            else:
                dices.append(2 * intersection / denom)

        valid = [v for v in dices if not np.isnan(v)]
        return {
            "per_class_dice": dices,
            "mean_dice": float(np.mean(valid)) if valid else 0.0,
        }

    @staticmethod
    def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray, ignore_index: int = -1) -> dict:
        """
        Computes overall pixel accuracy and mean class accuracy.

        Overall Accuracy = correct pixels / total pixels
        Mean Class Accuracy = mean of per-class recall
        """
        valid_mask = target != ignore_index
        correct = (pred[valid_mask] == target[valid_mask]).sum()
        total = valid_mask.sum()

        overall_acc = float(correct / total) if total > 0 else 0.0
        return {
            "overall_pixel_accuracy": overall_acc,
            "total_pixels": int(total),
            "correct_pixels": int(correct),
        }

    @staticmethod
    def _get_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
        """Extracts the boundary of a binary mask using erosion."""
        h, w = mask.shape
        # kernel size proportional to image size
        kernel_size = max(1, int(round(dilation_ratio * min(h, w))))
        struct = np.ones((kernel_size, kernel_size), dtype=bool)
        eroded = binary_erosion(mask, structure=struct)
        return mask.astype(bool) & ~eroded

    @classmethod
    def compute_boundary_f1(
        cls,
        pred: np.ndarray,
        target: np.ndarray,
        num_classes: int,
        ignore_index: int = -1,
        dilation_ratio: float = 0.02,
    ) -> dict:
        """
        Computes Boundary F1 score per class and mean Boundary F1.

        Boundary F1 measures how well predicted boundaries align with ground-truth boundaries.

        Args:
            dilation_ratio: Controls boundary thickness as a fraction of image size.
        """
        bf_scores = []
        for c in range(num_classes):
            if c == ignore_index:
                continue
            pred_boundary = cls._get_boundary((pred == c).astype(np.uint8), dilation_ratio)
            target_boundary = cls._get_boundary((target == c).astype(np.uint8), dilation_ratio)

            tp = np.logical_and(pred_boundary, target_boundary).sum()
            fp = np.logical_and(pred_boundary, ~target_boundary).sum()
            fn = np.logical_and(~pred_boundary, target_boundary).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            bf_scores.append(f1)

        valid = [v for v in bf_scores if not np.isnan(v)]
        return {
            "per_class_boundary_f1": bf_scores,
            "mean_boundary_f1": float(np.mean(valid)) if valid else 0.0,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Runs inference on a single PIL Image or numpy array.

        Returns:
            pred_mask (np.ndarray): Predicted class map with shape (H, W).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Upsample logits back to original image size
        logits = outputs.logits  # (1, num_classes, H', W')
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
        return pred_mask

    # ------------------------------------------------------------------
    # Dataset-level evaluation
    # ------------------------------------------------------------------

    def evaluate_single(self, image: Union[Image.Image, np.ndarray], target: np.ndarray) -> dict:
        """
        Evaluates a single image-mask pair.

        Args:
            image: PIL Image or numpy array (H, W, 3).
            target: Ground-truth mask (H, W) with integer class labels.

        Returns:
            dict with all metric results.
        """
        pred = self.predict(image)

        # Resize target to match prediction if needed
        if pred.shape != target.shape:
            target = np.array(
                Image.fromarray(target.astype(np.uint8)).resize(
                    (pred.shape[1], pred.shape[0]), Image.NEAREST
                )
            )

        results = {}
        results.update(self.compute_iou(pred, target, self.num_classes, self.ignore_index))
        results.update(self.compute_dice(pred, target, self.num_classes, self.ignore_index))
        results.update(self.compute_pixel_accuracy(pred, target, self.ignore_index))
        results.update(self.compute_boundary_f1(pred, target, self.num_classes, self.ignore_index))
        return results

    def evaluate_dataset(self, dataloader: DataLoader) -> dict:
        """
        Evaluates over a full dataset DataLoader.

        Expects each batch to yield (images, targets) where:
          - images: list of PIL Images or tensors (B, C, H, W)
          - targets: numpy arrays or tensors (B, H, W)

        Returns:
            dict with aggregated mean metrics across the dataset.
        """
        all_iou, all_dice, all_pixel_acc, all_bf1 = [], [], [], []

        for batch_idx, (images, targets) in enumerate(dataloader):
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()

            for i in range(len(images)):
                img = images[i]
                if isinstance(img, torch.Tensor):
                    img = Image.fromarray(img.permute(1, 2, 0).byte().numpy())

                target = targets[i]
                r = self.evaluate_single(img, target)

                all_iou.append(r["mean_iou"])
                all_dice.append(r["mean_dice"])
                all_pixel_acc.append(r["overall_pixel_accuracy"])
                all_bf1.append(r["mean_boundary_f1"])

            print(f"  Batch {batch_idx + 1}/{len(dataloader)} evaluated...", end="\r")

        print()
        return {
            "mean_iou": float(np.nanmean(all_iou)),
            "mean_dice": float(np.nanmean(all_dice)),
            "mean_pixel_accuracy": float(np.nanmean(all_pixel_acc)),
            "mean_boundary_f1": float(np.nanmean(all_bf1)),
            "num_samples": len(all_iou),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(results: dict):
        """Prints a formatted evaluation report."""
        print("\n" + "=" * 50)
        print("       SEGMENTATION EVALUATION REPORT")
        print("=" * 50)

        summary_keys = ["mean_iou", "mean_dice", "mean_pixel_accuracy", "mean_boundary_f1"]
        labels = {
            "mean_iou": "Mean IoU",
            "mean_dice": "Mean Dice Coefficient",
            "mean_pixel_accuracy": "Pixel Accuracy",
            "mean_boundary_f1": "Boundary F1 Score",
        }

        for key in summary_keys:
            if key in results:
                print(f"  {labels[key]:<25} {results[key]:.4f}  ({results[key]*100:.2f}%)")

        if "num_samples" in results:
            print(f"\n  Evaluated on {results['num_samples']} samples.")

        # Per-class breakdown if available
        if "per_class_iou" in results:
            print("\n  Per-class IoU:")
            for i, v in enumerate(results["per_class_iou"]):
                tag = "N/A" if np.isnan(v) else f"{v:.4f}"
                print(f"    Class {i:>3}: {tag}")

        print("=" * 50 + "\n")


# -----------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import Dataset

    # --- Minimal dummy dataset for demonstration ---
    class DummySegDataset(Dataset):
        def __init__(self, n=4, size=(512, 512), num_classes=19):
            self.n = n
            self.size = size
            self.num_classes = num_classes

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            image = Image.fromarray(
                np.random.randint(0, 255, (*self.size, 3), dtype=np.uint8)
            )
            target = np.random.randint(0, self.num_classes, self.size, dtype=np.int64)
            return image, target

    NUM_CLASSES = 19
    MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"  # replace with your HF model

    evaluator = SegmentationEvaluator(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        ignore_index=255,   # common void/ignore label
    )

    # --- Single image evaluation ---
    sample_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    sample_target = np.random.randint(0, NUM_CLASSES, (512, 512))

    print("Evaluating single image...")
    single_results = evaluator.evaluate_single(sample_image, sample_target)
    evaluator.print_report(single_results)

    # --- Dataset evaluation ---
    dataset = DummySegDataset(n=4, num_classes=NUM_CLASSES)

    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), np.stack(targets)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    print("Evaluating dataset...")
    dataset_results = evaluator.evaluate_dataset(loader)
    evaluator.print_report(dataset_results)
