import os
import glob

import torch
import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# DataLoader  (sample-list builder)
# ---------------------------------------------------------------------------

class RGBDataLoader:
    """
    Scans a dataset directory tree and returns a list of sample dicts::

        {'rgb': <path>, 'mask': <path>}

    Expected directory layout::

        <dataset_root>/
            <split>/                   e.g. Training / Validation
                <rgb_folder>/
                    <category>/
                        <template>/
                            image_001.png
                            ...
                <mask_folder>/
                    <category>/
                        <template>/
                            image_001.png
                            ...

    Args:
        mask_folder:  Folder name for mask images (e.g. ``'Mask'``).
        rgb_folder:   Folder name for RGB images  (e.g. ``'RGB'``).
        categories:   List of category sub-folder names.
        templates:    List of template sub-folder names.
    """

    def __init__(self, mask_folder: str, rgb_folder: str,
                 categories: list, templates: list):
        self.mask_folder = mask_folder
        self.rgb_folder  = rgb_folder
        self.categories  = categories
        self.templates   = templates

    def load_images(self, split: str, dataset_root: str) -> list:
        """
        Scan *dataset_root/split* and return a list of sample dicts.

        Samples where either the RGB image or the mask is missing are
        skipped with a warning.

        Args:
            split:         Sub-folder name, e.g. ``'Training'`` or ``'Validation'``.
            dataset_root:  Absolute path to the root of the dataset.

        Returns:
            List of dicts: ``[{'rgb': str, 'mask': str}, ...]``
        """
        samples     = []
        missing_log = []
        split_root  = os.path.join(dataset_root, split)

        for category in self.categories:
            for template in self.templates:
                mask_dir = os.path.join(split_root, self.mask_folder, category, template)
                rgb_dir  = os.path.join(split_root, self.rgb_folder,  category, template)

                if not os.path.isdir(mask_dir):
                    print(f'  WARNING: mask directory not found — {mask_dir}')
                    continue

                for mask_path in sorted(glob.glob(os.path.join(mask_dir, '*'))):
                    if not os.path.isfile(mask_path):
                        continue

                    fname    = os.path.basename(mask_path)
                    rgb_path = os.path.join(rgb_dir, fname)

                    if not os.path.isfile(rgb_path):
                        missing_log.append((fname, rgb_path))
                        continue

                    samples.append({'rgb': rgb_path, 'mask': mask_path})

        if missing_log:
            print(f'[{split}] {len(missing_log)} samples skipped (missing RGB):')
            for fname, path in missing_log[:10]:
                print(f'  {fname} → {path}')
            if len(missing_log) > 10:
                print(f'  … and {len(missing_log) - 10} more')

        print(f'[{split}] Total samples found: {len(samples)}')
        return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RGBDataset(Dataset):
    """
    PyTorch Dataset for RGB image + binary mask pairs.

    Each call to ``__getitem__`` returns a tuple::

        (rgb_t, mask_t)

    where:
        ``rgb_t``  — ``(3, H, W)`` float32 tensor, normalised to ``[-1, 1]``.
        ``mask_t`` — ``(1, H, W)`` float32 tensor, binary ``{0, 1}``.

    Args:
        samples:   List of dicts with keys ``'rgb'`` and ``'mask'``,
                   as returned by :class:`RGBDataLoader`.
        img_size:  Spatial size for resizing (both height and width).
        augment:   If ``True``, apply random flips / rotations during
                   training to improve generalisation.
    """

    def __init__(self, samples: list, img_size: int = 256, augment: bool = False):
        self.samples  = samples
        self.img_size = img_size

        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
            ], additional_targets={'mask': 'mask'})

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_rgb(self, path: str) -> np.ndarray:
        """Load image as an ``(H, W, 3)`` uint8 numpy array."""
        return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask as an ``(H, W)`` float32 array with values in ``{0, 1}``."""
        arr = np.array(Image.open(path).convert('L'), dtype=np.float32)
        return (arr > 127).astype(np.float32)

    @staticmethod
    def _to_tensor_rgb(arr: np.ndarray) -> torch.Tensor:
        """
        Convert ``(H, W, 3)`` uint8 array → ``(3, H, W)`` float32 tensor in ``[-1, 1]``.
        """
        arr = arr.astype(np.float32) / 255.0          # [0, 1]
        arr = (arr - 0.5) / 0.5                        # [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        rgb  = self._load_rgb(sample['rgb'])    # (H, W, 3)
        mask = self._load_mask(sample['mask'])  # (H, W)

        out  = self.transform(image=rgb, mask=mask)
        rgb  = out['image']  # (H, W, 3) after resize / augment
        mask = out['mask']   # (H, W)

        rgb_t  = self._to_tensor_rgb(rgb)                         # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()      # (1, H, W)

        return rgb_t, mask_t
