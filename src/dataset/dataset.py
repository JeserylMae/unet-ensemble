import torch
import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    """
    Returns a tuple of tensors per sample whose contents depend on the active `features`:

        Individual feature tensors — shape (1, H, W) each — for each active feature:
            prnu_t        if 'prnu'         in features
            illu_t        if 'illumination' in features
            freq_t        if 'frequency'    in features
        fused_t       — shape (C, H, W) where C = number of active features
        mask_t        — shape (1, H, W), binary ground-truth

    The return order is always: (*active_feature_tensors, fused_t, mask_t),
    with active features in the fixed order: prnu → illumination → frequency.

    Supported feature combinations:
        ('prnu', 'illumination', 'frequency')  — returns prnu_t, illu_t, freq_t, fused_t, mask_t
        ('prnu', 'frequency')                  — returns prnu_t, freq_t, fused_t, mask_t
        ('prnu', 'illumination')               — returns prnu_t, illu_t, fused_t, mask_t
        ('frequency', 'illumination')          — returns illu_t, freq_t, fused_t, mask_t

    If augment=True, random flips and rotations are applied consistently across all feature
    images and the mask during training to improve generalisation.
    """

    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, samples, img_size=256, augment=False,
                 features=('prnu', 'illumination', 'frequency')):
        self.samples  = samples
        self.img_size = img_size
        self.features = frozenset(f.lower() for f in features)

        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]

        additional_targets = {feat: 'image' for feat in self.active_features}

        if augment:
            self.spatial = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
            ], additional_targets={**additional_targets, 'mask': 'mask'})
        else:
            self.spatial = A.Compose([
                A.Resize(img_size, img_size),
            ], additional_targets={**additional_targets, 'mask': 'mask'})

    def __len__(self):
        return len(self.samples)

    def _load_gray(self, path):
        """Load image as grayscale numpy array (H, W) uint8."""
        img = Image.open(path).convert('L').resize((self.img_size, self.img_size))
        return np.array(img, dtype=np.uint8)

    def _norm_tensor(self, arr):
        """Normalise (H,W) uint8 -> (1, H, W) float tensor in [-1, 1]."""
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load only active feature images
        raw = {}
        for feat in self.active_features:
            raw[feat] = self._load_gray(sample[feat])

        mask = np.array(
            Image.open(sample['mask']).convert('L').resize((self.img_size, self.img_size)),
            dtype=np.float32,
        )
        mask = (mask > 127).astype(np.float32)

        # albumentations requires a 3-channel 'image' as primary input.
        # Use the first active feature (repeated 3×) as the dummy primary image;
        # all features are passed as additional_targets so they receive identical transforms.
        primary_feat = self.active_features[0]
        primary_img  = np.stack([raw[primary_feat]] * 3, axis=-1)  # (H, W, 3)

        aug_kwargs = {'image': primary_img, 'mask': mask}
        for feat in self.active_features:
            # Pass each feature as a 3-channel image so albumentations accepts it
            aug_kwargs[feat] = np.stack([raw[feat]] * 3, axis=-1)

        out = self.spatial(**aug_kwargs)

        # Extract single channel from augmented outputs and build tensors
        feature_tensors = {}
        for feat in self.active_features:
            aug_arr = out[feat][:, :, 0]          # (H, W)
            feature_tensors[feat] = self._norm_tensor(aug_arr)   # (1, H, W)

        mask_aug = out['mask']                    # (H, W)

        # fused_t: stack active features in canonical order → (C, H, W)
        fused_t = torch.cat(
            [feature_tensors[f] for f in self.active_features], dim=0
        )

        mask_t = torch.from_numpy(mask_aug).unsqueeze(0).float()  # (1, H, W)

        return (
            *[feature_tensors[f] for f in self.active_features],
            fused_t,
            mask_t,
        )
   

