import torch
import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    """
    Returns **5 tensors** per sample:
        - `prnu_t`, `illu_t`, `freq_t` — shape `(1, H, W)` each, for the MBEN branches
        - `fused_t` — shape `(3, H, W)`, channel-wise concatenation for the concat stem
        - `mask_t`  — shape `(1, H, W)`, binary ground-truth

    If augment=True, random flips and rotations are also applied to the image and mask simultaneously during 
    training to improve generalization.
    """

    def __init__(self, samples, img_size=256, augment=False):
        self.samples  = samples
        self.img_size = img_size

        if augment:
            self.spatial = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
            ], additional_targets={
                'prnu': 'image', 'illu': 'image', 'freq': 'image', 'mask': 'mask'
            })
        else:
            self.spatial = A.Compose([
                A.Resize(img_size, img_size),
            ], additional_targets={
                'prnu': 'image', 'illu': 'image', 'freq': 'image', 'mask': 'mask'
            })

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
        prnu_path, illu_path, freq_path, mask_path = self.samples[idx]

        prnu = self._load_gray(prnu_path)
        illu = self._load_gray(illu_path)
        freq = self._load_gray(freq_path)
        mask = np.array(Image.open(mask_path).convert('L').resize(
            (self.img_size, self.img_size)), dtype=np.float32)  # ← add resize here
        mask = (mask > 127).astype(np.float32)

        # albumentations requires a 3-ch 'image'; use channel-wise concat as the main image
        rgb = np.stack([prnu, illu, freq], axis=-1)   # (H, W, 3)

        out = self.spatial(
            image=rgb,
            prnu =np.stack([prnu, prnu, prnu], axis=-1),  # dummy 3-ch for aug consistency
            illu =np.stack([illu, illu, illu], axis=-1),
            freq =np.stack([freq, freq, freq], axis=-1),
            mask =mask,
        )

        # Extract single channel from augmented outputs
        prnu_aug = out['prnu'][:, :, 0]   # (H, W)
        illu_aug = out['illu'][:, :, 0]
        freq_aug = out['freq'][:, :, 0]
        mask_aug = out['mask']            # (H, W)

        # Individual tensors for MBEN — (1, H, W)
        prnu_t = self._norm_tensor(prnu_aug)
        illu_t = self._norm_tensor(illu_aug)
        freq_t = self._norm_tensor(freq_aug)

        # Channel-wise fused tensor — (3, H, W)
        fused_t = torch.cat([prnu_t, illu_t, freq_t], dim=0)

        mask_t = torch.from_numpy(mask_aug).unsqueeze(0).float()   # (1, H, W)

        return prnu_t, illu_t, freq_t, fused_t, mask_t
    

