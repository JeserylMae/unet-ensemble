import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from scipy.ndimage import gaussian_filter


class PRNU:
    def __init__(self, img_path):
        self.path = img_path

    def load_image(self):
        img = cv2.imread(self.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img
    
    def denoise_image(self, img):
        # Lighter NL-means — preserve more phone camera sensor noise
        img = cv2.fastNlMeansDenoising(img, h=2, templateWindowSize=5, searchWindowSize=21)
        means = img.astype(np.float32) / 255.0

        # wavelet denoising
        wavelet = denoise_wavelet(
            means,
            method='BayesShrink',
            mode='soft',
            wavelet='db4',       # db4 better for camera noise
            wavelet_levels=4,    # more levels = captures more frequency bands
            rescale_sigma=True
        )

        return wavelet, means
    
    def suppress_residual(self, wavelet, means):
        # Additive residual
        residual = means - wavelet

        # Remove Low-Frequency Leakage (increased sigma to catch more low-freq)
        # Smaller sigma — preserve high-freq phone sensor pattern
        residual = residual - gaussian_filter(residual, sigma=1.5)

        # Column suppression only — row suppression removes horizontal
        residual -= np.mean(residual, axis=0)

        return residual
    
    def visualize(self, img):
        # Clip to ±3 std — preserve more dynamic range
        # for authentic images which have wider noise distribution
        std = np.std(img)
        img = np.clip(img, -3 * std, 3 * std)

        # Stretch to full 0-255 range
        vis = img - img.min()
        vis = vis / (vis.max() + 1e-8)
        vis = (vis * 255).astype(np.uint8)

        # Lower CLAHE clip — don't over-amplify synthetic artifacts
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        vis = clahe.apply(vis)

        # Lighter gamma — preserve contrast difference between authentic and synthetic
        vis = np.power(vis / 255.0, 0.6)
        vis = (vis * 255).astype(np.uint8)

        return vis