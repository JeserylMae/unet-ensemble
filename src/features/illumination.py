import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


class Illumination:
    def load_image(self, img_path, size):
        image  = cv2.imread(img_path)
        image  = cv2.resize(image, (size, size))
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float64)

        return gray_f
    
    def get_mean(self, image, sigma):
        local_mean   = gaussian_filter(image, sigma=sigma)
        local_mean_n = cv2.normalize(local_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return local_mean_n

    def get_variance(self, image, window_size):
        lm_box      = uniform_filter(image, size=window_size)
        lm_sq       = uniform_filter(image ** 2, size=window_size)
        local_var   = np.clip(lm_sq - lm_box ** 2, 0, None)
        local_var_n = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return local_var_n

    def blend(self, mean_n, var_n):
        mean_f = mean_n.astype(np.float64)
        var_f  = var_n.astype(np.float64)

        combined_raw = mean_f * var_f
        combined     = cv2.normalize(combined_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return combined
    
