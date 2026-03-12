import cv2
import numpy as np
from src.features.prnu import PRNU
from src.features.frequency import Frequency
from src.features.illumination import Illumination


def extract_prnu(rgb_image: np.ndarray) -> np.ndarray:
    prnu = PRNU()
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    wavelet, means = prnu.denoise_image(img)
    residual = prnu.suppress_residual(wavelet, means)

    vis = prnu.visualize(residual)

    return vis

def extract_frequency(rgb_image: np.ndarray) -> np.ndarray:
    freq = Frequency()
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    f_spectrum = freq.compute_frequency_spectrum(img)

    return f_spectrum

def extract_illumination(rgb_image: np.ndarray, sigma, window_size) -> np.ndarray:
    illum = Illumination()
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    mean = illum.get_mean(img, sigma)
    var = illum.get_variance(img, window_size)
    combined = illum.blend(mean, var)

    return combined